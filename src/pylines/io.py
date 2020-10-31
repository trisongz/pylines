import random
import os
from . import line_count, create_idx_key, get_idx_key, get_write_fn, get_read_fn, _io_type
from . import _env, parser, json, glob, Timer
if _env['tqdm']:
    from tqdm.auto import tqdm, trange
if _env['tf']:
    from .tflow import setup_tf_serialization_features, serialize_tf_example, SerializeTFWorker, TFRWriter
    from .tflow import TFDatasetFromTensors, TFRDataset
if _env['torch']:
    from .torch import serialize_torch_example, SerializeTorchWorker, setup_torch_serialization_features
    from .torch import TorchWriter, DynamicCollate, PylinesDataset, PylinesIterableFunctionDataset, PylinesDatasetFromIterator

#if _env['ray']:
#    import ray.util.multiprocessing as mp
#else:
import math
from .logger import get_logger
import multiprocessing as mp
import hashlib
import gc
logger = get_logger()
_tokenize_fn = None

# https://stackoverflow.com/questions/620367/how-to-jump-to-a-particular-line-in-a-huge-text-file
class LineSeekableFile:
    def __init__(self, seekable):
        self.fin = seekable
        self.line_map = list() # Map from line index -> file position.
        self.line_map.append(0)
        while seekable.readline():
            self.line_map.append(seekable.tell())

    def index(self):
        return self.line_map
    
    def __len__(self):
        return len(self.line_map)

    def __getitem__(self, index):
        # NOTE: This assumes that you're not reading the file sequentially.  
        # For that, just use 'for line in file'.
        self.fin.seek(self.line_map[index])
        return self.fin.readline()


class LazyLoadFile:
    def __init__(self, filename, skip_broken=True):
        self.filename = filename
        self.reader = get_read_fn(filename)
        self._skip = skip_broken
        self.fseek = LineSeekableFile(self.reader)
        if self._skip:
            self.lineidx = 0
            self.badidx = 0
    
    def random_iter(self, num_lines=None):
        num_lines = num_lines if num_lines else len(self.fseek)
        total_idx = [i for i in range(num_lines)]
        random.shuffle(total_idx)
        for idx in total_idx:
            if self._skip:
                try:
                    yield self.loads(self.fseek[idx])
                    self.lineidx += 1
                except:
                    self.badidx += 1

            else:
                yield self.loads(self.fseek[idx])
    
    def quick_iter(self, num_lines=None):
        num_lines = num_lines if num_lines else len(self.fseek)
        for x, line in enumerate(self.reader):
            if self._skip:
                try:
                    yield self.loads(line)
                    self.lineidx += 1
                except:
                    self.badidx += 1
            else:
                yield self.loads(line)
                if x >= num_lines:
                    break

    def iter(self):
        for line in self.reader:
            if self._skip:
                try:
                    yield self.loads(line)
                    self.lineidx += 1
                except:
                    self.badidx += 1
            else:
                yield self.loads(line)

    def loads(self, v):
        return parser.parse(v).as_dict()

    def __getitem__(self, idx):
        return self.loads(self.fseek[idx])

    def __len__(self):
        return len(self.fseek)
    
    def stats(self):
        return {'loaded': self.lineidx, 'missed': self.badidx}
    
    def resetstats(self):
        self.lineidx = 0
        self.badidx = 0
        return {'loaded': self.lineidx, 'missed': self.badidx}


def setup_tokenize_fn(tokenizer_fn):
    assert _env['transformers'], 'Transformers must be installed to use tokenize function'
    global _tokenize_fn
    _tokenize_fn = tokenizer_fn

def TokenizerWorker(ex):
    try:
        result = _tokenize_fn(ex)
        return result
    except:
        return None

def setup_iter_fn(iter_fn):
    global _iter_func
    _iter_func = iter_fn

def IterWorker(ex):
    try:
        result = _iter_func(ex)
        return result
    except:
        return None

def setup_filter_fns(filter_fns):
    global _filter_func
    _filter_func = filter_fns

def FilterWorker(ex):
    result = {}
    for key in ex:
        if key not in _filter_func['bypass'] and key in _filter_func:
            res = _filter_func[key](ex[key])
            if res:
                result[key] = res
        elif key in _filter_func['bypass']:
            result[key] = ex[key]
    if bool(result):
        return None
    return result


def FileIterator(filename):
    with get_read_fn(filename) as f:
        for line in f:
            yield parser.parse(line).as_dict()
    raise StopIteration


def make_hashes(inputs):
    return hashlib.sha256(str.encode(inputs)).hexdigest()

def check_hashes(inputs, hashed_text):
	if make_hashes(inputs) == hashed_text:
		return hashed_text
	return False

class Pylines:
    def __init__(self, input_fns=None, output_fn=None, skip_broken=True, overwrite_output=False, use_lazy=False, use_mp=True, use_idx=False, total_lines=0):
        self._skip, self._lazy, self._mp, self._idx, self._overwrite = skip_broken, use_lazy, use_mp, use_idx, overwrite_output
        self.total_lines = total_lines
        self.writer, self.reader = None, None
        self.input_fns, self.output_fn = None, None
        self.stats = {}
        self.timer = Timer()
        self.stored_items = list()
        self._io(input_fns, output_fn)
    
    def as_tokenizer(self, tokenizer_fn=None, input_fns=None, use_mp=True):
        if tokenizer_fn:
            setup_tokenize_fn(tokenizer_fn)
        assert _tokenize_fn, 'tokenizer_fn must first be set before being able to run'
        self._io(input_fns, output_fn=None)
        for result in self._as_iter(_tokenize_fn, TokenizerWorker, use_mp, desc='Tokenization'):
            yield result
        logger.info(f'{self.timer.stop()} for Tokenizing {self.total_lines} Items')

    def run_tokenizer(self, tokenizer_fn=None, input_fns=None, output_fn=None, use_mp=True):
        self._io(input_fns, output_fn)
        for result in self.as_tokenizer(tokenizer_fn=tokenizer_fn, use_mp=use_mp):
            self.write(result)
        self.flush()

    def as_processor(self, iter_func=None, input_fns=None, use_mp=True):
        if iter_func:
            setup_iter_fn(iter_func)
        assert _iter_func, 'iter_func must first be set before running'
        self._io(input_fns, output_fn=None)
        for result in self._as_iter(_iter_func, IterWorker, use_mp, desc='Iterator Function'):
            yield result
        logger.info(f'{self.timer.stop()} for {self.total_lines} Items')

    def run_processor(self, iter_func=None, input_fns=None, output_fn=None, use_mp=True):
        self._io(input_fns, output_fn)
        for result in self.as_processor(iter_func=iter_func, use_mp=use_mp):
            self.write(result)
        self.flush()

    # filter_funcs = {'text': filter_fuc, 'target': filter_func, 'idx': filter_func, 'bypass': ['key_1', 'key_2']}
    def as_filter(self, filter_funcs=None, input_fns=None, use_mp=True):
        if filter_funcs:
            setup_filter_fns(filter_funcs)
        assert _filter_func, 'filter_funcs must first be set before running'
        self._io(input_fns, output_fn=None)
        for result in self._as_iter(FilterWorker, FilterWorker, use_mp, desc='Filtering Items'):
            yield result
        logger.info(f'{self.timer.stop()} for Filtering {self.total_lines} Items')
    
    def run_filter(self, filter_funcs=None, input_fns=None, output_fn=None, use_mp=True):
        self._io(input_fns, output_fn)
        for result in self.as_filter(filter_funcs=filter_funcs, use_mp=use_mp):
            self.write(result)
        self.flush()
    
    def _tftensordict(self, all_examples, dataset_features=None):
        _features = list()
        _tensor_examples = dict()
        for axis in dataset_features:
            _features += dataset_features[axis]['names']
        for feats in _features:
            _tensor_examples[feats] = list()
        for ex in all_examples:
            for key, v in ex.items():
                if key in _features:
                    _tensor_examples[key].extend(v)
        return _tensor_examples

    def _tfencoder(self, all_examples, dataset_features=None, slices=True, use_mp=True):
        if dataset_features:
            for axis in dataset_features:
                assert 'names' in dataset_features[axis], 'names is a required key for dataset features.'
            setup_tf_serialization_features(dataset_features)
        
        if slices:
            _tensor_ds = self._tftensordict(all_examples, dataset_features)
            return _tensor_ds
            
        else:
            for serialized_ex in self._as_iter_items(all_examples, serialize_tf_example, SerializeTFWorker, use_mp=use_mp, desc=f'Serializing to TFRecords'):
                yield serialized_ex

    def _tfwriter(self, all_examples, output_dir, dataset_features=None, start_idx=1, split_key='split', split='train', write_string='{}_shard_{}.tfrecords', shard_size=50000, overwrite=False, use_tempdir=False, use_mp=True):
        _total_match = self.count_matching(split_key, split) if split_key else self.total_lines
        with TFRWriter(output_dir, _total_match, start_idx, split, write_string, shard_size, overwrite, use_tempdir) as writer:
            for serialized_ex in self._tfencoder(all_examples, dataset_features, slices=False, use_mp=use_mp):
                writer.write(serialized_ex)
        
        tfrecord_files, total_items = writer.close()
        return tfrecord_files, total_items

    def _torchencoder(self, all_examples, dataset_features=None, use_mp=True):
        if dataset_features:
            setup_torch_serialization_features(dataset_features)

        for serialized_ex in self._as_iter_items(all_examples, serialize_torch_example, SerializeTorchWorker, use_mp=use_mp, desc=f'Serializing to Torch'):
            yield serialized_ex
    
    def _torchwriter(self, all_examples, output_dir, dataset_features=None, start_idx=1, split_key='split', split='train', write_string='{}_shard_{}.pkl', shard_size=50000, overwrite=False, use_tempdir=False, use_mp=True, compression=True):
        _total_match = self.count_matching(split_key, split) if split_key else self.total_lines
        with TorchWriter(output_dir, _total_match, start_idx, split, write_string, shard_size, overwrite, use_tempdir, compression) as writer:
            for serialized_ex in self._torchencoder(all_examples, dataset_features, use_mp):
                writer.write(serialized_ex)
        
        torch_files, total_items = writer.close()
        return torch_files, total_items

    def _tokenize_examples(self, tokenizer_fn, use_mp=True):
        all_results = list()
        if tokenizer_fn:
            for result in self.as_tokenizer(tokenizer_fn, use_mp=use_mp):
                all_results.append(result)
        else:
            logger.warning(f'No Tokenizer Function Provided. Assuming Input Files are Pretokenized.')
            for result in self.as_iterator():
                all_results.append(result)
            logger.info(f'Loaded {len(all_results)} Examples. Keys: {list(i for i in all_results[0])}')
        return all_results

    def as_encoder(self, dataset_features=None, tokenizer_fn=None, serialization='tf', input_fns=None, use_mp=True):
        _methods = ['tf', 'torch']
        assert serialization in _methods, f'Currently only {_methods} are supported'
        assert _env[serialization], f'{serialization} library is required to run Serialization'
        self._io(input_fns, output_fn=None)
        all_examples = self._tokenize_examples(tokenizer_fn, use_mp)
        if serialization == 'tf':
            for serialized_ex in self._tfencoder(all_examples, dataset_features, use_mp):
                yield serialized_ex
        elif serialization == 'torch':
            for serialized_ex in self._torchencoder(all_examples, dataset_features, use_mp):
                yield serialized_ex
       
        logger.info(f'{self.timer.stop()} for Serializing [{serialization}] {len(all_examples)} Examples')

    def run_encoder(self, output_dir, dataset_features=None, tokenizer_fn=None, serialization='tf', input_fns=None, start_idx=1, split_key='split', split='train', write_string='{}_shard_{}.tfrecords', shard_size=50000, overwrite=False, use_tempdir=False, use_mp=True, compression=True):
        self._io(input_fns, output_fn=None)
        all_examples = self._tokenize_examples(tokenizer_fn, use_mp)
        if serialization == 'tf':
            tfrecord_files, total_items = self._tfwriter(all_examples, output_dir, dataset_features, start_idx, split_key, split, write_string, shard_size, overwrite, use_tempdir, use_mp)
            return tfrecord_files, total_items
        elif serialization == 'torch':
            torch_files, total_items = self._torchwriter(all_examples, output_dir, dataset_features, start_idx, split_key, split, write_string, shard_size, overwrite, use_tempdir, use_mp, compression)
            return torch_files, total_items

    def as_dataset(self, batch_sizes, dataset_features=None, tokenizer_fn=None, framework='tf', input_fns=None, split_key='split', splits=['train', 'validation', 'test'], use_mp=True):
        self._io(input_fns, output_fn=None)
        all_examples = self._tokenize_examples(tokenizer_fn, use_mp)
        _dataset = dict()
        _encoder_fn = self._tfencoder if framework == 'tf' else None
        if splits:
            _splitdataset = self._dataset_splits(all_examples, split_key, splits)
            for split in splits:
                if _encoder_fn:
                    _encoded_examples = list()
                    for example in _encoder_fn(_splitdataset[split], dataset_features, use_mp):
                        _encoded_examples.append(example)
                else:
                    _encoded_examples = _splitdataset[split]
                _dataset[split] = {'examples': _encoded_examples, 'batch_size': batch_sizes[split]}
            _splitdataset = None
            gc.collect()
        else:
            if _encoder_fn:
                _encoded_examples = list()
                for example in _encoder_fn(all_examples, dataset_features, use_mp):
                    _encoded_examples.append(example)
            else:
                _encoded_examples = all_examples
            _dataset['train'] = {'examples': _encoded_examples, 'batch_size': batch_sizes}
            splits = ['train']
        
        if framework == 'tf':
            _tfdataset = TFDatasetFromTensors(_dataset, dataset_features)
            return _tfdataset
        elif framework == 'torch':
            _torchdataset = dict()
            for split in splits:
                _torchdataset[split] = PylinesDataset(num_examples=len(_dataset[split]['examples']), examples=_dataset[split]['examples'])
            logger.info('Torch Dataset should be used with DynamicCollate function with the DataLoader for Optimal Performance')
            return _torchdataset

    def _dataset_splits(self, all_examples, split_key, splits):
        split_results = dict()
        for split in splits:
            split_results[split] = list()
        for example in all_examples:
            ex_split = example[split_key]
            split_results[ex_split].append(example)
        return split_results

    def _as_iter(self, IterFunc, Worker, use_mp, desc):
        pbar = trange(self.total_lines, desc=desc) if _env['tqdm'] else None
        self.timer.start(desc)
        if use_mp:
            if isinstance(use_mp, int):
                pool = mp.Pool(use_mp)
            else:
                pool = mp.Pool()
            for fn in self.input_fns:
                for result in pool.imap_unordered(Worker, FileIterator(fn)):
                    if result:
                        yield result
                    if pbar:
                        pbar.update()
            
        else:
            for fn in self.input_fns:
                for result in self._file_iter(fn):
                    ex = IterFunc(result)
                    if ex:
                        yield ex
                    if pbar:
                        pbar.update()
        if pbar:
            pbar.close()

    def _as_iter_items(self, items, IterFunc, Worker, use_mp, desc):
        pbar = trange(len(items), desc=desc) if _env['tqdm'] else None
        self.timer.start(desc)
        if use_mp:
            if isinstance(use_mp, int):
                pool = mp.Pool(use_mp)
            else:
                pool = mp.Pool()
            for result in pool.imap_unordered(Worker, items):
                if result:
                    yield result
                if pbar:
                    pbar.update()
            
        else:
            for item in items:
                ex = IterFunc(item)
                if ex:
                    yield ex
                if pbar:
                    pbar.update()
        if pbar:
            pbar.close()

    def deduplicate(self, keys, input_fns=None, output_fn=None, write=True):
        self._io(input_fns, output_fn)
        _sets = {}
        results = list()
        assert _io_type(keys) == 'list', 'Keys must be in the form of a list'
        for key in keys:
            _sets[key] = set()
        for result in self.as_iterator():
            _pass = True
            for k in keys:
                hashed_key = make_hashes(result[k])
                if hashed_key in _sets[k]:
                    _pass = False
                else:
                    _sets[k].add(hashed_key)
            if _pass:
                if write:
                    self.write(result)
                else:
                    results.append(result)
        
        if not write:
            return results

    def find(self, key, value, results='first', filename=None, verbose=False):
        assert results in ['first', 'all'], 'Results should either be all or first to return'
        _matched_results = list()
        _matched = False
        if filename:
            for x, result in enumerate(self._fast_file_iter(filename)):
                if result[key] == value:
                    _matched_results.append((result if _io_type(result) == 'dict' else result.as_dict()))
                    _matched = True
                
                if _matched:
                    if verbose:
                        logger.info(f'Found Match on IDX: {x}')
                    if results == 'first':
                        break
                    elif results == 'all':
                        _matched = False

                
        else:
            for fn in self.input_fns:
                for x, result in enumerate(self._fast_file_iter(fn)):
                    if result[key] == value:
                        _matched_results.append((result if _io_type(result) == 'dict' else result.as_dict()))
                        _matched = True
                        if results == 'first':
                            break
                
                    if _matched:
                        if verbose:
                            logger.info(f'Found Match on IDX: {x} in {fn}')
                        if results == 'first':
                            break
                        elif results == 'all':
                            _matched = False

                if _matched and results == 'first':
                    break
        
        return _matched_results
    
    def merge(self, input_fns=None, output_fn=None):
        self._io(input_fns, output_fn)
        pbar = trange(self.total_lines, desc=f'Merging {len(self.input_fns)} Files') if _env['tqdm'] else None
        self.timer.start(f'Merging {len(self.input_fns)} Files')
        for result in self.as_iterator():
            self.write(result)
            if pbar:
                pbar.update()
        self.flush()
        if pbar:
            pbar.close()
        logger.info(f'{self.timer.stop()} with {self.total_lines} Items to {self.output_fn}')
        
    def count_matching(self, key, value, input_fns=None):
        _matches = 0
        self._io(input_fns)
        for result in self.as_iterator():
            if result[key] == value:
                _matches += 1
        
        return _matches

    def verify_linecount(self):
        _idx = 0
        for result in self.as_iterator():
            if result and result != '':
                _idx += 1

        logger.info(f'Original Count: {self.total_lines} Verified Count: {_idx}')
        if _idx >= self.total_lines:
            self.total_lines = _idx
        return _idx

    def write(self, item):
        if not self.writer:
            assert self.output_fn, 'Output File must be set prior to write. call .writefile(filename) to set the output file'
            self.writer_fn = get_write_fn(self.output_fn, overwrite=self._overwrite)
            self.writer = self.writer_fn.write
            self._writeidx = 0
            self.flushidx = math.ceil(self.total_lines / 10)
        
        self.writer(json.dumps(item, ensure_ascii=False))
        self.writer('\n')
        self._writeidx += 1
        if self._writeidx % self.flushidx == 0:
            self.writer_fn.flush()
            self._writeidx = 0

    def index(self, idx, fn=None):
        if fn:
            return self._file_idx(idx, fn)
        else:
            results = {}
            for fn in self.input_fns:
                results[fn] = self._file_idx(idx, fn)
            return results
    
    def to_dict(self, input_fns=None):
        results = {}
        self._io(input_fns)
        for result in self.as_iterator():
            for key in result:
                if key not in results:
                    results[key] = {'items': [result[key]], 'count': 1}
                else:
                    results[key]['items'].append(result[key])
                    results[key]['count'] += 1
        
        return results

    def to_list(self, input_fns=None):
        results = list()
        self._io(input_fns)
        for result in self.as_iterator():
            results.append(result)
        return results

    def from_list(self, all_items, write=True, clear_items=False, output_fn=None):
        self._io(input_fns=None, output_fn=output_fn)
        assert _io_type(all_items) == 'list', 'This function must be used with a list'
        if write:
            logger.info(f'Writing {len(all_items)} to {self.output_fn}')
            for item in all_items:
                self.write(item)
        else:
            if clear_items:
                logger.info(f'Flushing Existing Items from Memory')
                self.stored_items = None
                gc.collect()
                self.stored_items = list()
            self.stored_items += all_items
            logger.info(f'Stored {len(all_items)} to Memory. Total Stored Items: {len(self.stored_items)}. Call pylines.stored_items to access items.')

    def to_batches(self, all_items=None, batch_size=2):
        all_batches = list()
        all_items = all_items if all_items else self.stored_items
        for batch in self._build_batches(all_items, batch_size):
            all_batches += [batch]
        return all_batches

    def as_iterator(self):
        for fn in self.input_fns:
            for item in self._file_iter(fn):
                yield item


    def _build_batches(self, lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]

    def _file_idx(self, idx, fn):
        reader = get_read_fn(fn)
        for x, line in enumerate(reader):
            if x < idx:
                pass
            elif x == idx:
                return self.loads(line)
        reader.close()

    def _file_iter(self, fn):
        reader = get_read_fn(fn)
        if self._skip:
            self._reset_stats(fn)
        for line in reader:
            if self._skip:
                try:
                    yield self.loads(line)
                    self.stats[fn]['read'] += 1
                except:
                    self.stats[fn]['missed'] += 1
            else:
                yield self.loads(line)
        reader.close()

    def _fast_file_iter(self, fn):
        reader = get_read_fn(fn)
        if self._skip:
            self._reset_stats(fn)
        for line in reader:
            if self._skip:
                try:
                    yield self.loads(line)
                    self.stats[fn]['read'] += 1
                except:
                    self.stats[fn]['missed'] += 1

            else:
                yield self.loads(line)
        reader.close()

    def _get_file_lines(self, input_fns=None):
        if input_fns:
            for fn in input_fns:
                self.total_lines += line_count(fn)
        else:
            self.total_lines = 0
            for fn in self.input_fns:
                self.total_lines += line_count(fn)

    def __iter__(self):
        for fn in self.input_fns:
            for result in self._file_iter(fn):
                yield result
    
    def _io(self, input_fns=None, output_fn=None):
        if input_fns:
            self._setup_input_fns(input_fns)
            self._get_file_lines()
        if output_fn:
            self._setup_output_fn(output_fn)

    def _setup_input_fns(self, input_fns):
        in_files = []
        if _io_type(input_fns) == 'str':
            if input_fns.endswith('*'):
                in_files = glob(input_fns)
            else:
                in_files = [input_fns]
        elif _io_type(input_fns) == 'list':
            for fn in input_fns:
                if fn.endswith('*'):
                    in_files += glob(fn)
                else:
                    in_files.append(fn)
        else:
            raise ValueError('Input Filenames should be a string or list')
        
        if self.input_fns:
            in_files = [f for f in in_files if f not in self.input_fns]
            if len(in_files) != 0:
                self.input_fns += in_files
        else:
            self.input_fns = in_files
    
    def _setup_output_fn(self, output_fn):
        if self.writer:
            self.writer_fn.flush()
            self.writer_fn.close()
            self.writer, self.writer_fn = None, None
        if _io_type(output_fn) == 'str':
            self.output_fn = output_fn
        else:
            raise ValueError('Output Filenames should be a string')

    def parse(self, v):
        return parser.parse(v)

    def loads(self, v):
        if _io_type(v) == 'bytes':
            return parser.parse(v).as_dict()
        else:
            return json.loads(v)

    def load(self, v):
        return json.load(v)
    
    def dumps(self, v):
        return json.dumps(v, ensure_ascii=False)

    def dump(self, fn, v):
        if _io_type(fn) == 'str':
            json.dump(v, get_write_fn(fn, overwrite=self._overwrite))
        else:
            json.dump(v, fn)
    
    def _reset_stats(self, fn):
        self.stats[fn] = {'read': 0, 'missed': 0}

    def clear_input_files(self):
        self.stats = {}
        self.input_fns = None

    def set_input_files(self, input_fns):
        self.clear_input_files()
        self._setup_input_fns(input_fns)

    def add_files(self, input_fns):
        self._setup_input_fns(input_fns)

    def set_writefile(self, output_fn, overwrite=False):
        self._overwrite = overwrite
        self._setup_output_fn(output_fn)

    def close(self):
        if self.writer:
            self.writer_fn.flush()
            self.writer_fn.close()
        if self.reader:
            self.reader.close()

    def flush(self):
        if self.writer:
            self.writer_fn.flush()

    def linecount(self, filename=None):
        results = {}
        if filename:
            results[filename] = line_count(filename)
            return results
        for fn in self.input_fns:
            results[fn] = line_count(fn)
        return results

    def __len__(self):
        return self.total_lines        

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()