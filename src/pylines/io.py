import random
import os
from . import line_count, create_idx_key, get_idx_key, get_write_fn, get_read_fn, _io_type
from . import _env, parser, json, glob, Timer
if _env['tqdm']:
    from tqdm.auto import tqdm, trange
if _env['tf']:
    from .tflow import setup_tf_serialization_features, serialize_example, SerializeWorker, TFRWriter

#if _env['ray']:
#    import ray.util.multiprocessing as mp
#else:
from .logger import get_logger
import multiprocessing as mp
import hashlib
logger = get_logger()

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
    result = _tokenize_fn(ex)
    return result

def setup_iter_fn(iter_fn):
    global _iter_func
    _iter_func = iter_fn

def IterWorker(ex):
    result = _iter_func(ex)
    return result

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
    
    def as_encoder(self, dataset_features=None, tokenizer_fn=None, serialization='tf', input_fns=None, use_mp=True):
        _methods = ['tf']
        assert serialization in _methods, f'Currently only {_methods} are supported'
        assert _env['tf'], 'Tensorflow is required to run Serialization'
        if dataset_features:
            for axis in dataset_features:
                assert 'names' in dataset_features[axis], 'names is a required key for dataset features.'
            setup_tf_serialization_features(dataset_features)

        self._io(input_fns, output_fn=None)
        all_results = list()
        if _tokenize_fn or tokenizer_fn:
            for result in self.as_tokenizer(tokenizer_fn, use_mp=use_mp):
                all_results.append(result)
        else:
            logger.warning(f'No Tokenizer Function is Found. Assuming Input Files are Pretokenized.')
            for result in self.as_iterator():
                all_results.append(result)
        
        for serialized_ex in self._as_iter_items(all_results, serialize_example, SerializeWorker, use_mp=use_mp, desc=f'Serializing to {serialization}'):
            yield serialized_ex
        logger.info(f'{self.timer.stop()} for Serializing {len(all_results)} Examples')

    def run_encoder(self, output_dir, dataset_features=None, tokenizer_fn=None, serialization='tf', input_fns=None, start_idx=1, split_key='split', split='train', write_string='{}_shard_{}.tfrecords', shard_size=50000, overwrite=False, use_tempdir=False, use_mp=True):
        self._io(input_fns, output_fn=None)
        _total_match = self.count_matching(split_key, split) if split_key else self.total_lines
        with TFRWriter(output_dir, _total_match, start_idx, split, write_string, shard_size, overwrite, use_tempdir) as writer:
            for serialized_ex in self.as_encoder(dataset_features, tokenizer_fn, use_mp=use_mp):
                writer.write(serialized_ex)
        
        #writer.close()


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
                    yield result
                    if pbar:
                        pbar.update()
            
        else:
            for fn in self.input_fns:
                for result in self._file_iter(fn):
                    ex = IterFunc(result)
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
                yield result
                if pbar:
                    pbar.update()
            
        else:
            for item in items:
                ex = IterFunc(item)
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

    def find(self, key, value, results='first', filename=None):
        assert results in ['first', 'all'], 'Results should either be all or first to return'
        _matched = False
        if filename:
            for x, result in enumerate(self._fast_file_iter(filename)):
                if result[key] == value:
                    yield result.as_dict()
                    _matched = True
                
                if _matched:
                    print(f'Found Match on IDX: {x}')
                    if results == 'first':
                        break
                    elif results == 'all':
                        _matched = False

                
        else:
            for fn in self.input_fns:
                for x, result in enumerate(self._fast_file_iter(fn)):
                    if result[key] == value:
                        yield result.as_dict()
                        _matched = True
                        if results == 'first':
                            break
                
                    if _matched:
                        print(f'Found Match on IDX: {x} in {fn}')
                        if results == 'first':
                            break
                        elif results == 'all':
                            _matched = False

                if _matched and results == 'first':
                    break
    
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
        
        self.writer(json.dumps(item, ensure_ascii=False))
        self.writer('\n')

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

    def from_list(self, all_items, output_fn=None):
        self._io(input_fns=None, output_fn=output_fn)
        assert _io_type(all_items) == 'list', 'This function must be used with a list'
        for item in all_items:
            self.write(item)

    def to_batches(self, all_items, batch_size=2):
        all_batches = list()
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
        if _io_type(output_fn) == 'str':
            self.output_fn = output_fn
        else:
            raise ValueError('Output Filenames should be a string')


    def parse(self, v):
        return parser.parse(v)

    def loads(self, v):
        return parser.parse(v).as_dict()

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