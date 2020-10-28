import random
import os
from . import line_count, create_idx_key, get_idx_key, get_write_fn, get_read_fn, _io_type
from . import _env, parser, json, glob, Timer
if _env['tqdm']:
    from tqdm.auto import tqdm, trange
#if _env['ray']:
#    import ray.util.multiprocessing as mp
#else:
from .logger import get_logger
import multiprocessing as mp

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
    global tokenize_fn
    tokenize_fn = tokenizer_fn

def TokenizerWorker(ex):
    result = tokenize_fn(ex)
    return result

def setup_iter_fn(iter_fn):
    global _iter_func
    _iter_func = iter_fn

def IterWorker(ex):
    result = _iter_func(ex)
    return result

def FileIterator(filename):
    with get_read_fn(filename) as f:
        for line in f:
            yield parser.parse(line).as_dict()
    raise StopIteration


class Pylines:
    def __init__(self, input_fns=None, output_fn=None, skip_broken=True, overwrite_output=False, use_lazy=False, use_mp=True, use_idx=False, total_lines=0):
        self._skip, self._lazy, self._mp, self._idx, self._overwrite = skip_broken, use_lazy, use_mp, use_idx, overwrite_output
        self.total_lines = total_lines
        self.writer, self.reader = None, None
        self.input_fns, self.output_fn = None, None
        self.stats = {}
        self.timer = Timer()
        if input_fns:
            self._setup_input_fns(input_fns)
            
        if output_fn:
            self._setup_output_fn(output_fn)
    
    def tokenize(self, tokenizer_fn, return_results=False, input_fns=None, output_fn=None, use_mp=True):
        setup_tokenize_fn(tokenizer_fn)
        if input_fns:
            self._setup_input_fns(input_fns)
        if output_fn:
            self._setup_output_fn(output_fn)

        pbar = trange(self.total_lines, desc='Tokenization') if _env['tqdm'] else None
        self.timer.start('Tokenization')
        if use_mp:
            if isinstance(use_mp, int):
                pool = mp.Pool(use_mp)
            else:
                pool = mp.Pool()
            for fn in self.input_fns:
                for result in pool.imap_unordered(TokenizerWorker, FileIterator(fn)):
                    if return_results:
                        yield result
                    else:
                        self.write(result)
                    if pbar:
                        pbar.update()
                self.flush()
            
        else:
            for fn in self.input_fns:
                for result in self._file_iter(fn):
                    ex = tokenizer_fn(result)
                    if return_results:
                        yield ex
                    else:
                        self.write(ex)
                    if pbar:
                        pbar.update()
                self.flush()
        if pbar:
            pbar.close()
        logger.info(f'{self.timer.stop()} for {self.total_lines} Items')

    def run_function(self, iter_func, return_results=False, input_fns=None, output_fn=None, use_mp=True):
        setup_iter_fn(iter_func)
        if input_fns:
            self._setup_input_fns(input_fns)
        if output_fn:
            self._setup_output_fn(output_fn)

        pbar = trange(self.total_lines, desc='Iterator Function') if _env['tqdm'] else None
        self.timer.start('Iterator Function')
        if use_mp:
            if isinstance(use_mp, int):
                pool = mp.Pool(use_mp)
            else:
                pool = mp.Pool()
            for fn in self.input_fns:
                for result in pool.imap_unordered(IterWorker, FileIterator(fn)):
                    if return_results:
                        yield result
                    else:
                        self.write(result)
                    if pbar:
                        pbar.update()
                self.flush()
            
        else:
            for fn in self.input_fns:
                for result in self._file_iter(fn):
                    ex = iter_func(result)
                    if return_results:
                        yield ex
                    else:
                        self.write(ex)
                    if pbar:
                        pbar.update()
                self.flush()
        if pbar:
            pbar.close()
        logger.info(f'{self.timer.stop()} for {self.total_lines} Items')

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
        if input_fns:
            self._setup_input_fns(input_fns)
        if output_fn:
            self._setup_output_fn(output_fn)
        pbar = trange(self.total_lines, desc=f'Merging {len(self.input_fns)} Files') if _env['tqdm'] else None
        self.timer.start(f'Merging {len(self.input_fns)} Files')
        for result in self.iter():
            self.write(result)
            if pbar:
                pbar.update()
        self.flush()
        if pbar:
            pbar.close()
        logger.info(f'{self.timer.stop()} with {self.total_lines} Items to {self.output_fn}')
        

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
            for fn in self.input_fns:
                self.total_lines += line_count(fn)

    def __iter__(self):
        for fn in self.input_fns:
            for result in self._file_iter(fn):
                yield result
    

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
            self.input_fns += in_files
            self._get_file_lines(in_files)
        else:
            self.input_fns = in_files
            self._get_file_lines()
    
    def _setup_output_fn(self, output_fn):
        if _io_type(output_fn) == 'str':
            self.output_fn = output_fn
        else:
            raise ValueError('Output Filenames should be a string')

    def iter(self):
        for fn in self.input_fns:
            for item in self._file_iter(fn):
                yield item

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
        #self.lineidx = 0
        #self.badidx = 0
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