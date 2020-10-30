from . import _file_exists, _env, get_read_fn, get_write_fn, Timer
from . import save_pkle, load_pkle, load_data, save_data, iterator_function
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from threading import Thread
from itertools import chain, cycle
import tempfile
import math
import gc
import os
import glob
if _env['tqdm']:
    from tqdm.auto import tqdm, trange
if _env['numpy']:
    import numpy as np
if _env['tf']:
    import tensorflow as tf
import random
from .logger import get_logger

logger = get_logger()

def trim_batch(input_ids, pad_token_id, attention_mask=None):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])

def pad_seq(seq, max_batch_len, pad_value):
    return seq + (max_batch_len - len(seq)) * [pad_value]

class TorchIO:
    def __init__(self, outfile=None, tmpfile=None, copy=False, gcopy=False, remove_tmp=True, compress=True):
        self._outfile, self._rm, = outfile, remove_tmp
        self._tmpfile = tmpfile
        self._copy, self._gcopy = copy, gcopy
        self.filename = self._tmpfile if self._tmpfile else self._outfile
        self._gcopy = False if not _env['tf'] else self._gcopy
        self._compress = compress
        self.fn = get_write_fn(self.filename, require='smartopen') if compress else get_write_fn(self.filename)

    def write(self, x):
        save_data(x, self.fn)

    def close(self):
        self.fn.flush()
        self.fn.close()
        if (self._gcopy or self._copy) and self._tmpfile:
            _threaded_copy = Thread(target=self._copydaemon,)
            _threaded_copy.start()

    def _copydaemon(self):
        if self._gcopy:
            tf.io.gfile.copy(self._tmpfile, self._outfile, overwrite=True)
            if self._rm:
                if tf.io.gfile.exists(self._outfile):
                    tf.io.gfile.remove(self.filename)
                else:
                    logger.info('Cannot confirm copied file exists. Not removing tmp file')

        elif self._copy:
            require = 'smartopen' if self._compress else None
            with get_read_fn(self.filename, require=require) as read_fn:
                with get_write_fn(self._outfile, require=require) as write_fn:
                    data = load_data(read_fn)
                    save_data(data, write_fn)

            write_fn.flush()
            write_fn.close()
            read_fn.close()
            data = None
            gc.collect()
            if self._rm:
                os.remove(self.filename)
        logger.info(f'Copied {self.filename} -> {self._outfile}')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        logger.info(f"Closing File  - {self.filename}")
        self.close()


class TorchWriter(object):
    def __init__(self, dir_path, num_examples, start_idx=1, split='train', write_string='{}_shard_{}.pkl', shard_size=50000, overwrite=False, use_tempdir=False, compression=True):
        self._init_stats()
        self._dir, self._total, self._split, self._fstring,  self._shardsize, self._overwrite = dir_path, num_examples, split, write_string, shard_size, overwrite
        self._startidx = start_idx
        self._usetmp = use_tempdir
        self._compress = compression
        self._cache = list()
        self._shardsize = self._total if self._total < self._shardsize else self._shardsize
        self._numfiles = math.ceil(self._total / self._shardsize)
        self.pbar = trange(self._total, desc=f"[{self._split}] Torch Writer", dynamic_ncols=True) if _env['tqdm'] else None
        self._files = list()
        self._setup_files()
        self.fn = self.openfile()
        self.filename = self.fn.filename
        self.writer = self.fn.write
        self.timer = Timer()
        logger.info(f"- Writer Config for Split [{self._split}] -> Shard Size/Num Ex per Cache: {self._shardsize}, Total Items: {self._total}, Write Path: {self._dir}, Number of Files: {self._numfiles}")
        self.timer.start('Torch Writer')

    def write(self, x):
        self._cache.append(x)
        if self.curr_idx >= self._shardsize:
            self.writer(self._cache)
            self._clearcache()
            self.close()
            self.shard += 1
            self.idx += 1
            self.fn = self.openfile()
            self.filename = self.fn.filename
            self.writer = self.fn.write

        else:
            self.idx += 1
            self.curr_idx += 1
        
        if self.pbar:
            self.pbar.update()

    def close(self):
        logger.info(f'Closing. Shard {self.shard}/{self._numfiles}\nFilename: {self.filename}\n{self.idx} Examples written in {self.timer.time()}.\nRemaining Items: {self._total - self.idx}/{self._total}.')        
        try:
            self.fn.close()
        except:
            pass
        return self._files, self.idx
    
    def openfile(self):
        self.curr_idx = 0
        filename = self._fstring.format(self._split, self.shard)
        if self._compress:
            filename = filename + '.gz'
        _outfile = os.path.join(self._dir, filename)
        _tmpfile = os.path.join(self._tmpdir, filename) if self._tmpdir else None
        _file = TorchIO(outfile=_outfile, tmpfile=_tmpfile, copy=self._copy, gcopy=self._gcopy, remove_tmp=True, compress=self._compress)
        if self.pbar:
            self.pbar.set_description(f"[{self._split}] Torch Writer: {filename}")
        self._files.append(_file.filename)
        return _file

    def _init_stats(self):
        self.idx = 0
        self.curr_idx = 0
        self.shard = 1

    def _setup_files(self):
        self._tmpdir = None
        self._existing_fns = None
        self._copy, self._gcopy = False, False
        if self._dir.startswith('s3') or self._usetmp:
            self._tmpdir = tempfile.gettempdir()
            self._copy = True
            if self._usetmp and self._dir.startswith('gs://'):
                self._gcopy = True
        
        if self._dir.startswith('/') or self._dir.startswith('gs://'):
            if self._dir.startswith('gs://'):
                assert _env['tf'], 'Tensorflow should be installed to use gs files to ensure directory is set up properly'
                _exists = tf.io.gfile.exists
                _mkdir = tf.io.gfile.makedirs
                _glob = tf.io.gfile.glob
            else:
                _exists = os.path.exists
                _mkdir = os.makedirs
                _glob = glob.glob

            if self._dir.endswith('/'):
                self._dir = self._dir[:-1]
            if not _exists(self._dir):
                _mkdir(self._dir)
            _fpath = os.path.join(f'{self._dir}/{self._split}*')
            if _exists(self._dir):
                self._existing_fns = _glob(_fpath)
                logger.info(f'Existing Files: {self._existing_fns}')
            else:
                logger.info(f'{_fpath} does not exist. Creating Directory.')
                _mkdir(self._dir)

        if self._startidx > 1:
            self.shard = self._startidx
            logger.info(f' Manually setting Shard IDX to {self.shard}')
        elif not self._overwrite and self._existing_fns:
            self.shard = len(self._existing_fns) + 1
            logger.info(f' Setting Shard IDX to {self.shard}')
        elif self._overwrite:
            logger.info(f'Overwrite is enabled. Will Start at Shard IDX 1')
    
    def _clearcache(self):
        self._cache = None
        gc.collect()
        self._cache = list()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        logger.info(f"- Torch Writer is Closing")
        self.close()


_all_keys_base = {
    'input_ids': torch.long,
    'attention_mask': torch.long,
    'token_type_ids': torch.long,
    'labels': torch.float
}
_trim_batch_key = {
    0: 'input_ids',
    1: 'attention_mask'
}

def setup_torch_serialization_features(dataset_features):
    global _torch_example_features
    _torch_example_features = dataset_features

def serialize_torch_example(ex):
    for key in ex:
        ex[key] = torch.tensor(ex[key], dtype=_torch_example_features[key])
    return ex

def SerializeTorchWorker(ex):
    result = serialize_torch_example(ex)
    return result

def _to_list(lst):
    if isinstance(lst, list):
        return lst
    elif isinstance(lst, torch.Tensor):
        array = lst.numpy()
        lst = list(array)
        return lst
    elif isinstance(lst, numpy.ndarray):
        return lst.tolist()
    else:
        return lst

class DynamicCollate:
    def __init__(self, pad_token_id=0, all_keys=_all_keys_base, input_ids_key='input_ids', no_pad=['labels'], trim_batches=_trim_batch_key, lm_labels='labels'):
        self.pad_token_id = pad_token_id
        self._datakeys, self._inputkey, self._nopad, self._trim, self._lmlabels = all_keys, input_ids_key, no_pad, trim_batches, lm_labels
        self._basedict = {}
        for key in self._datakeys:
            self._basedict[key] = list()
        if not self._nopad:
            self._nopad = []
    
    def __call__(self, batch):
        batch_dict = {}
        for key in self._datakeys:
            batch_dict[key] = list()
        max_size = max([len(example[self._inputkey]) for example in batch])
        #for key in self._datakeys:
        for example in batch:
            for key, v in example.items():
                if key not in self._datakeys:
                    pass
                elif key == self._inputkey:
                    batch_dict[key] += [pad_seq(_to_list(v), max_size, self.pad_token_id)]
                elif self._nopad and key in self._nopad:
                    batch_dict[key] += [_to_list(v)]
                else:
                    batch_dict[key] += [pad_seq(_to_list(v), max_size, 0)]
        
        if self._trim:
            batch_dict[self._trim[0]], batch_dict[self._trim[1]] = trim_batch(batch_dict[self._trim[0]], self.pad_token_id, attention_mask=batch_dict[self._trim[1]])
        
        if self._lmlabels:
            lmlabels = np.ndarray(batch_dict[self._lmlabels])
            lmlabels[lmlabels[:, :] == self.pad_token_id] = -100
            batch_dict[self._lmlabels] = lmlabels

        for key in batch_dict:
            batch_dict[key] = torch.tensor(batch_dict[key], dtype=self._datakeys[key])

        return batch_dict

def load_from_cache_file(filename):
    require = 'smartopen' if filename.endswith('.gz') else None
    with get_read_fn(filename, require=require) as read_fn:
        return load_data(read_fn)

#def create_dataloader(dataset, tokenizer, batch_size):


class PylinesDataset(Dataset):
    def __init__(self, num_examples, examples=None, load_fn=None, cache_files=None, inital_load=2):
        self.examples = examples if examples else list()
        self._total, self._files = num_examples, cache_files
        self._loadfn = load_fn
        self.idx, self.cacheidx = 0, 0
        self._loadcache, self._loadfunction = False, False
        if not examples and self._loadfn:
            self._loadeditems = 0
            self._runloadfunction()
            logger.info(f'Loaded {self._loadeditems} Examples from Load Function')

        if self._loadfn and len(self.examples) < self._total:
            self._loadfunction = True
            self._loadidx = math.ceil((self._total / len(self.examples)) * .7)
            logger.info(f'Will Run Load Function every {self._loadidx} Examples')
            
        elif not examples and self._files:
            self._loadcache = True
            self._loadidx = math.ceil((self._total / len(self._files)) * .7)
            self._loadeditems = 0
            inital_load = len(self._files) if inital_load > len(self._files) else inital_load
            for i in range(inital_load):
                self._buildcache()

    def _buildcache(self):
        examples = load_from_cache_file(self._files[self.cacheidx])
        self._loadeditems += len(examples)
        self.examples += examples
        examples = None
        gc.collect()
        if self.cacheidx == 0:
            logger.info(f'Loaded First Cache File with {self._loadeditems}/{self._total} Examples in Memory')
            logger.info(f'Will Load Cache every {self._loadidx} Examples')

        self.cacheidx += 1
        if self.cacheidx >= len(self._files):
            self._loadcache = False
        elif self._loadeditems >= self._total:
            self._loadcache = False

    def _runloadfunction(self):
        examples = self._loadfn()
        self._loadeditems += len(examples)
        self.examples += examples
        examples = None
        gc.collect()
        if self._loadeditems >= self._total:
            self._loadfunction = False

    def __getitem__(self, idx):
        _idx = self.idx if (self._loadcache or self._loadfunction) else idx
        self.idx += 1
        if self._loadcache or self._loadfunction:
            if self.idx % self._loadidx == 0:
                target = self._buildcache if self._loadcache else self._runloadfunction
                _threaded_loader = Thread(target=target,)
                _threaded_loader.start()

        return self.examples[_idx]

    def __len__(self):
        return self._total



class PylinesIterableFunctionDataset(IterableDataset):
    def __init__(self, num_examples, iter_function, lazy=True):
        self.examples = list()
        self._total = num_examples
        self._iter = iter_function
        self.idx, self.iteridx = 0, 0
        self.lazy = lazy
        if not self.lazy:
            self._load_all_examples()
    
    def _iter_examples(self, examples):
        for ex in examples:
            yield ex

    def _create_iter_from_examples(self, examples):
        return chain.from_iterable(map(self._iter_examples, cycle(examples)))

    def _create_iter_from_function(self, func):
        return iterator_function(func)

    def _load_all_examples(self):
        for ex in self._iter():
            try:
                self.examples += ex
                self.iteridx += 1
            except StopIteration:
                break
        if self.iteridx < self._total:
            self._total = self.iteridx

    def __iter__(self):
        if self.lazy:
            return self._create_iter_from_function(self._iter)
        return self._create_iter_from_examples(self.examples)
            
    def __len__(self):
        return self._total

class PylinesDatasetFromIterator(Dataset):
    def __init__(self, num_examples, iter_function, batch_iter=.2):
        self.examples = list()
        self._total = num_examples
        self._iter = iter_function
        self.idx, self.iteridx = 0, 0
        self.lazy = False if batch_iter == 0 else True
        self._iteractive = True
        if not self.lazy:
            logger.info('Loading all Examples')
            self._load_all_examples()
        elif self.lazy:
            self._numiters = math.ceil(self._total * batch_iter)
            logger.info(f'Loading {self._numiters} Examples Per Call')
            self._iter_batch()
    
    def _iter_batch(self):
        for i in range(self._numiters):
            try:
                self.examples += next(self._iter())
                self.iteridx += 1
            except StopIteration:
                self._iteractive = False

    def _load_all_examples(self):
        for ex in self._iter():
            try:
                self.examples += ex
                self.iteridx += 1
            except StopIteration:
                break
        logger.info(f'Loaded {self.iteridx}/{self._total} Examples')
        if self.iteridx < self._total:
            logger.info(f'Setting new Total Examples to {self.iteridx}')
            self._total = self.iteridx
        self._iteractive = False
        
    def __getitem__(self, idx):
        _idx = self.idx if self._iteractive else idx
        if _idx > self._total:
            _idx = self._total
        self.idx += 1
        if self._iteractive:
            if self.idx % self._numiters == 0:
                self._iter_batch()
        return self.examples[_idx]

    def __len__(self):
        return self._total

