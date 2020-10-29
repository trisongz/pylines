from . import _file_exists, _env, get_read_fn, get_write_fn, Timer
import tensorflow as tf
from threading import Thread
import tempfile
import math
import os
if _env['tqdm']:
    from tqdm.auto import tqdm, trange

from .logger import get_logger
logger = get_logger()

'''
dataset_features = {
        'x': {
            'length': 416,
            'names': ["input_ids", "attention_mask"]
        },
        'y': {
            'length': 96,
            'names': ["target_ids", "target_attention_mask", "shifted_target_input_ids"]
        }
    }
'''


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def setup_tf_serialization_features(dataset_features):
    global _tf_example_features, _tf_dataset_features
    _tf_dataset_features = dataset_features
    _tf_example_features = []
    for axis in dataset_features:
        _tf_example_features += dataset_features[axis]['names']

def serialize_example(ex):
    feature = {feat: _int64_feature(ex[feat]) for feat in _tf_example_features}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def SerializeWorker(ex):
    result = serialize_example(ex)
    return result


class TFileIO:
    def __init__(self, outfile=None, tmpfile=None, copy=False, gcopy=False, remove_tmp=True):
        self._outfile, self._rm, = outfile, remove_tmp
        self._tmpfile = tmpfile
        self._copy, self._gcopy = copy, gcopy
        if self._tmpfile:
            self.filename = self._tmpfile
        else:
            self.filename = self._outfile
        self.fn = tf.io.TFRecordWriter(self.filename)
        self.writer = self.fn.write

    def write(self, x):
        self.writer(x)

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
            with get_write_fn(self._outfile) as write_fn:
                data = tf.io.read_file(self.filename)
                write_fn.write(data)
                write_fn.flush()
            write_fn.close()
            if self._rm:
                tf.io.gfile.remove(self.filename)
        logger.info(f'Copied {self.filename} -> {self._outfile}')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        logger.info(f"Closing File  - {self.filename}")
        self.close()



class TFRWriter(object):
    def __init__(self, dir_path, num_examples, start_idx=1, split='train', write_string='{}_shard_{}.tfrecords', shard_size=50000, overwrite=False, use_tempdir=False):
        self._init_stats()
        self._dir, self._total, self._split, self._fstring,  self._shardsize, self._overwrite = dir_path, num_examples, split, write_string, shard_size, overwrite
        self._startidx = start_idx
        self._usetmp = use_tempdir
        self._shardsize = self._total if self._total < self._shardsize else self._shardsize
        self._numfiles = math.ceil(self._total / self._shardsize)
        self.pbar = trange(self._total, desc=f"[{self._split}] TFRecords Writer", dynamic_ncols=True) if _env['tqdm'] else None
        self._setup_files()
        self.fn = self.openfile()
        self.filename = self.fn.filename
        self.writer = self.fn.write
        self.timer = Timer()
        logger.info(f"- Writer Config for Split [{self._split}] -> Shard Size/Num Ex per Record: {self._shardsize}, Total Items: {self._total}, Write Path: {self._dir}, Number of Files: {self._numfiles}")
        self.timer.start('TFRecords Writer')

    def write(self, x):
        if self.curr_idx >= self._shardsize:
            self.writer(x)
            self.close()
            self.shard += 1
            self.idx += 1
            self.fn = self.openfile()
            self.filename = self.fn.filename
            self.writer = self.fn.write

        else:
            self.writer(x)
            self.idx += 1
            self.curr_idx += 1
        
        if self.pbar:
            self.pbar.update()

    def close(self):
        logger.info(f'Closing. Shard {self.shard}/{self._numfiles}\nFilename: {self.filename}\n{self.idx} Examples written in {self.timer.time()}.\nRemaining Items: {self._total - self.idx}/{self._total}.')        
        self.fn.close()    
    
    def openfile(self):
        self.curr_idx = 0
        filename = self._fstring.format(self._split, self.shard)
        _outfile = os.path.join(self._dir, filename)
        _tmpfile = os.path.join(self._tmpdir, filename) if self._tmpdir else None
        _file = TFileIO(outfile=_outfile, tmpfile=_tmpfile, copy=self._copy, gcopy=self._gcopy, remove_tmp=True)
        if self.pbar:
            self.pbar.set_description(f"[{self._split}] TFRecords Writer: {filename}")
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
            if self._dir.endswith('/'):
                self._dir = self._dir[:-1]
            if not tf.io.gfile.exists(self._dir):
                tf.io.gfile.makedirs(self._dir)
            _fpath = os.path.join(f'{self._dir}/{self._split}*')
            if tf.io.gfile.exists(self._dir):
                self._existing_fns = tf.io.gfile.glob(_fpath)
                logger.info(f'Existing Files: {self._existing_fns}')
            else:
                logger.info(f'{_fpath} does not exist. Creating Directory.')
                tf.io.gfile.makedirs(self._dir)

        if self._startidx > 1:
            self.shard = self._startidx
            logger.info(f' Manually setting Shard IDX to {self.shard}')
        elif not self._overwrite and self._existing_fns:
            self.shard = len(self._existing_fns) + 1
            logger.info(f' Setting Shard IDX to {self.shard}')
        elif self._overwrite:
            logger.info(f'Overwrite is enabled. Will Start at Shard IDX 1')
        

    def __enter__(self):
        return self

    def __exit__(self, *_):
        logger.info(f"- TFRecords Writer is Closing")
        self.close()