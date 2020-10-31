from . import _file_exists, _env, get_read_fn, get_write_fn, Timer, _io_type
import tensorflow as tf
from threading import Thread
import tempfile
import math
import os
if _env['tqdm']:
    from tqdm.auto import tqdm, trange
import random
from .logger import get_logger

logger = get_logger()
AUTO = tf.data.experimental.AUTOTUNE

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

def serialize_tf_example(ex):
    feature = {feat: _int64_feature(ex[feat]) for feat in _tf_example_features}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def SerializeTFWorker(ex):
    result = serialize_tf_example(ex)
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
            self._files = list()

        else:
            self.writer(x)
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

''' args for TFRDataset
datasets = {
    'train': {
        'path': '/path/to/train/',
        'batch_size': 8,
    },
    'validation': {
        'path': 'gs://bucket/dataset/val/,
        'batch_size:' 16,
    },
    'test': {
        'path': '/path/to/test/',
        'batch_size': 12,
    }
}
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

def find_max(seq, curr_max):
    return max(len(seq), curr_max)

def find_maxes(seqs):
    m = 0
    for seq in seqs:
        m = find_max(seq, m)
    return m

def setup_detokenize_fn(detokenizer_fn):
    assert _env['transformers'], 'Transformers must be installed to use detokenize function'
    global _detokenize_fn
    _detokenize_fn = detokenizer_fn

def DetokenizerWorker(ex):
    result = _detokenize_fn(ex)
    return result

'''
tensor_datasets : {
    'train': {
        'examples': dataset,
        'batch_size': 8
    },
    'validation': {
        'examples': dataset,
        'batch_size': 8
    },
    'test': {
        'examples': dataset,
        'batch_size': 8
    }
}
'''

def count_tfdataset(dataset):
    idx = tf.data.experimental.cardinality(dataset).numpy()
    if _io_type(idx) == 'int':
        return idx

    idx = 0
    ds_iter = next(iter(dataset))
    while True:
        try:
            ds_iter()
            idx += 1
        except StopIteration:
            break
    return idx

class TFDatasetFromTensors:
    def __init__(self, datasets, dataset_features):
        self._setup_dataset_mapping(datasets, dataset_features)
        
    def _setup_dataset_mapping(self, datasets, dataset_features):
        self.axis, self._features, self._datafields, self.columns, self.splits = list(), dict(), dict(), dict(), list() 
        self._basedataset = {}
        for i, axis in enumerate(dataset_features):
            self.axis.append(axis)
            self.columns[axis] = dataset_features[axis]
            self._features[axis] = dataset_features[axis]['names']
            self._basedataset[axis] = {}
        self.num_axis = len(self.axis)
        self.datasets = datasets
        for split in datasets:
            self.splits.append(split)
            self.datasets[split]['examples'] = self._map_split(datasets[split]['examples'])
    
    def _map_split(self, dataset):
        _ds = {}
        for x in self.axis:
            _ds[x] = {i: dataset[i] for i in feat in self._features[x]}
            #for feat in self._features[x]:
            #    _ds[x][feat] = list()
        #for ex in dataset:
        #    for key, v in ex.items():
        #        for x in self.axis:
        #            if key in self._features[x]:
        #                _ds[x][key] += [v]
        
        #logger.info(f'Feats: {self._features}, Num Axis: {self.num_axis}')
        #logger.info(f'columns: {self.columns}, Base Dataset Dict: {self._basedataset}')
        return tuple((_ds[x]) for x in _ds)
    
    def get_dataset(self, split, shuffle=True, ordered=False, num_devices=1, return_total=True):
        _dataset = self.datasets[split]['examples']
        _batchsize = self.datasets[split]['batch_size'] * num_devices
        logger.info(f'- Building TF Dataset From Tensors for {split} with {_batchsize} Batch Size')
        ignore_order = tf.data.Options()
        if not ordered:
            ignore_order.experimental_deterministic = False # disable order, increase speed
        
        dataset = tf.data.Dataset.from_tensor_slices(_dataset)
        dataset = dataset.with_options(ignore_order)
        if shuffle:
            dataset = dataset.shuffle(2048)
        dataset = dataset.batch(_batchsize, drop_remainder=False)
        logger.info('Dataset Features')
        for x in self.axis:
            logger.info(f'{x} [{self.columns[x]["length"]}]: {self.columns[x]["names"]}')
        dataset = dataset.prefetch(buffer_size=AUTO)
        if return_total:
            num_items = count_tfdataset(dataset)
            return dataset, num_items
        return dataset


class TFRDataset:
    def __init__(self, datasets, dataset_features, tokenizer=None, validate=True, validate_num=5, verbose=False):
        self.validate, self._numvalidate, self.verbose = validate, validate_num, verbose
        self.tokenizer = tokenizer
        self._setup_dataset_mapping(datasets, dataset_features)
        
    def _setup_dataset_mapping(self, datasets, dataset_features):
        self.axis, self._features, self._datafields, self.columns, self.splits, self._tfrfiles = list(), dict(), dict(), dict(), list(), dict()
        self._baserecord = {}
        for i, axis in enumerate(dataset_features):
            self.axis.append(axis)
            self.columns[axis] = dataset_features[axis]
            self._datafields[axis] = dataset_features[axis]['names']
            self._features[i] = dataset_features[axis]['names']
            self._baserecord[i] = {}
        self.num_axis = len(self.axis)
        self.datasets = datasets
        for split in datasets:
            self.splits.append(split)
            if not self.datasets[split]['path'].endswith('/'):
                self.datasets[split]['path'] = self.datasets[split]['path'] + '/'
            self.datasets[split]['globpath'] = os.path.join(self.datasets[split]['path'], (split + '*'))
            self._tfrfiles[split] = tf.io.gfile.glob(self.datasets[split]['globpath'])

    def get_dataset(self, split, shuffle=True, ordered=False, num_devices=1, return_total=True):
        _dspath = self.datasets[split]['path']
        _tfrfiles = self._tfrfiles[split]
        _batchsize = self.datasets[split]['batch_size'] * num_devices
        assert len(_tfrfiles) != 0, 'Dataset loaded zero files. Something went wrong loading this file batch. Check File Paths.'
        logger.info(f'- Building TFRecords Dataset for {split}: {len(_tfrfiles)} files from {_dspath} with {_batchsize} Batch Size')
        if shuffle:
            random.shuffle(_tfrfiles)
        if len(_tfrfiles) >= 10:
            logger.info(f'Displaying First 10 Records{" [shuffled]" if shuffle else " "}')
        _displayfiles = _tfrfiles[:10] if len(_tfrfiles) >= 10 else _tfrfiles
        for _f in _displayfiles:
            _fn = _f.replace(_dspath, '')
            logger.info(f'- {_fn}')
        logger.info('')
        ignore_order = tf.data.Options()
        if not ordered:
            ignore_order.experimental_deterministic = False # disable order, increase speed
        dataset = tf.data.TFRecordDataset(_tfrfiles, num_parallel_reads=AUTO)
        dataset = dataset.with_options(ignore_order)
        if shuffle:
            dataset = dataset.shuffle(2048)
        dataset = dataset.map(self.parse_example, num_parallel_calls=AUTO)
        dataset = dataset.map(self.transform_record, num_parallel_calls=AUTO)
        dataset = dataset.batch(_batchsize, drop_remainder=False)
        logger.info('Dataset Features')
        for x in self.axis:
            logger.info(f'{x} [{self.columns[x]["length"]}]: {self.columns[x]["names"]}')

        dataset = dataset.prefetch(buffer_size=AUTO)
        if self.validate:
            self.validate_dataset(dataset, split)
        if return_total:
            num_items = count_tfdataset(dataset)
            return dataset, num_items

        return dataset


    def transform_record(self, record):
        _rec = self._baserecord
        for name in record:
            for i in range(self.num_axis):
                if name in self._features[i]:
                    _rec[i][name] = record[name]

        return (_rec[i] for i in self.axis)

    def parse_example(self, serialized_example):
        example = tf.io.parse_single_example(serialized_example, self._datafields)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t
        return example

    def validate_dataset(self, dataset, split):
        idx, valid = 0, 0
        for item in dataset.take(self._numvalidate):
            for i in range(self.num_axis):
                _axis = self.axis[i]
                for req in self._features[i]:
                    assert req in item[i].keys(), f'{req} is not in data {_axis}, your data keys: {item[i].keys()}'
                    assert len(item[i][req][0]) == self.columns[_axis]['length'], f"{req} dimension size is {len(item[i][req][0])}, not matching {self.columns[_axis]['length']}"
                    if idx == 0:
                        logger.info(f'IDX: {idx} [{req}] - Max Seq Length: {find_maxes(item[i][req])}, {len(item[i][req])}')
                        if self.verbose:
                            logger.info(f'({_axis} axis) {req} -> {item[i][req]}')
                            logger.info('\n')
            
                        if self.tokenizer and req == 'input_ids':
                            _batch = item[i][req]
                            for batch in _batch:
                                logger.info(f'(detokenized) -> {self.tokenizer.decode(batch)}')
                                logger.info('\n')


            logger.info('****' * 20)
            logger.info(f'IDX: {idx} Passed All Validation Checks')
            valid += 1
            idx += 1
        
        logger.info('****' * 20)
        logger.info(f'Dataset [{split}]: Passed {valid} Validation Checks')
        logger.info('****' * 20)