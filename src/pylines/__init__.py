__version__ = "0.0.4"

import platform

try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

import os
import subprocess
import time

import simdjson as json

parser = json.Parser()

try:
    import tensorflow as tf

    glob = tf.io.gfile.glob
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    _tf_available = True
except ImportError:
    import glob

    glob = glob.glob
    _tf_available = False

try:
    import redis

    _redis_available = True
except ImportError:
    _redis_available = False

try:
    import transformers

    _transformers_available = True
except ImportError:
    _transformers_available = False

try:
    import smart_open

    _smartopen_available = True
except ImportError:
    _smartopen_available = False

try:
    import ray

    _ray_available = True
except ImportError:
    _ray_available = False

try:
    import torch

    _torch_available = True
except ImportError:
    _torch_available = False

try:
    import tqdm.auto

    _tqdm_available = True
except ImportError:
    _tqdm_available = False

try:
    from google.cloud import storage
    from google.oauth2 import service_account

    USE_GCS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", None)
    if USE_GCS:
        if os.path.exists(USE_GCS):
            gcp_client = service_account.Credentials.service_account_info = json.load(open(USE_GCS, 'r'))
            gcp_storage_client = storage.Client.from_service_account_json(USE_GCS)
            _gcs_available = True
        else:
            gcp_client, gcp_storage_client = None, None
            _gcs_available = False
except ImportError:
    gcp_client, gcp_storage_client = None, None
    _gcs_available = False

try:
    import boto3

    _S3_ID = os.environ.get("AWS_ACCESS_KEY_ID", None)
    _S3_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", None)
    _BOTO_PATH_1 = os.environ.get("AWS_SHARED_CREDENTIALS_FILE", None)
    _BOTO_PATH_2 = os.path.join(os.environ.get("HOME", "/root"), ".aws/credentials")
    _BOTO_PATH_2_EXISTS = os.path.exists(_BOTO_PATH_2)
    _BOTO_PATH_3 = os.environ.get("BOTO_CONFIG", None)

    _BOTO_PATH_4 = os.environ.get("BOTO_PATH", None)
    _BOTO_PATH_5 = os.path.join(os.environ.get("HOME", "/root"), ".boto")
    _BOTO_PATH_5_EXISTS = os.path.exists(_BOTO_PATH_5)
    s3_client = None

    if _S3_ID and _S3_KEY:
        s3_client = boto3.Session(aws_access_key_id=_S3_ID, aws_secret_access_key=_S3_KEY)
    elif _BOTO_PATH_1 or _BOTO_PATH_2_EXISTS or _BOTO_PATH_3:
        s3_client = boto3.Session()
    elif _BOTO_PATH_4:
        os.environ['BOTO_CONFIG'] = _BOTO_PATH_4
        s3_client = boto3.Session()
    elif _BOTO_PATH_5_EXISTS:
        os.environ['BOTO_CONFIG'] = _BOTO_PATH_5
        s3_client = boto3.Session()

    if s3_client:
        _s3_available = True
    else:
        _s3_available = False
except ImportError:
    _s3_available = False
    s3_client = None

_env = {
    'gcs': _gcs_available,
    's3': _s3_available,
    'transformers': _transformers_available,
    'smartopen': _smartopen_available,
    'torch': _torch_available,
    'tf': _tf_available,
    'ray': _ray_available,
    'tqdm': _tqdm_available,
}


def auth_cloud(gcs=None, s3=None):
    # expects a json file for gcs
    global _gcs_available, _s3_available, s3_client, gcp_client, gcp_storage_client
    if gcs:
        try:
            gcp_client = service_account.Credentials.service_account_info = json.load(open(gcs, 'r'))
            gcp_storage_client = storage.Client.from_service_account_json(gcs)
            _gcs_available = True
        except Exception as e:
            print(f'Unable to Authenticate GCS: {str(e)}')
            _gcs_available = False

    if s3:
        if _io_type(s3) == 'str' and os.path.exists(s3):
            os.environ['AWS_SHARED_CREDENTIALS_FILE'] = s3
            try:
                s3_client = boto3.Session()
                _s3_available = True
            except Exception as e:
                print(f'Unable to Authenticate S3: {str(e)}')
                _s3_available = False
        elif _io_type(s3) == 'dict':
            if 'aws_access_key_id' in s3 and 'aws_secret_access_key' in s3:
                try:
                    s3_client = boto3.Session(aws_access_key_id=s3['aws_access_key_id'],
                                              aws_secret_access_key=s3['aws_secret_access_key'])
                    _s3_available = True
                except Exception as e:
                    print(f'Unable to Authenticate S3: {str(e)}')
                    _s3_available = False
            elif 'aws_access_key_id'.upper() in s3 and 'aws_secret_access_key'.upper() in s3:
                try:
                    s3_client = boto3.Session(aws_access_key_id=s3['aws_access_key_id'.upper()],
                                              aws_secret_access_key=s3['aws_secret_access_key'.upper()])
                    _s3_available = True
                except Exception as e:
                    print(f'Unable to Authenticate S3: {str(e)}')
                    _s3_available = False
            elif 'AWS_SHARED_CREDENTIALS_FILE' in s3:
                os.environ['AWS_SHARED_CREDENTIALS_FILE'] = s3
                try:
                    s3_client = boto3.Session()
                    _s3_available = True
                except Exception as e:
                    print(f'Unable to Authenticate S3: {str(e)}')
                    _s3_available = False
    else:
        _S3_ID = os.environ.get("AWS_ACCESS_KEY_ID", None)
        _S3_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", None)
        _S3_PATH = os.environ.get("AWS_SHARED_CREDENTIALS_FILE", None)
        _GCS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", None)
        if not _s3_available:
            if _S3_ID and _S3_KEY:
                try:
                    s3_client = boto3.Session(aws_access_key_id=_S3_ID, aws_secret_access_key=_S3_KEY)
                    _s3_available = True
                except Exception as e:
                    print(f'Unable to Authenticate S3: {str(e)}')
                    _s3_available = False
            elif _S3_PATH:
                try:
                    os.environ['AWS_SHARED_CREDENTIALS_FILE'] = _S3_PATH
                    s3_client = boto3.Session()
                    _s3_available = True
                except Exception as e:
                    print(f'Unable to Authenticate S3: {str(e)}')
                    _s3_available = False
        if not _gcs_available and _GCS_PATH:
            try:
                gcp_client = service_account.Credentials.service_account_info = json.load(open(_GCS_PATH, 'r'))
                gcp_storage_client = storage.Client.from_service_account_json(_GCS_PATH)
                _gcs_available = True
            except Exception as e:
                print(f'Unable to Authenticate GCS: {str(e)}')
                _gcs_available = False

    if _gcs_available:
        print('Authenticated GCS')
    else:
        gcp_client = None
        gcp_storage_client = None

    if _s3_available:
        print('Authenticated S3')
    else:
        s3_client = None


def get_read_fn(filename):
    if filename.startswith('gs://'):
        if _tf_available:
            return tf.io.gfile.GFile(filename, 'rb+')
        elif _smartopen_available:
            if _gcs_available:
                return smart_open.open(filename, 'rb', transport_params=dict(client=gcp_storage_client))
            else:
                return smart_open.open(filename, 'rb')
        else:
            raise ValueError('Tensorflow and SmartOpen are not available to open a GCS File')
    elif filename.startswith('s3://'):
        if _smartopen_available:
            if _s3_available:
                return smart_open.open(filename, 'rb', transport_params={'session': s3_client})
            else:
                return smart_open.open(filename, 'rb')
        else:
            raise ValueError('SmartOpen is not available to open a S3 File')
    elif filename.startswith('https://'):
        if _smartopen_available:
            return smart_open.open(filename, 'rb')
        else:
            raise ValueError('SmartOpen is not available to open a HTTP File')
    elif filename.startswith('hdfs://') or filename.startswith('webhdfs://'):
        if _smartopen_available:
            return smart_open.open(filename, 'rb')
        else:
            raise ValueError('SmartOpen is not available to open a HDFS File')
    else:
        if _tf_available:
            return tf.io.gfile.GFile(filename, 'rb+')
        elif _smartopen_available:
            return smart_open.open(filename, 'rb')
        else:
            return open(filename, 'rb')


def _file_exists(filename):
    if _tf_available:
        return tf.io.gfile.exists(filename)
    else:
        return os.path.exists(filename)


def get_write_fn(filename, overwrite=False):
    f_exists = _file_exists(filename)
    if overwrite:
        _write_mode = 'wb'
    elif f_exists:
        _write_mode = 'ab'
    else:
        _write_mode = 'wb'
    if _tf_available:
        _write_mode = _write_mode + '+'
    if filename.startswith('gs://'):
        if _tf_available:
            return tf.io.gfile.GFile(filename, _write_mode)
        elif _smartopen_available:
            if _gcs_available:
                return smart_open.open(filename, _write_mode, transport_params=dict(client=gcp_storage_client))
            else:
                return smart_open.open(filename, _write_mode)
        else:
            raise ValueError('Tensorflow and SmartOpen are not available to open a GCS File')
    elif filename.startswith('s3://'):
        if _smartopen_available:
            if _s3_available:
                return smart_open.open(filename, _write_mode, transport_params={'session': s3_client})
            else:
                return smart_open.open(filename, _write_mode)
        else:
            raise ValueError('SmartOpen is not available to open a S3 File')
    elif filename.startswith('https://'):
        if _smartopen_available:
            return smart_open.open(filename, _write_mode)
        else:
            raise ValueError('SmartOpen is not available to open a HTTP File')
    elif filename.startswith('hdfs://') or filename.startswith('webhdfs://'):
        if _smartopen_available:
            return smart_open.open(filename, _write_mode)
        else:
            raise ValueError('SmartOpen is not available to open a HDFS File')
    else:
        if _tf_available:
            return tf.io.gfile.GFile(filename, _write_mode)
        elif _smartopen_available:
            return smart_open.open(filename, _write_mode)
        else:
            return open(filename, _write_mode)


def line_count(filename):
    if platform.system() == "Windows":
        out = subprocess.check_output(["type", filename, "|", "find", "/c", "/v", ""], shell=True)
    else:
        out = subprocess.check_output(["wc", "-l", filename])
    return int(out.split()[0])


def create_idx_key(filename, line):
    idx = {i: key for i, key, v in enumerate(line.items())}
    idx_fn = (filename.split('.')[-1].strip() + '.idx')
    json.dump(idx, get_write_fn(idx_fn), overwrite=True)
    return idx


def get_idx_key(filename):
    idx_fn = (filename.split('.')[-1].strip() + '.idx')
    if _file_exists(idx_fn):
        idx_key = json.loads(get_read_fn(idx_fn))
        idx = {v: key for key, v in idx_key.items()}
        return idx
    else:
        return None


def _io_type(x):
    if isinstance(x, list):
        return 'list'
    elif isinstance(x, dict):
        return 'dict'
    elif isinstance(x, bool):
        return 'bool'
    elif isinstance(x, str):
        return 'str'
    elif isinstance(x, int):
        return 'int'
    elif isinstance(x, float):
        return 'float'


class Timer:
    def __init__(self):
        self.idx = 0
        self.active = False

    def start(self, task):
        if self.active:
            self.stop()

        self.task = task
        self.start_timer()
        self.active = True

    def start_timer(self):
        self.start_time = time.time()
        self.ckpt_time = time.time()
        self.total_time = 0

    def time(self):
        stop_time = time.time()
        time_string = f'- Current Time for {self.task}: {(stop_time - self.start_time) / 60:.2f} mins / {(stop_time - self.start_time) :.2f} secs'
        return time_string

    def secs(self):
        stop_time = time.time()
        s = (stop_time - self.start_time)
        return s

    def stop(self):
        stop_time = time.time()
        time_string = f'- Total Time for {self.task}: {(stop_time - self.start_time) / 60:.2f} mins / {(stop_time - self.start_time) :.2f} secs'
        self.start_time = 0
        return time_string


from . import logger
from . import io
from .io import Pylines, LazyLoadFile, LineSeekableFile
