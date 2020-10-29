# Pylines
Simplifying parsing of large jsonline files in NLP Workflows (but you can use it for anything)

## Highlights
- Memory Efficient Processing of Large (GB+) Jsonline files
- High performance through use of iterators, multiprocessing, and simdjson
- Deterministic open_function, allowing r/w to cloud storage objects (s3,gs)
- Removes need to handle dumps, newlines, flush when writing to file
- Minimal dependencies (only simdjson)
- Allows passing of functions directly to iterators w/ multiprocessing, writing to file or returning results

## Quickstart

#### Quick Installation + Various Flavors
```shell

# only installs pysimdjson as the requirement
pip install --upgrade pylines 

 # Installs tensorflow, smart_open[all], google-cloud-storage
pip install --upgrade pylines[cloud]

# Installs above + transformers, torch
pip install --upgrade pylines[all] 

```

#### Source Installation (probably a bad idea.)
```shell
pip install --upgrade git+https://github.com/trisongz/pylines
```


#### Why Pylines

After using numerous data formats, I came to appreciate jsonlines due to its transparency (being able to look at the data without it being in a binary format), as well as its ability to handle text-based serialization when compared to other data formats that often break. Pylines is used extensively in my private projects and is being released as a standalone project since I almost always use it when dealing with data.

Pylines is designed to be simplistic and deterministic, dealing with things such as:

- open function: will determine open based on the filename prefix, including from gs, s3, https, and hfds
- r/w/a: will determine write mode based on detecting a file present, unless overwrite is called
- broken lines: by default, will skip any unparsable lines rather than returning an error, which can be a pain in the middle of a large data processing batch, and then having to figure out how to resume
- flexibility in adding multiple files: deals with strings, lists, and/or globs
- flushing the file: will automatically flush file buffers before closing to prevent unwritten files when writing to storage objects

Pylines is backed by [pysimdjson](https://github.com/TkTech/pysimdjson), a python binding around [simdjson](https://github.com/lemire/simdjson), allowing high performance read/write ops, as well as being highly memory efficient when dealing with extremely large files (30GB+) through extensive use of generators.

Some additional current features include:

- Multi-threaded Tokenization from input files
- Tensorflow Records Writer with Tokenization
- Object Storage r/w support
- Useful iterators/writers that take external functions, and switching from one to another easily.
- A LazyLoadFile class (wip) that allows for sampling of the dataset without loading everything into memory
- functions that allow retrieving a line by index or with a key=value match


#### Example of Usage
```python3
'''
Base Default Variables for Pylines

input_fns: None, if no input file is set, can be later set or added. by calling , adding to existing files or  to override existing list of files, and set to the new files.
output_fn: None, if no output file is set, can be later set or added. - if no output file is set, can be later set by calling 
skip_broken: True, will skip unparsable lines by default
overwrite_output: False, will control overwriting of the output file
use_lazy: False, doesn't do anything right now
use_mp: True, doesn't explicitly do anything right now
use_idx: False, doesn't do anything right now
total_lines: 0, used to set the total lines expected, useful if you already know how many items are in your file and prevent counting of items during initialization.

'''
from pylines import Pylines
input_fn = 'myinputfile.json' # or ['inputfile1.json', 'inputfile2.json']

# Quick Iterator through all items
for item in Pylines(input_fn):
    # process items here
    print(item)

# Can be initialized as an object
pylines = Pylines(input_fn)
# returns total count of all lines in all input files
print(len(pylines))

# Can set output file and then iterate through input files, and writing to output file
output_file = 'myoutputfile.json' # only takes a string right now
pylines.set_writefile(output_file)

write = pylines.write
for item in pylines.as_iterator():
    # process items here
    write(item)

# alternatively, you can pass a function and run an iterator with mp, and have it write to the output file
pylines.run_function(processor_function)

# or get the results from the processor to do something else
for result in pylines.as_function(processor_function):
    # do something with results
```


#### Built in Generators and Writers
Pylines has several built in iterators that follow the same pattern, making it simple to switch between a generator and writing to a file

```python3

from pylines import Pylines

input_fn = 'myinputfile.json'
output_fn = 'tokenized_file.json'

# as_* indicates a generator, yielding results
# run_* indicates a writer, no returns

pylines = Pylines(input_file, output_file)

# Assuming defined functions

for result in pylines.as_tokenizer(tokenizer_function):
    yield result

# Then to write it all to file after verifying the results

pylines.run_tokenizer(tokenizer_function)

# Other functions

# Maps each line in the file to a custom defined processor function. expects a result
pylines.as_processor(custom_processor_function) -> Generator
pylines.run_processor(custom_processor_function) -> Writer

# Takes a filter function for each key, with 'bypass' being keys that are ignored.
# As such, since the function can also return a different value, the output from the filter function will update the value
# Filter function must either return None or a value.
# Example:
# filter_funcs = {'text': text_filter_fuc, 'target': text_processor_func, 'idx': filter_idx_func, 'bypass': ['key_1', 'key_2']}

pylines.as_filter(custom_processor_function) -> Generator
pylines.run_filter(custom_processor_function) -> Writer

# Encoder only supports tensorflow records at this time.

# Options 
# output_dir, dataset_features=None, tokenizer_fn=None, serialization='tf',
# input_fns=None, start_idx=1, split_key='split', split='train', 
# write_string='{}_shard_{}.tfrecords', shard_size=50000, overwrite=False, 
# use_tempdir=False, use_mp=True

# Requires an output_dir, dataset_features, and tokenizer_fn if not passed previously.
# For gs backed object storage, you can write directly to gs. If it's s3, it will create a temporary dir, and upon completion, run a background
# Thread to copy/move the file to s3.
# You can explicitly set use_tempdir=True to use local_storage during write, and it will write to local first.

# Write String requires two placeholders, which is formatted using 'split' and the current shard count.
# If files are detected in the dir, the shard count will automatically be the next shard number, preventing overwrites, unless explicitly called, or start_idx!=1.

# if multiple splits exist in the dataset, you can set split_key to the key that your split is defined as, and split to the value.
# It will run a filter to find all the items that match split_key=split before encoding.
# If no split_key exists in the dataset, set split_key=None which will process the entire dataset without filtering

# Expected Dataset Features Format:

dataset_features = {
        'x': {
            'names': ["input_ids", "attention_mask"]
        },
        'y': {
            'names': ["target_ids", "target_attention_mask", "shifted_target_input_ids"]
        }
    }

pylines.as_encoder(dataset_features=dataset_features, tokenizer_fn=tokenize_fn, split_key=None, use_mp=1) -> Generator

# Only output_dir is required for run_encoder
pylines.run_encoder(dataset_features=dataset_features, tokenizer_fn=tokenize_fn, output_dir=output_dir, split_key=None, use_mp=1) -> Writer

```


#### Example of Tokenization
```python3
from transformers import T5Tokenizer
from pylines import Pylines
import numpy as np

tokenizer = T5Tokenizer.from_pretrained('t5-base')
input_fn = 'myinputfile.json'
output_fn = 'tokenized_file.json'

def shift_to_right(input_ids, decoder_start_token_id):
    shifted_input_ids = np.zeros(input_ids.shape, dtype=input_ids.dtype)
    shifted_input_ids[..., 1:] = input_ids[..., :-1]
    shifted_input_ids[..., 0] = decoder_start_token_id
    return shifted_input_ids

def tokenize_fn(example):
    encoder_tokens = tokenizer(example['source_text'], truncation='longest_first', max_length=416, padding='max_length', add_special_tokens=True)
    decoder_tokens = tokenizer(example['target_text'], truncation='longest_first', max_length=96, padding='max_length', add_special_tokens=True)
    target_input_ids = np.copy(decoder_tokens['input_ids'])

    shifted_target_input_ids = shift_to_right(target_input_ids, tokenizer.pad_token_id)
    target_input_ids[target_input_ids == tokenizer.pad_token_id] = -100

    return {
        'input_ids': encoder_tokens['input_ids'],
        'attention_mask': encoder_tokens['attention_mask'],
        'target_ids': target_input_ids.tolist(),
        'target_attention_mask': decoder_tokens['attention_mask'],
        'shifted_target_input_ids': shifted_target_input_ids.tolist()
    }

# Initialize Pylines with input_fn and output_fn
pylines = Pylines(input_fn, output_fn, overwrite_output=False)

# Pass the above tokenization function through, which will be serialized and used to process every line.
# By default, use_mp=True will use all cores.
pylines.run_tokenizer(tokenize_fn, use_mp=True)
# or use_mp=int will use that many processes in mp.Pool
pylines.run_tokenizer(tokenize_fn, use_mp=4)

# or use as a generator
for ex in pylines.as_tokenizer(tokenize_fn, use_mp=True):
    # do something with ex

# Or serialize to tfrecords by defining your dataset features

dataset_features = {
        'x': {
            'names': ["input_ids", "attention_mask"]
        },
        'y': {
            'names': ["target_ids", "target_attention_mask", "shifted_target_input_ids"]
        }
    }

# will yield serialized examples after passing through tokenizer_fn if its provided.
for x, ex in enumerate(pylines.as_encoder(dataset_features=dataset_features, tokenizer_fn=tokenize_fn, use_mp=1)):
    if x == 5:
        print(ex)

# define a output_dir and it will write tfrecords
output_dir = '/my/dataset/dir'
pylines.run_encoder(dataset_features=dataset_features, tokenizer_fn=tokenize_fn, output_dir=output_dir, split_key=None, use_mp=1)


```
#### Automatically Reading/Writing to Object Storage

By Default, upon init, Pylines will look for common environment variables including:
- GOOGLE_APPLICATION_CREDENTIALS
- AWS_SHARED_CREDENTIALS_FILE
- AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY

If those are set, then the associated client will be initialized
- google.cloud.storage - GCS
- boto3 - S3

These will be passed to smart_open for S3, and GCS (if tensorflow is not installed).

If these are not automatically detected during first initialization, they can also be explicitly set
```python3

import os
from pylines import auth_cloud, Pylines

# Can be set at the env level 
os.environ["AWS_ACCESS_KEY_ID"] = ''
os.environ["AWS_SECRET_ACCESS_KEY"] = ''
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/path/to/adc.json'
os.environ["AWS_SHARED_CREDENTIALS_FILE"] = '/path/to/file'

auth_cloud()

# or passed explictly
my_gcs_cred = '/path/to/adc.json'
my_s3_cred = '/path/to/file'

auth_cloud(gcs=my_gcs_cred, s3=my_s3_cred)

# or passed as a dict, only for s3

s3 = {
    'aws_access_key_id': 'key',
    'aws_secret_access_key': 'key'
}
auth_cloud(gcs=my_gcs_cred, s3=s3)
```


## Quick Function Cheatsheet
```python3
'''

# I/O
Pylines.add_files(input_files): str or list. Used to add new files to the input feed
Pylines.set_input_files(input_files): str or list. Used to override existing files.

Pylines.set_writefile(output_file, overwrite=False): str, bool. Used to set the output file. Will append by default if file exists.

Pylines.clear_input_files(): No args. Clears all input files. Use set_input_files or add_files to add new input files.

# Json Functions - Majority of default json functions are wrappers around simdjson
Pylines.parse(x): Will read from binary and return as bytes. Use .loads() instead for a python dict. Useful for direct serialization, ie from json to pickle.
Pylines.loads(x): Will read from binary or any json string, and return as deserialized json.

Pylines.dumps(x): Serializes json input
Pylines.dump(filename, x): Serializes json input (non-jsonlines), and will take an open_fn or a raw file string.

Pylines.index(idx, filename=None): -> dict
Returns a line from a specific index. If filename is set, will use that specific file instead of existing files added. If no filename, will return the same index from all files added.

Pylines.write(x): Serializes and writes json to the output_fn and appends with a newline. Do not call with json.dumps() as this is already handled.

# Other Functions
Pylines.find(key, value, results=['first', 'all'], filename=None): -> dict [generator]
Yields results when key=value from file, assuming all lines have key within the line. If used with filename, will search that filename instead of added files, else will search through all files added. If results='first', will only return the first line that matches. If results='all', will return all lines in all file(s) that match.

Pylines.as_iterator(): -> dict [generator]
Takes no args. Iterates over all files added and yields deserialized json lines.

Pylines.merge(input_fns=None, output_fn=None):
Adds input_fns to existing files, and sets output_fn if not none. Merges them all into the output fn.

Pylines.linecount(filename=None): -> dict in {filename: num_lines} for each file
If filename, will find the total number of lines in the file. Else will iterate through input_files and returns dict

Pylines.stats -> dict [property] in {filename: {'read': int, 'missed': int}} for each filename that have been processed. Will reset upon each call of iterators.

# Example usage below.
Pylines.run_tokenizer(tokenizer_fn, use_mp=[True, or int for num_processes]): -> tokenized examples to output_fn.
Iterates through all files and tokenizes with provided tokenizer_fn. use_mp = False will not use multiprocessing. Otherwise will use all available cores by default.

Pylines.run_processor(iter_fn, use_mp=[True, or int for num_processes]): ->  results from function(line) to output_fn.
Iterates through all files and tokenizes with provided iter_fn. use_mp = False will not use multiprocessing. Otherwise will use all available cores by default.
'''

```


Roadmap:

- Different format of serialization outputs (pickle, torch) // tfrecords done
- Creation of Dataset objects for Tensorflow and Pytorch, to easily plug into training pipelines, with or without caching
- Support for sharding of files rather than a single big output file
- Conditional merging of input files // done
- Support for mapping of jsonlines into functions
- Allow for caching through redis backend
- Creating an index file that maps keys <-> int to save precious bytes when writing large files
  