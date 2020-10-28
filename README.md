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

#### Editable Installation
```shell
git clone https://github.com/trisongz/pylines.git && cd pylines && pip install -e .
```

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
for item in pylines.iter():
    # process items here
    write(item)

# alternatively, you can pass a function and run an iterator with mp, and have it write to the output file
pylines.run_function(processor_function)

# or get the results from the processor to do something else
for result in pylines.run_function(processor_function, return_results=True):
    # do something with results

'''
Quick Function Cheatsheet

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

Pylines.iter(): -> dict [generator]
Takes no args. Iterates over all files added and yields deserialized json lines.

Pylines.merge(input_fns=None, output_fn=None):
Adds input_fns to existing files, and sets output_fn if not none. Merges them all into the output fn.

Pylines.linecount(filename=None): -> dict in {filename: num_lines} for each file
If filename, will find the total number of lines in the file. Else will iterate through input_files and returns dict

Pylines.stats -> dict [property] in {filename: {'read': int, 'missed': int}} for each filename that have been processed. Will reset upon each call of iterators.

# Example usage below.
Pylines.tokenize(tokenizer_fn, return_results=False, use_mp=[True, or int for num_processes]): -> tokenized examples to output_fn.
Iterates through all files and tokenizes with provided tokenizer_fn. use_mp = False will not use multiprocessing. Otherwise will use all available cores by default.
If return_results=True, yields results as a generator.

Pylines.run_function(iter_fn, return_results=False, use_mp=[True, or int for num_processes]): ->  results from function(line) to output_fn.
Iterates through all files and tokenizes with provided iter_fn. use_mp = False will not use multiprocessing. Otherwise will use all available cores by default.
If return_results=True, yields results as a generator.
'''

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
- A LazyLoadFile class (wip) that allows for sampling of the dataset without loading everything into memory
- functions that allow retrieving a line by index or with a key=value match



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
processor = Pylines(input_fn, output_fn, overwrite_output=False)

# Pass the above tokenization function through, which will be serialized and used to process every line.
# By default, use_mp=True will use all cores.
processor.tokenize(tokenize_fn, use_mp=True)
# or use_mp=int will use that many processes in mp.Pool
processor.tokenize(tokenize_fn, use_mp=4)

# or use as a generator
for ex in processor.tokenize(tokenize_fn, use_mp=True, return_results=True):
    # do something with ex

```

Roadmap:

- Different format of serialization outputs (pickle, torch, tfrecords)
- Creation of Dataset objects for Tensorflow and Pytorch, to easily plug into training pipelines, with or without caching
- Support for sharding of files rather than a single big output file
- Conditional merging of input files
- Support for mapping of jsonlines into functions
- Allow for caching through redis backend
- Creating an index file that maps keys <-> int to save precious bytes when writing large files