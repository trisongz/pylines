# Pylines
Simplifying parsing of large jsonline files in NLP Workflows (but you can use it for anything)

## Quickstart

#### Quick Installation
```shell
pip install --upgrade git+https://github.com/trisongz/pylines.git
```

#### Editable Installation
```shell
git clone https://github.com/trisongz/pylines.git && cd pylines && pip install -e .
```

#### Example of Usage
```python3
'''
Base Default Variables for Pylines

input_fns=None - if no input file is set, can be later set by calling Pylines.add_files(input_files), adding to existing files or Pylines.set_input_files(input_files) to override existing list of files, and set to the new files.
output_fn=None - if no output file is set, can be later set by calling Pylines.set_writefile(output_file)
skip_broken=True - will skip unparsable lines by default
overwrite_output=False - will control overwriting of the output file
use_lazy=False - doesn't do anything right now
use_mp=True - doesn't explicitly do anything right now
use_idx=False - doesn't do anything right now
total_lines=0 - used to set the total lines expected, useful if you already know how many items are in your file and prevent counting of items during initialization.

'''
from pylines import Pylines
input_fn = 'myinputfile.json' # or ['inputfile1.json', 'inputfile2.json']

# Quick Iterator through all items
for item in Pylines(input_fn):
    # process items here
    print(item)

# Can be initialized as an object
lines = Pylines(input_fn)
# returns total count of all lines in all input files
print(len(lines))

# Can set output file and then iterate through input files, and writing to output file
output_file = 'myoutputfile.json' # only takes a string right now
lines.set_writefile(output_file)

write = lines.write
for item in lines.iter():
    # process items here
    write(item)

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
processor.tokenize(tokenize_fn, use_mp=True)

```

Roadmap:

- Different format of serialization outputs (pickle, torch, tfrecords)
- Creation of Dataset objects for Tensorflow and Pytorch, to easily plug into training pipelines, with or without caching
- Support for sharding of files rather than a single big output file
- Conditional merging of input files
- Support for mapping of jsonlines into functions