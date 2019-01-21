# time-attention
Implementation of RNN for Time Series prediction from the paper 
https://arxiv.org/abs/1704.02971.

Some parameters' names, variables and configurations keys are
derived from the paper.


## Reproducing the results

* Download the necessary data through `./get_data.sh`

* All the parameters for the training and features used are stored in
  `conf/*.json`.

### Training file

```
USAGE: train.py [flags]
flags:

train.py:
  --config: Path to json file with the configuration to be run
    (default: 'conf/SML2010.json')

Try --helpfull to get a list of all flags.
```

## Configurations generator

Used for generating multiple configurations to be run for performance
analysis

```
usage: generate_configs.py [-h] [--src SRC] [--dest DEST]

Generates config files for multiple configurations. It requires the source
directory containing the jsons of the base configurations from which the new
configurations have to be generated.

optional arguments:
  -h, --help   show this help message and exit
  --src SRC    Source directory where base configurations are found
  --dest DEST  Destination directory where the files will be created
```