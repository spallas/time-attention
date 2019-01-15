
import tensorflow as tf
import json

if __name__ == '__main__':
    # load hyper-parameters from configuration file
    with open("conf/experiment1.json") as f:
        conf = json.load(f)

