import shutil
import time

from pathlib import Path

import numpy as np
import tensorflow as tf
from config import Config

import model as _model
from data_loader import get_datasets
import matplotlib.pyplot as plt

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('config', 'conf/SML2010.json', 'Path to json file with the configuration to be run')

def plot(session, model, next_element, log_path: Path):
    all_true = []
    all_predicted = []
    while True:
        try:
            x, y = session.run(next_element)
            predictions = session.run(model.predictions, {model.driving_series: x, model.past_history: y})
            true = np.reshape(y[:, -1], [-1]).tolist()
            predicted = np.reshape(predictions, [-1]).tolist()
            all_true += true
            all_predicted += predicted

        except tf.errors.OutOfRangeError:
            break

    all_true = np.array(all_true)
    all_predicted = np.array(all_predicted)

    plt.figure()
    plt.plot(all_true, label="true")
    plt.plot(all_predicted, label="predicted")
    plt.legend(loc='upper left')
    plt.title("Test data")
    plt.ylabel("target serie")
    plt.xlabel("time steps")
    plt.show()


def main(argv):
    # load hyper-parameters from configuration file
    config = Config.from_file(FLAGS.config)

    _, _, test_set = get_datasets(config)
    test_set = test_set.batch(config.batch_size, drop_remainder=True)

    model = _model.TimeAttnModel(config)
    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        test_iterator = test_set.make_initializable_iterator()

        test_next_element = test_iterator.get_next()

        # Restore from last evaluated epoch
        print("Restoring from: {}".format(config.log_path / "model-max-ckpt"))
        saver.restore(session, str(config.log_path / "model-max-ckpt"))

        session.run(test_iterator.initializer)
        test_scores = model.evaluate(session, test_next_element)

        print("RMSE: {:.5f}".format(test_scores["RMSE"]))
        print("MAE: {:.5f}".format(test_scores["MAE"]))
        print("MAPE: {:.5f}".format(test_scores["MAPE"]))

        session.run(test_iterator.initializer)
        plot(session, model, test_next_element, config.log_path)       

if __name__ == '__main__':
    tf.app.run(main=main)
