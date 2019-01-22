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


def copy_checkpoint(source, target):
    for ext in (".index", ".data-00000-of-00001"):
        shutil.copyfile(source.with_suffix(ext), target.with_suffix(ext))


def make_summary(value_dict):
    return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k, v in value_dict.items()])


def plot(session, model, next_element, i, log_path: Path):
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
    plt.title("Validation data")
    plt.ylabel("target serie")
    plt.xlabel("time steps")
    plt.savefig(log_path / f"plot{i}.png", dpi=300)
    plt.close()


def main(argv):
    # load hyper-parameters from configuration file
    config = Config.from_file(FLAGS.config)
    # set seeds for reproducibility
    tf.set_random_seed(config.seed)
    np.random.seed(config.seed)

    train_set, val_set, test_set = get_datasets(config)
    train_set = train_set.batch(config.batch_size, drop_remainder=True)
    val_set = val_set.batch(config.batch_size, drop_remainder=True)
    test_set = test_set.batch(config.batch_size, drop_remainder=True)

    model = _model.TimeAttnModel(config)

    report_frequency = config.report_frequency
    saver = tf.train.Saver(max_to_keep=1)
    log_path = config.log_path
    writer = tf.summary.FileWriter(log_path, flush_secs=20)

    best_RMSE = float("inf")
    best_MAE = float("inf")
    best_MAPE = float("inf")
    accumulated_loss = 0.0
    initial_time = time.time()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        tf_global_step = 0

        train_iterator = train_set.make_initializable_iterator()
        val_iterator = val_set.make_initializable_iterator()
        test_iterator = test_set.make_initializable_iterator()

        train_next_element = train_iterator.get_next()
        val_next_element = val_iterator.get_next()
        test_next_element = test_iterator.get_next()

        # Restore from last evaluated epoch
        ckpt = tf.train.get_checkpoint_state(log_path)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring from: {}".format(ckpt.model_checkpoint_path))
            saver.restore(session, ckpt.model_checkpoint_path)
            init_global_step = session.run(model.global_step)
        else:
            init_global_step = 0

        for i in range(config.num_epochs):
            session.run(train_iterator.initializer)
            print(f"====================================== EPOCH {i} ======================================")
            while True:
                try:
                    x, y = session.run(train_next_element)
                    tf_loss, tf_global_step, _, _ = session.run([model.loss, model.global_step,
                                                                  model.train_op_en, model.train_op_dec],
                                                                 feed_dict={model.driving_series: x,
                                                                            model.past_history: y})
                    accumulated_loss += tf_loss

                    if tf_global_step % report_frequency == 0:
                        total_time = time.time() - initial_time
                        steps_per_second = (tf_global_step - init_global_step) / total_time
                        average_loss = accumulated_loss / report_frequency
                        print("[{}] loss={:.5f}, steps/s={:.5f}".format(tf_global_step, average_loss, steps_per_second))
                        writer.add_summary(make_summary({"loss": average_loss}), tf_global_step)
                        accumulated_loss = 0.0

                except tf.errors.OutOfRangeError:
                    break

            session.run(val_iterator.initializer)
            saver.save(session, log_path / "model", global_step=tf_global_step)
            val_scores = model.evaluate(session, val_next_element)

            if val_scores["RMSE"] < best_RMSE:
                best_RMSE = val_scores["RMSE"]
                copy_checkpoint(log_path / f"model-{tf_global_step}",
                                log_path / "model-max-ckpt")

            writer.add_summary(make_summary(val_scores), tf_global_step)
            writer.add_summary(make_summary({"min RMSE = ": best_RMSE}), tf_global_step)
            print("----------------------")
            print("RMSE: {:.5f}".format(val_scores["RMSE"]))
            print("MAE: {:.5f}".format(val_scores["MAE"]))
            print("MAPE: {:.5f}".format(val_scores["MAPE"]))
            print("best_RMSE={:.5f}".format(best_RMSE))

            if i % config.plot_frequency == 0:
                session.run(val_iterator.initializer)
                plot(session, model, val_next_element, i, config.log_path)


if __name__ == '__main__':
    tf.app.run(main=main)
