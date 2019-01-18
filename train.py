import shutil
import time
import os

import numpy as np
import tensorflow as tf
from config import Config

import model
from data_loader import get_datasets
import matplotlib.pyplot as plt


def copy_checkpoint(source, target):
    for ext in (".index", ".data-00000-of-00001"):
        shutil.copyfile(source + ext, target + ext)


def make_summary(value_dict):
    return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k, v in value_dict.items()])


if __name__ == '__main__':
    # load hyper-parameters from configuration file

    with open("conf/NASDAQ100.json") as f:
        config = Config.from_json(f.read())

    train_set, val_set, test_set = get_datasets(config)
    train_set = train_set.batch(config.batch_size, drop_remainder=True)
    val_set = val_set.batch(config.batch_size, drop_remainder=True)
    test_set = test_set.batch(config.batch_size, drop_remainder=True)
    model = model.TimeAttnModel(config)

    report_frequency = config.report_frequency

    saver = tf.train.Saver()

    log_dir = config.log_dir
    writer = tf.summary.FileWriter(log_dir, flush_secs=20)

    best_RMSE = float("inf")
    best_MAE = float("inf")
    best_MAPE = float("inf")
    accumulated_loss = 0.0
    initial_time = time.time()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        tf_global_step = 0

        iterator = train_set.make_initializable_iterator()
        val_iterator = val_set.make_initializable_iterator()
        test_iterator = test_set.make_initializable_iterator()
        next_element = iterator.get_next()
        val_next_element = val_iterator.get_next()
        test_next_element = test_iterator.get_next()

        # Restore from last evaluated epoch
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring from: {}".format(ckpt.model_checkpoint_path))
            saver.restore(session, ckpt.model_checkpoint_path)
            init_global_step = session.run(model.global_step)
        else:
            init_global_step = 0

        session.run(val_iterator.initializer)
        for i in range(config.num_epochs):
            session.run(iterator.initializer)
            print(f"====================================== EPOCH {i} ======================================")
            while True:
                try:
                    x, y = session.run(next_element)
                    tf_loss, tf_global_step, _, __ = session.run([model.loss, model.global_step,
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

            saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
            eval_summary, eval_RMSE, eval_MAE, eval_MAPE = model.evaluate(session, val_next_element)
            session.run(val_iterator.initializer)

            if eval_RMSE < best_RMSE:
                best_RMSE = eval_RMSE
                copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)),
                                os.path.join(log_dir, "model.max.ckpt"))

            writer.add_summary(eval_summary, tf_global_step)
            writer.add_summary(make_summary({"min RMSE = ": best_RMSE}), tf_global_step)
            print("----------------------")
            print("RMSE: {:.5f}".format(eval_RMSE))
            print("MAE: {:.5f}".format(eval_MAE))
            print("MAPE: {:.5f}".format(eval_MAPE))
            print("best_RMSE={:.5f}".format(best_RMSE))

            if i % config.plot_frequency == 0:
                session.run(test_iterator.initializer)
                all_true = []
                all_predicted = []
                while True:
                    try:
                        x, y = session.run(test_next_element)
                        predictions = session.run(model.predictions, {model.driving_series: x, model.past_history: y})
                        true = np.reshape(y[:, -1], [-1]).tolist()
                        predicted = np.reshape(predictions, [-1]).tolist()
                        all_true += true
                        all_predicted += predicted

                    except tf.errors.OutOfRangeError:
                        break

                all_true = np.array(all_true)
                all_predicted = np.array(all_predicted)
                plt.plot(all_true, label="true")
                plt.plot(all_predicted, label="predicted")
                plt.legend(loc='upper left')
                plt.title("Test data")
                plt.ylabel("target series")
                plt.xlabel("time steps")
                plt.savefig(f"log/plot{i}.png", dpi=300)
                plt.show()
