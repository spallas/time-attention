import shutil
import time
import os

import tensorflow as tf
from config import Config

import model
from data_loader import get_datasets


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

    report_frequency = 10  # config["report_frequency"]
    eval_frequency = 100  # config["eval_frequency"]

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
        iterator = train_set.make_initializable_iterator()
        next_element = iterator.get_next()
        for _ in range(config.num_epochs):
            session.run(iterator.initializer)
            while True:
                try:
                    x, y = session.run(next_element)
                    tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op],
                                                             feed_dict={model.driving_series: x,
                                                                        model.past_history: y})
                    accumulated_loss += tf_loss

                    if tf_global_step % report_frequency == 0:
                        total_time = time.time() - initial_time
                        steps_per_second = tf_global_step / total_time
                        average_loss = accumulated_loss / report_frequency
                        print("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
                        writer.add_summary(make_summary({"loss": average_loss}), tf_global_step)
                        accumulated_loss = 0.0

                    if tf_global_step % eval_frequency == 0:
                        saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
                        eval_summary, eval_RMSE, eval_MAE, eval_MAPE = model.evaluate(session)

                        if eval_RMSE < best_RMSE:
                            best_RMSE = eval_RMSE
                            copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)),
                                            os.path.join(log_dir, "model.max.ckpt"))

                        writer.add_summary(eval_summary, tf_global_step)
                        writer.add_summary(make_summary({"min RMSE = ": best_RMSE}), tf_global_step)

                        print("[{}] eval_RMSE={:.2f}, best_RMSE={:.2f}".format(tf_global_step, eval_RMSE, best_RMSE))
                except tf.errors.OutOfRangeError:
                    break
