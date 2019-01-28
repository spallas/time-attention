import itertools
from pathlib import Path

import numpy as np
import tensorflow as tf
from config import Config

import model as _model
from data_loader import get_datasets
import matplotlib.pyplot as plt

plt.style.use("seaborn-darkgrid")


def get_np_array(session, model, next_element):
    all_true = []
    all_predicted = []
    while True:
        try:
            x, y = session.run(next_element)
            predictions = session.run(
                model.predictions,
                {model.driving_series: x, model.past_history: y},
            )
            true = np.reshape(y[:, -1], [-1]).tolist()
            predicted = np.reshape(predictions, [-1]).tolist()
            all_true += true
            all_predicted += predicted

        except tf.errors.OutOfRangeError:
            break
    return np.array(all_true), np.array(all_predicted)


def plot(
    session,
    model,
    train_next_element,
    val_next_element,
    test_next_element,
    name="tmp",
    show=True,
):

    val_true, val_predicted = get_np_array(session, model, val_next_element)
    test_true, test_predicted = get_np_array(session, model, test_next_element)

    val_size, test_size = len(val_true), len(test_true)

    plt.figure()
    plt.plot(range(val_size), val_true, label="val true")
    plt.plot(range(val_size), val_predicted, label="val predicted")
    plt.plot(
        range(val_size, val_size + test_size), test_true, label="test true"
    )
    plt.plot(
        range(val_size, val_size + test_size),
        test_predicted,
        label="test predicted",
    )
    plt.ylabel("target serie")
    plt.xlabel("time steps")
    plt.legend(loc="upper left")
    plt.title(name)
    if show:
        plt.show()
    else:
        plt.savefig(name, dpi=400)
        plt.close()


def evaluate(config, show=True, name="tmp"):

    with tf.Graph().as_default():
        train_set, val_set, test_set = get_datasets(config, shuffled=False)
        train_set = train_set.batch(config.batch_size, drop_remainder=True)
        val_set = val_set.batch(config.batch_size, drop_remainder=True)
        test_set = test_set.batch(config.batch_size, drop_remainder=True)

        model = _model.TimeAttnModel(config)
        saver = tf.train.Saver(max_to_keep=1)
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            train_iterator = train_set.make_initializable_iterator()
            val_iterator = val_set.make_initializable_iterator()
            test_iterator = test_set.make_initializable_iterator()

            train_next_element = train_iterator.get_next()
            val_next_element = val_iterator.get_next()
            test_next_element = test_iterator.get_next()

            # Restore from last evaluated epoch
            print(
                "Restoring from: {}".format(config.log_path / "model-max-ckpt")
            )
            saver.restore(session, str(config.log_path / "model-max-ckpt"))

            session.run(train_iterator.initializer)
            train_scores = model.evaluate(session, train_next_element)
            print("============Train=============")
            print("RMSE: {:.5f}".format(train_scores["RMSE"]))
            print("MAE: {:.5f}".format(train_scores["MAE"]))
            print("MAPE: {:.5f}".format(train_scores["MAPE"]))

            session.run(val_iterator.initializer)
            val_scores = model.evaluate(session, val_next_element)
            print("============Validation=============")
            print("RMSE: {:.5f}".format(val_scores["RMSE"]))
            print("MAE: {:.5f}".format(val_scores["MAE"]))
            print("MAPE: {:.5f}".format(val_scores["MAPE"]))

            session.run(test_iterator.initializer)
            test_scores = model.evaluate(session, test_next_element)
            print("============Test=============")
            print("RMSE: {:.5f}".format(test_scores["RMSE"]))
            print("MAE: {:.5f}".format(test_scores["MAE"]))
            print("MAPE: {:.5f}".format(test_scores["MAPE"]))

            session.run(train_iterator.initializer)
            session.run(val_iterator.initializer)
            session.run(test_iterator.initializer)
            plot(
                session,
                model,
                train_next_element,
                val_next_element,
                test_next_element,
                show=show,
                name=name,
            )

    return train_scores, val_scores, test_scores


def main(argv=None):
    # load hyper-parameters from configuration file
    config_files = list(Path("gen_confs").glob("**/*.json"))
    conf_names = [filepath.name for filepath in config_files]
    configs = [Config.from_file(conf_file) for conf_file in config_files]
    sets = ["train", "val", "test"]
    scores = ["rmse", "mae", "mape"]
    header = "conf," + (
        ",".join(["_".join(tup) for tup in itertools.product(sets, scores)])
    )
    with open("results.csv", "w") as f:
        f.write(header + "\n")
        for name, config in zip(conf_names, configs):
            train_scores, val_scores, test_scores = evaluate(
                config,
                show=False,
                name="gen_imgs/" + str(Path(name).with_suffix("")),
            )
            f.write(
                ",".join(
                    [
                        name,
                        str(train_scores["RMSE"]),
                        str(train_scores["MAE"]),
                        str(train_scores["MAPE"]),
                        str(val_scores["RMSE"]),
                        str(val_scores["MAE"]),
                        str(val_scores["MAPE"]),
                        str(test_scores["RMSE"]),
                        str(test_scores["MAE"]),
                        str(test_scores["MAPE"]),
                    ]
                )
                + "\n"
            )


if __name__ == "__main__":
    tf.app.run(main=main)
