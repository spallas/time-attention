from math import sqrt

import numpy as np
import tensorflow as tf

from sklearn.metrics.regression import mean_absolute_error, mean_squared_error
from config import Config

import model as _model
from data_loader import get_datasets
import matplotlib.pyplot as plt

plt.style.use("seaborn-darkgrid")

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "config",
    "conf/SML2010.json",
    "Path to json file with the configuration to be run",
)


def compute_scores(true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = sqrt(mean_squared_error(true, pred))
    mape = np.abs((true - pred) / true).mean() * 100
    return mae, rmse, mape


def show_scores(
    model, session, next_element, y_scaler, set_name, plot_offset=0
):
    true, pred = model.predict(session, next_element)
    pred = y_scaler.inverse_transform(np.append(pred[0], pred[1:]))
    true = y_scaler.inverse_transform(np.append(true[0], true[1:]))
    mae, rmse, mape = compute_scores(true, pred)

    plt.plot(
        range(plot_offset, plot_offset + len(true)),
        true,
        label=f"{set_name} true",
    )
    plt.plot(
        range(plot_offset, plot_offset + len(pred)),
        pred,
        label=f"{set_name} pred",
    )

    print(f"============{set_name}=============")
    print(f"MAE: {mae:.5f}")
    print(f"RMSE: {rmse:.5f}")
    print(f"MAPE: {mape:.5f}")

    return plot_offset + len(true)


def main(argv):
    # load hyper-parameters from configuration file
    config = Config.from_file(FLAGS.config)

    train_set, val_set, test_set, x_scaler, y_scaler = get_datasets(
        config, shuffled=False, normalized=True
    )
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

        # Restore the best model
        print("Restoring from: {}".format(config.log_path / "model-max-ckpt"))
        saver.restore(session, str(config.log_path / "model-max-ckpt"))

        plt.figure()

        session.run(train_iterator.initializer)
        offset = show_scores(
            model, session, train_next_element, y_scaler, "Train"
        )

        session.run(val_iterator.initializer)
        offset = show_scores(
            model,
            session,
            val_next_element,
            y_scaler,
            "Validation",
            plot_offset=offset,
        )

        session.run(test_iterator.initializer)
        offset = show_scores(
            model,
            session,
            test_next_element,
            y_scaler,
            "Test",
            plot_offset=offset,
        )

        plt.ylabel("target serie")
        plt.xlabel("time steps")
        plt.legend(loc="upper left")
        plt.show()


if __name__ == "__main__":
    tf.app.run(main=main)
