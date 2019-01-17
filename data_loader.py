import tensorflow as tf
import pandas as pd
import numpy as np

from config import Config

tf.enable_eager_execution()


# TODO(xhebraj) implement return of train/test/validation
def get_train_test_dataset(config: Config, cat_before_window=False):
    """
    Returns X and y of the data passed as config.

    Parameters
    ----------
    config : Config
    cat_before_window : bool
        Whether to concatenate the files before transforming it
        into windows

    Returns
    -------
    d : tensorflow.data.Dataset(tuples)
        The tuples are
            X : (config.batch_size, config.n, config.T)
            y : (config.batch_size, config.T, 1)
    Usage
    -----
    ```
        dataset = get_train_test_dataset(config)
        it = dataset.batch(batch_size)
        for x, y in it:
            print(x, y)
    ```
    """

    def window(df, size, driving_series, target_series):
        X = df[driving_series].values
        y = df[target_series].values
        X_T = []
        y_T = []
        for i in range(len(X) - size):
            X_T.append(X[i : i + size])
            y_T.append(y[i : i + size])

        return np.array(X_T), np.array(y_T)

    path = config.data_paths[0]

    with open(path) as f:
        header = f.readline().strip().split(config.sep)

    usecols = [col for col in header if col not in config.drop_cols]

    driving_series = [col for col in usecols if col not in config.target_cols]

    dfs = []
    for path in config.data_paths:
        dfs.append(pd.read_csv(path, sep=config.sep, usecols=usecols))

    df = None
    X_T = None
    y_T = None
    if cat_before_window:
        df = pd.concat(dfs)
        X_T, y_T = window(df, config.T, driving_series, config.target_cols)
        X_T = X_T.transpose((0, 2, 1))
    else:
        X_Ts = []
        y_Ts = []
        for df in dfs:
            X_T, y_T = window(df, config.T, driving_series, config.target_cols)
            X_T = X_T.transpose((0, 2, 1))
            X_Ts.append(X_T)
            y_Ts.append(np.squeeze(y_T))
        X_T = np.vstack(X_Ts)
        y_T = np.vstack(y_Ts)

    y_T = np.expand_dims(y_T, axis=-1)

    X_dataset = tf.data.Dataset.from_tensor_slices(X_T)
    y_dataset = tf.data.Dataset.from_tensor_slices(y_T)

    dataset = tf.data.Dataset.zip((X_dataset, y_dataset))
    return dataset


# Test
if __name__ == "__main__":
    with open("conf/experiment2.json") as f:
        config = Config.from_json(f.read())

    dataset = get_train_test_dataset(config)

    it = dataset.batch(config.batch_size)

    for x, y in it:
        print("X", x)
        print("Y", y)
        break

"""

it = data_w.make_one_shot_iterator()
for x in it:
    print(x)
"""
