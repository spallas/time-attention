import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from typing import Tuple, List
from config import Config


def window(
    df: pd.DataFrame,
    size: int,
    driving_series: List[str],
    target_series: List[str],
):
    X = df[driving_series].values
    y = df[target_series].values
    X_T = []
    y_T = []
    for i in range(len(X) - size + 1):
        X_T.append(X[i : i + size])
        y_T.append(y[i : i + size])

    return np.array(X_T), np.array(y_T)


def get_np_dataset(
    config: Config, normalized: bool = False
) -> Tuple[np.ndarray, np.ndarray]:

    dfs = []
    for path in config.data_paths:
        dfs.append(pd.read_csv(path, sep=config.sep, usecols=config.usecols))

    df = pd.concat(dfs)

    x_scaler = None
    y_scaler = None
    if normalized:
        x_scaler = StandardScaler().fit(
            df[config.driving_series][: int(config.train_ratio * len(df))]
        )
        y_scaler = StandardScaler().fit(
            df[config.target_cols][: int(config.train_ratio * len(df))]
        )
        df[config.driving_series] = x_scaler.transform(df[config.driving_series])
        df[config.target_cols] = y_scaler.transform(df[config.target_cols])

    X_T, y_T = window(df, config.T, config.driving_series, config.target_cols)
    X_T = X_T.transpose((0, 2, 1))
    y_T = np.squeeze(y_T)

    return X_T, y_T, x_scaler, y_scaler


def get_datasets(
    config: Config,
    cat_before_window: bool = False,
    shuffled: bool = True,
    normalized: bool = False,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, StandardScaler, StandardScaler]:
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
    train_d : tensorflow.data.Dataset(tuples)
        The tuples are
            X : (config.batch_size, config.n, config.T)
            y : (config.batch_size, config.T, 1)
    val_d
    test_d

    Usage
    -----

    Graph Mode:
    ```
    dataset = get_train_test_dataset(config)
    dataset = dataset.batch(batch_size) # .shuffle() if necessary
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    for _ in range(epochs):
        sess.run(iterator.initializer)
        while True:
            try:
                sess.run(next_element)
            except tf.errors.OutOfRangeError:
                break

        # [Perform end-of-epoch calculations here.]
    ```

    Eager Mode:
    ```
        dataset = get_train_test_dataset(config)
        it = dataset.batch(batch_size)
        for x, y in it:
            print(x, y)
    ```
    """
    X_T, y_T, x_scaler, y_scaler = get_np_dataset(config, normalized=normalized)
    train_size = int(len(X_T) * config.train_ratio)
    val_size = int(config.val_ratio * len(X_T))
    test_size = val_size

    dataset = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices(X_T),
            tf.data.Dataset.from_tensor_slices(y_T),
        )
    )

    train_dataset = dataset.take(train_size)
    if shuffled:
        train_dataset = train_dataset.shuffle(
            train_size, reshuffle_each_iteration=True
        )
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size).take(test_size)
    return train_dataset, val_dataset, test_dataset, x_scaler, y_scaler


# Test
if __name__ == "__main__":
    tf.enable_eager_execution()
    with open("conf/NASDAQ100.json") as f:
        config = Config.from_json(f.read())

    tr, val, te, x_scaler, y_scaler = get_datasets(config, normalized=True)

    it = tr.make_one_shot_iterator()
    lit = val.make_one_shot_iterator()
    tit = te.make_one_shot_iterator()
    print(f"len(list(it)) {len(list(it))}")
    print(f"len(list(lit)) {len(list(lit))}")
    print(f"len(list(tit)) {len(list(tit))}")

    it = tr.make_one_shot_iterator()
    print(next(it))

    it = tr.make_one_shot_iterator()
    print(next(it))
