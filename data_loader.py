import tensorflow as tf
import pandas as pd
import numpy as np

tf.enable_eager_execution()


config = {
    "data_paths": ["data/NEW-DATA-1.T15.csv", "data/NEW-DATA-2.T15.csv"],
    "drop_cols": [
        "1:Date",
        "2:Time",
        "19:Exterior_Entalpic_1",  # All zeros
        "20:Exterior_Entalpic_2",
        "21:Exterior_Entalpic_turbo",
        "24:Day_Of_Week",
    ],
    "target_cols": ["4:Temperature_Habitacion_Sensor"],
    "sep": " ",
    "window_size": 10,
}


def get_train_test_dataset(config, cat_before_window=False):
    """
    cat_before_window : whether to concatenate the files before transforming it
                        into windows
    """

    def window(df, size, driving_series, target_series):
        X = df[driving_series].values
        y = df[target_series].values
        X_T = []
        y_T = []
        for i in range(len(X) - size):
            X_T.append(X[i: i + size])
            y_T.append(y[i: i + size])

        return np.array(X_T), np.array(y_T)

    path = config["data_paths"][0]

    with open(path) as f:
        header = f.readline().strip().split(config["sep"])

    usecols = [col for col in header if col not in config["drop_cols"]]

    driving_series = [
        col for col in usecols if col not in config["target_cols"]
    ]

    dfs = []
    for path in config["data_paths"]:
        dfs.append(pd.read_csv(path, sep=config["sep"], usecols=usecols))

    df = None
    X_T = None
    y_T = None
    if cat_before_window:
        df = pd.concat(dfs)
        X_T, y_T = window(
            df, config["window_size"], driving_series, config["target_cols"]
        )
        X_T = X_T.transpose((0, 2, 1))
    else:
        X_Ts = []
        y_Ts = []
        for df in dfs:
            X_T, y_T = window(
                df, config["window_size"], driving_series, config["target_cols"]
            )
            X_T = X_T.transpose((0, 2, 1))
            X_Ts.append(X_T)
            y_Ts.append(y_Ts)
        print("Ended transforms")
        X_T = np.vstack(X_Ts)
        print("Ended X_T")
        y_T = np.vstack(y_Ts)
        print("Ended y_T")

    X_dataset = tf.data.Dataset.from_tensor_slices(X_T)
    y_dataset = tf.data.Dataset.from_tensor_slices(y_T)

    dataset = tf.data.Dataset.zip((X_dataset, y_dataset))
    return dataset


dataset = get_train_test_dataset(config)

it = dataset.batch(3)

for x, y in it:
    print("X", x)
    print("Y", y)
    break

"""
record_defaults = [tf.float32] * (len(header) - len(config["drop_cols"]))


data = tf.data.experimental.CsvDataset(
    config["data_path"],
    record_defaults,
    field_delim=config["sep"],
    select_cols=select_cols,
    header=True,
)

it = data_w.make_one_shot_iterator()
for x in it:
    print(x)
"""
