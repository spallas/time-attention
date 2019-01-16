import tensorflow as tf

tf.enable_eager_execution()

config = {
    "data_path": ["data/NEW-DATA-1.T15.csv", "data/NEW-DATA-2.T15.csv"],
    "drop_cols": [
        0,  # Date
        1,  # Time
        18,  # Entalpic motor 1      , All zeros
        19,  # Entalpic motor 2      , All zeros
        20,  # Entalpic motor turbo  , All zeros
    ],
    "sep": " ",
    "window_size": 10,
}

with open(config["data_path"][0]) as f:
    header = f.readline().strip().split(config["sep"])

record_defaults = [tf.float32] * (len(header) - len(config["drop_cols"]))
select_cols = list(
    filter(lambda x: x not in config["drop_cols"], range(len(header)))
)

data = tf.data.experimental.CsvDataset(
    config["data_path"],
    record_defaults,
    field_delim=config["sep"],
    select_cols=select_cols,
    header=True,
)

data_w = data.window(config["window_size"], drop_remainder=True).flat_map(
    lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19: x1
)
it = data_w.make_one_shot_iterator()
for x in it:
    print(x)
