import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import LSTMCell, RNN, Dense
from keras.layers import concatenate
from sklearn.metrics.regression import mean_squared_error, mean_absolute_error

from config import Config
from data_loader import get_np_dataset

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "config",
    "conf/NASDAQ100_keras.json",
    "Path to json file with the configuration to be run",
)

config = Config.from_file(FLAGS.config)

tf.set_random_seed(config.seed)
np.random.seed(config.seed)

num_steps_ahead = 1
n_layers = 1

layers = [config.m] * n_layers

# Encoder

# driving series with shape (T, n)
encoder_inputs = Input(shape=(None, config.n))  # add endogenous series

encoder = RNN([LSTMCell(units) for units in layers], return_state=True)

encoder_out_and_states = encoder(encoder_inputs)
encoder_states = encoder_out_and_states[1:]

# Decoder
decoder_inputs = Input(shape=(config.T - num_steps_ahead,))

z_i = Dense(64)(decoder_inputs)

encoder_out = concatenate(encoder_states + [z_i])

z = Dense(256)(encoder_out)
decoder_outs = Dense(len(config.target_cols), activation="linear")(z)

model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outs)
model.compile(optimizer="adam", loss="mse")

# Training and testing

X_t, y_t = get_np_dataset(config)
X_t = X_t.transpose((0, 2, 1))
y_t = np.expand_dims(y_t, axis=2)

"""
The dataset is a sequence of overlapping windows (stride 1) of
length `config.T`.
In this scenario the model can look the values of the driving series
in the timesteps [1, `config.T` - `num_steps_ahead` + 1] and of the
target serie in the timesteps [1, `config.T` - `num_steps_ahead`] (Note
the absence of +1) in order to predict the target serie values in the
timesteps [`config.T` - `num_steps_ahead`, `config.T`].
"""

# Last `num_steps_ahead` of window of size T are the target
target_y_t = np.copy(y_t[:, -num_steps_ahead:, :])
X_t = X_t[:, :config.T - num_steps_ahead + 1, :]
decoder_input = y_t.transpose((0, 2, 1))[:, :, :-num_steps_ahead]

test_size = 2500
train_target_y_t = target_y_t[:-test_size]
train_X_t = X_t[:-test_size]
train_decoder_input = np.tile(
    decoder_input[:-test_size], (1, num_steps_ahead, 1)
)

test_target_y_t = target_y_t[-test_size:]
test_X_t = X_t[-test_size:]
test_decoder_input = np.tile(
    decoder_input[-test_size:], (1, num_steps_ahead, 1)
)

train_target_y_t = np.reshape(train_target_y_t, [train_target_y_t.shape[0],
                                                 train_target_y_t.shape[1]])

train_decoder_input = np.squeeze(train_decoder_input)
test_decoder_input = np.squeeze(test_decoder_input)

print(train_target_y_t.shape)

model.fit(
    x=[train_X_t, train_decoder_input],
    y=train_target_y_t,
    epochs=config.num_epochs,
    validation_split=0.1,
    batch_size=config.batch_size,
    callbacks=[
        EarlyStopping(
            min_delta=0.01, patience=75, mode="min", restore_best_weights=True
        )
    ],
)

model.save(str(config.log_dir) + "/saved_model.h5")

y_pred = model.predict(x=[test_X_t, test_decoder_input])

plot_target = test_target_y_t[::num_steps_ahead, :, :].reshape(-1)
plot_pred = y_pred[::num_steps_ahead, :].reshape(-1)

plt.figure()
plt.plot(plot_target, label="true")
plt.plot(plot_pred, label="predicted")
plt.legend(loc="upper left")
plt.title("Test data")
plt.ylabel("target series")
plt.xlabel("time steps")
plt.show()
print(f"RMSE {math.sqrt(mean_squared_error(plot_target, plot_pred))}")
print(f"MAE {mean_absolute_error(plot_target, plot_pred)}")
print(f"MAPE {np.abs((plot_target - plot_pred) / plot_target).mean() * 100}")
