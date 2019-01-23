import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import LSTM, Concatenate, Permute, Reshape, LSTMCell, RNN, Dense

from config import Config
from data_loader import get_np_dataset

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('config', 'conf/SML2010.json', 'Path to json file with the configuration to be run')

config = Config.from_file(FLAGS.config)
# set seeds for 'reproducibility'
tf.set_random_seed(config.seed)
np.random.seed(config.seed)

layers = [32, 32]

num_steps_ahead = 1

X_t, y_t = get_np_dataset(config)
X = Input(shape=(config.n, config.T))
past = Input(shape=(config.T,))
past_r = Reshape((1, config.T))(past)
z = Concatenate(axis=1)([X, past_r])

num_input_features = 1  # The dimensionality of the input at each time step. In this case a 1D signal.
num_output_features = 1  # The dimensionality of the output at each time step. In this case a 1D signal.

encoder_inputs = Input(shape=(None, num_input_features))

# Create a list of RNN Cells, these are then concatenated into a single layer
# with the RNN layer.
encoder_cells = []
for hidden_neurons in layers:
    encoder_cells.append(LSTMCell(hidden_neurons))

encoder = RNN(encoder_cells, return_state=True)

encoder_outputs_and_states = encoder(encoder_inputs)

# Discard encoder outputs and only keep the states.
# The outputs are of no interest to us, the encoder's
# job is to create a state describing the input sequence.
encoder_states = encoder_outputs_and_states[1:]

# The decoder input will be set to zero (see random_sine function of the utils module).
# Do not worry about the input size being 1, I will explain that in the next cell.
decoder_inputs = Input(shape=(None, 1))

decoder_cells = []
for hidden_neurons in layers:
    decoder_cells.append(LSTMCell(hidden_neurons))

decoder = RNN(decoder_cells, return_sequences=True, return_state=True)

# Set the initial state of the decoder to be the output state of the encoder.
# This is the fundamental part of the encoder-decoder.
decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

# Only select the output of the decoder (not the states)
decoder_outputs = decoder_outputs_and_states[0]

# Apply a dense layer with linear activation to set output to correct dimension
# and scale (tanh is default activation for GRU in Keras, our output sine function can be larger then 1)

decoder_outputs = Dense(num_output_features)(decoder_outputs)


model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.summary()

y_history = y_t
y_future = np.copy(y_t[:, -num_steps_ahead:])
y_history = y_history[:, :-num_steps_ahead]
y_dec_input = np.zeros([y_history.shape[0], y_history.shape[1], 1])
y_history = np.expand_dims(y_history, axis=2)
y_future = np.expand_dims(y_future, axis=2)

model.fit(x=[y_history, y_dec_input], y=y_future,
          epochs=100, batch_size=128,
          validation_split=0.2)

