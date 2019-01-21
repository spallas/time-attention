import tensorflow as tf

from tensorflow.python.layers.core import dense
from tensorflow.python.ops.rnn_cell_impl import LSTMCell

from config import Config

from math import sqrt


class TimeAttnModel:

    def __init__(self, config: Config):
        self.config = config

        self.driving_series = tf.placeholder(tf.float32, [None,  # batch size
                                                          self.config.n,  # n (number of supporting series)
                                                          self.config.T])  # T (length of a time window)
        self.past_history = tf.placeholder(tf.float32, [None,  # batch size
                                                        self.config.T])  # T

        self.predictions, self.loss = self.get_predictions_and_loss(self.driving_series,
                                                                    self.past_history)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.reset_global_step = tf.assign(self.global_step, 0)
        learning_rate = tf.train.exponential_decay(self.config.learning_rate, self.global_step,
                                                   self.config.decay_frequency, self.config.decay_rate,
                                                   staircase=True)

        trainable_params_en = tf.trainable_variables(scope="EncoderRNN")
        trainable_params_dec = tf.trainable_variables(scope="DecoderRNN")

        gradients_en = tf.gradients(self.loss, trainable_params_en)
        gradients_dec = tf.gradients(self.loss, trainable_params_dec)

        gradients_en, _ = tf.clip_by_global_norm(gradients_en, self.config.max_gradient_norm)
        gradients_dec, _ = tf.clip_by_global_norm(gradients_dec, self.config.max_gradient_norm)

        optimizers = {
            "adam": tf.train.AdamOptimizer,
            "sgd": tf.train.GradientDescentOptimizer
        }

        optimizer_en = optimizers[self.config.optimizer](learning_rate)
        optimizer_dec = optimizers[self.config.optimizer](learning_rate)

        self.train_op_en = optimizer_en.apply_gradients(zip(gradients_en, trainable_params_en))
        self.train_op_dec = optimizer_dec.apply_gradients(zip(gradients_dec, trainable_params_dec), global_step=self.global_step)

        self.RMSE = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(tf.reshape(self.past_history[:, -1], [-1]),
                                                                 tf.reshape(self.predictions, [-1])))))
        self.MAE = tf.reduce_mean(
            tf.abs(
                tf.subtract(tf.reshape(self.past_history[:, -1], [-1]), tf.reshape(self.predictions, [-1]))
            )
        )
        self.MAPE = tf.reduce_mean(
            tf.abs(
                tf.divide(
                    tf.subtract(tf.reshape(self.past_history[:, -1], [-1]), tf.reshape(self.predictions, [-1])),
                    tf.reshape(self.past_history[:, -1], [-1])
                )
            )
        ) * 100

    def _attention(self, hidden_state, cell_state, input):
        attn_input = tf.concat([hidden_state, cell_state], axis=1)
        attn_input = tf.reshape(tf.tile(attn_input, [1, input.shape[1]]),
                                [self.config.batch_size, input.shape[1], 2 * hidden_state.shape[1]]
        )
        z = tf.tanh(dense(attn_input, input.shape[2]) + dense(input, input.shape[2], use_bias=False))

        pre_softmax_attn = tf.layers.dense(z, 1)
        return tf.nn.softmax(pre_softmax_attn)

    def get_predictions_and_loss(self, driving_series, past_history):

        # define encoder
        with tf.variable_scope("EncoderRNN"):
            cell = LSTMCell(self.config.m, state_is_tuple=True)

            initial_state = cell.zero_state(self.config.batch_size, tf.float32)
            state = initial_state
            s, h = state
            outputs = []

            for t in range(self.config.T):
                # if t > 0: tf.get_variable_scope().reuse_variables()
                if self.config.inp_att_enabled:
                    alpha = self._attention(h, s, driving_series)

                    # input weighted with attention weights
                    x_tilde = tf.squeeze(alpha) * driving_series[:, :, t]
                else:
                    x_tilde = driving_series[:, :, t]

                (cell_output, state) = cell(x_tilde, state)
                s, h = state
                outputs.append(h)

        encoder_outputs = tf.concat(outputs, axis=1)
        encoder_outputs = tf.reshape(encoder_outputs, [self.config.batch_size, self.config.T, -1])

        # define decoder
        with tf.variable_scope("DecoderRNN"):
            # TODO: check order of hidden states and cell states
            cell = LSTMCell(self.config.p, state_is_tuple=True)
            initial_state = cell.zero_state(self.config.batch_size, tf.float32)
            c_t = tf.get_variable("c_t", [self.config.batch_size, self.config.m])
            state = initial_state
            s_, d = state

            for t in range(self.config.T):
                # if t > 0: tf.get_variable_scope().reuse_variables()
                if self.config.temporal_att_enabled:
                    beta = self._attention(d, s_, encoder_outputs)

                    c_t = tf.reduce_sum(beta * encoder_outputs, axis=1)
                else:
                    c_t = encoder_outputs[:, t, :]

                if t < self.config.T - 1:
                    y_c = tf.concat([tf.expand_dims(past_history[:, t], -1), c_t], axis=1)
                    y_tilde = tf.layers.dense(y_c, 1)
                    (cell_output, state) = cell(y_tilde, state)
                    s_, d = state

            d_c = tf.concat([d, c_t], axis=1)
            y_T = tf.layers.dense(tf.layers.dense(d_c, self.config.p), 1)
            y_T = tf.squeeze(y_T)

        loss = tf.losses.mean_squared_error(y_T, past_history[:, - 1])
        return y_T, loss

    def evaluate(self, session, next_element):
        RMSE_tot = 0.0
        MAE_tot = 0.0
        MAPE_tot = 0.0

        num_batches = 0

        while True:
            try:
                x, y = session.run(next_element)
                RMSE, MAE, MAPE = session.run([self.RMSE, self.MAE, self.MAPE],
                                              feed_dict={self.driving_series: x, self.past_history: y})

                RMSE_tot += (RMSE ** 2) * self.config.batch_size
                MAE_tot += MAE * self.config.batch_size
                MAPE_tot += MAPE * self.config.batch_size
                num_batches += 1

            except tf.errors.OutOfRangeError:
                break

        scores = {}
        scores["RMSE"] = sqrt(RMSE_tot / (num_batches * self.config.batch_size))
        scores["MAE"] = MAE_tot / (num_batches * self.config.batch_size)
        scores["MAPE"] = MAPE_tot / (num_batches * self.config.batch_size)
        return scores
