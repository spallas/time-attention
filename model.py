import tensorflow as tf

from tensorflow.python.ops.rnn_cell_impl import LSTMCell


class TimeAttnModel:

    def __init__(self, config):
        self.config = config

        self.driving_series = tf.placeholder(tf.float32, [None,  # batch size
                                                          None,  # n (number of supporting series)
                                                          None])  # T (length of a time window)
        self.past_history = tf.placeholder(tf.float32, [None,  # batch size
                                                        None,  # T
                                                        1])  # y_i

        self.predictions, self.loss = self.get_predictions_and_loss(self.driving_series,
                                                                    self.past_history)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.reset_global_step = tf.assign(self.global_step, 0)
        learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                                   self.config["decay_frequency"], self.config["decay_rate"],
                                                   staircase=True)

        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
        optimizers = {
            "adam": tf.train.AdamOptimizer,
            "sgd": tf.train.GradientDescentOptimizer
        }
        optimizer = optimizers[self.config["optimizer"]](learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)

    def get_predictions_and_loss(self, driving_series, past_history):
        # define encoder

        with tf.variable_scope("EncoderRNN"):
            cell = LSTMCell(self.config["m"], state_is_tuple=True)

            initial_state = cell.zero_state(self.config["batch_size"], tf.float32)
            state = initial_state
            s, h = state
            outputs = []

            v_e = tf.get_variable("v_e", [self.config["batch_size"], 1,
                                          self.config["T"]])

            W_e = tf.get_variable("W_e", [self.config["batch_size"], 1,
                                          self.config["T"], 2 * self.config["m"]])

            U_e = tf.get_variable("U_e", [self.config["batch_size"], 1,
                                          self.config["T"], self.config["T"]])

            v_e = tf.tile(v_e, [1, self.config["n"], 1])
            U_e = tf.tile(U_e, [1, self.config["n"], 1, 1])
            W_e = tf.tile(W_e, [1, self.config["n"], 1, 1])

            for t in range(self.config["T"]):
                # if t > 0: tf.get_variable_scope().reuse_variables()
                attn_input = tf.concat([h, s], axis=1)
                attn_input = tf.reshape(tf.tile(attn_input, [1, self.config["n"]]),
                                        [self.config["batch_size"], self.config["n"], 2*self.config["m"], 1])
                driving_series = tf.expand_dims(driving_series, -1)

                # @ is matrix multiplication
                z = tf.tanh(tf.squeeze(W_e @ attn_input) + tf.squeeze(U_e @ driving_series))

                e_t = tf.reduce_sum(v_e * z, axis=-1)
                alpha = tf.nn.softmax(e_t)  # attention weights

                driving_series = tf.squeeze(driving_series)
                x_tilde = alpha * driving_series[:, :, t]  # input weighted with attention weights

                (cell_output, state) = cell(x_tilde, state)
                s, h = state
                outputs.append(h)

        encoder_outputs = tf.concat(outputs, axis=1)
        encoder_outputs = tf.reshape(encoder_outputs, [self.config["batch_size"], self.config["T"], -1])

        # define decoder
        with tf.variable_scope("DecoderRNN"):
            cell = LSTMCell(self.config["m"], state_is_tuple=True)

            initial_state = cell.zero_state(self.config["batch_size"], tf.float32)
            state = initial_state
            s, h = state
            outputs = []

            for t in range(self.config["T"]):
                # if t > 0: tf.get_variable_scope().reuse_variables()
                pass

        return []
