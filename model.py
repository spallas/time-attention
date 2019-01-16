import os
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
            b_ve = tf.get_variable("b_ve", [self.config["batch_size"], 1])
            W_e = tf.get_variable("W_e", [self.config["batch_size"], 1,
                                          self.config["T"], 2 * self.config["m"]])
            b_we = tf.get_variable("b_we", [self.config["batch_size"], 1, self.config["T"]])
            U_e = tf.get_variable("U_e", [self.config["batch_size"], 1,
                                          self.config["T"], self.config["T"]])

            v_e = tf.tile(v_e, [1, self.config["n"], 1])
            U_e = tf.tile(U_e, [1, self.config["n"], 1, 1])
            W_e = tf.tile(W_e, [1, self.config["n"], 1, 1])
            b_we = tf.tile(b_we, [1, self.config["n"], 1])
            b_ve = tf.tile(b_ve, [1, self.config["n"]])

            for t in range(self.config["T"]):
                # if t > 0: tf.get_variable_scope().reuse_variables()
                attn_input = tf.concat([h, s], axis=1)
                attn_input = tf.reshape(tf.tile(attn_input, [1, self.config["n"]]),
                                        [self.config["batch_size"], self.config["n"], 2*self.config["m"], 1])
                driving_series = tf.expand_dims(driving_series, -1)

                # @ is matrix multiplication
                z = tf.tanh(tf.squeeze(W_e @ attn_input) + b_we + tf.squeeze(U_e @ driving_series))

                e_t = tf.reduce_sum(v_e * z, axis=-1) + b_ve
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
            # TODO: check order of hidden states and cell states
            cell = LSTMCell(self.config["p"], state_is_tuple=True)
            initial_state = cell.zero_state(self.config["batch_size"], tf.float32)
            state = initial_state
            s_, d = state

            c_t = tf.get_variable("c_t", [self.config["batch_size"], self.config["m"]])
            v_d = tf.get_variable("v_e", [self.config["batch_size"], 1,
                                          self.config["m"]])
            b_vd = tf.get_variable("b_vd", [self.config["batch_size"], 1])
            w_tilde = tf.get_variable("w_tilde", [self.config["batch_size"],
                                                  self.config["m"] + 1])
            b_tilde = tf.get_variable("b_tilde", [self.config["batch_size"]])
            W_d = tf.get_variable("W_e", [self.config["batch_size"], 1,
                                          self.config["m"], 2 * self.config["p"]])
            b_wd = tf.get_variable("b_vd", [self.config["batch_size"], 1, self.config["m"]])
            U_d = tf.get_variable("U_e", [self.config["batch_size"], 1,
                                          self.config["m"], self.config["m"]])
            v_y = tf.get_variable("v_y", [self.config["batch_size"], self.config["p"]])
            W_y = tf.get_variable("W_y", [self.config["batch_size"], self.config["p"],
                                          self.config["p"] + self.config["m"]])
            b_w = tf.get_variable("b_w", [self.config["batch_size"], self.config["p"]])
            b_v = tf.get_variable("b_v", [self.config["batch_size"]])
            v_d = tf.tile(v_d, [1, self.config["T"], 1])
            U_d = tf.tile(U_d, [1, self.config["T"], 1, 1])
            W_d = tf.tile(W_d, [1, self.config["T"], 1, 1])
            b_vd = tf.tile(b_vd, [1, self.config["T"]])
            b_wd = tf.tile(b_wd, [1, self.config["T"], 1])

            for t in range(self.config["T"]):
                # if t > 0: tf.get_variable_scope().reuse_variables()
                attn_input = tf.concat([d, s_], axis=1)
                attn_input = tf.reshape(tf.tile(attn_input, [1, self.config["T"]]),
                                        [self.config["batch_size"], self.config["T"], 2 * self.config["p"], 1])
                encoder_outputs = tf.expand_dims(encoder_outputs, -1)

                # @ is matrix multiplication
                z = tf.tanh((tf.squeeze(W_d @ attn_input) + b_wd) + tf.squeeze(U_d @ encoder_outputs))

                l_t = tf.reduce_sum(v_d * z, axis=-1) + b_vd
                beta = tf.nn.softmax(l_t)  # attention weights

                encoder_outputs = tf.squeeze(encoder_outputs)
                c_t = tf.reduce_sum(tf.expand_dims(beta, -1) * encoder_outputs, axis=1)

                if t < self.config["T"] - 1:
                    y_c = tf.concat([tf.expand_dims(past_history[:, t], -1), c_t], axis=1)
                    y_tilde = tf.reduce_sum(w_tilde * y_c, axis=-1) + b_tilde
                    (cell_output, state) = cell(tf.expand_dims(y_tilde, -1), state)
                    s_, d = state

            d_c = tf.expand_dims(tf.concat([d, c_t], axis=1), -1)
            y_T = tf.reduce_sum(v_y * (tf.squeeze(W_y @ d_c) + b_w)) + b_v

        loss = tf.losses.mean_squared_error(y_T, past_history[:, self.config["T"] - 1])

        return y_T, loss

    def restore(self, session):
        vars_to_restore = [v for v in tf.global_variables()]
        saver = tf.train.Saver(vars_to_restore)
        checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
        print("Restoring from {}".format(checkpoint_path))
        session.run(tf.global_variables_initializer())
        saver.restore(session, checkpoint_path)
