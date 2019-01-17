import tensorflow as tf
import json
from tensorflow.python.ops.rnn_cell_impl import LSTMCell

tf.enable_eager_execution()

print("begin")

with open("conf/experiment1.json") as f:
    config = json.load(f)

driving_series = tf.ones([config["batch_size"], config["n"], config["T"]])
past_history = tf.ones([config["batch_size"], config["T"]])

with tf.variable_scope("EncoderRNN"):
    cell = LSTMCell(config["m"], state_is_tuple=True)

    initial_state = cell.zero_state(config["batch_size"], tf.float32)
    state = initial_state
    c, h = state
    outputs = []

    for t in range(config["T"]):
        # if t > 0: tf.get_variable_scope().reuse_variables()
        attn_input = tf.concat([h, c], axis=1)

        attn_input = tf.reshape(tf.tile(attn_input, [1, config["n"]]), [config["batch_size"],
                                                                        config["n"], 2*config["m"]])
        print("attn_input\t", attn_input.shape)

        x = tf.layers.dense(attn_input, config["T"])
        print("x\t", x.shape)
        print(driving_series.shape)

        y = tf.layers.dense(driving_series, config["T"], use_bias=False)
        print("y\t", y.shape)

        z = tf.tanh(x + y)
        print("z\t", z.shape)

        e_t = tf.layers.dense(z, 1)

        print("e_t\t", e_t.shape)

        alpha = tf.nn.softmax(e_t)

        print("a_t\t", alpha.shape)

        alpha = tf.squeeze(alpha)

        print(driving_series[:, :, t].shape)
        x_tilde = alpha * driving_series[:, :, t]

        print("x_tilde\t", x_tilde.shape)

        (cell_output, state) = cell(x_tilde, state)
        c, h = state

        print(h.shape)
        outputs.append(h)

        print("\n\n\n")


encoder_outputs = tf.concat(outputs, axis=1)
encoder_outputs = tf.reshape(encoder_outputs, [config["batch_size"], config["T"], config["m"]])

print(encoder_outputs)

# define decoder
with tf.variable_scope("DecoderRNN"):
    cell = LSTMCell(config["p"], state_is_tuple=True)

    initial_state = cell.zero_state(config["batch_size"], tf.float32)
    state = initial_state
    s_, d = state
    c_t = tf.get_variable("c_t", [config["batch_size"], config["m"]])

    outputs = []

    for t in range(config["T"]):
        # if t > 0: tf.get_variable_scope().reuse_variables()
        attn_input = tf.concat([d, s_], axis=1)
        attn_input = tf.reshape(tf.tile(attn_input, [1, config["T"]]),
                                [config["batch_size"], config["T"], 2*config["p"]])

        print("encoder_outputs", encoder_outputs.shape)
        print("attn_input", attn_input.shape)

        x = tf.layers.dense(attn_input, config["m"])
        print("x", x.shape)

        y = tf.layers.dense(encoder_outputs, config["m"], use_bias=False)
        print("y", y.shape)

        z = tf.tanh(x + y)

        print("z", z.shape)

        l_t = tf.layers.dense(z, 1)
        beta = tf.nn.softmax(l_t)  # attention weights
        # beta = tf.squeeze(beta)

        encoder_outputs = tf.squeeze(encoder_outputs)
        print("encoder_outputs", encoder_outputs.shape)
        print("beta", beta.shape)
        c_t = tf.reduce_sum(beta * encoder_outputs, axis=1)

        print("c_t", c_t.shape)
        if t < config["T"] - 1:
            y_c = tf.concat([tf.expand_dims(past_history[:, t], -1), c_t], axis=1)
            print("y_c", y_c.shape)

            y_tilde = tf.layers.dense(y_c, 1)

            (cell_output, state) = cell(y_tilde, state)
            s_, d = state

        print("\n\n\n")

    d_c = tf.concat([d, c_t], axis=1)

    y_T = tf.layers.dense(tf.layers.dense(d_c, config["p"]), 1)
    y_T = tf.squeeze(y_T)

    print(y_T.shape)


