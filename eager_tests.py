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
    # TODO: add bias to Ws, check order of hidden states and cell states
    cell = LSTMCell(config["m"], state_is_tuple=True)

    initial_state = cell.zero_state(config["batch_size"], tf.float32)
    state = initial_state
    c, h = state
    outputs = []

    v_e = tf.get_variable("v_e", [config["batch_size"], 1,
                                  config["T"]])

    W_e = tf.get_variable("W_e", [config["batch_size"], 1,
                                  config["T"], 2 * config["m"]])

    U_e = tf.get_variable("U_e", [config["batch_size"], 1,
                                  config["T"], config["T"]])

    v_e = tf.tile(v_e, [1, config["n"], 1])
    U_e = tf.tile(U_e, [1, config["n"], 1, 1])
    W_e = tf.tile(W_e, [1, config["n"], 1, 1])

    for t in range(config["T"]):
        # if t > 0: tf.get_variable_scope().reuse_variables()
        attn_input = tf.concat([h, c], axis=1)
        print("W_e\t", W_e.shape)

        attn_input = tf.reshape(tf.tile(attn_input, [1, config["n"]]), [config["batch_size"], config["n"], 2*config["m"], 1])
        print("attn_input\t", attn_input.shape)

        x = tf.matmul(W_e, attn_input)  # tf.layers.dense(attn_input, config["T"])
        x = tf.squeeze(x)
        print("x\t", x.shape)
        driving_series = tf.expand_dims(driving_series, -1)
        print("U_e\t", U_e.shape)
        print(driving_series.shape)

        y = tf.matmul(U_e, driving_series)
        y = tf.squeeze(y)
        print("y\t", y.shape)

        z = tf.tanh(x + y)
        print("z\t", z.shape)

        print("v_e\t", v_e.shape)
        e_t = tf.reduce_sum(v_e * z, axis=-1)

        print("e_t\t", e_t.shape)

        alpha = tf.nn.softmax(e_t)

        print("a_t\t", alpha.shape)

        driving_series = tf.squeeze(driving_series)
        print(driving_series.shape)
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
    # TODO: add bias to Ws, check order of hidden states and cell states
    cell = LSTMCell(config["p"], state_is_tuple=True)

    initial_state = cell.zero_state(config["batch_size"], tf.float32)
    state = initial_state
    s_, d = state
    c_t = tf.get_variable("c_t", [config["batch_size"], config["m"]])

    outputs = []

    v_d = tf.get_variable("v_e", [config["batch_size"], 1,
                                  config["m"]])

    w_tilde = tf.get_variable("w_tilde", [config["batch_size"],
                                          config["m"] + 1])

    b_tilde = tf.get_variable("b_tilde", [config["batch_size"]])

    W_d = tf.get_variable("W_e", [config["batch_size"], 1,
                                  config["m"], 2 * config["p"]])

    U_d = tf.get_variable("U_e", [config["batch_size"], 1,
                                  config["m"], config["m"]])

    v_y = tf.get_variable("v_y", [config["batch_size"], config["p"]])
    W_y = tf.get_variable("W_y", [config["batch_size"], config["p"], config["p"] + config["m"]])
    b_w = tf.get_variable("b_w", [config["batch_size"], config["p"]])
    b_v = tf.get_variable("b_v", [config["batch_size"]])

    v_d = tf.tile(v_d, [1, config["T"], 1])
    U_d = tf.tile(U_d, [1, config["T"], 1, 1])
    W_d = tf.tile(W_d, [1, config["T"], 1, 1])

    for t in range(config["T"]):
        # if t > 0: tf.get_variable_scope().reuse_variables()
        attn_input = tf.concat([d, s_], axis=1)
        attn_input = tf.reshape(tf.tile(attn_input, [1, config["T"]]),
                                [config["batch_size"], config["T"], 2*config["p"], 1])
        encoder_outputs = tf.expand_dims(encoder_outputs, -1)

        print("encoder_outputs", encoder_outputs.shape)
        print("attn_input", attn_input.shape)

        x = tf.squeeze(W_d @ attn_input)
        print("x", x.shape)

        y = tf.squeeze(U_d @ encoder_outputs)
        print("y", y.shape)

        # @ is matrix multiplication
        z = tf.tanh(x + y)

        print("z", z.shape)

        l_t = tf.reduce_sum(v_d * z, axis=-1)
        beta = tf.nn.softmax(l_t)  # attention weights

        encoder_outputs = tf.squeeze(encoder_outputs)
        print("encoder_outputs", encoder_outputs.shape)
        print("beta", beta.shape)
        c_t = tf.reduce_sum(tf.expand_dims(beta, -1) * encoder_outputs, axis=1)

        print("c_t", c_t.shape)
        if t < config["T"] - 1:
            y_c = tf.concat([tf.expand_dims(past_history[:, t], -1), c_t], axis=1)
            print("y_c", y_c.shape)

            y_tilde = tf.reduce_sum(w_tilde * y_c, axis=-1) + b_tilde

            (cell_output, state) = cell(tf.expand_dims(y_tilde, -1), state)
            s_, d = state

        print("\n\n\n")

        # outputs.append(h)

    d_c = tf.expand_dims(tf.concat([d, c_t], axis=1), -1)

    y_T = tf.reduce_sum(v_y * (tf.squeeze(W_y @ d_c) + b_w)) + b_v

    print(y_T)
