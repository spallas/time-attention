import tensorflow as tf
import json
from tensorflow.python.ops.rnn_cell_impl import LSTMCell

tf.enable_eager_execution()

print("begin")

with open("conf/experiment1.json") as f:
    config = json.load(f)

driving_series = tf.ones([config["batch_size"], config["n"], config["T"]])

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

with tf.variable_scope("EncoderRNN"):
    for t in range(config["T"]):
        # if t > 0: tf.get_variable_scope().reuse_variables()
        attn_input = tf.concat([c, h], axis=1)
        print("W_e\t", W_e.shape)

        attn_input = tf.reshape(tf.tile(attn_input, [1, config["n"]]),
                                [config["batch_size"], config["n"], 2*config["m"], 1])
        print("attn_input\t", attn_input.shape)

        x = tf.matmul(W_e, attn_input)  # tf.layers.dense(attn_input, config["T"])
        x = tf.squeeze(x)
        print("x\t", x.shape)
        driving_series = tf.reshape(driving_series, [config["batch_size"], config["n"], config["T"], 1])
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
        outputs.append(h)

        print("\n\n\n")


encoder_outputs = tf.concat(outputs, axis=1)
encoder_outputs = tf.reshape(encoder_outputs, [config["batch_size"], config["T"], -1])
print(encoder_outputs)
