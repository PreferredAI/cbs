import tensorflow as tf

def create_rnn_cell(cell_type, state_size, default_initializer, reuse=None):
    if cell_type == 'GRU':
        return tf.nn.rnn_cell.GRUCell(state_size, activation=tf.nn.tanh, reuse=reuse)
    elif cell_type == 'LSTM':
        return tf.nn.rnn_cell.LSTMCell(state_size, initializer=default_initializer, activation=tf.nn.tanh, reuse=reuse)
    else:
        return tf.nn.rnn_cell.BasicRNNCell(state_size, activation=tf.nn.tanh, reuse=reuse)


def create_rnn_encoder(x, rnn_units, dropout_rate, seq_length, rnn_cell_type, param_initializer, seed, name="Bseq_Encoder", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        rnn_cell = create_rnn_cell(rnn_cell_type, rnn_units, param_initializer)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, input_keep_prob=1 - dropout_rate, seed=seed)
        init_state = rnn_cell.zero_state(tf.shape(x)[0], tf.float32)
        # RNN Encoder: Iteratively compute output of recurrent network
        rnn_outputs, _ = tf.nn.dynamic_rnn(rnn_cell, x, initial_state=init_state, sequence_length=seq_length, dtype=tf.float32)
        return rnn_outputs


def create_basket_encoder(x, dense_units, param_initializer, activation_func=None, name="Basket_Encoder", reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        return tf.layers.dense(x, dense_units, kernel_initializer=param_initializer,
                            bias_initializer=tf.zeros_initializer, activation=activation_func)


# Inspired by https://github.com/ilivans/tf-rnn-attention/blob/master/attention.py
def attentional_combination(inputs, attention_size, name="attentional_combination", return_alphas=False):
    with tf.variable_scope(name):
        if isinstance(inputs, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            inputs = tf.concat(inputs, 2)

        inputs_shape = inputs.shape
        nb_inputs = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
        hidden_size = inputs_shape[2].value  # hidden size of the RNN layer

        # Attention mechanism
        W_omega = tf.get_variable(initializer=tf.random_normal([hidden_size, attention_size], stddev=0.01), name="W_omega")
        b_omega = tf.get_variable(initializer=tf.random_normal([attention_size], stddev=0.01), name="b_omega")

        # Leaky ReLU
        x = (tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))
        v = tf.maximum(x, 0.01*x)
        vu = tf.matmul(v, tf.ones([attention_size, 1]))
        exps = tf.reshape(tf.exp(vu), [-1, nb_inputs])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

        # Output of Bi-RNN is reduced with attention vector
        output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, nb_inputs, 1]), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


def attention(inputs, attention_size, name="bseq_attention", time_major=False, return_alphas=False, reuse=None):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article
    
    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """
    with tf.variable_scope(name, reuse=reuse):
        if isinstance(inputs, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            inputs = tf.concat(inputs, 2)

        if time_major:
            # (T,B,D) => (B,T,D)
            inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

        hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

        # Trainable parameters
        w_omega = tf.get_variable(dtype=tf.float32, initializer=tf.random_normal([hidden_size, attention_size], stddev=0.01), name="w_omega")
        b_omega = tf.get_variable(dtype=tf.float32, initializer=tf.random_normal([attention_size], stddev=0.01), name="b_omega")
        u_omega = tf.get_variable(dtype=tf.float32, initializer=tf.random_normal([attention_size], stddev=0.01), name="u_omega")

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


def get_last_right_output(full_output, max_length, actual_length, dim):
    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(full_output)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * max_length + (actual_length - 1)
    # Indexing
    return tf.gather(tf.reshape(full_output, [-1, dim]), index)


def get_last_N_output(full_output, max_length, actual_length, rnn_units, N=1):
    subset_length = tf.zeros_like(actual_length) + N
    subset_length = tf.reduce_max([actual_length, subset_length], axis=0)

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(full_output)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * max_length + tf.range(actual_length-subset_length, actual_length - 1)
    # Indexing
    return tf.gather(tf.reshape(full_output, [-1, rnn_units]), index)