import tensorflow as tf

from abc import abstractmethod
import layers


class Model:
    def __init__(self, sess, seed, learning_rate, model_type):
        self.session = sess
        self.seed = seed
        self.learning_rate = tf.constant(learning_rate)
        self.scope = model_type
        self.model_type = model_type

    @abstractmethod
    def train_batch(self, x_batch, x_batch_length, y_batch):
        pass

    @abstractmethod
    def validate_batch(self, x_batch, x_batch_length, y_batch):
        pass

    @abstractmethod
    def generate_prediction(self, x_batch, x_batch_length):
        pass  

    def create_optimization_block(self, logits, y, top_k):
        self.prediction = tf.nn.softmax(logits)
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        self.top_k_acc = top_k_categorical_accuracy(y, self.prediction, top_k)

        # Adam optimizer
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # Op to calculate every variable gradient
        self.grads = train_op.compute_gradients(self.loss, tf.trainable_variables())
        self.update_grads = train_op.apply_gradients(self.grads)

        # Summarize all variables and their gradients
        print("-------------------- SUMMARY ----------------------")
        total_parameters = 0
        for grad, var in self.grads:
            print(var.name, " ", grad)
            tf.summary.histogram(var.name, var)
            tf.summary.histogram(var.name + '/grad', grad)
            
            shape = var.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
                print("+ {:<64} {:<10,} parameter(s)".format(var.name, variable_parameters))
                total_parameters += variable_parameters

        print("Total number of parameters: {:,}".format(total_parameters))
        print("----------------- END SUMMARY ----------------------\n")

        # Create a summary to monitor cost tensor
        tf.summary.scalar("Batch_Train_Loss", self.loss)
        tf.summary.scalar("Batch_Train_Acc", self.top_k_acc)

        # Create a summary to monitor cost tensor
        tf.summary.scalar("Batch_Val_Loss", self.loss, collections=['validation'])
        tf.summary.scalar("Batch_Val_Acc", self.top_k_acc, collections=['validation'])

        # Merge all summaries into a single op
        self.merged_summary_op = tf.summary.merge_all()
        self.val_merged_summary_op = tf.summary.merge_all(key='validation')

    def get_model_type(self):
        return self.model_type


class MCNET(Model):
    def __init__(self, sess, dense_unit, max_length, nb_items, model_type, batch_size, top_k, seed, learning_rate):
        super().__init__(sess, seed, learning_rate, model_type)
        self.dense_unit = dense_unit
        self.max_length = max_length
        self.nb_items = nb_items

        with tf.variable_scope(self.scope):
            self.bseq = tf.placeholder(tf.float32, shape=(batch_size, self.max_length, self.nb_items), name='raw_bseq')
            self.bseq_length = tf.placeholder(tf.int32, shape=(batch_size,), name='raw_bseq_length')
            self.y = tf.placeholder(tf.float32, shape=(batch_size, nb_items), name='target_item')

            # Basket encoder
            basket_encoder = layers.create_basket_encoder(self.bseq, self.dense_unit, 
                param_initializer=tf.initializers.he_uniform(), activation_func=tf.nn.relu)
            # Hack to build the indexing and retrieve the right output.
            last_output = layers.get_last_right_output(basket_encoder, self.max_length, self.bseq_length,
                                                           self.dense_unit)
            with tf.variable_scope("Aggregate_Layer"):
                W_Agg = tf.get_variable(dtype=tf.float32,
                                             initializer=tf.random_normal((self.dense_unit, self.nb_items), stddev=0.01),
                                             name="W_Agg")
                B_Agg = tf.get_variable(initializer=tf.random_normal((1, self.nb_items), stddev=0.01), name="B_Agg")
                logits = tf.matmul(last_output, W_Agg) + B_Agg

            with tf.name_scope("Optimization"):
                self.create_optimization_block(logits, self.y, top_k)

    def train_batch(self, x, x_length, y):
        _, loss, acc, summary = self.session.run([self.update_grads, self.loss, self.top_k_acc, self.merged_summary_op],
                                                 feed_dict={self.bseq: x, self.bseq_length: x_length, self.y: y})
        return loss, acc, summary

    def validate_batch(self, x, x_length, y):
        return  self.session.run([self.loss, self.top_k_acc, self.val_merged_summary_op],
                                            feed_dict={self.bseq: x, self.bseq_length: x_length, self.y: y})

    def generate_prediction(self, x, x_length):
        return self.session.run(self.prediction, feed_dict={self.bseq: x, self.bseq_length: x_length})


class BSEQ(Model):
    def __init__(self, sess, dense_units, rnn_units, max_length, nb_items,
                 model_type, batch_size, top_k, rnn_cell_type, rnn_dropout_rate, seed, learning_rate):
        super().__init__(sess, seed, learning_rate, model_type)
        self.dense_units = dense_units
        self.rnn_units = rnn_units
        self.max_length = max_length
        self.nb_items = nb_items
        self.rnn_cell_type = rnn_cell_type

        with tf.variable_scope(self.scope):
            self.bseq = tf.placeholder(tf.float32, shape=(batch_size, self.max_length, self.nb_items),
                                         name='raw_bseq')
            self.bseq_length = tf.placeholder(tf.int32, shape=(batch_size, ), name='raw_bseq_length')
            self.y = tf.placeholder(tf.float32, shape=(None, nb_items), name='target_item')
           
            basket_encoder = layers.create_basket_encoder(self.bseq, self.dense_units, 
                param_initializer=tf.initializers.he_uniform(), activation_func=tf.nn.relu)

            rnn_outputs = layers.create_rnn_encoder(basket_encoder, self.rnn_units, rnn_dropout_rate,
                            self.bseq_length, rnn_cell_type, param_initializer=tf.initializers.glorot_uniform(), seed=self.seed)

            with tf.variable_scope("Aggregate_Layer"):
                # Hack to build the indexing and retrieve the right output.
                last_rnn_output = layers.get_last_right_output(rnn_outputs, self.max_length, self.bseq_length,
                                                                    self.rnn_units)
                W_Agg = tf.get_variable(dtype=tf.float32,
                                        initializer=tf.random_normal((self.rnn_units, self.nb_items), stddev=0.01),
                                        name="W_Agg")
                B_Agg = tf.get_variable(dtype=tf.float32,
                                        initializer=tf.random_normal((1, self.nb_items), stddev=0.01), name="B_Agg")
                logits = tf.matmul(last_rnn_output, W_Agg) + B_Agg

            with tf.name_scope("Optimization"):
                self.create_optimization_block(logits, self.y, top_k)

    def train_batch(self, x, x_length, y):
        _, loss, acc, summary = self.session.run([self.update_grads, self.loss, self.top_k_acc, self.merged_summary_op],
                                                 feed_dict={self.bseq: x, self.bseq_length: x_length, self.y: y})
        return loss, acc, summary

    def validate_batch(self, x, x_length, y):
        return self.session.run([self.loss, self.top_k_acc, self.val_merged_summary_op],
                                    feed_dict={self.bseq: x, self.bseq_length: x_length, self.y: y})

    def generate_prediction(self, x, x_length):
        return self.session.run(self.prediction, feed_dict={self.bseq: x, self.bseq_length: x_length})


class MULTIPLE_BSEQ(Model):
    def __init__(self, sess, dense_units, rnn_units, max_length, nb_items, use_attention,
                 model_type, batch_size, top_k, rnn_cell_type, rnn_dropout_rate, seed, learning_rate):
        super().__init__(sess, seed, learning_rate, model_type)
        self.dense_units = dense_units
        self.rnn_units = rnn_units
        self.max_length = max_length
        self.nb_items = nb_items
        self.rnn_cell_type = rnn_cell_type

    def train_batch(self, x, x_length, y):
        _, loss, acc, summary = self.session.run(
            [self.update_grads, self.loss, self.top_k_acc, self.merged_summary_op],
            feed_dict={self.bseq_support: x[0], self.bseq_support_length: x_length[0],
                       self.bseq_target: x[1], self.bseq_target_length: x_length[1], self.y: y})
        return loss, acc, summary

    def validate_batch(self, x, x_length, y):
        return self.session.run([self.loss, self.top_k_acc, self.val_merged_summary_op],
                    feed_dict={self.bseq_support: x[0], self.bseq_support_length: x_length[0],
                    self.bseq_target: x[1], self.bseq_target_length: x_length[1], self.y: y})

    def generate_prediction(self, x, x_length):
        return self.session.run(self.prediction, feed_dict={self.bseq_support: x[0], self.bseq_support_length: x_length[0],
                                    self.bseq_target: x[1], self.bseq_target_length: x_length[1]})


class SN(MULTIPLE_BSEQ):
    def __init__(self, sess, dense_units, rnn_units, max_length, nb_items, use_attention,
                 model_type, batch_size, top_k, rnn_cell_type, rnn_dropout_rate, seed, learning_rate):
        super().__init__(sess, dense_units, rnn_units, max_length, nb_items, use_attention, model_type, batch_size, top_k, rnn_cell_type, rnn_dropout_rate, seed, learning_rate)

        with tf.variable_scope(self.scope):
            self.bseq_support = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.max_length, self.nb_items),
                                               name='bseq_support')
            self.bseq_support_length = tf.placeholder(dtype=tf.int32, shape=(batch_size,), name='bseq_support_length')

            self.bseq_target = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.max_length, self.nb_items),
                                              name='bseq_target')
            self.bseq_target_length = tf.placeholder(dtype=tf.int32, shape=(batch_size,), name='bseq_target_length')
            self.y = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_items), name='target_item')


            # Encode the support basket sequence
            bseq_support_encoder = layers.create_basket_encoder(self.bseq_support, self.dense_units, 
                                            param_initializer=tf.initializers.he_uniform(), activation_func=tf.nn.relu)
            bseq_support_encoder = layers.create_rnn_encoder(bseq_support_encoder, self.rnn_units, rnn_dropout_rate, self.bseq_support_length, rnn_cell_type, 
                                            param_initializer=tf.initializers.glorot_uniform(), seed=self.seed)

            # Encode the target basket sequence
            bseq_target_encoder = layers.create_basket_encoder(self.bseq_target, self.dense_units, 
                                            param_initializer=tf.initializers.he_uniform(), activation_func=tf.nn.relu, reuse=True)
            bseq_target_encoder = layers.create_rnn_encoder(bseq_target_encoder, self.rnn_units, rnn_dropout_rate, self.bseq_target_length, rnn_cell_type, 
                                            param_initializer=tf.initializers.glorot_uniform(), seed=self.seed, reuse=True)

            with tf.variable_scope("Aggregate_Layer"):
                if use_attention:
                    support_output = layers.attention(bseq_support_encoder, self.rnn_units)
                    target_output = layers.attention(bseq_target_encoder, self.rnn_units, reuse=True)
                else:
                    # Hack to build the indexing and retrieve the right output.
                    support_output = layers.get_last_right_output(bseq_support_encoder, self.max_length, self.bseq_support_length, self.rnn_units)
                    target_output = layers.get_last_right_output(bseq_target_encoder, self.max_length, self.bseq_target_length, self.rnn_units)
                
                concat = tf.concat([support_output, target_output], axis=1)
                W_Agg = tf.get_variable(dtype=tf.float32, initializer=tf.random_normal((self.rnn_units * 2, self.nb_items), stddev=0.01), name="W_Agg")
                B_Agg = tf.get_variable(dtype=tf.float32, initializer=tf.random_normal((1, self.nb_items), stddev=0.01), name="B_Agg")
                logits = tf.matmul(concat, W_Agg) + B_Agg
            
            with tf.name_scope("Optimization"):
                self.create_optimization_block(logits, self.y, top_k)
            

class CFN(MULTIPLE_BSEQ):
    def __init__(self, sess, dense_units, rnn_units, max_length, nb_items, use_attention,
                 model_type, batch_size, top_k, rnn_cell_type, rnn_dropout_rate, seed, learning_rate):
        super().__init__(sess, dense_units, rnn_units, max_length, nb_items, use_attention, model_type, batch_size, top_k, rnn_cell_type, rnn_dropout_rate, seed, learning_rate)

        with tf.variable_scope(self.scope):
            self.bseq_support = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.max_length, self.nb_items),
                                               name='bseq_support')
            self.bseq_support_length = tf.placeholder(dtype=tf.int32, shape=(batch_size,), name='bseq_support_length')

            self.bseq_target = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.max_length, self.nb_items),
                                              name='bseq_target')
            self.bseq_target_length = tf.placeholder(dtype=tf.int32, shape=(batch_size,), name='bseq_target_length')
            self.y = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_items), name='target_item')

            # Encode the support basket sequence
            bseq_support_encoder = layers.create_basket_encoder(self.bseq_support, self.dense_units, 
                                            param_initializer=tf.initializers.he_uniform(), activation_func=tf.nn.relu)
            bseq_support_encoder = layers.create_rnn_encoder(bseq_support_encoder, self.rnn_units, rnn_dropout_rate, self.bseq_support_length, rnn_cell_type, 
                                            param_initializer=tf.initializers.glorot_uniform(), seed=self.seed, name="Bseq_Support_Encoder")

            # Encode the target basket sequence
            bseq_target_encoder = layers.create_basket_encoder(self.bseq_target, self.dense_units, 
                                            param_initializer=tf.initializers.he_uniform(), activation_func=tf.nn.relu, reuse=True)
            bseq_target_encoder = layers.create_rnn_encoder(bseq_target_encoder, self.rnn_units, rnn_dropout_rate, self.bseq_target_length, rnn_cell_type, 
                                            param_initializer=tf.initializers.glorot_uniform(), seed=self.seed, name="Bseq_Target_Encoder")

            with tf.variable_scope("Aggregate_Layer"):
                # Hack to build the indexing and retrieve the right output.
                if use_attention:
                    support_output = layers.attention(bseq_support_encoder, self.rnn_units, name="bseq_support_attention")
                    target_output = layers.attention(bseq_target_encoder, self.rnn_units, name="bseq_target_attention")
                else:
                    support_output = layers.get_last_right_output(bseq_support_encoder, self.max_length, self.bseq_support_length, self.rnn_units)
                    target_output = layers.get_last_right_output(bseq_target_encoder, self.max_length, self.bseq_target_length, self.rnn_units)
                
                W_Agg_S = tf.get_variable(dtype=tf.float32, initializer=tf.random_normal((self.rnn_units, self.nb_items), stddev=0.01), name="W_Agg_S")
                W_Agg_T = tf.get_variable(dtype=tf.float32, initializer=tf.random_normal((self.rnn_units, self.nb_items), stddev=0.01), name="W_Agg_T")

                B_Agg = tf.get_variable(dtype=tf.float32, initializer=tf.random_normal((1, self.nb_items), stddev=0.01), name="B_Agg")
                logits = tf.matmul(support_output, W_Agg_S) + tf.matmul(target_output, W_Agg_T) + B_Agg

            with tf.name_scope("Optimization"):
                self.create_optimization_block(logits, self.y, top_k)


class DFN(MULTIPLE_BSEQ):
    def __init__(self, sess, dense_units, rnn_units, max_length, nb_items, use_attention,
                 model_type, batch_size, top_k, rnn_cell_type, rnn_dropout_rate, seed, learning_rate):
        super().__init__(sess, dense_units, rnn_units, max_length, nb_items, use_attention, model_type, batch_size, top_k, rnn_cell_type, rnn_dropout_rate, seed, learning_rate)

        with tf.variable_scope(self.scope):
            self.bseq_support = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.max_length, self.nb_items),
                                               name='bseq_support')
            self.bseq_support_length = tf.placeholder(dtype=tf.int32, shape=(batch_size,), name='bseq_support_length')

            self.bseq_target = tf.placeholder(dtype=tf.float32, shape=(batch_size, self.max_length, self.nb_items),
                                              name='bseq_target')
            self.bseq_target_length = tf.placeholder(dtype=tf.int32, shape=(batch_size,), name='bseq_target_length')
            self.y = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_items), name='target_item')

             # Encode the support basket sequence
            bseq_support_encoder = layers.create_basket_encoder(self.bseq_support, self.dense_units, 
                                            param_initializer=tf.initializers.he_uniform(), activation_func=tf.nn.relu)
            bseq_support_encoder = layers.create_rnn_encoder(bseq_support_encoder, self.rnn_units, rnn_dropout_rate, self.bseq_support_length, rnn_cell_type, 
                                            param_initializer=tf.initializers.glorot_uniform(), seed=self.seed, name="Bseq_Support_Encoder")

            # Encode the target basket sequence
            bseq_target_encoder = layers.create_basket_encoder(self.bseq_target, self.dense_units, 
                                            param_initializer=tf.initializers.he_uniform(), activation_func=tf.nn.relu, reuse=True)

            with tf.variable_scope("Aggregate_Layer"):
                if use_attention:
                    support_output = layers.attention(bseq_support_encoder, self.rnn_units, name="bseq_support_attention")
                else:
                    # Hack to build the indexing and retrieve the right output.
                    support_output = layers.get_last_right_output(bseq_support_encoder, self.max_length, self.bseq_support_length, self.rnn_units)

                target_output = layers.get_last_right_output(bseq_target_encoder, self.max_length, self.bseq_target_length, self.dense_units)
                
                W_Agg_S = tf.get_variable(dtype=tf.float32, initializer=tf.random_normal((self.rnn_units, self.nb_items), stddev=0.01), name="W_Agg_S")
                W_Agg_T = tf.get_variable(dtype=tf.float32, initializer=tf.random_normal((self.dense_units, self.nb_items), stddev=0.01), name="W_Agg_T")
                B_Agg = tf.get_variable(dtype=tf.float32, initializer=tf.random_normal((1, self.nb_items), stddev=0.01), name="B_Agg")
                logits = tf.matmul(support_output, W_Agg_S) + tf.matmul(target_output, W_Agg_T) + B_Agg
            
            with tf.name_scope("Optimization"):
                self.create_optimization_block(logits, self.y, top_k)            


def categorical_accuracy(y_true, y_pred):
    correct_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Follow keras
def top_k_categorical_accuracy(y_true, y_pred, k=5):
    top_k = tf.nn.in_top_k(y_pred, tf.argmax(y_true, 1), k)
    return tf.reduce_mean(tf.cast(top_k, tf.float32), axis=-1)
