import tensorflow as tf
import numpy as np
import os
import procedure, utils

# Parameters
# ###########################
# GPU & Seed

tf.flags.DEFINE_string("device_id", None, "GPU device is to be used in training (default: None)")
tf.flags.DEFINE_integer("seed", 89, "Seed value for reproducibility (default: 89)")

# Model hyper-parameters
tf.flags.DEFINE_string("data_dir", None, "The input data directory (default: None)")
tf.flags.DEFINE_string("output_dir", None, "The output directory (default: None)")
tf.flags.DEFINE_string("tensorboard_dir", None, "The tensorboard directory (default: None)")
tf.flags.DEFINE_string("model_type", None, "The model type (default: None)")

tf.flags.DEFINE_integer("dense_unit", 2, "The dimensionality of the dense layer (default: 2)")
tf.flags.DEFINE_integer("rnn_unit", 4, "The number of hidden units of RNN (default: 4)")

# Training hyper-parameters
tf.flags.DEFINE_integer("nb_epoch", 20, "Number of epochs (default: 20)")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate (default: 0.001)")
tf.flags.DEFINE_float("dropout_rate", 0.3, "Dropout keep probability for RNN (default: 0.3)")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size (default: 32)")
tf.flags.DEFINE_integer("display_step", 10, "Show loss/acc for every display_step batches (default: 10)")
tf.flags.DEFINE_string("rnn_cell_type", "LSTM", " RNN Cell Type like LSTM, GRU, etc. (default: LSTM)")
tf.flags.DEFINE_integer("top_k", 10, "Top K Accuracy (default: 10)")

tf.flags.DEFINE_integer("early_stopping_k", 5, "Early stopping patience (default: 5)")
tf.flags.DEFINE_float("epsilon", 1e-8, "The epsilon threshold in training (default: 1e-8)")\

tf.flags.DEFINE_boolean("train_mode", False, "Turn on/off the training mode (default: False)")
tf.flags.DEFINE_boolean("prediction_mode", False, "Turn on/off the prediction mode (default: False)")

config = tf.flags.FLAGS
print("---------------------------------------------------")
print("SeedVal = " + str(config.seed))
print("\nParameters: " + str(config.__len__()))
for iterVal in config.__iter__():
    print(" + {}={}".format(iterVal, config.__getattr__(iterVal)))
print("Tensorflow version: ", tf.__version__)
print("---------------------------------------------------")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config.device_id

# for reproducibility
np.random.seed(config.seed)
tf.set_random_seed(config.seed)

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
gpu_config.log_device_placement = False
sess = tf.Session(config=gpu_config)

# ----------------------- MAIN PROGRAM -----------------------

data_dir = config.data_dir
output_dir = config.output_dir
tensorboard_dir=config.tensorboard_dir

training_file = data_dir + "/all_train.txt"
validate_file = data_dir + "/all_validate.txt"
testing_file = data_dir + "/all_test.txt"
print("***************************************************************************************")
print("Output Dir: " + output_dir)

# Create directories
print("@Create directories")
utils.create_folder(output_dir + "/models")
utils.create_folder(output_dir + "/topN")

if tensorboard_dir is not None:
    utils.create_folder(tensorboard_dir)

# Load train, validate & test
print("@Load train,validate&test data")
training_instances = utils.read_file_as_lines(training_file)
nb_train = len(training_instances)
total_train_batches = utils.compute_total_batches(nb_train, config.batch_size)
print(" + Total training sequences: ", nb_train)
print(" + #batches in train ", total_train_batches)

validate_instances = utils.read_file_as_lines(validate_file)
nb_validate = len(validate_instances)
total_validate_batches = utils.compute_total_batches(nb_validate, config.batch_size)
print(" + Total validating sequences: ", nb_validate)
print(" + #batches in validate ", total_validate_batches)

testing_instances = utils.read_file_as_lines(testing_file)
nb_test = len(testing_instances)
total_test_batches = utils.compute_total_batches(nb_test, config.batch_size)
print(" + Total testing sequences: ", nb_test)
print(" + #batches in test ", total_test_batches)

# Create dictionary
print("@Build knowledge")
MAX_SEQ_LENGTH, item_dict = utils.build_knowledge(training_instances, validate_instances)

print("#Statistic")
NB_ITEMS = len(item_dict)
print(" + Maximum sequence length: ", MAX_SEQ_LENGTH)
print(" + Total items: ", NB_ITEMS)

model_dir=output_dir + "/models"
if config.train_mode:
    with tf.Session(config=gpu_config) as sess:
        # Init the network
        net = procedure.create_network(sess, MAX_SEQ_LENGTH, NB_ITEMS, config)
        sess.run(tf.global_variables_initializer())

        # Train the network
        train_generator = utils.seq_batch_generator(training_instances, item_dict, MAX_SEQ_LENGTH, config.batch_size, True)
        validate_generator = utils.seq_batch_generator(validate_instances, item_dict, MAX_SEQ_LENGTH, config.batch_size, False)
        procedure.train_network(sess, net, train_generator, validate_generator, total_train_batches, total_validate_batches, 
                    config.nb_epoch, config.epsilon, config.early_stopping_k, config.display_step, tensorboard_dir, model_dir)
    tf.reset_default_graph()

# Generate prediction
if config.prediction_mode:
    with tf.Session(config=gpu_config) as sess:
        # Init the network
        net = procedure.create_network(sess, MAX_SEQ_LENGTH, NB_ITEMS, config)
        sess.run(tf.global_variables_initializer())

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        recent_dir = utils.recent_model_dir(model_dir)
        saver.restore(sess, model_dir + "/" + recent_dir + "/model.ckpt")
        print("Model restored from file: %s" % recent_dir)

        test_generator = utils.seq_batch_generator(testing_instances, item_dict, MAX_SEQ_LENGTH, config.batch_size, False)
        procedure.evaluate_network(sess, net, model_dir, test_generator, total_test_batches, nb_test, item_dict, 
            config.display_step, config.seed, output_dir + "/topN/out.txt")
    tf.reset_default_graph()