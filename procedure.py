import tensorflow as tf
import numpy as np
import models, utils
import time


def create_network(sess, MAX_SEQ_LENGTH, NB_ITEMS, config):
    model_type = config.model_type
    if "mcnet" in model_type:
        net = models.MCNET(sess, config.dense_unit, MAX_SEQ_LENGTH, NB_ITEMS, model_type, 
                            config.batch_size, config.top_k, config.seed, config.learning_rate)
    elif "bseq" in model_type:
        net = models.BSEQ(sess, config.dense_unit, config.rnn_unit, MAX_SEQ_LENGTH, NB_ITEMS, model_type,
                                config.batch_size, config.top_k, config.rnn_cell_type, config.dropout_rate, config.seed, config.learning_rate)
    elif "sn" in model_type:
        if model_type.endswith("att"):
            net = models.SN(sess, config.dense_unit, config.rnn_unit, MAX_SEQ_LENGTH, NB_ITEMS, True, model_type,
                                config.batch_size, config.top_k, config.rnn_cell_type, config.dropout_rate, config.seed, config.learning_rate)
        else:
            net = models.SN(sess, config.dense_unit, config.rnn_unit, MAX_SEQ_LENGTH, NB_ITEMS, False, model_type,
                                config.batch_size, config.top_k, config.rnn_cell_type, config.dropout_rate, config.seed, config.learning_rate)
    elif "cfn" in model_type:
        if model_type.endswith("att"):
            net = models.CFN(sess, config.dense_unit, config.rnn_unit, MAX_SEQ_LENGTH, NB_ITEMS, True, model_type,
                                config.batch_size, config.top_k, config.rnn_cell_type, config.dropout_rate, config.seed, config.learning_rate)
        else:
            net = models.CFN(sess, config.dense_unit, config.rnn_unit, MAX_SEQ_LENGTH, NB_ITEMS, False, model_type,
                                config.batch_size, config.top_k, config.rnn_cell_type, config.dropout_rate, config.seed, config.learning_rate)
    elif "dfn" in model_type:
        if model_type.endswith("att"):
            net = models.DFN(sess, config.dense_unit, config.rnn_unit, MAX_SEQ_LENGTH, NB_ITEMS, True, model_type,
                            config.batch_size, config.top_k, config.rnn_cell_type, config.dropout_rate, config.seed, config.learning_rate)
        else:
            net = models.DFN(sess, config.dense_unit, config.rnn_unit, MAX_SEQ_LENGTH, NB_ITEMS, False, model_type,
                            config.batch_size, config.top_k, config.rnn_cell_type, config.dropout_rate, config.seed, config.learning_rate)
    return net

def train_network(sess, net, train_generator, validate_generator, total_train_batches, total_validate_batches, 
                nb_epoch, epsilon, early_stopping_k, display_step, tensorboard_dir, output_dir):

    model_type = net.get_model_type()
    summary_writer = None
    if tensorboard_dir is not None:
        summary_writer = tf.summary.FileWriter(tensorboard_dir)
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    val_best_acc = [-1.0]
    patience_cnt = 0
    for epoch in range(0, nb_epoch):
        print("\n-----------------------------------------------------")
        print("@Epoch#" + str(epoch))


        print("\n----------------- TRAIN ----------------------------")
        train_loss = 0.0
        train_acc = 0.0

        for batch_id, data in train_generator:
            start_time = time.time()
            if model_type.startswith("bseq") or model_type.startswith("mcnet"):
                if "target" in model_type:
                    loss, acc, summary = net.train_batch(data[1]['X_t'], data[1]["L_t"], data[2]['Y'])
                else:
                    loss, acc, summary = net.train_batch(data[0]['X_s'], data[0]["L_s"], data[2]['Y'])
            else:
                loss, acc, summary = net.train_batch([data[0]['X_s'],  data[1]['X_t']],
                                                     [data[0]["L_s"], data[1]["L_t"]], data[2]['Y'])

            train_loss += loss
            avg_train_loss = train_loss / (batch_id + 1)

            train_acc += acc
            avg_train_acc = train_acc / (batch_id + 1)

            # Write logs at every iteration
            if summary_writer is not None:
                summary_writer.add_summary(summary, epoch * total_train_batches + batch_id)

                loss_sum = tf.Summary()
                loss_sum.value.add(tag="Loss/Train_Loss", simple_value=avg_train_loss)
                summary_writer.add_summary(loss_sum, epoch * total_train_batches + batch_id)

                acc_sum = tf.Summary()
                acc_sum.value.add(tag="Accuracy/Train_Acc", simple_value=avg_train_acc)
                summary_writer.add_summary(acc_sum, epoch * total_train_batches + batch_id)

            if batch_id % display_step == 0 or batch_id == total_train_batches - 1:
                running_time = time.time() - start_time
                print("Training | Epoch " +  str(epoch) + " | "  + str(batch_id + 1) + "/" + str(total_train_batches) 
                    + " | Loss= " + "{:.8f}".format(avg_train_loss) + ", Accuracy= " + "{:.8f}".format(avg_train_acc)
                    + " | Time={:.2f}".format(running_time) + "s")

            if batch_id >= total_train_batches - 1:
                break

        print("\n----------------- VALIDATION ----------------------------")
        val_loss = 0.0
        val_acc = 0.0
        for batch_id, data in validate_generator:
            if model_type.startswith("bseq") or model_type.startswith("mcnet"):
                if "target" in model_type:
                    loss, acc, summary = net.validate_batch(data[1]['X_t'], data[1]["L_t"], data[2]['Y'])
                else:
                    loss, acc, summary = net.validate_batch(data[0]['X_s'], data[0]["L_s"], data[2]['Y'])
            else:
                loss, acc, summary = net.validate_batch([data[0]['X_s'], data[1]['X_t']],
                                                     [data[0]["L_s"], data[1]["L_t"]], data[2]['Y'])
            val_loss += loss
            avg_val_loss = val_loss / (batch_id + 1)

            val_acc += acc
            avg_val_acc = val_acc / (batch_id + 1)

            # Write logs at every iteration
            if summary_writer is not None:
                summary_writer.add_summary(summary, epoch * total_validate_batches + batch_id)

                loss_sum = tf.Summary()
                loss_sum.value.add(tag="Loss/Val_Loss", simple_value=avg_val_loss)
                summary_writer.add_summary(loss_sum, epoch * total_validate_batches + batch_id)

                acc_sum = tf.Summary()
                acc_sum.value.add(tag="Accuracy/Val_Acc", simple_value=avg_val_acc)
                summary_writer.add_summary(acc_sum, epoch * total_validate_batches + batch_id)
            
            if batch_id % display_step == 0 or batch_id == total_validate_batches - 1:
                print("Validating | Epoch " + str(epoch) + " | " + str(batch_id + 1) + "/" + str(total_validate_batches) 
                    + " | Loss = " + "{:.8f}".format(avg_val_loss) + " | Accuracy = " + "{:.8f}".format(avg_val_acc))

            if batch_id >= total_validate_batches - 1:
                break

        avg_val_acc = val_acc / total_validate_batches
        print("\n@ The validation's accuracy = " + str(avg_val_acc))
        imprv_ratio = (avg_val_acc - val_best_acc[-1] )/val_best_acc[-1]

        if val_best_acc[-1] < 0 or imprv_ratio > epsilon:
            print("# The validation's accuracy is improved from " + "{:.6f}".format(val_best_acc[-1]) + \
                  " to " + "{:.6f}".format(avg_val_acc))
            val_best_acc.append(avg_val_acc)

            patience_cnt = 0

            save_dir = output_dir + "/epoch_" + str(epoch)
            utils.create_folder(save_dir)

            save_path = saver.save(sess, save_dir + "/model.ckpt")
            print("The model is saved in: %s" % save_path)
        else:
            patience_cnt += 1

        if patience_cnt >= early_stopping_k:
            print("# The training is early stopped at Epoch " + str(epoch))
            break


def evaluate_network(sess, net, model_dir, test_generator, total_test_batches, actual_nb_test, item_dict, display_step, seed, output_file):
    model_type = net.get_model_type()
    nb_items = len(item_dict)

    print("@Save evaluation metrics to " + output_file)
    f = open(output_file, "w")
    ranks = []
    for batch_id, data in test_generator:
        if model_type.startswith("bseq") or model_type.startswith("mcnet"):
            if model_type.endswith("target"):
                Y_pred = net.generate_prediction(data[1]['X_t'], data[1]["L_t"])
            else:
                Y_pred = net.generate_prediction(data[0]['X_s'], data[0]["L_s"])
        else:
            Y_pred = net.generate_prediction([data[0]['X_s'], data[1]['X_t']],
                                             [data[0]["L_s"], data[1]["L_t"]])
        
        for i, probs in enumerate(Y_pred):
            target_item = data[2]['O'][i]
            score = probs[item_dict[target_item]]
            rank = sum(v > score for v in probs) + 1
            ranks.append(rank)
        
        if batch_id % display_step == 0 or batch_id == total_test_batches - 1:
            print(str(batch_id + 1) + "/" + str(total_test_batches))

        if batch_id >= total_test_batches - 1:
            break
    
    ranks = ranks[:actual_nb_test]
    ranks = np.asarray(ranks)

    print("+ Metrics: ")
    for k in [1, 5, 10, 20, 50]:
        recall_at_k = (ranks <= k).sum(0) * 100.0 / actual_nb_test
        f.write(str(k) + "," + str(recall_at_k) + "\n")
        print("    R@" + str(k), "=", recall_at_k)


    mrrs = 1.0 / ranks
    mrrs[ranks > 200] = 0
    mrr = np.sum(mrrs) / actual_nb_test
    f.write("mrr," + str(mrr) + "\n")
    print("    MRR", "=", mrr)

    f.close()