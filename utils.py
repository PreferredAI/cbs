import numpy as np, os, re

def build_knowledge(train_instances, validate_instances):
    MAX_LENGTH = 0
    item_dict = {}
    item_set = set()

    for ins in train_instances:
        parts = ins.split("=>")
        for i in range(0,2):
            seq = parts[i].split("|")
            if len(seq) > MAX_LENGTH:
                MAX_LENGTH = len(seq)

            item_list = re.split('[#| ]+', parts[i])
            for item in item_list:
                item_set.add(item)
        item_set.add(parts[2].rstrip())

    for ins in validate_instances:
        parts = ins.split("=>")
        for i in range(0,2):
            seq = parts[i].split("|")
            if len(seq) > MAX_LENGTH:
                MAX_LENGTH = len(seq)

            item_list = re.split('[#| ]+', parts[i])
            for item in item_list:
                item_set.add(item)
        item_set.add(parts[2].rstrip())

    items = sorted(list(item_set))
    for o in items:
        item_dict[o] = len(item_dict)

    return MAX_LENGTH, item_dict


def seq_batch_generator(raw_lines, item_dict, max_length, batch_size=32, is_train=True):
    NB_ITEMS = len(item_dict)
    total_batches = compute_total_batches(len(raw_lines), batch_size)

    X_support = []
    X_target = []
    L_support = []
    L_target = []
    Y = []
    O = []

    batch_id = 0
    while 1:
        lines = raw_lines[:]
        if is_train:
            np.random.shuffle(lines)

        for line in lines:
            parts = line.split("=>")
            support_seq = parts[0].split("|")
            target_seq = parts[1].split("|")
            target_item = parts[2].rstrip()

            # Truncate if a sequence is too long
            support_seq = truncate_seq(support_seq, max_length)
            target_seq = truncate_seq(target_seq, max_length)

            # Keep the length for dynamic_rnn
            L_support.append(len(support_seq))
            L_target.append(len(target_seq))

            # The support sequence
            Xs = np.zeros(shape=(max_length, NB_ITEMS), dtype=np.int32)
            for t, basket in enumerate(support_seq):
                basket = basket.split("#")[0]  # Ignore not trained items
                item_list = basket.rstrip().split()
                Xs[t] = create_binary_vector(item_list, item_dict)
            X_support.append(Xs)

            # The target sequence
            Xt = np.zeros(shape=(max_length, NB_ITEMS), dtype=np.int32)
            for t, basket in enumerate(target_seq):
                item_list = basket.rstrip().split()
                Xt[t] = create_binary_vector(item_list, item_dict)
            X_target.append(Xt)

            # The predicted item
            Y.append(create_binary_vector([target_item], item_dict))
            O.append(target_item)

            if len(Y) % batch_size == 0:
                yield batch_id, ({'X_s': np.asarray(X_support), 'L_s': np.asarray(L_support)},
                                 {'X_t': np.asarray(X_target), 'L_t': np.asarray(L_target)},
                                 {'Y': np.asarray(Y), 'O': np.asarray(O)})
                X_support = []
                X_target = []
                L_support = []
                L_target = []
                Y = []
                O = []

                batch_id += 1

            if batch_id == total_batches:
                batch_id = 0
                if not is_train:
                    break

def create_binary_vector(item_list, item_dict):
    v = np.zeros(len(item_dict), dtype='int32')
    for item in item_list:
        v[item_dict[item]] = 1
    return v


def truncate_seq(seq, max_length):
    seq_length = len(seq)
    new_seq = seq[:]
    if seq_length > max_length:
        new_seq = new_seq[(seq_length - max_length):]
    return new_seq


def list_dir(dir, directory_only=False):
    rtn_list = []
    for f in os.listdir(dir):
        if directory_only and os.path.isdir(os.path.join(dir, f)):
            rtn_list.append(f)
        elif not directory_only and os.path.isfile(os.path.join(dir, f)):
            rtn_list.append(f)
    return rtn_list


def create_folder(dir):
    try:
        os.makedirs(dir)
    except OSError:
        pass


def read_file_as_lines(filePath):
    with open(filePath, "r") as f:
        lines = [line.rstrip('\n') for line in f]
        return lines


def compute_total_batches(nb_intances, batch_size):
    total_batches = int(nb_intances / batch_size)
    if nb_intances % batch_size != 0:
        total_batches += 1
    return total_batches


def recent_model_dir(dir):
    folder_list = list_dir(dir, True)
    folder_list = sorted(folder_list, key=get_epoch)
    return folder_list[-1]


def get_epoch(x):
    idx = x.index('_') + 1
    return int(x[idx:])

