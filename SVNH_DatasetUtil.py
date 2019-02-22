import numpy as np
import scipy.io as sio
import os.path

train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "SVHN_train_32x32.mat")
test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "SVHN_test_32x32.mat")
extra_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "SVHN_extra_32x32.mat")

#default parameters for argparse
# default_params = {
#     "learning_rate": 0.001,
#     "num_epochs": 25,
#     "batch_size": 128,
#     "train_data_file": "./assignment_httm_data/SVHN_train_32x32.mat",
#     "test_data_file": "./assignment_httm_data/SVHN_test_32x32.mat",
#     "extra_data_file": "./assignment_httm_data/SVHN_extra_32x32.mat",
#     "load_extra": False,
#     "model": "CNN1",
#     "validation_percentage": 0.1,
#     "data_shuffle": True,
#     "preprocess": False,
#     "mode": 'train',
#     "runs_name": None,
#     "tensorboard_dir": '~/tensorboard_runs'
# }

def load_raw_data(train_data_file, test_data_file, load_extra_data, extra_data_file):
    """
    Load RAW Google SVHN Digit Localization from .mat files
    """
    loading_information = "with Extra" if load_extra_data else "without Extra"
    print("Loading SVHN dataset {}...".format(loading_information))
    raw_train_data = sio.loadmat(train_data_file)
    raw_test_data = sio.loadmat(test_data_file)
    if load_extra_data:
        raw_extra_data = sio.loadmat(extra_data_file)
        print("Train size: {}, Test size: {}, Extra size: {}".format(raw_train_data['X'].shape[3],
                                                                     raw_test_data['X'].shape[3],
                                                                     raw_extra_data['X'].shape[3]))
        return [raw_train_data, raw_test_data, raw_extra_data]
    else:
        print("Train size: {}, Test size: {}".format(raw_train_data['X'].shape[3],
                                                     raw_test_data['X'].shape[3]))
        return [raw_train_data, raw_test_data]


def format_data(raw_data, number_of_examples):
    """
    Reshape RAW data to regular shape
    """
    old_shape = raw_data.shape
    new_data = []
    for i in range(number_of_examples):
        new_data.append(raw_data[:, :, :, i])
    new_data = np.asarray(new_data)
    print("Data has been reshaped from {} to {}".format(raw_data.shape, new_data.shape))
    return new_data / 255.


def one_hot_encoder(data, number_of_labels):
    """
    One-hot encoder for labels
    """
    data_size = len(data)
    one_hot_matrix = np.zeros(shape=(data_size, number_of_labels))
    for i in range(data_size):
        current_row = np.zeros(shape=(number_of_labels))
        current_number = data[i][0]
        if current_number == 10:
            current_row[0] = 1
        else:
            current_row[current_number] = 1
        one_hot_matrix[i] = current_row
    return one_hot_matrix


def load_svhn_data(train_path, test_path, extra_path, load_extra, eval_percentage):
    """
    Load SVHN Dataset
    """
    print("Loading SVHN dataset for classification...")
    # Load raw dataset
    if load_extra:

        print("Found extra dataset, loading it...")
        train, test, extra = load_raw_data(train_path, test_path, load_extra, extra_path)
        train['X'] = np.concatenate((train['X'], extra['X']), axis=3)
        train['y'] = np.concatenate((train['y'], extra['y']), axis=0)
    else:
        train, test = load_raw_data(train_path, test_path, load_extra, extra_path)

    # get values and labels
    train_all_values = format_data(train['X'], train['X'].shape[3])
    train_all_labels = one_hot_encoder(train['y'], 10)
    test_values = format_data(test['X'], test['X'].shape[3])
    test_labels = one_hot_encoder(test['y'], 10)

    np.random.seed(41)
    shuffle_indices = np.random.permutation(np.arange(len(train_all_values)))
    train_values_shuffled = train_all_values[shuffle_indices]
    train_labels_shuffled = train_all_labels[shuffle_indices]

    # Seperate into training and eval set
    # Original setting split the data into training and validation samples
    train_index = -1 * int(eval_percentage * float(len(train_values_shuffled)))
    train_values, eval_values = train_values_shuffled[:train_index], train_values_shuffled[train_index:]
    train_labels, eval_labels = train_labels_shuffled[:train_index], train_labels_shuffled[train_index:]
    print("Train/Eval split: {:d}/{:d}".format(len(train_labels), len(eval_labels)))
    print("Loading data completed")
    return [train_values, train_labels, eval_values, eval_labels, test_values, test_labels]

def my_load_svhn_data(train_path, test_path, extra_path, load_extra):
    """
    Load SVHN Dataset
    """
    print("Loading SVHN dataset for classification...")
    # Load raw dataset
    if load_extra:

        print("Found extra dataset, loading it...")
        train, test, extra = load_raw_data(train_path, test_path, load_extra, extra_path)
        train['X'] = np.concatenate((train['X'], extra['X']), axis=3)
        train['y'] = np.concatenate((train['y'], extra['y']), axis=0)
    else:
        train, test = load_raw_data(train_path, test_path, load_extra, extra_path)

    # get values and labels
    train_all_values = format_data(train['X'], train['X'].shape[3])
    train_all_labels = one_hot_encoder(train['y'], 10)
    test_values = format_data(test['X'], test['X'].shape[3])
    test_labels = one_hot_encoder(test['y'], 10)

    np.random.seed(41)
    shuffle_indices = np.random.permutation(np.arange(len(train_all_values)))
    train_values_shuffled = train_all_values[shuffle_indices]
    train_labels_shuffled = train_all_labels[shuffle_indices]
    print("Loading data completed")

    return train_values_shuffled, train_labels_shuffled, test_values, test_labels

    # Seperate into training and eval set
    # # Original setting split the data into training and validation samples
    # train_index = -1 * int(eval_percentage * float(len(train_values_shuffled)))
    # train_values, eval_values = train_values_shuffled[:train_index], train_values_shuffled[train_index:]
    # train_labels, eval_labels = train_labels_shuffled[:train_index], train_labels_shuffled[train_index:]
    # print("Train/Eval split: {:d}/{:d}".format(len(train_labels), len(eval_labels)))
    # print("Loading data completed")
    # return [train_values, train_labels, eval_values, eval_labels, test_values, test_labels]

def load_data():
    train_X, train_Y, test_X, test_Y = my_load_svhn_data(train_path = train_path,
                                                                      test_path = test_path,
                                                                      extra_path = extra_path,
                                                                      load_extra = False)

    return (train_X, train_Y), (test_X, test_Y)


if __name__ == "__main__":


    # train_X, train_Y, eval_X, eval_Y, test_X, test_Y = load_svhn_data(train_path = train_path,
    #                                                                   test_path = test_path,
    #                                                                   extra_path = extra_path,
    #                                                                   load_extra = True,
    #                                                                   eval_percentage = 0.1
    #                                                                  )
    (train_X, train_Y), (test_X, test_Y) = load_data()
    print(np.shape(train_X))
    print(np.shape(train_Y))
    print(np.shape(test_X))
    print(np.shape(test_Y))
