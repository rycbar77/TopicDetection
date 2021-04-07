import random
from configs import *
import os

from file_utils import *


def init_test_files(data_path, train_path, test_path):
    files = get_files(data_path)
    # files = random.sample(files, 40)
    random.shuffle(files)
    train_size = int(len(files) * 0.8)
    train_file_list = files[:train_size]
    test_file_list = files[train_size:]
    # train_files = list(set([i.split('\\')[1] for i in train_file_list]))
    # test_files = list(set([i.split('\\')[1] for i in test_file_list]))
    recreate_dir(train_path)
    recreate_dir(test_path)
    # for i in train_dirs:
    #     os.mkdir(train_path + i)
    # for i in test_dirs:
    #     os.mkdir(test_path + i)
    for tests in test_file_list:
        shutil.copy(data_path + tests, test_path + tests.split('\\')[1])
    for trains in train_file_list:
        shutil.copy(data_path + trains, train_path + trains.split('\\')[1])

    return train_file_list, test_file_list


if __name__ == "__main__":
    init_test_files(data_path, train_path, test_path)
