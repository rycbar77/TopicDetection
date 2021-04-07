import random
from configs import *
import os
import jieba
from file_utils import *
from pprint import pprint


def cut_txt(file_list):
    res = []
    for old_file in file_list:
        fi = open(old_file, 'r', encoding='utf-8', errors='ignore')
        text = fi.read()
        new_text = jieba.cut(text, cut_all=False)
        new_str = " ".join(new_text).replace('\n', '').replace('\u3000', '')
        # pprint(new_str)
        res.append(new_str)
    return res


if __name__ == "__main__":
    cut_txt(get_files(train_path))
