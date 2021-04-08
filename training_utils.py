import random
from configs import *
import os
import jieba
from file_utils import *
from pprint import pprint
from munkres import Munkres
from sklearn.metrics import accuracy_score
import numpy as np


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


def best_map(L1, L2):
    """

    :param L1: 真实标签
    :param L2: 聚类标签
    :return:
    """
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return accuracy_score(L1, newL2), newL2


if __name__ == "__main__":
    cut_txt(get_files(train_path))
