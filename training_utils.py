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


def get_list(X):
    X_list = []
    for doc in X:
        if not doc:
            tmp_doc = [0.0 for _ in range(X.obj.num_topics)]
        else:
            tmp_doc = [i[1] for i in doc]
            if len(tmp_doc) < X.obj.num_topics:
                tmp_doc += [0 for _ in range(X.obj.num_topics - len(tmp_doc))]
        X_list.append(tmp_doc)
    return X_list


def scores(l, label_pred, X_array):
    from sklearn import metrics
    info = metrics.mutual_info_score(l, label_pred)
    mutual_info = metrics.adjusted_mutual_info_score(l, label_pred)
    normal_info = metrics.normalized_mutual_info_score(l, label_pred)
    print("互信息：{0}\n调整互信息：{1}\n标准化互信息：{2}".format(info, mutual_info, normal_info))
    rand = metrics.adjusted_rand_score(l, label_pred)
    print('兰德系数：', rand)
    score = metrics.silhouette_score(X_array, label_pred)
    print("轮廓系数：", score)


def draw(X, label):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca = pca.fit(X)
    array1 = pca.transform(X)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 5))
    plt.style.use('seaborn-pastel')
    ax = fig.add_subplot(121)

    fig.patch.set_alpha(0.)
    plt.cla()
    for i in range(20):
        ax.scatter(array1[label == i, 0], array1[label == i, 1], s=10)

    pca = PCA(n_components=3)
    pca = pca.fit(X)
    array1 = pca.transform(X)

    ax = fig.add_subplot(122, facecolor=None, projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    fig.patch.set_alpha(0.)
    plt.cla()
    for i in range(20):
        ax.scatter(array1[label == i, 0], array1[label == i, 1], array1[label == i, 2], s=10)
    plt.show()


if __name__ == "__main__":
    cut_txt(get_files(train_path))
