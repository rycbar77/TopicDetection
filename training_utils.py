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
        new_str = " ".join(new_text).replace('\n', '')
        # pprint(new_str)
        res.append(new_str)
    return res


def best_map(real_label, cluster_label):
    """
    KM
    :param real_label: 真实标签
    :param cluster_label: 聚类标签
    :return:
    """
    label1 = np.unique(real_label)
    n_class1 = len(label1)
    label2 = np.unique(cluster_label)
    n_class2 = len(label2)
    n_class = np.maximum(n_class1, n_class2)
    G = np.zeros((n_class, n_class))
    for i in range(n_class1):
        index_class1 = (real_label == label1[i]).astype(float)
        for j in range(n_class2):
            index_class2 = (cluster_label == label2[j]).astype(float)
            G[i, j] = np.sum(index_class2 * index_class1)
    m = Munkres()
    c = np.array(m.compute(-G.T))[:, 1]
    label = np.zeros(cluster_label.shape)
    for i in range(n_class2):
        label[cluster_label == label2[i]] = label1[c[i]]
    return accuracy_score(real_label, label), label


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


def scores(real_label, cluster_label):
    from sklearn import metrics
    info = metrics.mutual_info_score(real_label, cluster_label)
    mutual_info = metrics.adjusted_mutual_info_score(real_label, cluster_label)
    normal_info = metrics.normalized_mutual_info_score(real_label, cluster_label)
    print("互信息：{0}\n调整互信息：{1}\n标准化互信息：{2}".format(info, mutual_info, normal_info))
    rand = metrics.adjusted_rand_score(real_label, cluster_label)
    print('兰德系数：', rand)


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
    # cut_txt(get_files(train_path))
    a = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])
    b = np.array([2, 2, 2, 1, 1, 4, 4, 4, 4])
    best_map(a, b)
