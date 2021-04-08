from gensim import corpora, models, similarities
from pprint import pprint
from configs import *
from collections import defaultdict
from file_utils import get_files
from training_utils import cut_txt
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
stop_list = {"the", "of", "is", "and", "to", "in", "that", "we", "for", "an", "are", "by", "be", "as", "on",
             "with", "can", "if", "from", "which", "you", "it", "this", "then", "at", "have", "all", "not", "one",
             "has", "or", "that", "..", "...", "---"}


def get_stop_words():
    with open('./hit_stopwords.txt', 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            s = line.strip()
            stop_list.add(s)


def process_corpus(path):
    get_stop_words()
    text_corpus = cut_txt(get_files(path))

    texts = [[word for word in document.split() if word not in stop_list]
             for document in text_corpus]

    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
    return processed_corpus


def train_model(path="", corpus_path='corpus.mm', dic_path='model.dic', tfidf_path='model.tfidf',
                lsi_path='model.lsi', mSimilar_path='model.mSimilar'):
    """
    To train and save models
    :param path: If all the models are stored in the same directory, use path to describe the dir path
    :param corpus_path: corpus path
    :param dic_path: dictionary path
    :param tfidf_path: tfidf model path
    :param lsi_path: lsi model path
    :param mSimilar_path: MatrixSimilarity path
    :return:
    """
    processed_corpus = process_corpus(train_path)
    dictionary = corpora.Dictionary(processed_corpus)
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
    corpora.MmCorpus.serialize(path + corpus_path, bow_corpus)
    dictionary.save(path + dic_path)

    tfidf = models.TfidfModel(bow_corpus)
    tfidf.save(path + tfidf_path)
    corpus_tfidf = tfidf[bow_corpus]

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=20)
    corpus_lsi = lsi[corpus_tfidf]
    lsi.save(path + lsi_path)
    mSimilar = similarities.MatrixSimilarity(corpus_lsi)
    mSimilar.save(path + mSimilar_path)


def load_model(path="", corpus_path='corpus.mm', dic_path='model.dic', tfidf_path='model.tfidf',
               lsi_path='model.lsi', mSimilar_path='model.mSimilar'):
    """
    To load trained models
    :param path: If all the models are stored in the same directory, use path to describe the dir path
    :param corpus_path: corpus path
    :param dic_path: dictionary path
    :param tfidf_path: tfidf model path
    :param lsi_path: lsi model path
    :param mSimilar_path: MatrixSimilarity path
    :return:
    corpus: corpora.MmCorpus
    dict: corpora.Dictionary
    tfidf: models.TfidfModel
    lsi: models.LsiModel
    mSimilar: similarities.MatrixSimilarity
    """
    corpus = corpora.MmCorpus(path + corpus_path)
    dict = corpora.Dictionary.load(path + dic_path)
    tfidf = models.TfidfModel.load(path + tfidf_path)
    lsi = models.LsiModel.load(path + lsi_path)
    mSimilar = similarities.MatrixSimilarity.load(path + mSimilar_path)
    return corpus, dict, tfidf, lsi, mSimilar


if __name__ == "__main__":
    # train_model()

    corpus, dict, tfidf, lsi, mSimilar = load_model("./model_test/")

    # pprint(len(corpus))
    # pprint(len(dict.token2id))
    # doc=corpus[0]
    # 把测试语料转成词袋向量
    # vec_bow = dict.doc2bow(doc.split())
    # pprint(vec_bow)
    # 求tfidf值
    # vec_tfidf = tfidf[doc]
    # pprint(vec_tfidf)
    # 转成lsi向量
    # vec_lsi = lsi[vec_tfidf]
    # pprint(vec_lsi)
    # 求解相似性文档
    # sims = mSimilar[vec_lsi]
    # sims = sorted(enumerate(sims), key=lambda item: -item[1])
    # print(sims)
