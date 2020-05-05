# -*- coding: utf-8 -*-
import jieba
import numpy
import json
import copy
import os
import re
import time
import logging
import csv
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA,KernelPCA
from gensim.models import word2vec
#########配置log日志方便打印#############

LOG_FORMAT = "%(asctime)s -%(filename)s[line:%(lineno)d]- %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m-%d-%Y %H:%M:%S"

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

logger = logging.getLogger(__name__)

#------------------处理数据-----------------------#


# stopword_list = []
# rumor_corpus = []
# unrumor_corpus = []
# training_data = []
# validation_data = []
# test_data = []
# bag_of_word_count = {}


def readrumorfile(filename, bag_of_word_count, stopword_list, rumor_corpus):
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            # line.encode('utf-8')
            text_json = json.loads(line)
            s = jieba.cut(re.sub("[A-Za-z0-9\!\%\[\]\,\。]", "", text_json["rumorText"]))  # 过滤掉句子中的数字和字母以及标点符号
            line_list = list(s)

            cp_line = copy.deepcopy(line_list)
            for word in line_list:
                if word in stopword_list:
                    word.encode('utf-8')
                    cp_line.remove(word)  # 去掉停用词
            for word in cp_line:
                if word not in bag_of_word_count:
                    bag_of_word_count[word] = 1
                else:
                    bag_of_word_count[word] += 1
            rumor_corpus.append(",".join(cp_line))
            if len(rumor_corpus) >= 1000:
                break
        f.close()
def readnewsfile(filename,bag_of_word_count,stopword_list,unrumor_corpus):
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if len(line) <= 43:  # 过滤掉短文本   毫无意义的文本 len("①柔道冠军称父亲深夜被带走真相是这样http://t.cn/EJuoyyO")=38
                continue
            s = jieba.cut(re.sub("[A-Za-z0-9\!\%\[\]\,\。]", "", line.strip('\n')))
            line_list = list(s)

            cp_line = copy.deepcopy(line_list)
            for word in line_list:
                if word in stopword_list:
                    word.encode('utf-8')
                    cp_line.remove(word)  # 去掉停用词
            for word in cp_line:
                if word not in bag_of_word_count:
                    bag_of_word_count[word] = 1
                else:
                    bag_of_word_count[word] += 1
            unrumor_corpus.append(",".join(cp_line))
            if len(unrumor_corpus) >= 1000:
                break
        f.close()

#将词袋中小于frequ的直接去掉

def removeWord(rumor_corpus,unrumor_corpus,bag_of_word_count,frequ):
    rumor_cor = []
    unrumor_cor = []
    for s_r,s_u in zip(rumor_corpus,unrumor_corpus):
        list_s_r = s_r.split(",")
        list_s_u = s_u.split(",")

        list_r = copy.deepcopy(list_s_r)
        list_u = copy.deepcopy(list_s_u)

        for w in list_s_r:
            if w not in bag_of_word_count:
                logger.info(w)
                continue
            if bag_of_word_count[w] < frequ:
                list_r.remove(w)
        for w in list_s_u:
            if w not in bag_of_word_count:
                logger.info(w)
                continue
            if bag_of_word_count[w] < frequ:
                list_u.remove(w)

        if list_s_r:
            rumor_cor.append(",".join(list_r))
        if list_s_u:
            unrumor_cor.append(",".join(list_u))

    return rumor_cor,unrumor_cor



def getdata(stopword_list, bag_of_word_count, rumor_corpus, unrumor_corpus):
    # remove stopwords from list_corpus

    with open("../data/stopword.txt", "r", encoding="utf-8") as fp:
        for line in fp:
            stopword_list.append(line[:-1])
        fp.close()

    logger.info("读取停用词，构造stopword_list集合")

    # 谣言
    # 数据处理    list_corpus = [rumorText,rumorText,rumorText,...]
    readrumorfile("../data/rumors_v170613.json", bag_of_word_count, stopword_list, rumor_corpus)

    logger.info("从 rumors_v170613.json 谣言文本中获取 %g条数据" % (len(rumor_corpus)))
    # 非谣言
    readnewsfile("../data/news20190407-214236.txt", bag_of_word_count, stopword_list, unrumor_corpus)
    if len(unrumor_corpus) <= 1000:
        readnewsfile("../data/news20190407-214412.txt", bag_of_word_count, stopword_list, unrumor_corpus)
    # 释放堆内存
    stopword_list.clear()

    logger.info("从 news20190407-214236.txt | news20190407-214412.txt文本中获取到 %g" % (len(unrumor_corpus)))
    logger.info("词袋长度：%s" % (len(bag_of_word_count)))
    corpus = rumor_corpus + unrumor_corpus

    return corpus,bag_of_word_count, rumor_corpus, unrumor_corpus


def Sklearn_getfeature(corpus):
    # 将list_corpus里面所有的谣言短文本转换向量化，构建词袋
    vectoerizer = CountVectorizer(min_df=1, max_df=1.0, token_pattern='\\b\\w+\\b')

    X = vectoerizer.fit_transform(corpus)

    # 计算TF-IDF
    tfidf_transformer = TfidfTransformer()

    tfidf = tfidf_transformer.fit_transform(X)

    logger.info("用sklearn构建词袋，TFIDF计算完成")
    # logger.info(tfidf[0][0])
    # logger.info(type(tfidf.toarray()))

    # 构造tupple，准备测试：
    # label = numpy.zeros((1000, 2))
    # for i in range(0, 500):
    #     label[i][0] = 1
    # for i in range(500, 1000):
    #     label[i][1] = 1
    # label = numpy.asarray(label)
    data_tfidf = tfidf.toarray()

    with open('../data/roumordataset.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_tfidf)

    #利用PCA降维
    pca = PCA(n_components=841)
    data_pca = pca.fit_transform(data_tfidf)
    with open('../data/roumordatasetPCA.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_pca)

    #利用PCA核方法进行降维
    kpca = KernelPCA(kernel="rbf")
    data_kpca = kpca.fit_transform(data_tfidf)
    with open('../data/roumordatasetKPCA.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data_kpca)
    return tfidf


def gensim_getfeature(corpus):

    return


# 测试时使用的函数----毫无用处
def WriteFile(data, target):
    if os.path.exists(target):
        path, suffix = os.path.splitext(target)
        s = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        target = path + s + suffix
    with open(target, 'w', encoding="utf-8") as fp:
        for str in data:
            fp.write(str)
            fp.write("\n")
    fp.close()

#做数据集   按照训练集：验证集 = 4：1

def main():
    stopword_list = []
    rumor_corpus = []
    unrumor_corpus = []
    training_data = []
    validation_data = []
    test_data = []
    bag_of_word_count = {}
    frequ = 2

    corpus, bag_of_word_count, rumor_corpus, unrumor_corpus = getdata(stopword_list, bag_of_word_count, rumor_corpus,
                                                                      unrumor_corpus)
    logger.info(len(rumor_corpus))
    logger.info(len(unrumor_corpus))

    rumor_cor, unrumor_cor = removeWord(rumor_corpus, unrumor_corpus, bag_of_word_count, frequ)
    logger.info(len(rumor_cor))
    logger.info(len(unrumor_cor))

    with open("../data/bag_word.json", "w", encoding='utf-8') as f:
        json.dump(bag_of_word_count, f, ensure_ascii=False)

    Sklearn_getfeature(rumor_cor + unrumor_cor)


if __name__ == '__main__':
    c = []
    print(len(c))
    # main()
    # w = "急找孩子，求转 实验小学寻人启事13930886687帮忙扩散，今天上午一个三岁多小女孩在锦绣花园小区附近被人拐走了，小女孩能说出她爸爸的手机号码 从监控上看是被一个四十多岁男人抱走了现大人都急疯了 有知情者请告之 万分感谢 看到信息的兄弟姐妹留意一下 联系人 张静杰13930886687如果看一眼懒得 "
    # s = jieba.cut(re.sub("[A-Za-z0-9\!\%\[\]\,\。]","",w))
    # bag = {}
    # r =list(s)
    # # list_s = ",".join(r)
    # print(r)
    # l = ",".join(r)
    # print(l)
    # print(l.split(","))
    # # print(list_s)
    # Word = ['孩子','生活','求转','帮忙','今天','一个三岁','小女孩','今天','小女孩']
    # words = ['孩子,生活,求转,帮忙', '今天,一个三岁,小女孩', '今天']










