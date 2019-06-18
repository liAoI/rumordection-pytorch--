# -*- coding: utf-8 -*-
import logging
import json
import numpy as np
import jieba
import os
import re
from gensim.models import word2vec
from config import myConfig
LOG_FORMAT = "%(asctime)s -%(filename)s[line:%(lineno)d]- %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m-%d-%Y %H:%M:%S"

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

logger = logging.getLogger(__name__)


'''
    把每一个文本按照id:text这样的形式精简成一个json文件
    文本去重，将重复的文本保留其中一个，剩下的都删除
'''

def Getlabe(file):
    label= {}
    with open(file,encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line =line.replace('eid:','')
            line = line.replace('label:', '')
            line = line.split('\t')
            label[line[0]] = line[1]
    return label

def Id_Text(file):
    ID_TEXT = {}
    with open(file,'r',encoding='utf-8') as fjson:
        dataj = json.load(fjson)
    for i in range(len(dataj)):
        ID_TEXT[dataj[i]['id']] =dataj[i]['original_text']

    return ID_TEXT

def Getfiles(dir):
    f = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                f.append(os.path.join(root, file))
    return f

def createfile(path,name,data):
    name = os.path.split(name)[1]
    if os.path.exists(path) == False:
        os.makedirs(path)
    os.chdir(path)
    with open(name,'w+',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False)

#返回需要删除的文件名列表
def GetremoveFiles(dir):
    removefiles = []
    f = []
    count = 0
    for root, dirs, files in os.walk(dir):
        for file in files:
            if os.path.splitext(file)[1] == '.txt':
                f.append(os.path.join(root, file))

    for name in f:
        with open(name,'r',encoding='utf-8') as ftxt:
            for line in ftxt:
                count+=1
                line =line.split(':')
                if count ==1:
                    continue
                else:
                    removefiles.append(line[0]+'.json')
        count =0
    return removefiles

def removef(filelist):
    os.chdir('../data')
    for file in filelist:
        if os.path.exists(file):
            os.remove(file)
            logger.info(file + ' 文件已删除！')
        else:
            logger.info("删除文件 "+file+' 失败')

#对../data里面的数据进行去特殊符号，分词，整理成一个文本，然后利用word2vec向量化文本
def processdata(dir):
    datalist = []
    files = Getfiles(dir)
    #第一步，读取一个文本的所有内容，将原始文本和评论都作为内容，分词然后作为一个样本放到datalist里面

    for f in files:
        Jf = {}
        onetext = []
        _,filename=os.path.split(f)
        id,_ = os.path.splitext(filename)
        with open(f,'r',encoding='utf-8') as fj:

            fjson = json.load(fj)

            for key in fjson:
                if key == 'label':
                    continue
                else:
                    onetext.append(fjson[key])

            s = jieba.cut(re.sub("[A-Za-z0-9\!\%\[\]\,\。]", "", ','.join(onetext)))
            s = ' '.join(s)

            Jf['text'] =s
            Jf['label'] = fjson['label']
        with open('./data/'+id+'.json','w',encoding='utf-8') as lo:
            json.dump(Jf, lo, ensure_ascii=False)
        datalist.append(s)

    # 第二步，将datalist写到./data目录下，命名为word
    with open('./data/word.txt','w',encoding='utf-8') as f:
        for line in datalist:
            f.write(line)
            f.write('\n')
    #最后一步，调用词向量工具，向量化词袋
    sentense = word2vec.Text8Corpus(r'./data/word.txt')
    model = word2vec.Word2Vec(sentense, sg=1, size=100, window=5, min_count=1, negative=4, sample=0.001, hs=1,
                              workers=4)
    model.save('./WordModel')
def MAIN(Config):
    sentense = word2vec.Text8Corpus(r'./data/word.txt')
    model = word2vec.Word2Vec(sentense, sg=1, size=40, window=5, min_count=1, negative=4, sample=0.001, hs=1,
                              workers=4)
    model.save('./WordModel')
    # processdata(myConfig.work_data)
    # label = Getlabe(Config.labelfile)
    # logger.info(label)
    # files = Getfiles(Config.original_data)
    # for f in files:
    #     _,filename=os.path.split(f)
    #     name,_ = os.path.splitext(filename)
    #     dataj = Id_Text(f)
    #     dataj['label'] = label[name]
    #     createfile('../data',f,dataj)
    # lt = GetremoveFiles(Config.repeatfile)
    # removef(lt)
    # os.chdir('../data')
    # os.remove('4010312877.json')
    # with open(r'D:\paper\谣言检测论文\群里面的资料\rumdect\重复文本\rumor1.txt','r',encoding='utf-8') as f:
    #     a = f.read()
    #     a =a.split(':')
    #     print(a)

    # count =0
    # label = Getlabe(r'D:\paper\谣言检测论文\群里面的资料\rumdect\Weibo.txt')
    # print(label)
    # with open(r'D:\paper\谣言检测论文\群里面的资料\rumdect\Weibo.txt',encoding='utf-8') as f:
    #     for line in f:
    #         line = line.strip()
    #         line =line.replace('eid:','')
    #         line = line.replace('label:', '')
    #         line = line.split('\t')
    #
    #         print(line)
    #         count+=1
    #         if count ==4:
    #             break
    # a = Getfiles(r'D:\paper\谣言检测论文\群里面的资料\rumdect\Weibo')
    # logger.info(a[0])
    # b = Id_Text(a[0])
    #
    # logger.info(b)
    # createfile('../data',os.path.split(a[0])[1],b)
    return

    # with open(r'D:\paper\谣言检测论文\群里面的资料\rumdect\Weibo\4016873519.json',encoding='utf-8') as f:
    #     datajson = json.load(f)
    #     print(datajson[0]['original_text'])

if __name__ == '__main__':
    MAIN(myConfig)