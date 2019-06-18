class myConfig(object):
    #原始数据存放路径
    original_data = r'D:\paper\谣言检测论文\群里面的资料\rumdect\Weibo'
    #原始数据处理后放到以下路径，方便后续处理
    work_data = '../data'
    #微博标签的文本
    labelfile = r'D:\paper\谣言检测论文\群里面的资料\rumdect\Weibo.txt'
    #重复的文本信息
    repeatfile = r'D:\paper\谣言检测论文\群里面的资料\rumdect\重复文本'

    #RNN训练的参数设置
    hidden_size =80
    inputsize = 40    #这是根据word2vec训练出来的词向量size决定的
    layers = 2
    epochs = 100
    lr=1e-2