from torch.utils import data
import json
import os
import torch.nn as nn
import torch
import numpy as np
from gensim.models import word2vec
import logging
from config import myConfig
import visdom

viz = visdom.Visdom()
loss_win = viz.scatter(X=np.asarray([[0,0]]))
acc_win = viz.scatter(X=np.asarray([[0,0]]))
print(torch.cuda.is_available())

#########配置log日志方便打印#############

LOG_FORMAT = "%(asctime)s -%(filename)s[line:%(lineno)d]- %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m-%d-%Y %H:%M:%S"

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

logger = logging.getLogger(__name__)

#将数据分成四份，一份测试，其余作训练
class ShipDataset(data.Dataset):
    def __init__(self,vali = 1,Train = True,dir='./data'):
        super(ShipDataset, self).__init__()
        self.train = []
        self.vaild = []
        #获取数据清单
        flist = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if os.path.splitext(file)[1] == '.json':
                    flist.append(os.path.join(root, file))
        flist = [flist[i:i + len(flist)//5] for i in range(0, len(flist), len(flist)//5)]

        for i in range(len(flist)):
            if i == vali:
                self.vaild = flist[i]
            else:
                self.train +=flist[i]

        self.ModeTrain = Train

        self.MO = word2vec.Word2Vec.load('./WordModel')

    def __getitem__(self, item):
        X = []
        Y = None
        if self.ModeTrain:
            filename = self.train[item]
        else:
            filename = self.vaild[item]

        with open(filename, 'r', encoding='utf-8') as f:
            fjson = json.load(f)
        for i in fjson['text']:
            try:
                X.append(self.MO[i])
            except (KeyError):
                # logger.info(i +'没有向量化！')
                continue
        X = torch.tensor(X[0:30])  # 这里由于每条句子长度不一致，导致无法封装到一个batch里，所以才设置取前30
        Y = int(fjson['label'])
        # Y = torch.tensor(int(fjson['label'])).float()
        if Y == 0 :    #非谣言
            Y = torch.tensor([0.0,1.0])
        else:
            Y = torch.tensor([1.0,0.0])

        return X,Y

    def __len__(self):
    #返回数据的数量
        if self.ModeTrain:
            return len(self.train)
        else:
            return len(self.vaild)


class RNNModel(nn.Module):
    def __init__(self,input_size,hidden_size,n_layers,lstm,GPU):
        super(RNNModel, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.lstm = lstm
        self.hidden_size = hidden_size
        self.gpu = GPU
        if self.gpu == True:
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True ).cuda()
            self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True).cuda()
            # self.linear = nn.Linear(self.hidden_size,2)  #二分类，最后结果[0,1] [1,0]
            self.layer = nn.Sequential(nn.Linear(self.hidden_size, 2), nn.Sigmoid()).cuda()
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
            self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True)
            self.layer = nn.Sequential(nn.Linear(self.hidden_size, 2), nn.Sigmoid())

    def forward(self,input,state=None):
        batch, _, _ = input.size()
        if self.gpu == True:
            if self.lstm == True:
                if state is None:
                    h = torch.randn(self.n_layers, batch, self.hidden_size).cuda().float()
                    c = torch.randn(self.n_layers, batch, self.hidden_size).cuda().float()
                else:
                    h, c = state

                # output [batchsize,time,hidden_size]
                output, state = self.rnn(input, (h, c))
            else:
                if state is None:
                    state = torch.randn(self.n_layers, batch, self.hidden_size).cuda().float()
                output, state = self.gru(input, state)
        else:
            if self.lstm == True:
                if state is None:
                    h = torch.randn(self.n_layers, batch, self.hidden_size).float()
                    c = torch.randn(self.n_layers, batch, self.hidden_size).float()
                else:
                    h, c = state

                # output [batchsize,time,hidden_size]
                output, state = self.rnn(input, (h, c))
            else:
                if state is None:
                    state = torch.randn(self.n_layers, batch, self.hidden_size).float()
                output, state = self.gru(input, state)
        #最后输出结果
        output = self.layer(output[:, -1, :])
        return output,state

def adjust_learning_rate(lr, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
	lr = lr * (0.5 ** (epoch // 10))
	return lr

def RunCPU(opt):
    lr = 1e-3
    train_data = ShipDataset()
    train_loader = data.DataLoader(train_data, batch_size=4, num_workers=0, shuffle=False)
    valid_data = ShipDataset(Train = False)
    valid_loader = data.DataLoader(valid_data, batch_size=4,num_workers=0, shuffle=False)

    # for i, validset in enumerate(valid_loader):
    #     X,Y = validset
    #     print(X.size())

    model = RNNModel(opt.inputsize,opt.hidden_size,opt.layers,lstm=True,GPU=False)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    criterion = nn.BCELoss()


    for epoch in range(opt.epochs):
        # lr = adjust_learning_rate(lr,epoch)
        for i,trainset in enumerate(train_loader):
            X,Y=trainset

            out_y,_ = model(X)

            loss = criterion(out_y,Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('batch_loss: {0},学习率为{1}'.format(loss,lr))


        #训练完一个epoch之后，计算一下在验证集上的准确度
        out = torch.tensor([[0.0]])
        y = torch.tensor([0])
        for valiset in valid_loader:
            v_X,v_Y = valiset
            out_y, _ = model(v_X)
            # logger.info(out_y)
            # logger.info(out)
            out = torch.cat((out,out_y),0)
            y = torch.cat((y,v_Y),0)
            # logger.info(out)
            # logger.info(y)
        correct_pred = torch.eq(torch.argmax(out, 1), y)
        acc = correct_pred.sum().item()/ y.size(0)

        print('第 {0} 轮训练精度为 {1}'.format(epoch+1,acc))
        torch.save({
            'epoch': epoch + 1,
            # 'arch': args.arch,
            'state_dict': model.state_dict(),
            'loss': loss,
        }, 'checkpoint{0}.tar'.format(epoch+1))


def RunGPU(opt):
    #将模型迁移到GPU上
    lr = 1e-3
    model = RNNModel(opt.inputsize, opt.hidden_size, opt.layers,lstm=True,GPU=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss().cuda()
    # 查看可更新的参数
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # return
    #将数据迁移到GPU`
    train_data = ShipDataset(vali=2)
    train_loader = data.DataLoader(train_data, batch_size=6, num_workers=2, shuffle=False)

    valid_data = ShipDataset(vali=2,Train=False)
    valid_loader = data.DataLoader(valid_data, batch_size=8, num_workers=2, shuffle=False)

    for epoch in range(opt.epochs):
        # lr = adjust_learning_rate(lr, epoch)
        for i, trainset in enumerate(train_loader):
            X, Y = trainset
            X = torch.tensor(X).cuda()
            Y = torch.tensor(Y).cuda()
            out_y, _ = model(X)
            # logger.info(Y)
            # logger.info(out_y)
            loss = criterion(out_y, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # for param in model.named_parameters():
            #     logger.info(param)
            if i % 100 == 0:
                print('batch_loss: {0},学习率为{1}'.format(loss, lr))



        out = torch.tensor([[0.0,0.0]]).cuda()
        y = torch.tensor([[0.0,0.0]]).cuda()
        for valiset in valid_loader:
            v_X,v_Y = valiset
            v_X = torch.tensor(v_X).cuda()
            v_Y = torch.tensor(v_Y).cuda()
            out_y, _ = model(v_X)
            # logger.info(out_y)
            # logger.info(out)
            out = torch.cat((out,out_y),0)
            y = torch.cat((y,v_Y),0)
            # logger.info(out)
            # logger.info(y)
        correct_pred = torch.eq(torch.argmax(out, 1), torch.argmax(y, 1))
        acc = correct_pred.sum().item()/ y.size(0)
        print('第 {0} 轮训练精度为 {1}'.format(epoch + 1, acc))
        viz.scatter(X=np.array([[epoch + 1, loss.item()]]), name='loss', win=loss_win, update='append')
        viz.scatter(X=np.array([[epoch + 1, acc]]), name='acc', win=acc_win, update='append')
        if acc > 0.90:
            torch.save({
                'epoch': epoch + 1,
                # 'arch': args.arch,
                'state_dict': model.state_dict(),
                'loss': loss,
            }, 'checkpoint{0}.tar'.format(epoch + 1))
if __name__ == '__main__':
    if torch.cuda.is_available():
        RunGPU(myConfig)
    else:
        RunCPU(myConfig)
