
20190618    created by LiAo

所使用数据集来自论文https://www.ijcai.org/Proceedings/16/Papers/537.pdf里面的。

后续所有实现的代码我都会放到这个地址上：https://github.com/liAoI/rumorDetection  请小伙伴们赐教！

下图为普通RNN（lstm）跑出来的结果：

在测试集上的精度变化情况

![image](https://github.com/liAoI/RNN-pytorch--/blob/master/images_result/2list.png)

在测试集上的损失函数下降情况

![image](https://github.com/liAoI/RNN-pytorch--/blob/master/images_result/2listloss.png)

上面结果可以说比较正常，因为loss函数在不断下降，而精度也随loss的下降而上升。

下面是CNN+lstm做出来的结果：

测试集上的结果

![image](https://github.com/liAoI/RNN-pytorch--/blob/master/images_result/newplot(3).png) ![image](https://github.com/liAoI/RNN-pytorch--/blob/master/images_result/newplot(2).png)

由于精度上升的同时，loss也在上升，这反应出模型或者数据设计的有问题，或者说cnn+lstm这个模型里面的一些细节没有设计好。这个loss只是测试集上的，如果把训练集上的loss拿出来比较，就可以得出一些更确定的结论。之前也画过，但是由于最近开始学seq2seq和GAN模型，就先放一放吧！虽然论文的作者在用cnn+lstm进行文本分类的时候，精度能达到0.9以上，这就说明我没有复现成功他的论文。惭愧，惭愧！

从图中可以发现：cnn+lstm做出来的模型其实没有单独的lstm的好，这有可能是我复现别人论文的时候，没有设计好，和论文作者想的有点差别。

还有一点就是，本实验都是采用词向量，没有论文里面的句向量，也没有用单词索引或者TFIDF,词向量用google的word2vec做的。这也是导致实验结果和论文说的有差别的原因之一！
