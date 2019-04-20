# deepLearning
Step 1, Minibatch validation Loss= 332.2242, Minibatch train Loss=418.1009, validation Accuracy= 0.481,train Accuracy = 0.480

Step 2000, Minibatch validation Loss= 1.1952, Minibatch train Loss=0.6876, validation Accuracy= 0.676,train Accuracy = 0.720

Step 4000, Minibatch validation Loss= 0.5778, Minibatch train Loss=0.4372, validation Accuracy= 0.767,train Accuracy = 0.768

Step 6000, Minibatch validation Loss= 0.3925, Minibatch train Loss=0.3610, validation Accuracy= 0.800,train Accuracy = 0.793

Step 8000, Minibatch validation Loss= 0.3296, Minibatch train Loss=0.2590, validation Accuracy= 0.848,train Accuracy = 0.878

Step 9000, Minibatch validation Loss= 0.3764, Minibatch train Loss=0.2433, validation Accuracy= 0.867,train Accuracy = 0.900

Step 9100, Minibatch validation Loss= 0.4105, Minibatch train Loss=0.2469, validation Accuracy= 0.848,train Accuracy = 0.920

Step 9200, Minibatch validation Loss= 0.2904, Minibatch train Loss=0.2292, validation Accuracy= 0.890,train Accuracy = 0.902

Step 9300, Minibatch validation Loss= 0.2769, Minibatch train Loss=0.1764, validation Accuracy= 0.871,train Accuracy = 0.930

Step 9400, Minibatch validation Loss= 0.2665, Minibatch train Loss=0.2101, validation Accuracy= 0.871,train Accuracy = 0.920

Step 9500, Minibatch validation Loss= 0.3453, Minibatch train Loss=0.2646, validation Accuracy= 0.857,train Accuracy = 0.880

Step 9600, Minibatch validation Loss= 0.3651, Minibatch train Loss=0.2407, validation Accuracy= 0.848,train Accuracy = 0.878

Step 9700, Minibatch validation Loss= 0.2878, Minibatch train Loss=0.2530, validation Accuracy= 0.876,train Accuracy = 0.880

Step 9800, Minibatch validation Loss= 0.3769, Minibatch train Loss=0.2424, validation Accuracy= 0.824,train Accuracy = 0.880

Step 9900, Minibatch validation Loss= 0.3710, Minibatch train Loss=0.1932, validation Accuracy= 0.867,train Accuracy = 0.910

Step 10000, Minibatch validation Loss= 0.2808, Minibatch train Loss=0.2062, validation Accuracy= 0.871,train Accuracy = 0.878

Final test accuracy = 82%

上面是用CNN训练得到的准确度，数据集2000个样本，采用ITF-IDF向量化文本。在训练中，CNN用了两层卷积层，对于全连接层的权重没有正则化，看出在训练集上达到了百分之90多的
准确率，但是在验证集上始终在百分之80多，所以测试集上最后一次显示不到百分之90的成功率。
接下来，我想试试google的word2vec这个处理文本，用RNN来训练数据，看是否会有提升。
