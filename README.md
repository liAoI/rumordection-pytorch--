
20190618    created by LiAo

所使用数据集来自论文https://www.ijcai.org/Proceedings/16/Papers/537.pdf里面的。

后续所有实现的代码我都会放到这个地址上：https://github.com/liAoI/rumorDetection  

下图为普通RNN（lstm）跑出来的结果：

在测试集上的精度变化情况

![image](https://github.com/liAoI/RNN-pytorch--/blob/master/images_result/2list.png)

在测试集上的损失函数下降情况

![image](https://github.com/liAoI/RNN-pytorch--/blob/master/images_result/2listloss.png)

上面结果可以说比较正常，因为loss函数在不断下降，而精度也随loss的下降而上升。


生成对抗的结果，但是生成模型可能有些问题，正在调试！
![image](https://github.com/liAoI/rumorDetection/blob/master/papercode/result.png)
