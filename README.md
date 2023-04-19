此项目完全是照搬的rasbt大佬的内容，自己只是照着抄了一遍，上传github，只是为了监督自己，原github地址：https://github.com/rasbt

### 2023-04-19
    ```
    完成pytorch/cnn/cnn-alexnet-cifar10.py

    但是训练的时候训练集能达到90%+,但是验证集只有70%+
    
    最后在训练接，验证集和测试集上的准确率全部都是70%+

    暂时还不清楚是什么原因！！！


    完成pytorch/cnn/cnn-allconv.py

    模型训练的步骤：
        1. 加载模型和数据到GPU
        2. 经过模型训练
        3. 通过损失函数计算损失
        4. 优化器参数清零
        5. 调用损失函数的反向传播
        6. 利用反向传播进行参数更新 
        7. 优化器更新值
        8. 循环1-7步
    ```


### 2023-04-17
   ```
   完成pytorch-lightning/cnn/cnn-alexnet-cifar10.py
   
   from collections import Counter 函数可以跟踪可迭代对象中每个元素出现的次数
   Counter().most_common() 函数可以获取计数器字典中出现次数最多的前n个元素，并按照他们的频率进行排序
   ```