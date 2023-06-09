1.数据集
  机器学习数据集定义为训练模型和进行预测所需的数据集合。这些数据集分为结构化和非结构化数据集，其中结构化数据集采用表格格式，其中数据集的行对应于记录，列对应于要素，非结构化数据集对应于图像，文本，语音，音频等。通过数据采集，数据整理和数据探索获得的数据，在学习过程中，这些数据集被分为训练，验证和测试集，用于训练和测量模式的性能。

2.模型
  模型是一个数学表达式或算法，用于对数据进行建模，预测或分类。它可以捕捉数据中的模式，并使用这些模式来做出预测或分类新的数据。

3.梯度下降：
  神经网络会有很多权重参数，这个权重参数作用于输入会产生对应的输出，输出与目标之间又有一定的差距，这个差距我们可以定义一个loss来进行描述，我们优化的过程就相当于减少loss的过程，与是我们可以用loss对每个参数求导，根据导数和学习率来优化权重的值。

4.学习率：
  学习率是一个超参数，在机器学习中，我们往往采用梯度下降算法来进行优化，学习率体现了每次优化的步长.

5.batch_size:
  在机器学习中，往往不是把训练集一个一个丢进去学习，而是一个组，一个batch丢进去学习的。batch_size体现的是一次训练中样本的个数。他可以加快训练速度，提升泛化性能。

6.momentum（动量）
  Momentum 是一个动量因子β，用于加速参数更新的方向。Momentum方法在每次迭代中，不仅考虑当前的梯度信息，还考虑之前迭代中的更新方向，通过对历史梯度的加权平均来计算当前梯度的方向。这样可以使参数更新的方向更加平滑，从而加速模型收敛的过程。

7.epoch:
  训练的轮数

8.全联接层：
  也就是线性层，把数据展开的一维后，前前一层的所有数据和后一层的所有数据都用权重连接。

9.卷积层：
  多用与图像识别之中，基本思想是通过一个滑动的卷积核对图片进行扫描，相当于根据卷积核权重的不同对图片不同方面的特征的扫描和提取。

10，激活函数
  激活函数是一个数学函数，它用于在每个神经元的输出上执行非线性转换。激活函数的主要目的是引入非线性性质，使神经网络可以处理非线性模式。

11.损失函数
  损失函数是在机器学习中常用的一个概念，它是用于评估模型预测结果与真实结果之间差距的函数。其中，交叉熵损失函数是深度学习中广泛使用的一种损失函数。

