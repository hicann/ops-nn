# SparseSoftmaxCrossEntropyWithLogits

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：计算稀疏类标签场景下的Softmax交叉熵损失，并同步输出反向传播所需的梯度。与稠密标签不同，labels直接给出每个样本的类别索引，无需进行one-hot展开。

- 计算公式：

  设features为$x$（shape为$[N, C]$），labels为$y$（shape为$[N]$，取值范围$[0, C)$），对第$n$个样本：

  $$
  \text{softmax}(x)_{n,c} = \frac{\exp(x_{n,c})}{\sum_{k=0}^{C-1}\exp(x_{n,k})}
  $$

  $$
  \text{loss}_n = -\log\big(\text{softmax}(x)_{n,y_n}\big)
  $$

  $$
  \text{backprop}_{n,c} = \text{softmax}(x)_{n,c} - \mathbb{1}\{c = y_n\}
  $$

  其中：

  - $N$为batch数，$C$为类别数。
  - loss为每个样本的损失（长度为$N$的向量）。
  - backprop为反向传播的梯度（shape为$[N, C]$的矩阵）。

## 参数说明

| 参数名   | 输入/输出/属性 | 描述                                                         | 数据类型                 | 数据格式 |
| -------- | -------------- | ------------------------------------------------------------ | ------------------------ | -------- |
| features | 输入           | 预测的logits，对应公式中的$x$，是一个batch_size * num_classes的矩阵。 | FLOAT16、FLOAT、BFLOAT16 | ND       |
| labels   | 输入           | 类别标签，对应公式中的$y$，是一个长度为batch_size的向量，取值范围为[0, num_classes)。 | INT32、INT64             | ND       |
| loss     | 输出           | 每个样本的损失值，长度为batch_size的向量。数据类型与features一致。 | FLOAT16、FLOAT、BFLOAT16 | ND       |
| backprop | 输出           | 反向传播的梯度，shape与features一致。数据类型与features一致。 | FLOAT16、FLOAT、BFLOAT16 | ND       |

## 约束说明

- features与loss、backprop的数据类型保持一致。
- labels的维度数须等于features的维度数减1，且除最后一维外各维度大小必须与features对应维度相同。
- features的最后一维（num_classes）必须大于0，即至少包含一个类别。
- labels的取值需落在[0, num_classes)范围内。
- 兼容TensorFlow的SparseSoftmaxCrossEntropyWithLogits算子。

## 调用说明

| 调用方式 | 样例代码                                                     | 说明                                                         |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式调用 | [test_geir_sparse_softmax_cross_entropy_with_logits](./examples/test_geir_sparse_softmax_cross_entropy_with_logits.cpp) | 通过[算子IR](./op_graph/sparse_softmax_cross_entropy_with_logits_proto.h)构图方式调用SparseSoftmaxCrossEntropyWithLogits算子。 |
