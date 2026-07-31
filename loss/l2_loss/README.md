# L2Loss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：计算输入x中所有元素平方和的一半，即L2正则化损失。输出y为标量。

- 计算公式：

  $$
  y = \frac{1}{2} \sum_{i} x_i^2
  $$

  其中$x$是输入`self`，$i$遍历$x$中的所有元素。

## 参数说明

- x (aclTensor*，计算输入)：公式中的输入`x`，Device侧的aclTensor。支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT。
- y (aclTensor*，计算输出)：公式中的输出`y`，Device侧的aclTensor。数据类型与x一致。shape为0维（标量）。支持[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)，[数据格式](../../docs/zh/context/data_format.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT。

## 约束说明

无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GEIR入图  | [test_geir_l2_loss.cpp](examples/test_geir_l2_loss.cpp) | 通过算子IR构图方式调用L2Loss算子。 |
