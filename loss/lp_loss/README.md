# LpLoss


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

- 算子功能：计算输入self和目标target中每个元素之间的平均绝对误差（Mean Absolute Error，简称MAE）。reduction指定要应用到输出的缩减，支持 'none'、'mean'、'sum'。'none' 表示不应用缩减，'mean' 表示输出的总和将除以输出中的元素数，'sum' 表示输出将被求和。

- 计算公式：

  当`reduction`为`none`时

  $$
  \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
  l_n = \left| x_n - y_n \right|,
  $$

  其中$x$是self，$y$是target，$N$是batch的大小。如果`reduction`不是`'none'`, 那么

  $$
  \ell(x, y) =
  \begin{cases}
  \operatorname{mean}(L), &  \text{if reduction} = \text{'mean';}\\
  \operatorname{sum}(L),  &  \text{if reduction} = \text{'sum'.}
  \end{cases}
  $$
## 参数说明

- self (aclTensor*，计算输入)：公式中的输入`self`，Device侧的aclTensor。数据类型与target的数据类型满足数据类型推导规则（参见[互推导关系](../../docs/zh/context/互推导关系.md)）。shape支持0-8维，shape需要与target满足[broadcast规则](../../docs/zh/context/broadcast关系.md)。支持[非连续的Tensor](../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32、INT64。
- target (aclTensor*，计算输入)：公式中的输入`target`，Device侧的aclTensor。数据类型与self的数据类型满足数据类型推导规则（参见[互推导关系](../../docs/zh/context/互推导关系.md)）。shape支持0-8维，shape需要与self满足[broadcast规则](../../docs/zh/context/broadcast关系.md)。支持[非连续的Tensor](../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32、INT64。
- reduction (int64_t，计算输入，为属性)：用于指定要应用到输出的缩减公式中的输入，公式中的`reduction`，Host侧的整型。支持 0('none') | 1('mean') | 2('sum')。'none' 表示不应缩减，'mean' 表示输出的总和将除以输出中的元素数，'sum' 表示输出将被求和。
- out (aclTensor*，计算输出)：公式中的`out`，Device侧的aclTensor。且数据类型需要是self与target推导之后可转换的数据类型（参见[互转换关系](../../docs/zh/context/互转换关系.md)）。支持[非连续的Tensor](../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../docs/zh/context/数据格式.md)支持ND。当reduction的值为0时，out的shape与self和target broadcast后的shape一致；当reduction的值不为0时，out的shape支持0维。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32、INT64、COMPLEX64、COMPLEX128。

## 约束说明

- 确定性计算：
    - aclnnL1Loss默认确定性实现。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_l1_loss.cpp](examples/test_aclnn_l1_loss.cpp) | 通过[aclnnL1Loss](docs/aclnnL1Loss.md)接口方式调用LpLoss算子。 |