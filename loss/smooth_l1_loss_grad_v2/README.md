# SmoothL1LossBackward

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

- 算子功能：计算[aclnnSmoothL1Loss](../../smooth_l1_loss_v2/docs/aclnnSmoothL1Loss.md) api的反向传播。
- 计算公式：

  SmoothL1Loss的反向传播可以通过求导计算。对于SmoothL1Loss的第一种情况，即|x-y|<1，其导数为：
  $$
  \frac{\partial SmoothL1Loss(x,y)}{\partial x} = x - y
  $$

  对于SmoothL1Loss的第二种情况，即|x-y|≥1，其导数为：
  $$
  \frac{\partial SmoothL1Loss(x,y)}{\partial x} = sign(x - y)
  $$
  
  其中sign(x)表示x的符号函数，即：
  $$
  sign(x) =\begin{cases}
  1,& if　x>0 \\
  0,& if　x=0 \\
  -1,& if　x<0
  \end{cases}
  $$

## 参数说明

  - gradOut（aclTensor*，计算输入）：梯度反向输入，公式中的`SmoothL1Loss`，Device侧的aclTensor。shape需要与self，target满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)，数据类型与self、target的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)），支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)， [数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。

  - self（aclTensor*，计算输入）：公式中的`x`，Device侧的aclTensor。shape需要与gradOut，target满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)，数据类型与gradOut、target的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)），支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)， [数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。

  - target（aclTensor*，计算输入）：公式中的`y`，Device侧的aclTensor。shape需要与gradOut，self满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)，数据类型与gradOut、target的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)），支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)， [数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。

  - reduction（int64_t，计算输入）：计算属性，指定要应用到输出的缩减，Host侧的整型。取值支持0('none')|1('mean')|2('sum')。'none'表示不应用减少，'mean'表示输出的总和将除以输出中的元素数，'sum'表示输出将被求和。

  - beta（float，计算输入）：计算属性，指定在L1和L2损失之间更改的数值，数据类型为float，该值必须是非负的。

  - gradInput（aclTensor*，计算输出）：shape为gradOut，self，target的broadcast结果，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。

## 约束说明
- 确定性计算： 
    - aclnnSmoothL1LossBackward默认确定性实现。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_smooth_l1_loss_backward.cpp](examples/test_aclnn_smooth_l1_loss_backward.cpp) | 通过[aclnnSmoothL1LossBackward](docs/aclnnSmoothL1LossBackward.md)接口方式调用SmoothL1LossBackward算子。 |