# L1LossGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |


## 功能说明

- 算子功能：计算[aclnnL1Loss](../../lp_loss/docs/aclnnL1Loss.md)的反向传播。reduction指定损失函数的计算方式，支持 'none'、'mean'、'sum'。'none' 表示不应用减少，'mean' 表示输出的总和将除以输出中的元素数，'sum' 表示输出将被求和。

## 参数说明

- gradOutput (aclTensor*，计算输入)：梯度反向输入，公式中的输入`gradOutput`，Device侧的aclTensor。数据类型与self、target的数据类型满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。shape需要与self、target满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)，且小于9维。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。
- self (aclTensor*，计算输入)：公式中的输入`self`，Device侧的aclTensor。数据类型与gradOutput、target满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。shape需要与gradOutput、target满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)，且小于9维。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。
- target (aclTensor*，计算输入)：公式中的输入`target`，Device侧的aclTensor。数据类型要与self、gradOutput满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。shape需要与gradOutput、self满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)，且小于9维。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。
- reduction (int64_t, 计算输入)：用于指定损失函数的计算方式，Host侧的整型。取值支持 0('none') | 1('mean') | 2('sum')。'none' 表示不应用减少，'mean' 表示输出的总和将除以输出中的元素数，'sum' 表示输出将被求和。
- gradInput (aclTensor*，计算输出)：公式中的输出`gradInput`，Device侧的aclTensor。数据类型要与self满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。shape需要与target、self、gradOutput满足broadcast关系[broadcast关系](../../../docs/zh/context/broadcast关系.md)，且小于9维。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。

## 约束说明

无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                                                      |
| ---------------- | --------------------------- |-------------------------------------------------------------------------|
| aclnn接口  | [test_aclnn_l1_loss_grad.cpp](examples/test_aclnn_l1_loss_grad.cpp) | 通过[aclnnL1LossBackward](docs/aclnnL1LossBackward.md)接口方式调用L1LossGrad算子。 |