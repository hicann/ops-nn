# KlDivLossGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

算子功能：进行[aclnnKlDiv](https://gitcode.com/cann/ops-math/blob/master/math/kl_div_v2/docs/aclnnKlDiv.md) api的结果的反向计算。

## 参数说明


- gradOutput(aclTensor*, 计算输入)：Device侧的aclTensor，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。shape需要与self满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16。
- self(aclTensor*, 计算输入)：Device侧的aclTensor。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16。
- target(aclTensor*, 计算输入)：Device侧的aclTensor，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。shape需要与self满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16。
- reduction(int64_t, 计算输入)：Host侧的int64_t，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。指定要应用到输出的缩减。支持0（‘none’）| 1（‘mean’）| 2（‘sum’）|3（‘batchmean’）。‘none’表示不应用减少，‘mean’表示输出的总和将除以输出中的元素数，‘sum’表示输出将被求和，‘batchmean’表示输出的总和将除以batch的个数。
- logTarget(bool, 计算输入)：Host侧的BOOL类型，是否对target进行log空间转换。
- out(aclTensor*, 计算输出)：Device侧的aclTensor。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16。

## 约束说明

无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_kl_div_loss_grad.cpp](examples/test_aclnn_kl_div_loss_grad.cpp) | 通过[aclnnKlDivBackward](docs/aclnnKlDivBackward.md)接口方式调用KlDivLossGrad算子。 |