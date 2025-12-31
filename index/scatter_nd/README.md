# ScatterNd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明
算子功能：拷贝data的数据至out，同时在指定indices处根据updates更新out中的数据。

## 参数说明
  - data(aclTensor*,计算输入)：Device侧的aclTensor, 数据类型与updates、out一致，shape满足1<=rank(data)<=8。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、BOOL、BFLOAT16

  - indices(aclTensor*,计算输入)：Device侧的aclTensor，数据类型支持INT32、INT64。indices.shape[-1] <= rank(data)，且1<=rank(indices)<=8。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。仅支持非负索引。indices中的索引数据不支持越界。

  - updates(aclTensor*,计算输入)：Device侧的aclTensor, 数据类型与data、out一致。shape要求rank(updates)=rank(data)+rank(indices)-indices.shape[-1] -1, 且满足1<=rank(updates)<=8。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、BOOL、BFLOAT16

  - out(aclTensor*，计算输出)：Device侧的aclTensor，数据类型与data、out一致，shape与data一致。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT、BOOL、BFLOAT16

## 约束说明
无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_scatter_nd](examples/test_aclnn_scatter_nd.cpp) | 通过[aclnnScatterNd](docs/aclnnScatterNd.md)接口方式调用ScatterAdd算子。 |