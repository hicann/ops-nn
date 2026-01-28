# QuantUpdateScatter

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> |√|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

先将updates在quantAxis轴上进行量化：quantScales对updates做缩放操作，quantZeroPoints做偏移。然后将量化后的updates中的值按指定的轴axis，根据索引张量indices逐个更新selfRef中对应位置的值。

## 参数说明

  - selfRef（aclTensor*, 计算输入|计算输出）：Device侧的Tensor，源数据张量，数据类型支持INT8，支持3-8维，[数据格式](../../docs/zh/context/数据格式.md)支持ND，支持[非连续的Tensor](../../docs/zh/context/非连续的Tensor.md)。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：其最后一维的大小必须32B对齐。
  - indices（aclTensor*, 计算输入）：Device侧的Tensor，索引张量，数据类型支持INT32、INT64类型，仅支持1维或2维，[数据格式](../../docs/zh/context/数据格式.md)支持ND，支持[非连续的Tensor](../../docs/zh/context/非连续的Tensor.md)。 \
  当indices shape为1维时，indices的取值范围为[0, selfRef.shape(axis) - updates.shape(axis))； \
  当indices shape为2维时，indices每项的第0个数据取值范围[0, selfRef.shape(0))，indices每项的第1个数据取值范围[0, selfRef.shape(axis) - updates.shape(axis))。
  - updates（aclTensor*, 计算输入）：Device侧的Tensor，更新数据张量，updates的维数需要与selfRef的维数一样，其第1维的大小等于indices的第1维的大小，且不大于selfRef的第1维的大小；其axis轴的大小不大于selfRef的axis轴的大小；其余维度的大小要跟selfRef对应维度的大小相等。[数据格式](../../docs/zh/context/数据格式.md)支持ND，支持[非连续的Tensor](../../docs/zh/context/非连续的Tensor.md)。
    - <term>Atlas 推理系列产品</term>：数据类型支持FLOAT16。其最后一维的大小必须32B对齐。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16。其最后一维的大小必须32B对齐。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16。
  - quantScales（aclTensor*, 计算输入）：Device侧的Tensor，量化缩放张量，支持1-8维，quantScales的元素个数需要等于updates在quantAxis轴的大小。支持[非连续的Tensor](../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 推理系列产品</term>：数据类型支持FLOAT32。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT32。
  - quantZeroPoints（aclTensor*, 计算输入）：Device侧的Tensor，量化偏移张量，支持1-8维，quantZeroPoints的元素个数需要等于updates在quantAxis轴的大小。支持[非连续的Tensor](../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../docs/zh/context/数据格式.md)支持ND，可选参数。
    - <term>Atlas 推理系列产品</term>：数据类型支持INT32。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、INT32。
  - axis（int64_t, 计算输入）：updates上用来更新的轴，数据类型为INT64，取值范围[-len(updates.shape) + 1, -1)或者[1, len(updates.shape) - 1)。
  - quantAxis（int64_t, 计算输入）：updates上用来量化的轴，数据类型为INT64，取值支持-1或者len(updates.shape) - 1。
  - reduction（int64_t, 计算输入）：指定数据操作方式，数据类型为INT64，取值支持1（update）。

## 约束说明

- indices的维数只能是1维或者2维；如果是2维，其第2维的大小必须是2；不支持索引越界，索引越界不校验；indices映射的selfRef数据段不能重合，若重合则会因为多核并发原因导致多次执行结果不一样。
- selfRef，indices，updates，quantScales，quantZeroPoints数据类型输入组合包括：
  - <term>Atlas 推理系列产品</term>：

    |selfRef|indices|updates|quantScales|quantZeroPoints|
    |---|---|---|---|---|
    |INT8|INT32|FLOAT16|FLOAT32|INT32|
    |INT8|INT64|FLOAT16|FLOAT32|INT32|
  - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：

    |selfRef|indices|updates|quantScales|quantZeroPoints|
    |---|---|---|---|---|
    |INT8|INT32|BFLOAT16|BFLOAT16|BFLOAT16|
    |INT8|INT64|BFLOAT16|BFLOAT16|BFLOAT16|
    |INT8|INT32|FLOAT16|FLOAT32|INT32|
    |INT8|INT64|FLOAT16|FLOAT32|INT32|

## 调用示例

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_inplace_quant_scatter](example/test_aclnn_inplace_quant_scatter.cpp) | 通过[aclnnInplaceQuantScatter](docs/aclnnInplaceQuantScatter.md)接口方式调用QuantUpdateScatter算子。 |