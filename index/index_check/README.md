# IndexCheck

## 产品支持情况

| 产品                                           | 是否支持 |
| :------------------------------------------- | :--: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：检查索引是否越界。当索引超出边界值范围时，算子会触发硬件异常并终止程序。
- 应用场景：用于索引类aclnn l2接口，在执行索引操作前进行索引合法性检查。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| bounds | 输入 | 各维度的边界值（即各维度的size），长度需与indices列表中tensor个数一致。 | INT64 | ND |
| indices | 输入 | 索引张量列表，列表中每个张量按顺序对应bounds中的一个边界值。 | INT32、INT64 | ND |

## 约束说明

- bounds必须为1维Tensor，长度范围为1～8。
- bounds的长度需要等于indices列表中tensor的个数。
- indices列表不能为空，列表中Tensor的个数不能超过8。
- indices列表中的每个Tensor支持0～8维、空Tensor；不支持9维及以上Tensor。
- 空indices Tensor不执行索引检查；空indices列表为非法输入。
- indices中的索引值支持负数。
- 若索引越界，算子会触发硬件异常并终止程序。
