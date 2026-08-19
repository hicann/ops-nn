# GroupNorm

## 产品支持情况

<!-- markdownlint-disable MD033 -->

| 产品 | 是否支持 |
| :----------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

<!-- markdownlint-enable MD033 -->

## 功能说明

- 算子功能：GroupNorm将输入`x`的通道轴划分为`num_groups`组，对每个样本、每个组独立计算总体均值和总体方差。

- 计算公式：

  $$
  y = \frac{x - mean}{\sqrt{variance + eps}} \times gamma + beta
  $$

  输出`mean`和`variance`的shape均为`(N, num_groups)`。`data_format`和`is_training`仅用于兼容既有GEIR接口，不参与Kernel计算。

## 参数说明

以下参数说明适用于GE图模式，aclnn API参数说明请参见[aclnnGroupNorm.md](docs/aclnnGroupNorm.md)。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 公式中的输入`x`，shape至少为2维，按`(N, C, *)`解释。 | FLOAT16、FLOAT | ND |
| gamma | 输入 | 公式中的`gamma`，为长度为`C`的一维Tensor，类型与`x`相同。 | FLOAT16、FLOAT | ND |
| beta | 输入 | 公式中的`beta`，为长度为`C`的一维Tensor，类型与`x`相同。 | FLOAT16、FLOAT | ND |
| num_groups | 必选属性 | 通道分组数，取值为正整数，且`C`必须能被其整除。 | INT | - |
| data_format | 可选属性 | 输入数据格式，默认值为`NCHW`，当前仅用于接口兼容。 | STRING | - |
| eps | 可选属性 | 添加到方差上的数值稳定项，默认值为`0.0001`。 | FLOAT | - |
| is_training | 可选属性 | 是否为训练模式，默认值为`true`，当前仅用于接口兼容。 | BOOL | - |
| y | 输出 | 公式中的输出`y`，shape和数据类型均与`x`相同。 | FLOAT16、FLOAT | ND |
| mean | 输出 | 每组均值，shape为`(N, num_groups)`。 | FLOAT16、FLOAT | ND |
| variance | 输出 | 每组总体方差，shape为`(N, num_groups)`。 | FLOAT16、FLOAT | ND |

## 约束说明

- GE图模式下，所有输入和输出的数据类型必须一致。
- Ascend 950的GE图模式要求`C`大于0；当`N=0`时三个输出均为空且不下发Kernel。
- Ascend 950的GE图模式在`N`大于0时不支持包含零维的空Tensor，Host校验失败后不下发Kernel。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | [test_geir_group_norm.cpp](examples/test_geir_group_norm.cpp) | 通过[GroupNorm IR](op_graph/group_norm_proto.h)构图调用。 |
| aclnn API | [test_aclnn_GroupNorm.cpp](examples/test_aclnn_GroupNorm.cpp) | 通过[aclnnGroupNorm](docs/aclnnGroupNorm.md)接口调用。 |
