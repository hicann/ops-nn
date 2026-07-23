# SwigluGroup

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：实现SwiGLU激活。输入`x`的最后一维被均分为`A`和`B`，计算`silu(A) * B`，直接输出与输入`x`相同数据类型的激活结果。

- 计算公式：

  $$
  y=silu(A) \times B
  $$

  当传入`clamp_limit`时：

  $$
  A=min(A, clamp\_limit)
  $$

  $$
  B=min(max(B, -clamp\_limit), clamp\_limit)
  $$

  当传入`weight`时：

  $$
  y=y \times weight
  $$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 待计算的输入张量，最后一维被均分为两部分用于SwiGLU，需为正且能被2整除。 | FLOAT16、BFLOAT16、FLOAT32 | ND |
| weight | 可选输入 | 每个token的权重，乘到SwiGLU结果上。元素个数需等于`x`除最后一维外的维度乘积。 | FLOAT32 | ND |
| group_index | 可选输入 | count模式下的分组token数量。 | INT64 | ND |
| y | 输出 | SwiGLU计算结果，数据类型与`x`一致，最后一维为`x`最后一维的一半。 | FLOAT16、BFLOAT16、FLOAT32 | ND |
| clamp_limit | 可选属性 | 默认值-1.0，表示不进行clamp；若设置为正数，则在激活前对SwiGLU输入做clamp。 | FLOAT | - |

## 约束说明

- 输入`x`的最后一维需为正，且能被2整除（被均分为`A`、`B`两部分，输出最后一维为输入的一半）。
- 输入`x`的数据类型仅支持FLOAT16、BFLOAT16、FLOAT32，且`y`的数据类型需与`x`一致。
- 当传入`weight`时，数据类型为FLOAT32，且元素个数需等于`x`除最后一维外的维度乘积。
- 当传入`group_index`时，数据类型为INT64，为count模式分组token数，实际处理行数为`group_index`所有元素之和与`bs`两者中的较小值（`bs`为`x`除最后一维外的维度乘积）。
- 当传入`group_index`时，调用者需保证`group_index`所有元素之和非负。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn调用 | [test_aclnn_swiglu_group](./examples/test_aclnn_swiglu_group.cpp) | 通过[aclnnSwigluGroup](./docs/aclnnSwigluGroup.md)接口方式调用SwigluGroup算子。 |
| 图模式调用 | - | 通过[算子IR](./op_graph/swiglu_group_proto.h)构图方式调用SwigluGroup算子。 |
