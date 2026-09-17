# INTrainingReduceV2

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | √ |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：作为实例归一化（Instance Normalization）训练前向的规约阶段，
  对每个实例通道的空间维分别计算元素和`sum`及平方和`square_sum`，并保留规约维度。
  本算子与INTrainingUpdateV2配合使用。
- 计算公式：

  $$
  sum_{n,c} = \sum_{r \in R} x_{n,c,r}
  $$

  $$
  square\_sum_{n,c} = \sum_{r \in R} x_{n,c,r}^{2}
  $$

  其中，$R$表示空间维集合；输出为原始和，不进行均值缩放。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| x | 输入 | 待规约的输入Tensor，对应公式中的`x`。 | FLOAT、FLOAT16 | NC1HWC0、NDC1HWC0、ND、NCHW、NHWC、NCDHW、NDHWC |
| sum | 输出 | `x`沿空间维求和的结果，保留实例维和通道维，规约维大小为1。 | FLOAT | NC1HWC0、NDC1HWC0、ND |
| square_sum | 输出 | `x`沿空间维求平方和的结果，shape与`sum`一致。 | FLOAT | NC1HWC0、NDC1HWC0、ND |

### 产品差异说明

| 产品 | 静态shape能力 | 动态shape能力 | shape/rank及格式映射 |
| --- | --- | --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | `x`支持NCHW、NCDHW、ND，`sum`和`square_sum`为ND | 支持动态shape和动态rank，输入、输出均为ND | NCHW输入为4维，NCDHW输入为5维；ND输入的rank为2～8，dim0和dim1分别为实例维和通道维 |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term><br><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term><br><term>Atlas 200I/500 A2 推理产品</term><br><term>Atlas 推理系列产品</term><br><term>Atlas 训练系列产品</term> | 4维NCHW/NHWC原始格式映射为NC1HWC0，5维NCDHW/NDHWC原始格式映射为NDC1HWC0；输入、输出格式一致 | 支持动态shape和动态rank，输入、输出均为NC1HWC0或NDC1HWC0 | 实际执行时，原始输入为4维或5维 |

## 约束说明

- 不支持空Tensor，`x`的任一维度均不能为0。

## 调用说明

本算子不提供aclnn单算子接口，通过图引擎（Graph Engine，GE）图模式调用。

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| GE图模式 | [test_geir_in_training_reduce_v2](examples/arch35/test_geir_in_training_reduce_v2.cpp) | 通过[INTrainingReduceV2算子IR](op_graph/in_training_reduce_v2_proto.h)构图并调用算子。 |
