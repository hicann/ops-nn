# INInferV2

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

- 算子功能：根据外部提供的均值和方差执行实例归一化推理，并可选进行缩放和平移；
  输出归一化结果，并可选透传均值和方差。本算子不计算均值和方差。
- 计算公式：

  当`gamma`和`beta`均提供时：

  $$
  y = (x - mean) \times {\gamma\over\sqrt {variance + \epsilon}} + \beta
  $$

  当`gamma`和`beta`均未提供时：

  $$
  y = {{x - mean}\over\sqrt {variance + \epsilon}}
  $$

  可选输出`batch_mean`和`batch_variance`分别为`mean`和`variance`的副本。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| x | 输入 | 待归一化的Tensor，对应公式中的`x`。 | FLOAT、FLOAT16 | NC1HWC0、ND |
| gamma | 可选输入 | 缩放参数，对应公式中的`gamma`；必须与`beta`同时提供或同时省略。 | FLOAT | NC1HWC0、ND |
| beta | 可选输入 | 平移参数，对应公式中的`beta`；必须与`gamma`同时提供或同时省略。 | FLOAT | NC1HWC0、ND |
| mean | 可选输入 | 均值，对应公式中的`mean`；当前各支持产品调用时均必须提供，接口声明差异见产品差异说明。 | FLOAT | NC1HWC0、ND |
| variance | 可选输入 | 方差，对应公式中的`variance`；当前各支持产品调用时均必须提供，接口声明差异见产品差异说明。 | FLOAT | NC1HWC0、ND |
| epsilon | 可选属性 | 加到`variance`上的数值，用于提高数值稳定性，默认值为1e-5。 | FLOAT | - |
| y | 输出 | 归一化结果，shape和数据类型与`x`一致。 | FLOAT、FLOAT16 | NC1HWC0、ND |
| batch_mean | 可选输出 | 提供该输出时，返回`mean`的副本；省略时不生成该输出。 | FLOAT | NC1HWC0、ND |
| batch_variance | 可选输出 | 提供该输出时，返回`variance`的副本；省略时不生成该输出。 | FLOAT | NC1HWC0、ND |

### 产品差异说明

| 产品 | 静态shape能力 | 动态shape能力 | 可选参数 | shape/rank及组合限制 |
| --- | --- | --- | --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | 各输入、输出均为ND | 支持动态shape和动态rank，各输入、输出均为ND | `gamma`和`beta`必须同时提供或同时省略；`mean`和`variance`注册为可选输入，但调用时必须提供；`batch_mean`和`batch_variance`为可选输出 | `x`的rank为2～8，shape为`[N, C, R...]`；`gamma`、`beta`、`mean`、`variance`的元素数均为`N*C`；支持空Tensor |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term><br><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term><br><term>Atlas 200I/500 A2 推理产品</term><br><term>Atlas 推理系列产品</term><br><term>Atlas 训练系列产品</term> | 各输入、输出均为NC1HWC0 | 不支持 | `gamma`和`beta`必须同时提供或同时省略；`mean`和`variance`为必选输入；`batch_mean`和`batch_variance`为可选输出 | `x`为5维`[N, C1, H, W, C0]`；`gamma`、`beta`、`mean`、`variance`为`[N, C1, 1, 1, C0]`；不支持空Tensor |

## 约束说明

无

## 调用说明

本算子不提供aclnn单算子接口，通过图引擎（Graph Engine，GE）图模式调用。

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| GE图模式 | [test_geir_in_infer_v2](examples/arch35/test_geir_in_infer_v2.cpp) | 通过[INInferV2算子IR](op_graph/in_infer_v2_proto.h)构图并调用算子，样例覆盖提供和省略`gamma`、`beta`两种场景。 |
