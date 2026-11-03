# Centralization

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

- 算子功能：计算输入张量在指定轴上的均值，并从输入对应元素中逐元素减去。
- 计算公式：

  $$
  y = x - mean(x, axes)
  $$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 待中心化的张量。 | FLOAT、FLOAT16 | ND |
| axes | 属性 | 均值归约轴，LIST_INT。支持负轴，轴必须唯一且位于 `[-rank, rank)`；是否可选及默认值见约束说明。 | LIST_INT | - |
| y | 输出 | 与x的shape、dtype相同。 | FLOAT、FLOAT16 | ND |

## 约束说明

- <term>Ascend 950PR&950DT系列产品</term>：
  - 输入和输出支持FLOAT、FLOAT16数据类型和ND格式，且shape与dtype必须相同。
  - 输入rank为1至8。支持动态shape/rank建图，执行时各维度必须已经确定。
  - 每个执行时维度和输入总元素数均不超过 9223372036854775807（int64 上限；实际受设备内存约束）。
  - `axes`可选，默认值为 `{-1}`；未设置或设置为空时使用最后一个轴。
  - 支持空tensor，输出保持与输入相同的shape和dtype。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| 图模式调用 | [test_geir_centralization](examples/test_geir_centralization.cpp) | 通过算子IR构图方式调用Centralization算子，IR定义见 [centralization_proto.h](op_graph/centralization_proto.h)。 |
| TensorFlow 解析通路 | - | 通过 TensorFlow 自定义算子 `Centralization`（OriginOpType=Centralization）经 GE 解析执行，参数与算子IR一致（x 输入、axes 可选属性、y 输出）。 |

当前支持GE图模式和TensorFlow解析通路，不提供aclnn、PyTorch或ONNX专用接口。本算子仅支持 ND 格式，不支持 FRACTAL_NZ、FRACTAL_Z 等格式。
