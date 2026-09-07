# Centralization

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | √ |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：计算输入张量在指定轴上的均值，并从输入对应元素中逐元素减去。
- 计算公式：

  $$
  y = x - mean(x, axes)
  $$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 待中心化的张量。rank为1至8，执行时各维度必须为具体非负整数。 | FLOAT、FLOAT16 | ND |
| axes | 属性 | 均值归约轴，LIST_INT，默认值为 `{-1}`。支持负轴；轴必须唯一且位于 `[-rank, rank)`。未设置或设置为空时使用最后一个轴。 | LIST_INT | - |
| y | 输出 | 与x的shape、dtype相同。 | FLOAT、FLOAT16 | ND |

## 约束说明

- <term>Ascend 950PR/Ascend 950DT</term>：
  - 输入和输出必须具有相同的shape与dtype；归约轴必须唯一且位于有效范围内。
  - 支持动态 shape/rank 建图；执行时各维度必须已经确定。
  - 支持空 tensor，输出保持与输入相同的 shape 和 dtype。
  - 输入和输出支持 ND 格式。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| 图模式调用 | [test_geir_centralization](examples/test_geir_centralization.cpp) | 通过算子 IR 构图方式调用 Centralization 算子。 |

当前支持GE图模式和TensorFlow解析通路，不提供aclnn、PyTorch或ONNX专用接口。
