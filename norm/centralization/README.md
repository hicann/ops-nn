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

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 | 约束 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 待中心化的张量。 | FLOAT、FLOAT16 | ND | rank为1至8；各执行时维度及输入总元素数均不超过9223372036854775807；支持动态shape/rank建图，执行时各维度必须已经确定；支持空tensor。 |
| axes | 属性 | 均值归约轴。 | LIST_INT | - | 可选，默认`{-1}`；支持负轴，范围`[-rank, rank)`；重复轴去重；未设置时使用最后一个轴；显式空列表表示不归约（输出为0）。 |
| y | 输出 | 中心化结果。 | FLOAT、FLOAT16 | ND | shape、dtype、format 与 x 相同。 |

## 约束说明

- <term>Ascend 950PR&950DT系列产品</term>：
  - 仅支持ND，不支持FRACTAL_NZ、FRACTAL_Z。
  - 支持动态shape/rank建图，执行时各维度必须已经确定（未解析的`-1`/未知rank由tiling拒绝）。
  - 支持空tensor，输出保持与输入相同的shape和dtype。
  - 参数级约束（dtype、rank、axes默认值/范围/去重/空列表、shape一致性）统一见「参数说明」表。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| 图模式调用 | [test_geir_centralization](examples/test_geir_centralization.cpp) | 通过算子IR构图方式调用Centralization算子，IR定义见 [centralization_proto.h](op_graph/centralization_proto.h)。 |
| TensorFlow 解析通路 | - | 通过 TensorFlow 自定义算子 `Centralization`（OriginOpType=Centralization）经 GE 解析执行；x 输入、y 输出，`axes` 在 TF 图中为 `list(int)`，GE IR 侧可选（默认 `{-1}`）。 |

当前支持GE图模式和TensorFlow解析通路，不提供aclnn、PyTorch或ONNX专用接口。本算子仅支持 ND 格式，不支持 FRACTAL_NZ、FRACTAL_Z 等格式。
