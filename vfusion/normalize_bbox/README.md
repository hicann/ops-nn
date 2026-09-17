# NormalizeBBox

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

- 算子功能：根据每幅图像的高度和宽度，将目标检测框的绝对坐标换算为相对尺度。
  算子不截断结果，因此输入框超出图像范围时，输出也可能超出`[0, 1]`。
- 计算公式：

  对第`b`个batch，令$h=shape\_hw[b,0]$、$w=shape\_hw[b,1]$，则每个框按以下方式计算：

  $$
  y_{b,i} = boxes_{b,i} / [h, w, h, w]
  $$

  `reversed_box`仅改变坐标轴的位置，不改变计算语义：

  - `reversed_box=false`：坐标轴为最后一维，`boxes`的shape为`[batch, ..., 4]`。
  - `reversed_box=true`：坐标轴为第1维，`boxes`的shape为`[batch, 4, ...]`。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| boxes | 输入 | 目标检测框的绝对坐标，对应公式中的`boxes`。 | FLOAT、FLOAT16 | ND |
| shape_hw | 输入 | 每个batch的图像尺寸，shape为`[batch, 3]`；前两个元素依次为高度和宽度，第3个元素为保留位。 | INT32 | ND |
| reversed_box | 可选属性 | 指定`boxes`的坐标轴位置，默认值为false。 | BOOL | - |
| y | 输出 | 换算后的相对坐标，shape和数据类型与`boxes`一致。 | FLOAT、FLOAT16 | ND |

### 产品差异说明

| 产品 | 静态shape能力 | 动态shape能力 | shape/rank及空Tensor限制 |
| --- | --- | --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | 输入、输出均为ND | 支持动态shape和动态rank，输入、输出均为ND | `boxes`的rank为2～8；支持batch或非坐标维为0的空Tensor，坐标维必须为4 |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term><br><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term><br><term>Atlas 200I/500 A2 推理产品</term><br><term>Atlas 推理系列产品</term><br><term>Atlas 训练系列产品</term> | 输入、输出均为ND | 不支持 | `boxes`仅支持3维，不支持空Tensor |

## 约束说明

- `shape_hw`的batch必须与`boxes`的batch一致。
- 当`reversed_box=false`时，`boxes`的最后一维必须为4；当`reversed_box=true`时，
  `boxes`的第1维必须为4。
- `shape_hw`中的高度和宽度直接作为除数，算子不校验其取值范围。

## 调用说明

本算子不提供aclnn单算子接口，支持图引擎（Graph Engine，GE）图模式及TensorFlow图导入。

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| GE图模式 | [test_geir_normalize_bbox.cpp](examples/arch35/test_geir_normalize_bbox.cpp) | 通过[NormalizeBBox算子IR](op_graph/normalize_bbox_proto.h)构图并调用算子。 |
| TensorFlow图导入 | - | 通过[TensorFlow适配定义](framework/normalize_bbox_tf_plugin.cpp)将TensorFlow的`NormalizeBBox`节点解析为GE算子。 |
