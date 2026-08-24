# BatchNorm3D

## 产品支持情况

|产品 | 是否支持 |
|---|---|
|Ascend 950PR/Ascend 950DT|√|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|
|Atlas 200I/500 A2 推理产品|×|
|Atlas 推理系列产品|√|
|Atlas 训练系列产品|√|

## 功能说明

BatchNorm3D对5D输入张量做批归一化，支持训练和推理模式。

计算公式：

```text
y = (x - mean) / sqrt(variance + epsilon) * scale + offset
```

训练模式下，`mean`和`variance`由输入批次按通道归约得到；推理模式下使用输入`mean`和`variance`。

## 参数说明

|参数名|输入/输出|数据类型|格式|说明|
|---|---|---|---|---|
|x|输入|FLOAT16、FLOAT32|NCDHW、NDHWC|5D输入，shape的通道维与`scale`一致。|
|scale|输入|FLOAT32|ND|缩放系数，长度等于通道数。|
|offset|输入|FLOAT32|ND|偏置，长度等于通道数。|
|mean|输入|FLOAT32|ND|推理模式使用，训练模式可为空。|
|variance|输入|FLOAT32|ND|推理模式使用，训练模式可为空。|
|y|输出|FLOAT16、FLOAT32|NCDHW、NDHWC|输出，shape与`x`一致。|
|batch_mean|输出|FLOAT32|ND|均值输出，长度等于通道数。|
|batch_variance|输出|FLOAT32|ND|方差输出，长度等于通道数。|
|reserve_space_1|输出|FLOAT32|ND|辅助输出，长度等于通道数。|
|reserve_space_2|输出|FLOAT32|ND|辅助输出，长度等于通道数。|
|epsilon|属性|FLOAT32|-|防止除零的小数，默认值为0.0001。|
|data_format|属性|STRING|-|支持`NCDHW`、`NDHWC`，默认值为`NCDHW`。|
|is_training|属性|BOOL|-|是否为训练模式，默认值为true。|

## 约束说明

- 当前实现支持`NCDHW`、`NDHWC`逻辑格式。
- `scale`、`offset`、`mean`、`variance`均为一维张量，长度需等于通道数。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| 图模式调用 | [test_geir_batch_norm3d](examples/test_geir_batch_norm3d.cpp) | 通过[算子IR](op_graph/batch_norm3d_proto.h)构图方式调用BatchNorm3D算子（含两组手工可算用例）。 |
