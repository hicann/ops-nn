# BatchNormGradExt2

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

BatchNormGradExt2对BatchNorm的反向梯度进行计算，输出输入梯度、scale梯度、offset梯度以及辅助输出。

输入支持4D/5D张量，支持NCDHW/NCHW/NHWC/NDHWC格式。

## 参数说明

|参数名|输入/输出|数据类型|格式|说明|
|---|---|---|---|---|
|y_backprop|输入|FLOAT16、FLOAT32、BFLOAT16|NCDHW、NCHW、NHWC、NDHWC|反向传播梯度。|
|x|输入|FLOAT16、FLOAT32、BFLOAT16|NCDHW、NCHW、NHWC、NDHWC|正向输入。|
|scale|输入|FLOAT32|ND|通道缩放系数。|
|reserve_space_1|输入|FLOAT32|ND|正向保存的均值。|
|reserve_space_2|输入|FLOAT32|ND|正向保存的统计量。|
|x_backprop|输出|FLOAT16、FLOAT32、BFLOAT16|NCDHW、NCHW、NHWC、NDHWC|输入梯度。|
|scale_backprop|输出|FLOAT32|ND|scale梯度。|
|offset_backprop|输出|FLOAT32|ND|offset梯度。|
|reserve_space_3|输出|FLOAT32|ND|辅助输出。|
|reserve_space_4|输出|FLOAT32|ND|辅助输出。|
|epsilon|属性|FLOAT32|-|默认0.0001。|
|data_format|属性|STRING|-|默认NHWC。|
|is_training|属性|BOOL|-|默认true。|

## 约束说明

- `y_backprop`和`x`的shape一致。
- `scale`、`reserve_space_1`、`reserve_space_2`均为1D，长度等于C。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| 图模式调用 | [batch_norm_grad_ext2_proto.h](op_graph/batch_norm_grad_ext2_proto.h) | 通过[算子IR](op_graph/batch_norm_grad_ext2_proto.h)构图方式调用BatchNormGradExt2算子。 |
