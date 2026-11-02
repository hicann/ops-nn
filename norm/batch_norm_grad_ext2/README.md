# BatchNormGradExt2

## 产品支持情况

|产品 | 是否支持 |
|---|---|
|Ascend 950PR&950DT系列产品|√|
|Atlas A3系列产品|√|
|Atlas A2系列产品|√|
|Atlas 200I/500 A2推理产品|√|
|Atlas推理系列产品|√|
|Atlas训练系列产品|√|

## 功能说明

BatchNormGradExt2对BatchNorm的反向梯度进行计算，输出输入梯度、scale梯度、offset梯度以及辅助输出。

输入支持4D/5D张量，支持NCDHW/NCHW/NHWC/NDHWC格式。

## 训练/推理模式说明

本算子通过 `is_training` 属性区分训练与推理两种模式。

### 训练模式（`is_training=true`）

```text
scale_backprop  = sum(y_backprop * (x - mean) / sqrt(var + epsilon))
offset_backprop = sum(y_backprop)
x_backprop      = scale / sqrt(var + epsilon) *
                  (y_backprop - mean(y_backprop) - mean(y_backprop * x_hat) * x_hat)
reserve_space_3 = 0
reserve_space_4 = 0
```

此时 `reserve_space_2` 的语义为正向保留的 `save_invstd`（即 `rstd = 1 / sqrt(var + epsilon)`），**不是** variance；扩展输出 `reserve_space_3`、`reserve_space_4` 固定置零。

### 推理模式（`is_training=false`）

```text
x_backprop = y_backprop * scale / sqrt(var + epsilon)
```

`x_backprop` 的 shape 与输入梯度 `y_backprop` 一致；此模式下 `reserve_space_2` 才按 variance 处理。`scale_backprop`、`offset_backprop` 为扩展输出，按接口语义固定输出。

## 参数说明

|参数名|输入/输出|数据类型|格式|说明|
|---|---|---|---|---|
|y_backprop|输入|FLOAT16、FLOAT32|NCDHW、NCHW、NHWC、NDHWC|反向传播梯度。|
|x|输入|FLOAT16、FLOAT32|NCDHW、NCHW、NHWC、NDHWC|正向输入。|
|scale|输入|FLOAT32|ND|通道缩放系数。|
|reserve_space_1|输入|FLOAT32|ND|正向保存的均值。|
|reserve_space_2|输入|FLOAT32|ND|正向保存的统计量。|
|x_backprop|输出|FLOAT16、FLOAT32|NCDHW、NCHW、NHWC、NDHWC|输入梯度。|
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
- **不支持空 tensor**：`y_backprop`/`x` 的所有维度必须大于 0（shape 中不能含 0 维，如 `(4, 8, 0, 16)`）。传入含 0 维的空 tensor 时，tiling 校验失败并返回 `y_backprop ... dim ... should be greater than 0`。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| 图模式调用 | [test_geir_batch_norm_grad_ext2](examples/test_geir_batch_norm_grad_ext2.cpp) | 通过[算子IR](op_graph/batch_norm_grad_ext2_proto.h)构图方式调用BatchNormGradExt2算子。 |
