# BatchNormGradExt2

English | [简体中文](./README.md)

## Applicable Products

|Product | Support |
|---|---|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|√|
|Atlas inference products|√|
|Atlas training products|√|

## Function

BatchNormGradExt2 computes the backward gradients of BatchNorm, and outputs the input gradient, scale gradient, offset gradient, and auxiliary outputs.

The input supports 4D/5D tensors and the NCDHW/NCHW/NHWC/NDHWC formats.

## Training/Inference Mode Description

This operator distinguishes training and inference modes through the `is_training` attribute.

### Training mode (`is_training=true`)

```text
scale_backprop  = sum(y_backprop * (x - mean) / sqrt(var + epsilon))
offset_backprop = sum(y_backprop)
x_backprop      = scale / sqrt(var + epsilon) *
                  (y_backprop - mean(y_backprop) - mean(y_backprop * x_hat) * x_hat)
reserve_space_3 = 0
reserve_space_4 = 0
```

In this case, the semantics of `reserve_space_2` is the `save_invstd` retained from the forward pass (that is, `rstd = 1 / sqrt(var + epsilon)`), **not** the variance. The extended outputs `reserve_space_3` and `reserve_space_4` are fixed to zero.

### Inference mode (`is_training=false`)

```text
x_backprop = y_backprop * scale / sqrt(var + epsilon)
```

The shape of `x_backprop` is the same as that of the input gradient `y_backprop`. Only in this mode is `reserve_space_2` treated as the variance. `scale_backprop` and `offset_backprop` are extended outputs and are fixedly output according to the API semantics.

## Parameters

|Parameter name|Input/Output|Data type|Format|Description|
|---|---|---|---|---|
|y_backprop|Input|FLOAT16, FLOAT32|NCDHW, NCHW, NHWC, NDHWC|Backpropagation gradient.|
|x|Input|FLOAT16, FLOAT32|NCDHW, NCHW, NHWC, NDHWC|Forward input.|
|scale|Input|FLOAT32|ND|Channel scaling coefficient.|
|reserve_space_1|Input|FLOAT32|ND|Mean saved from the forward pass.|
|reserve_space_2|Input|FLOAT32|ND|Statistic saved from the forward pass.|
|x_backprop|Output|FLOAT16, FLOAT32|NCDHW, NCHW, NHWC, NDHWC|Input gradient.|
|scale_backprop|Output|FLOAT32|ND|Scale gradient.|
|offset_backprop|Output|FLOAT32|ND|Offset gradient.|
|reserve_space_3|Output|FLOAT32|ND|Auxiliary output.|
|reserve_space_4|Output|FLOAT32|ND|Auxiliary output.|
|epsilon|Attribute|FLOAT32|-|Default value: 0.0001.|
|data_format|Attribute|STRING|-|Default value: NHWC.|
|is_training|Attribute|BOOL|-|Default value: true.|

## Constraints

- The shapes of `y_backprop` and `x` are identical.
- `scale`, `reserve_space_1`, and `reserve_space_2` are all 1D, and the length equals C.

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| Graph-mode invocation | [test_geir_batch_norm_grad_ext2](examples/test_geir_batch_norm_grad_ext2.cpp) | Calls the BatchNormGradExt2 operator through graph construction with the [operator IR](op_graph/batch_norm_grad_ext2_proto.h). |
