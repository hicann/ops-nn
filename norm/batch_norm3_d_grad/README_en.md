# BatchNorm3DGrad

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

BatchNorm3DGrad computes the backward gradients of BatchNorm3D and outputs the input gradient, the scale gradient, the offset gradient, and auxiliary outputs.

The input supports 4D/5D tensors and the NCDHW/NCHW/NHWC/NDHWC formats.

## Parameters

|Parameter name|Input/Output|Data type|Format|Description|
|---|---|---|---|---|
|y_backprop|Input|FLOAT16, FLOAT32|NCDHW, NCHW, NHWC, NDHWC|Backpropagation gradient.|
|x|Input|FLOAT16, FLOAT32|NCDHW, NCHW, NHWC, NDHWC|Forward input.|
|scale|Input|FLOAT32|ND|Channel scaling coefficient.|
|reserve_space_1|Input|FLOAT32|ND|Mean saved in the forward pass.|
|reserve_space_2|Input|FLOAT32|ND|Reciprocal of the standard deviation or an equivalent statistic saved in the forward pass.|
|x_backprop|Output|FLOAT16, FLOAT32|NCDHW, NCHW, NHWC, NDHWC|Input gradient.|
|scale_backprop|Output|FLOAT32|ND|Scale gradient.|
|offset_backprop|Output|FLOAT32|ND|Offset gradient.|
|reserve_space_4|Output|FLOAT32|ND|Auxiliary output.|
|reserve_space_5|Output|FLOAT32|ND|Auxiliary output.|
|epsilon|Attribute|FLOAT32|-|The default value is 0.0001.|
|data_format|Attribute|STRING|-|The default value is NCDHW.|
|is_training|Attribute|BOOL|-|The default value is true.|

## Constraints

- The shape of `y_backprop` is the same as that of `x`.
- `scale`, `reserve_space_1`, and `reserve_space_2` are all 1D, with a length equal to C.

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| Graph-mode invocation | [test_geir_batch_norm3d_grad](examples/test_geir_batch_norm3d_grad.cpp) | Calls the BatchNorm3DGrad operator by building a graph through [operator IR](op_graph/batch_norm3_d_grad_proto.h). |
