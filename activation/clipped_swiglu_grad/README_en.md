# ClippedSwigluGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Description: Backward operator of aclnnClippedSwiglu, which computes the gradient `gradXOut` of the input x.

- Formula:

  For the given forward input tensor x (with dimensions [a,b,c,d,e,f,g...]) and the output gradient gradY, ClippedSwigluGrad performs the following computation:

  1. Merges the axes of x based on the input parameter dim, and the merged dimensions are [pre, dim2H]; after axis merging, the dimensions of gradY are [pre, dim2H//2].

  2. Filters the pre axis of x based on the input parameter group_index, so that only the first group rows participate in the computation and the gradients of the remaining rows are 0.

  3. Splits x into parts a and b based on the input parameter interleaved (consistent with the forward propagation).

  4. Recomputes the forward intermediate quantities:
     - A = clamp(a, max=limit)
     - B = clamp(b, -limit, limit)
     - s = sigmoid(alpha * A)

  5. Computes the backward gradients:
     - maskX0 = (a <= limit)
     - maskX1 = (-limit <= b <= limit)
     - da = gradY * (B + bias) * s * (1 + alpha * A * (1 - s)) * maskX0
     - db = gradY * A * s * maskX1

  6. Scatters da and db back to dx:
     - When interleaved is true: dx[..., ::2] = da, dx[..., 1::2] = db
     - When interleaved is false: dx[..., :h] = da, dx[..., h:] = db

  7. Reshapes the output tensor grad_x so that its dimensions are consistent with those of x.

## Parameters

|Parameter name|Input/Output/Attribute|Description|Data type|Data format|
|---|---|---|---|---|
|grad_y|Input|Gradient of the forward output y. Its dimension on the dim axis is half of that of x, and the other dimensions are consistent with those of x.|FLOAT, FLOAT16, BFLOAT16|ND|
|x|Input|Forward input x.|FLOAT, FLOAT16, BFLOAT16|ND|
|group_index|Optional input|Grouping index. 1D, with no more than 8192 elements.|INT64|-|
|dim|Optional attribute|Index of the dimension for axis merging and splitting. Value range: [-x.dim(), x.dim()-1]. Default: -1.|INT64|-|
|alpha|Optional attribute|SwiGLU activation coefficient. Default: 1.702.|FLOAT|-|
|limit|Optional attribute|Clip threshold value. It must be greater than 0. Default: 7.0.|FLOAT|-|
|bias|Optional attribute|Deviation of the linear computation. Default: 1.0.|FLOAT|-|
|interleaved|Optional attribute|Splitting method. true = odd-even splitting, false = front-back splitting. Default: true.|BOOL|-|
|grad_x|Output|Gradient of x, with the same shape as x.|FLOAT, FLOAT16, BFLOAT16|ND|

## Constraints

None.

## Invocation Description

| Invocation method | Invocation sample | Description |
|---|---|---|
| aclnn invocation | [test_aclnn_clipped_swiglu_grad](./examples/test_aclnn_clipped_swiglu_grad.cpp) | Invokes the ClippedSwigluGrad operator through the [aclnnClippedSwigluGrad](./docs/aclnnClippedSwigluGrad_en.md) interface. |
