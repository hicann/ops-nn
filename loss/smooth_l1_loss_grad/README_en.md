# SmoothL1LossGrad

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Computes the backward propagation gradient of SmoothL1Loss. reduction='none', and only the element-wise gradient is output; mean/sum reduction is performed by the forward SmoothL1Loss.
- Formula:

  $$
  gradient = clamp\left(\frac{predict - label}{sigma},\ -1,\ 1\right) \times dout
  $$

  where sigma is equivalent to the `beta` parameter of PyTorch `smooth_l1_loss`, controlling the smooth transition threshold between L1 and L2. When the difference `predict - label = 0` and `dout` is finite, `gradient = 0`.

## Parameters

|Parameter name|Input/Output/Attribute|Description|Data type|Data format|
|:-----|:-----------|:----|:---------|:------|
|predict|Input|Prediction value, corresponding to predict in the formula.|FLOAT16, FLOAT, BFLOAT16|ND|
|label|Input|Target value, corresponding to label in the formula. Its data type must be the same as that of predict.|FLOAT16, FLOAT, BFLOAT16|ND|
|dout|Input|Upstream gradient, corresponding to dout in the formula. Its data type must be the same as that of predict.|FLOAT16, FLOAT, BFLOAT16|ND|
|gradient|Output|Output gradient, corresponding to gradient in the formula. Its data type is the same as that of predict.|FLOAT16, FLOAT, BFLOAT16|ND|
|sigma|Attribute|Smooth L1 transition threshold, corresponding to sigma in the formula, equivalent to the `beta` parameter of PyTorch. The value must be a finite positive number, and the default value is 1.0.|FLOAT|-|

## Constraints

- Deterministic computation: The operator is a pure element-wise operation and uses a deterministic implementation by default.
- The data types of predict, label, and dout must be the same, and each must be one of FLOAT16, FLOAT, and BFLOAT16; the data type of gradient is the same as that of predict.
- The shapes of predict, label, dout, and gradient must be the same (broadcast is not supported currently).
- The value of sigma must be a finite positive number (1.0 by default). Degrading sigma≤0 to the L1 Loss backward semantics is not implemented.

## Invocation Description

| Invocation method | Invocation sample | Description |
| ---------------- | --------------------------- |-------------------------------------------------------------------------|
| GE graph mode  | [test_geir_smooth_l1_loss_grad.cpp](examples/arch35/test_geir_smooth_l1_loss_grad.cpp)   | Calls the SmoothL1LossGrad operator by constructing a graph with the [operator IR](op_graph/smooth_l1_loss_grad_proto.h). |
