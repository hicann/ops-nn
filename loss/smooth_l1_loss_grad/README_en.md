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

- Operator function: computes the backpropagation gradient of SmoothL1Loss. reduction='none', and only the element-wise gradient is output. The mean/sum reduction is handled by the forward SmoothL1Loss.
- Formula:

  $$
  gradient = clamp\left(\frac{predict - label}{sigma},\ -1,\ 1\right) \times dout
  $$

  Here sigma is equivalent to the `beta` parameter of the PyTorch `smooth_l1_loss`, and it controls the smooth transition threshold between L1 and L2. When the difference `predict - label = 0`, `gradient = 0`.

## Parameters

|Parameter name|Input/Output/Attribute|Description|Data type|Data format|
|:-----|:-----------|:----|:---------|:------|
|predict|Input|The predicted value, corresponding to predict in the formula.|FLOAT16, FLOAT, BFLOAT16|ND|
|label|Input|The target value, corresponding to label in the formula. The data type is the same as predict.|FLOAT16, FLOAT, BFLOAT16|ND|
|dout|Input|The upstream gradient, corresponding to dout in the formula. The data type is the same as predict.|FLOAT16, FLOAT, BFLOAT16|ND|
|gradient|Output|The output gradient, corresponding to gradient in the formula. The data type is the same as predict.|FLOAT16, FLOAT, BFLOAT16|ND|
|sigma|Attribute|The Smooth L1 smoothing threshold, corresponding to sigma in the formula, equivalent to the beta of PyTorch. The value must be greater than 0. The default value is 1.0.|FLOAT|-|

## Constraints

- Deterministic Computation: the operator is a pure element-wise operation and is deterministic by default.
- The data types of predict, label, and dout must be consistent, and they must all be one of FLOAT16, FLOAT, and BFLOAT16. The data type of gradient is the same as predict.
- The shapes of predict, label, dout, and gradient must be consistent (broadcast is currently not supported), and 0-8 dimensions are supported.
- The value of sigma must be greater than 0 (default 1.0). The semantics of degrading to L1 Loss backward when sigma≤0 is not implemented.
- Empty tensors are supported. In the Tiling stage, empty tensors short-circuit successfully, and the Kernel does not initiate GM reads or writes.
- The ACLNN interface performs `AutoContiguous()` to make non-contiguous inputs contiguous. The GE graph path uses the ND format according to the registration constraints.

## Invocation Description

| Invocation method | Invocation sample | Description |
| ---------------- | --------------------------- |-------------------------------------------------------------------------|
| GE graph mode  | [test_geir_smooth_l1_loss_grad.cpp](examples/arch35/test_geir_smooth_l1_loss_grad.cpp)   | Invoke the SmoothL1LossGrad operator by constructing a graph through the [operator IR](op_graph/smooth_l1_loss_grad_proto.h). |
