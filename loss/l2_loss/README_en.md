# L2Loss

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Computes half of the sum of squares of all elements in the input x, that is, the L2 regularization loss. The output y is a scalar.

- Formula:

  $$
  y = \frac{1}{2} \sum_{i} x_i^2
  $$

  Here, $x$ is the input `self`, and $i$ traverses all elements in $x$.

## Parameters

- x (aclTensor*, computation input): The input `x` in the formula, an aclTensor on the Device side. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, and the [data format](../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: The supported data types are BFLOAT16, FLOAT16, and FLOAT.
- y (aclTensor*, computation output): The output `y` in the formula, an aclTensor on the Device side. Its data type is consistent with x. Its shape is 0-dimensional (scalar). [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, and the [data format](../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: The supported data types are BFLOAT16, FLOAT16, and FLOAT.

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GEIR graph construction  | [test_geir_l2_loss.cpp](examples/test_geir_l2_loss.cpp) | Calls the L2Loss operator by building a graph through the operator IR. |
