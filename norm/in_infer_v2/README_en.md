# INInferV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: performs instance normalization inference based on the externally provided mean and variance, and optionally performs scaling and shifting;
  it outputs the normalized result and passes the mean and variance through; you can connect the outputs in the graph as needed. This operator does not compute the mean and variance.
- Formula:

  When both `gamma` and `beta` are provided:

  $$
  y = (x - mean) \times {\gamma\over\sqrt {variance + \epsilon}} + \beta
  $$

  When neither `gamma` nor `beta` is provided:

  $$
  y = {{x - mean}\over\sqrt {variance + \epsilon}}
  $$

  The outputs `batch_mean` and `batch_variance` are copies of `mean` and `variance` respectively.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| x | Input | Tensor to be normalized, corresponding to `x` in the formula. | FLOAT, FLOAT16 | NC1HWC0, ND |
| gamma | Optional input | Scaling parameter, corresponding to `gamma` in the formula; it must be provided or omitted together with `beta`. | FLOAT | NC1HWC0, ND |
| beta | Optional input | Shifting parameter, corresponding to `beta` in the formula; it must be provided or omitted together with `gamma`. | FLOAT | NC1HWC0, ND |
| mean | Optional input | Mean, corresponding to `mean` in the formula; currently it must be provided when invoking the operator on all supported products. For the differences in API declarations, see the product difference description. | FLOAT | NC1HWC0, ND |
| variance | Optional input | Variance, corresponding to `variance` in the formula; currently it must be provided when invoking the operator on all supported products. For the differences in API declarations, see the product difference description. | FLOAT | NC1HWC0, ND |
| epsilon | Optional attribute | Value added to `variance` to improve numerical stability. The default value is 1e-5. | FLOAT | - |
| y | Output | Normalized result; the shape and data type are consistent with `x`. | FLOAT, FLOAT16 | NC1HWC0, ND |
| batch_mean | Optional output | Returns a copy of `mean`; this output does not need to be connected in the graph. | FLOAT | NC1HWC0, ND |
| batch_variance | Optional output | Returns a copy of `variance`; this output does not need to be connected in the graph. | FLOAT | NC1HWC0, ND |

### Product Difference Description

| Product | Static shape capability | Dynamic shape capability | Optional parameters | Shape/rank and combination limits |
| --- | --- | --- | --- | --- |
| <term>Ascend 950PR&950DT products</term> | All inputs and outputs are ND | Supports dynamic shape and dynamic rank; all inputs and outputs are ND | `gamma` and `beta` must be provided or omitted together; `mean` and `variance` are registered as optional inputs but must be provided at invocation; `batch_mean` and `batch_variance` are optional outputs and do not need to be connected in the graph | The rank of `x` is 2～8, and the shape is `[N, C, R...]`; the numbers of elements of `gamma`, `beta`, `mean`, and `variance` are all `N*C`; empty tensors are supported |
| <term>Atlas A3 products</term><br><term>Atlas A2 products</term><br><term>Atlas 200I/500 A2 inference products</term><br><term>Atlas inference products</term><br><term>Atlas training products</term> | All inputs and outputs are NC1HWC0 | Not supported | `gamma` and `beta` must be provided or omitted together; `mean` and `variance` are mandatory inputs; `batch_mean` and `batch_variance` are optional outputs | `x` is a 5-dimension `[N, C1, H, W, C0]`; `gamma`, `beta`, `mean`, and `variance` are `[N, C1, 1, 1, C0]`; empty tensors are not supported |

For the <term>Ascend 950PR&950DT products</term>, `batch_mean` and `batch_variance` are optional outputs, and the graph can connect only the required results.
A static GE graph depends on a precompiled binary that matches the operator registration configuration. During deployment, you need to install the complete operator package that contains the binary of this operator.

## Constraints

None

## Invocation Description

This operator does not provide an aclnn single-operator API, and it is invoked in Graph Engine (GE) graph mode.

| Invocation method | Invocation sample | Description |
| --- | --- | --- |
| GE graph mode | [test_geir_in_infer_v2](examples/arch35/test_geir_in_infer_v2.cpp) | Constructs the graph through the [INInferV2 operator IR](op_graph/in_infer_v2_proto.h) and invokes the operator. The sample covers both scenarios of providing and omitting `gamma` and `beta`. |
