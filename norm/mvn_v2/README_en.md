# MVNV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----: |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |
| Atlas 200I/500 A2 inference products | × |
| Atlas inference products | × |
| Atlas training products | × |

## Function

- Operator function: computes the mean and standard deviation of an ND input tensor along the specified axes, performs mean variance normalization, and makes the output have a mean of 0 and a variance close to 1 on the reduction axes.

- Formula:

$$
y = \frac{x - mean}{\sqrt{var} + eps}
$$

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | ND input tensor, corresponding to x in the formula. | FLOAT16, FLOAT | ND |
| y | Output | Normalized output tensor, with the same shape and dtype as x, corresponding to y in the formula. | FLOAT16, FLOAT | ND |
| eps | Optional attribute | Small constant to prevent division by zero, added to std (std + eps), with the default value 1e-9. | FLOAT | - |
| axes | Optional attribute | List of reduction axes. Each element must be in [0, rank(x)). By default the valid axes in [0, 2, 3] are taken. | INT64 | - |

## Constraints

- Input x supports 1-D to 8-D ND tensors.
- Each element of axes must be in [0, rank(x)); out-of-range values are rejected by the tiling stage.
- For an empty tensor input, an empty tensor output of the same shape is returned directly, and the kernel computation is not executed.
- Only the FLOAT16 and FLOAT data types are supported, and the output data type is consistent with the input.

## Invocation Description

| Invocation method | Invocation sample | Description |
| :--- | :--- | :--- |
| GE graph mode | [test_geir_mvn_v2](./examples/test_geir_mvn_v2.cpp) | Integrates into the Graph Engine graph mode through the operator IR definition [op_graph/mvn_v2_proto.h](./op_graph/mvn_v2_proto.h). |
