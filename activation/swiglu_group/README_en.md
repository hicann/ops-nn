# SwigluGroup

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: Implements the SwiGLU activation. The last dimension of the input `x` is evenly split into `A` and `B`, and the operator computes `silu(A) * B`, directly outputting the activation result with the same data type as the input `x`.

- Formula:

  $$
  y=silu(A) \times B
  $$

  When `clamp_limit` is passed:

  $$
  A=min(A, clamp\_limit)
  $$

  $$
  B=min(max(B, -clamp\_limit), clamp\_limit)
  $$

  When `weight` is passed:

  $$
  y=y \times weight
  $$

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Input tensor to be computed. Its last dimension is evenly split into two parts for SwiGLU and must be divisible by 2. | FLOAT16, BFLOAT16, FLOAT32 | ND |
| weight | Optional input | Weight of each token, multiplied onto the SwiGLU result. The number of elements must equal the product of the dimensions of `x` except the last dimension. | FLOAT32 | ND |
| group_index | Optional input | Number of grouped tokens in count mode. | INT64 | ND |
| y | Output | SwiGLU computation result. Its data type is consistent with `x`, its last dimension is half of the last dimension of `x`, and its other dimensions are consistent with `x`. | FLOAT16, BFLOAT16, FLOAT32 | ND |
| clamp_limit | Optional attribute | Truncates A and B before activation. The default value is -1.0, and passing -1.0 means no truncation. When truncation is enabled, a value greater than 0 must be passed. | FLOAT | - |

## Constraints

- When `group_index` is passed, its data type is INT64, and it is the number of grouped tokens in count mode. The values in `group_index` must be greater than or equal to 0. The number of actually processed rows is the smaller of the sum of all elements of `group_index` and `bs` (`bs` is the product of the dimensions of `x` except the last dimension).

## Invocation Description

| Invocation method | Invocation sample | Description |
| :--- | :--- | :--- |
| aclnn invocation | [test_aclnn_swiglu_group](./examples/test_aclnn_swiglu_group.cpp) | Calls the SwigluGroup operator through the [aclnnSwigluGroup](./docs/aclnnSwigluGroup_en.md) interface. |
| Graph-mode invocation | [test_geir_swiglu_group](./examples/test_geir_swiglu_group.cpp) | Calls the SwigluGroup operator by building a graph through [operator IR](./op_graph/swiglu_group_proto.h). |
