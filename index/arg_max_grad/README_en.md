# ArgMaxGrad

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    √     |

## Function

- Operator function: the backward of ArgMax. Along the axis specified by `dimension`, it writes `updates` into the positions specified by `indices`, and keeps the original values of `var` at the other positions.

- Formula:

  Assume that the shape of $var$ is $(d_0, ..., d_{m-1})$, $a$ is the normalized `dimension`, and $D=d_a$. The shapes of $indices$ and $updates$ are the shape of $var$ with dimension $a$ removed (the formula below describes broadcasting with that dimension of length 1), and $k$ is the index of the element on dimension $a$. Then:

  $$
  y[i_0,...,k,...,i_{m-1}] = \begin{cases} updates[i_0,...,0,...,i_{m-1}], & k = indices[i_0,...,0,...,i_{m-1}] \\ var[i_0,...,k,...,i_{m-1}], & \text{otherwise} \end{cases}
  $$

  Here:

  - When `dimension` is negative, it is normalized as $dimension + m$.
  - When the value of `indices` is not within the range $[0, D)$, no position on the whole axis hits, and the output keeps the original value of `var`.
  - This operator performs only conditional selection and involves no arithmetic operations; the values of `var` and `updates` are copied bit by bit as is.

## Parameters

| Parameter name | Input/Output/Attribute | Description                                                         | Data type        | Data format |
| ------ | -------------- | ------------------------------------------------------------ | --------------- | -------- |
| var    | Input           | Base tensor to be written. The output `y` has the same shape and data type as it.                          | FLOAT16, FLOAT, INT32, INT8 | ND |
| indices | Input          | Target indices along the `dimension` axis, with the shape of `var` with the `dimension` axis removed (when `rank(var)==1`, $(1,)$). | INT32     | ND |
| updates | Input          | Values to write, with the same shape as `indices` and the same data type as `var`.         | FLOAT16, FLOAT, INT32, INT8 | ND |
| y      | Output           | Hit positions take `updates`, and the others keep `var`; the shape and data type are both the same as those of `var`. | FLOAT16, FLOAT, INT32, INT8 | ND |
| dimension | Attribute        | Required. Specifies the axis along which to compare, with the value range $[-m, m-1]$, where $m$ is the number of dimensions of `var`. | INT       | -    |

## Constraints

- Let $m$ be the rank of `var` and $a$ the normalized `dimension`. When $m \ge 2$, both `indices` and `updates` have rank $m-1$, and their shapes are $(d_0,\ ...,\ d_{a-1},\ d_{a+1},\ ...,\ d_{m-1})$; when $m=1$, both shapes are $(1,)$.
- The shapes of `indices` and `updates` must be exactly the same.
- Shapes that keep the `dimension` axis (length 1, e.g. $[2,3,1,5]$) or have that axis longer than 1 (e.g. $[2,3,2,5]$) are not supported.
- The data type of `updates` must be consistent with that of `var`; the data type of `indices` is fixed to INT32.
- The rank of `var` must be at least 1, with no upper bound (rank 0 is rejected).
- The selected axis length $D=var.shape[dimension]$ must not exceed INT32_MAX (axis indices and `indices` are represented as int32).
- The indices of elements on the `dimension` axis are generated inside the operator based on `dimension` and the shape of `var`; users do not need to construct them, and they are not passed in as inputs.

## Invocation Description

| Invocation method | Sample code | Description |
| -------- | -------- | ---- |
| Graph-mode invocation | [test_geir_arg_max_grad.cpp](examples/test_geir_arg_max_grad.cpp) | Calls the ArgMaxGrad operator by constructing a graph through the [operator IR](./op_graph/arg_max_grad_proto.h). For details, see [Operator Invocation](../../docs/zh/invocation/quick_op_invocation.md). |
