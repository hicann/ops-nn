# BroadcastGradientArgs

English | [简体中文](./README.md)


## Applicable Products

| Product | Support |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×   |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function:

  BroadcastGradientArgs is an operator in TensorFlow used to compute the reduction indices required for gradient propagation.

  Its core function: during backpropagation, based on the original shapes of the two tensors in the forward pass, the operator automatically identifies the dimensions that were expanded by the broadcast mechanism, and outputs the dimensions on which the gradient needs to be reduced, so that the gradient can be restored from the broadcast shape to the shape of each original tensor.

- Example:

  ```text
  Example 1:
    The shape of the original tensor a is [2, 1, 4, 1, 6]
    The shape of the original tensor b is [2, 3, 1, 5, 1]
    x1_data: [2, 1, 4, 1, 6]   # x1_shape=[5]
    x2_data: [2, 3, 1, 5, 1]   # x2_shape=[5]

    y1_data: [1, 3]            # y1_shape=[2]
    y2_data: [2, 4]            # y2_shape=[2]

  Example 2:
    The shape of the original tensor a is [4, 1, 6]
    The shape of the original tensor b is [2, 3, 1, 5, 1]
    x1_data: [4, 1, 6]         # x1_shape=[3]
    x2_data: [2, 3, 1, 5, 1]   # x2_shape=[5]

    y1_data: [0, 1, 3]         # y1_shape=[3]
    y2_data: [2, 4]            # y2_shape=[2]

  Example 3:
    The shape of the original tensor a is [2, 1, 4, 1, 6]
    The shape of the original tensor b is [2, 1, 4, 1, 6]
    x1_data: [2, 1, 4, 1, 6]   # x1_shape=[5]
    x2_data: [2, 1, 4, 1, 6]   # x2_shape=[5]

    y1_data: []            # y1_shape=[0]
    y2_data: []            # y2_shape=[0]
  ```

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>The shape must be 1-dimensional, and the data is the shape of the original tensor a in the example.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>The shape must be 1-dimensional, and the data is the shape of the original tensor b in the example; the data type must be the same as that of x1.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td>The shape must be 1-dimensional, denoting the dimension indices that need reduction in the tensor shape corresponding to x1; the data type must be the same as that of x1.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td>The shape must be 1-dimensional, denoting the dimension indices that need reduction in the tensor shape corresponding to x2; the data type must be the same as that of x1.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Only the Ascend 950PR&950DT products are supported; other architectures are not supported.
- The inputs x1 and x2 must satisfy the broadcast rule: the corresponding dimensions must be equal, or at least one of them must be 1; otherwise, the operator reports an error.
- y1 and y2 are dynamic shape outputs, and the output memory must be preallocated by max(length of x1, length of x2); the actual number of output elements may be smaller than this value.
- The data types of x1 and x2 must be consistent, and the data types of y1 and y2 must be consistent with x1; only INT32 and INT64 are supported.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_broadcast_gradient_args](./examples/test_geir_broadcast_gradient_args.cpp)   | Calls the BroadcastGradientArgs operator by building a graph through [operator IR](./op_graph/broadcast_gradient_args_proto.h). |
| aclnn invocation | [test_aclnn_broadcast_gradient_args](./examples/test_aclnn_broadcast_gradient_args.cpp)   | Calls the BroadcastGradientArgs operator through the [aclnnBroadcastGradientArgs](./docs/aclnnBroadcastGradientArgs_en.md) interface. |
