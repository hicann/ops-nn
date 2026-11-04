# ForeachMaximumScalarList

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function

- Description: Performs element-wise comparison between the tensor list x and the scalar list scalars, and computes the maximum value corresponding to each element.

- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  scalars = [{scalars_0}, {scalars_1}, ... {scalars_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = \max(x_i, \text{scalars}_i) (i=0,1,...n-1)
  $$

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
      <td>x</td>
      <td>Input</td>
      <td>Denotes the input tensor list for the maximum value computation, corresponding to `x` in the formula. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalars</td>
      <td>Input</td>
      <td>Denotes the input scalar list for the maximum value computation, corresponding to `scalars` in the formula. The number of elements equals the number of tensors in `x`. The data types are paired according to the dtype of `x`: when `x` is FLOAT32, FLOAT16, or BFLOAT16, `scalars` supports FLOAT32; when `x` is INT32, `scalars` supports INT64. The aclnn interface additionally supports DOUBLE (`x` is of a floating-point type, which is internally converted to FLOAT32 before participating in the computation) and INT32 (`x` is INT32, which is internally converted to INT64 before participating in the computation).</td>
      <td>FLOAT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Denotes the output tensor list for the maximum value computation, corresponding to `y` in the formula. The data type and data format are consistent with those of the input parameter `x`, and the shape size is greater than or equal to the shape size of the input parameter `x`. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: the data types of x and y do not support BFLOAT16.

## Constraints

- The maximum number of tensors contained in the input `x` and the output `y` is 256 for each, and the TensorList length of `y` must equal the TensorList length of `x`.
- The shape size of the output `y` must be greater than or equal to the shape size of the tensor at the corresponding position in `x`.
- The ScalarList length of the input `scalars` must equal the TensorList length of `x`, and the maximum number of supported elements is 256.
- `scalars` is constrained only by the number of elements (ScalarList length) and the dtype, and its tensor rank is not limited. The kernel reads the i-th element by the flattened index.
- Empty TensorLists are supported: when the TensorList lengths of both `x` and `y` are 0, the aclnn interface and kernel/graph-mode direct invocation directly return success (no-op).
- Non-contiguous tensor support: the aclnn interface supports non-contiguous input `x` and output `y` (non-contiguous inputs are first converted to contiguous tensors to participate in the computation, and non-contiguous outputs are written back with the contiguous results). In graph mode, non-contiguous tensors are handled by the operator prototype `AutoContiguous` at the framework layer. Kernel direct invocation supports only contiguous tensors.
- The DOUBLE data type of the input `scalars` is supported only in the aclnn interface path (internally converted to FLOAT32 to participate in the computation). When `x` is INT32, the INT32 data type of `scalars` is also supported only in the aclnn interface path (internally converted to INT64 to participate in the computation). In graph mode (operator IR), `scalars` supports only FLOAT32 and INT64.
- The preceding constraints on length, dtype, shape, and so on return ACLNN_ERR_PARAM_INVALID in the aclnn interface path. In non-aclnn paths such as graph mode and operator direct invocation, they are uniformly validated by tiling, and a failure returns a tiling error (corresponding to the ACLNN_ERR_INNER_TILING_ERROR semantics).

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_foreach_maximum_scalar_list](examples/test_aclnn_foreach_maximum_scalar_list.cpp) | Invokes the ForeachMaximumScalarList operator through the [aclnnForeachMaximumScalarList](docs/aclnnForeachMaximumScalarList_en.md) interface. |
| Graph mode | -  | Invokes the ForeachMaximumScalarList operator by building a graph through the [operator IR](op_graph/foreach_maximum_scalar_list_proto.h).         |
