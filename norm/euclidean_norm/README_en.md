# EuclideanNorm

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function:

  Computes the Euclidean norm (L2 norm) of the input tensor along the given axes. It squares the tensor element-wise first, sums along the specified axes, and finally takes the square root. It is equivalent to `tf.math.reduce_euclidean_norm` in TensorFlow.

- Formula:

  $$
  y = \sqrt{ \sum_{i \in axes} x_i^{2} }
  $$

  Where $axes$ is the set of dimensions to reduce. When the attribute `keep_dims = true` is set, the reduced dimensions are kept as length 1 in the output; otherwise they are removed from the output shape.

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
      <td>Represents the input tensor for computing the Euclidean norm, x in the formula. Any number of dimensions is supported, with shape [D0, D1, ..., Dn-1].</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axes</td>
      <td>Input</td>
      <td><ul><li>Represents the dimensions to reduce, axes in the formula. It is a 1-D tensor whose element values are in the range [-rank(x), rank(x)), and duplicates are not allowed.</li><li>This input is value-dependent: both infershape and tiling depend on its specific values, so it must be injected as a Const operator in graph mode.</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keep_dims</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates whether to keep the reduced dimensions in the output. When it is true, the reduced dimensions are kept as length 1; when it is false, they are removed from the output shape.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the computation result of the Euclidean norm, y in the formula. Its data type is the same as x; its shape is jointly determined by x.shape, axes, and keep_dims.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- axes is a value-dependent input. In graph-mode invocation it must be injected with `op::Const`; injecting it at runtime through a placeholder Data operator makes it impossible to infer the output shape and tiling at compilation time.
- The elements of axes must fall in the range [-rank(x), rank(x)), and duplicates are not allowed.
- The data type of y is the same as x. When x is INT32, the output is still INT32 (no implicit type promotion), and the caller must ensure that the intermediate accumulation does not overflow.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_euclidean_norm](./examples/arch35/test_geir_euclidean_norm.cpp)   | Invokes the EuclideanNorm operator through the operator IR-based [graph construction](./op_graph/euclidean_norm_proto.h) method. |
