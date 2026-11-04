# QuantizeAddLayerNorm

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

- Operator function: An operation that combines quantization, addition, and layer normalization. It performs the following operations in sequence:
  - Addition: Adds the two input tensors.
  - Layer normalization: Performs layer normalization on the tensor after addition.
  - Quantization: Quantizes the input tensor from floating point to low-precision integer or floating point.

- Formula:

  $$
  x = x1 + x2 + bias
  $$

  $$
  rstd = {{1}\over\sqrt {Var(x)+eps}}
  $$

  $$
  norm_out = (x-\bar{x}) * rstd * gamma + beta
  $$

  $$
  out = round((norm_out/scales)+zero_points)
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
      <td>x1</td>
      <td>Input</td>
      <td>First input of the addition computation, corresponding to `x1` in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Second input of the addition computation, corresponding to `x2` in the formula. The shape is the same as the shape of `x1`.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Used to scale the normalized output, corresponding to `gamma` in the formula. The shape is the same as the last dimension of the shape of `x1`.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>Used to offset the normalized output, corresponding to `beta` in the formula. The shape is the same as the shape of `gamma`.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td>Offset used for the addition computation, corresponding to `bias` in the formula. The shape is the same as the shape of `gamma`.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Input</td>
      <td>Scale factor of quantization, corresponding to `scales` in the formula. The shape is the same as the shape of `gamma`.</td>
      <td>FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points1</td>
      <td>Optional input</td>
      <td>Offset tensor for obtaining y during quantization, corresponding to `zero_points1` in the formula. The shape is the same as the shape of `gamma`.</td>
      <td>FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Required attribute</td>
      <td>
      <ul><li>Specifies the data type of the quantized output tensor y.</li><li>No default value.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Optional attribute</td>
      <td><ul><li>Element-wise axis to quantize; other axes are broadcast. The specified axis cannot exceed the number of dimensions of input x. Currently only -1 is supported, and other values do not take effect.</li><li>Default value: -1.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Value added to the denominator to ensure numerical stability, corresponding to `eps` in the formula.</li><li>Default value: 1e-5.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additional_output</td>
      <td>Optional attribute</td>
      <td><ul><li>Whether to enable the output of x.</li><li>Default value: false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Final output result of QuantizeAddLayerNorm, corresponding to `out` in the formula.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Output</td>
      <td>Output result of the addition operation, corresponding to `x` in the formula. The shape is the same as the shape of `x1`.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: The data types of all inputs and outputs do not support BFLOAT16.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_quantize_add_layer_norm](examples/test_geir_quantize_add_layer_norm.cpp)  | Calls the QuantizeAddLayerNorm operator by building a graph with the [operator IR](op_graph/quantize_add_layer_norm_proto.h).         |
