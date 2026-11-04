# InplaceAddLayerNorm

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: an algorithm that combines in-place addition and layer normalization. The output parameters `x1` (corresponding to `y` in the formula) and `x2` (corresponding to `x` in the formula) reuse the input addresses of the input parameters `x1` and `x2`, implementing the InplaceAddLayerNorm function.
- Computation formula:

  $$
  x = x1 + x2 + biasOptional
  $$

  $$
  rstd = {{1}\over\sqrt {Var(x)+eps}}
  $$

  $$
  y = (x-\bar{x}) * rstd * \gamma + \beta
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
      <td>Denotes the input of the addition computation in AddLayerNorm, corresponding to `x1` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Denotes the input of the addition computation in AddLayerNorm, corresponding to `x2` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Denotes the `gamma` parameter in layer normalization, corresponding to `γ` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>Denotes the `beta` parameter in layer normalization, corresponding to `β` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Optional input</td>
      <td>Denotes the input of the addition computation in AddLayerNorm, corresponding to `biasOptional` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Value added to the denominator to ensure numerical stability, corresponding to eps in the formula.</li><li>The default value is 1e-5.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additional_output</td>
      <td>Optional attribute</td>
      <td><ul><li>Denotes whether to enable the output of x=x1+x2+biasOptional.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x1</td>
      <td>Output</td>
      <td>Denotes the result output of LayerNorm, corresponding to `y` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Output</td>
      <td>Outputs the mean of the result of (x1 + x2 + biasOptional) during the LayerNorm computation, corresponding to the mean of x in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Output</td>
      <td>Outputs the result of `rstd` during the LayerNorm computation, corresponding to `rstd` in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Output</td>
      <td>Denotes the result output of Add, corresponding to `x` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas inference products</term> and <term>Atlas training products</term>:
    - The data type BFLOAT16 is not supported for all input parameters and the output parameters `x1` and `x2`.
    - The tail-axis length of the five inputs x1, x2, beta, gamma, and bias must be greater than or equal to 32 Bytes.

  - <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: the data type BFLOAT16 is not supported for all inputs and outputs.

## Constraints

- Empty tensor support: <term>Ascend 950PR&950DT products</term> support empty input and empty output.
- Non-contiguous tensor support: the inputs and outputs do not support non-contiguous tensors.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | [test_geir_inplace_add_layer_norm](examples/test_geir_inplace_add_layer_norm.cpp)  | Calls the InplaceAddLayerNorm operator by constructing a graph through the [operator IR](op_graph/inplace_add_layer_norm_proto.h).         |
