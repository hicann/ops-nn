# AddLayerNorm

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

- Operator function: implements the AddLayerNorm function.
- Formula:

  $$
  x = x1 + x2 + bias
  $$

  $$
  rstd = {{1}\over\sqrt {Var(x)+epsilon}}
  $$

  $$
  y = (x-\bar{x}) * rstd * \gamma + \beta
  $$

  Here:

  - $\bar{x}$:

    $$
    \operatorname{\bar{x}}=\frac{1}{D}\sum_{1}^{D}{x_i}
    $$

    D is the number of elements in x that participate in the mean computation.

  - $Var(x)$:
    $$
    Var(x) = E(x-{\bar{x}})^2
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
      <td>Input of the addition computation in AddLayerNorm, corresponding to `x1` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input of the addition computation in AddLayerNorm, corresponding to `x2` in the formula. The shape must be consistent with `x1`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Denotes the `gamma` parameter in layer normalization, corresponding to `γ` in the formula. The shape is the same as the norm dimension value of `x1`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>Denotes the `beta` parameter in layer normalization, corresponding to `β` in the formula. The shape is the same as the norm dimension value of `x1`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Optional input</td>
      <td>Input of the addition computation in AddLayerNorm, corresponding to `bias` in the formula. The shape can be consistent with `gamma`/`beta` or with `x1`/`x2`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Value added to the denominator to ensure numerical stability, corresponding to `epsilon` in the formula.</li><li>The default value is 1e-5.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additional_output</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates whether to enable the output of x=x1+x2+bias.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Denotes the result output `y` of LayerNorm, corresponding to y in the formula. The shape must be consistent with input `x1`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Output</td>
      <td>Outputs the mean of the result of (x1 + x2 + bias) computed during LayerNorm, corresponding to the average of `x` in the formula. The shape must satisfy a broadcast relationship with `x1` (the first few dimensions are the same as the first few dimensions of `x1`, the later dimensions are 1, the total number of dimensions equals the number of dimensions of `x1`, and the first few dimensions refer to the number of dimensions of `x1` minus the number of dimensions of `gamma`, denoting the dimensions that do not require normalization).</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Output</td>
      <td>Outputs the result of `rstd` computed during LayerNorm, corresponding to `rstd` in the formula. The shape must satisfy a broadcast relationship with `x1` (the first few dimensions are the same as the first few dimensions of `x1`, the later dimensions are 1, the total number of dimensions equals the number of dimensions of `x1`, and the first few dimensions refer to the number of dimensions of `x1` minus the number of dimensions of gamma, denoting the dimensions that do not require normalization).</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Output</td>
      <td>Denotes the result output `x` of LayerNorm, corresponding to `x` in the formula. The shape must be consistent with input `x1`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas inference products</term>, <term>Atlas training products</term>:
  - The data types of all input parameters and of output parameters `y` and `x` do not support BFLOAT16.
  - In the usage scenario under the current products, the output parameters `mean` and `rstd` are invalid, and the output values do not take effect.
  - The tail axis lengths of the five inputs x1, x2, beta, gamma, and bias must be greater than or equal to 32 Bytes.

- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- Empty tensor support: empty input and empty output are not supported.
- Non-contiguous tensor support: inputs support non-contiguous tensors, and outputs do not support non-contiguous tensors.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_add_layer_norm](examples/test_aclnn_add_layer_norm.cpp) | Invokes the AddLayerNorm operator through the [aclnnAddLayerNorm](docs/aclnnAddLayerNorm_en.md) interface. |
| Graph-mode invocation | -  | Invokes the AddLayerNorm operator through graph construction with the [operator IR](op_graph/add_layer_norm_proto.h).         |
