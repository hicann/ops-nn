# Selu

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: computes the SELU (Scaled Exponential Linear Unit) activation function element-wise on the input tensor.
- Formula:

  $$
  \text{SELU}(x) = \text{scale} \times \left( \max(0, x) + \min\left(0, \alpha \times \left(e^x - 1\right)\right) \right)
  $$

  Here:

  - $\alpha = 1.6732632423543772848170429916717$
  - $\text{scale} = 1.0507009873554804934193349852946$

  Piecewise behavior:

  - When $x > 0$: $\text{SELU}(x) = \text{scale} \times x$ (linear region)
  - When $x \le 0$: $\text{SELU}(x) = \text{scale} \times \alpha \times (e^x - 1)$ (exponential saturation region)

## Parameters

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
<col style="width: 170px">
</colgroup>
<thead>
  <tr>
    <th>Parameter name</th>
    <th>Input/Output</th>
    <th>Description</th>
    <th>Data type</th>
    <th>Data format</th>
    <th>Dimensions (shape)</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x</td>
    <td>Input</td>
    <td>The input x in the formula.</td>
    <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT8</td>
    <td>ND</td>
    <td>Any dimension</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>The output y in the formula. The shape is the same as that of the input x.</td>
    <td>FLOAT, FLOAT16, BFLOAT16, INT32, INT8</td>
    <td>ND</td>
    <td>Any dimension</td>
  </tr>
</tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_selu](./examples/test_aclnn_selu.cpp) | Invokes the Selu operator through the [aclnnSelu&aclnnInplaceSelu](./docs/aclnnSelu&aclnnInplaceSelu_en.md) API. |
| aclnn invocation | [test_aclnn_inplace_selu](./examples/test_aclnn_inplace_selu.cpp) | Invokes the Selu operator through the [aclnnSelu&aclnnInplaceSelu](./docs/aclnnSelu&aclnnInplaceSelu_en.md) API. |
| GE graph mode | [test_geir_selu](./examples/test_geir_selu.cpp) | Invokes the Selu operator through graph construction using the [operator IR](./op_graph/selu_proto.h). |
