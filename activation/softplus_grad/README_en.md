# SoftplusGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

Description: Backpropagation of the Softplus activation function, aligned with TensorFlow `raw_ops.SoftplusGrad`.

Formula:

$$
backprops = gradients \times sigmoid(features) = \frac{gradients}{1 + exp(-features)}
$$

NumPy-style broadcast is supported: gradients and features can have different shapes, and the output shape is the broadcast result.

## Parameters

  <table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
  <col style="width: 120px">
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
      <td>gradients</td>
      <td>Input</td>
      <td>Gradient.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>features</td>
      <td>Input</td>
      <td>Input tensor.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>backprops</td>
      <td>Output</td>
      <td>Backpropagation result.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas inference products</term> and <term>Atlas training products</term>: The supported data types are FLOAT and FLOAT16.

## Constraints

- The data types of gradients, features, and backprops must be the same.
- This delivery supports only the data types FLOAT, FLOAT16, and BFLOAT16, and does not support DOUBLE (a DOUBLE input reports an error in the tiling phase of operator compilation. The IR type declaration is the cross-backend full set aligned with the built-in CANN registration, including DOUBLE).
- gradients and features support NumPy-style broadcast.
- Inputs of 0 to 8 dimensions are supported.
- Dynamic shapes are supported: the input shape can contain -1 unknown dimensions. During deduction, they are merged with the specific dimensions of the other input. When both inputs are -1, the output remains -1.
- Unknown rank [-2] is supported: when either input has an unknown rank, the output shape is conservatively deduced as [-2].
- The sigmoid computation uses positive and negative branches to prevent intermediate exponential overflow caused by large negative features, and no additional numerical lower bound is applied. NaN and Inf follow the IEEE floating-point operation rules. For example, Inf multiplied by the computation result 0 produces NaN.

## Invocation Description

| Invocation method   | Sample code | Description |
| ---------------- | -------- | ---- |
| GE graph mode  | [test_geir_softplus_grad.cpp](examples/arch35/test_geir_softplus_grad.cpp) | Calls the SoftplusGrad operator by building a graph through the GE IR. For the operator IR definition, see [softplus_grad_proto.h](op_graph/softplus_grad_proto.h). |
