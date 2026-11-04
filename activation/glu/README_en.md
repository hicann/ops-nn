# Glu

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Description: GLU is a gated linear unit function. It evenly splits the input tensor along the specified dimension dim into two tensors, and elementwise multiplies the front-half tensor by the result of the Sigmoid function on the back-half tensor.
- Formula:

  $$
  GLU(a,b)=a \otimes \sigma(b)
  $$

  a denotes the front-half tensor after the input tensor is evenly split along the specified dim, and b denotes the back-half tensor.

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
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
      <td>x</td>
      <td>Input</td>
      <td>Inputs a tensor.</td>
      <td>DT_BF16, FLOAT16, DT_FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Input parameter for the GLU computation. The input tensor is evenly split along the specified dimension dim into the two tensors a and b in the formula.</td>
      <td>DOUBLE, FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>dim</td>
      <td>Input</td>
      <td>Denotes the dimension along which the input self is split.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
     <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output parameter of the computation.</td>
      <td>DOUBLE, FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_glu](examples/test_aclnn_glu.cpp) | Calls the GLU operator through the [aclnnGlu](docs/aclnnGlu_en.md) interface. |
