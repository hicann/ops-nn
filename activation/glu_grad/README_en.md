# GLUGrad

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

- Description: Backward gradient computation of GLU. It evenly splits the input tensor self along the specified dimension dim into two tensors a and b, and computes the gradient based on gradOut.
- Formula:

  $$
  \frac{\partial GLU(a,b)}{\partial(a,b)}=cat(\sigma(b),\sigma(b) \otimes a \otimes (1-\sigma(b)))
  $$

  Mathematical computation expressions:
  Assume the GLUGrad output consists of two parts: out=[a_grad, b_grad]. Then:
  - sig_b=sigmoid(b)
  - **a_grad**=grad_out*sig_b
  - **b_grad**=a_grad*(a - a*sig_b)

  Here, grad_out is the gradient output, a is the first half of the input tensor self after even splitting along the specified dim, and b is the second half.

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
      <td>grad_out</td>
      <td>Input</td>
      <td>Gradient output tensor.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Input tensor of forward propagation.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Attribute</td>
      <td>Dimension along which to split input self.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Gradient output, with the same shape as self.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_glu_backward](examples/test_aclnn_glu_backward.cpp) | Calls the GluGrad operator through the [aclnnGluBackward](docs/aclnnGluBackward_en.md) interface. |
