# SiluGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Description: Backward propagation of [aclnnSilu](../swish/docs/aclnnSilu_en.md). It computes the gradient input of silu based on the silu backward propagation gradient and the forward output.
- Formula:

  $$
  \sigma(x) = {\frac{1} {1+{e}^{-x}}}
  $$

  $$
  s(x) = x\sigma(x)
  $$

  $$
  s^\prime(x) = \sigma(x)(1+x-x\sigma(x))
  $$

  $$
  gradInput = gradOutput * s^\prime(x)
  $$

Here, $\sigma(x)$ is the sigmoid function, $s(x)$ is the silu function, and $s^\prime(x)$ is the derivative of the silu function.

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
      <td>gradOutput</td>
      <td>Input</td>
      <td>Denotes the input gradient. It is gradOutput in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Denotes the input data. It is x in the formula and corresponds to the forward input parameter.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>gradInput</td>
      <td>Output</td>
      <td>Denotes the gradient computed for the input data self. It is gradInput in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
   </tbody>
  </table>

 - <term>Atlas training products</term>: the data types support FLOAT16 and FLOAT.

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                                                |
| ---------------- | --------------------------- |-------------------------------------------------------------------|
| aclnn interface  | [test_aclnn_silu_grad.cpp](examples/test_aclnn_silu_grad.cpp) | Invokes the SiluGrad operator through the [aclnnSiluBackward](docs/aclnnSiluBackward_en.md) interface. |
