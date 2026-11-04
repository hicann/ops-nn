# BinaryCrossEntropy

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Computes the binary cross entropy of x and y.
- Formula:

  When reduction is None:

  $$
  \ell(x, y)= L = \{l_{1},...,l_{n}\}^{T}, \ell_{n} = -  weight_{n}[y_{n}·log(x_{n}) + (1 - y_{n}·log(1-x_{n}))]
  $$

  When reduction is not None:

  $$
  \ell(x, y)
  \begin{cases}
  mean(L), & if\ reduction = mean \\
  sum(L), & if\ reduction = sum \\
  \end{cases}
  $$

## Parameters

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">Parameter name</th>
      <th style="white-space: nowrap">Input/Output/Attribute</th>
      <th style="white-space: nowrap">Description</th>
      <th style="white-space: nowrap">Data type</th>
      <th style="white-space: nowrap">Data format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Represents the predicted probability values, input `x` in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Input</td>
      <td>Represents the target tensor, input `y` in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Represents the weight tensor, input `weight` in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>Represents the computation output</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Attribute</td>
      <td>Represents the reduction method, input `reduction` in the formula, the enumeration value of output reduction.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>

  </tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_binary_cross_entropy.cpp](examples/test_aclnn_binary_cross_entropy.cpp) | Calls the BinaryCrossEntropy operator through the [aclnnBinaryCrossEntropy](docs/aclnnBinaryCrossEntropy_en.md) interface. |
