# Gelu

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
|  <term>KirinX90 processor series products</term>     |     √    |
|  <term>Kirin9030 processor series products</term>    |     √    |

## Function

- Operator function: Gaussian error linear unit activation function.

- Formula:

  $$
  out=GELU(self)=self × Φ(self)=0.5 * self * (1 + tanh( \sqrt{2 / \pi} * (self + 0.044715 * self^{3})))
  $$

## Parameters

  <table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 250px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage instructions</th>
      <th>Data type</th>
      <th>Data format</th>
      <th>Dimensions (shape)</th>
      <th>Non-contiguous tensor</th>
    </tr></thead>
  <tbody>
      <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input of the GELU activation function, self in the formula.</td>
      <td><ul><li>The data type must be the same as out.</li><li>The shape must be the same as out.</li><li>Empty tensors are supported.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output of the GELU activation function.</td>
      <td><ul><li>The data type must be consistent with self.</li><li>The shape must be consistent with self.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
   </tr>
  </tbody>
  </table>

  - <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, and <term>Atlas training products</term>: The supported data types are FLOAT and FLOAT16.
  - <term>KirinX90 processor series products</term>, <term>Kirin9030 processor series products</term>: The supported data types are FLOAT and FLOAT16.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code                                                                         | Description                                                                                 |
| ---------------- |------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| aclnn invocation  | [test_aclnn_gelu.cpp](examples/test_aclnn_gelu.cpp) | Calls the Gelu operator through the [aclnnGelu](docs/aclnnGelu_en.md) interface. |
