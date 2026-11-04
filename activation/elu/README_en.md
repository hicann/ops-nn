# Elu

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

- Description: Calls the exponential linear unit activation function ELU on each element x in the input tensor self, and stores the obtained result in the output tensor out.
- Formula:

  $$
  ELU(x) =
  \begin{cases}
  scale \ast x, \quad x > 0\\
  \alpha \ast scale \ast (exp(x \ast inputScale)-1), \quad x \leq 0
  \end{cases}
  $$

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1300px"><colgroup>
  <col style="width: 101px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 200px">
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
      <td>self</td>
      <td>Input</td>
      <td>Represents the input of the ELU activation function; it is x in the formula.</td>
      <td>Empty tensors are supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>Represents the activation coefficient of the ELU activation function; it is α in the formula.</td>
      <td>The data type must be a data type that can be converted to FLOAT.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Represents the scaling coefficient of the ELU activation function; it is scale in the formula.</td>
      <td>The data type must be a data type that can be converted to FLOAT.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>inputScale</td>
      <td>Input</td>
      <td>Represents the scaling coefficient of the input of the ELU activation function; it is inputScale in the formula.</td>
      <td>The data type must be a data type that can be converted to FLOAT.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Represents the output of the ELU activation function.</td>
      <td>The shape must be consistent with that of self.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas training products</term>: The supported data types are FLOAT and FLOAT16.

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_elu](./examples/test_aclnn_elu.cpp) | Invokes the Elu operator through the [aclnnElu&aclnnInplaceElu](./docs/aclnnElu&aclnnInplaceElu_en.md) interface. |
| aclnn invocation | [test_aclnn_inplace_elu](./examples/test_aclnn_inplace_elu.cpp) | Invokes the Elu operator through the [aclnnElu&aclnnInplaceElu](./docs/aclnnElu&aclnnInplaceElu_en.md) interface. |
