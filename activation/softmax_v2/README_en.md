# SoftmaxV2

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

- Operator function: computes the Softmax value of the input tensor.

- Formula:

  $$
  Softmax\left ( {{x}_{i}} \right )= {\frac {{e}^{{x}_{i}}} {\sum ^{n-1}_{j=0} {{e}^{{x}_{j}}}}}
  $$

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
      <td>x</td>
      <td>Input</td>
      <td>The input parameter for the Softmax computation, the x in the formula.</td>
      <td>FLOAT,FLOAT16,BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axes</td>
      <td>Input</td>
      <td>The axes on which Softmax is performed. Only 1 dimension is supported. The default value is -1.</td>
      <td>INT32_ARRAY</td>
      <td>-</td>
    </tr>
    <tr>
      <td>half_to_float</td>
      <td>Input</td>
      <td>An optional boolean value. This parameter determines whether the output data type is converted to float32 when the input data type is float16. The default value is false. When the value is false, the input and output data types must be consistent. If the value is true and the input data type is float16, the output data type should be float32.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output tensor in the formula</td>
      <td>FLOAT,FLOAT16,BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API | [test_aclnn_softmax_v2](examples/test_aclnn_softmax_v2.cpp) | Invokes the SoftmaxV2 operator through the [aclnnSoftmax](docs/aclnnSoftmax_en.md) API. |
