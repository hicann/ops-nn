# ConfusionSoftmaxGrad

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

- Operator function: Fused operator, which fuses mul, sum, and sub.

- Formula:

  $$
  y = grad - sum(grad \cdot x, axis=-1)
  $$

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 813px"><colgroup>
  <col style="width: 103px">
  <col style="width: 144px">
  <col style="width: 194px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>grad</td>
      <td>Input</td>
      <td>Gradient value of backpropagation, that is, the output gradient of the upper layer. It is grad in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>x in the formula. The data type and shape are the same as those of grad.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>y in the formula. The data type and shape are the same as those of grad.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    </tbody>
     </table>
- <term>Atlas inference products</term>, <term>Atlas training products</term>: The data types support FLOAT16 and FLOAT32.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_confusion_softmax_grad](./examples/test_geir_confusion_softmax_grad.cpp)   | Calls the ConfusionSoftmaxGrad operator through graph construction with the [operator IR](./op_graph/confusion_softmax_grad_proto.h). |
