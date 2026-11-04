# SoftmaxCrossEntropyWithLogits

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

- Description: computes the cross entropy loss of softmax and cross entropy, and provides the backward gradient of the input logits.
- Formula:

  $$
  \text{loss} = - \sum_i y_i \cdot \log(\text{softmax}(x_i))
  $$

$$
\text{backprop} = \text{softmax}(x_i) - y_i
$$

Here $x_i$ corresponds to the input features, and $y_i$ corresponds to the input labels.

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
      <th>Input/Output</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
      <tr>
      <td>features</td>
      <td>Input</td>
      <td>Output logits of the model, that is, x_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>labels</td>
      <td>Input</td>
      <td>Input labels, that is, y_i in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>loss</td>
      <td>Output</td>
      <td>Cross entropy loss of softmax and cross entropy, that is, loss in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>backprop</td>
      <td>Output</td>
      <td>Backward gradient of the input logits, that is, backprop in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraints

None

## Invocation Description

  | Invocation method   | Sample code           | Description                                         |
  | ---------------- | --------------------------- | --------------------------------------------------- |
  | aclnn interface | [test_aclnn_softmax_cross_entropy_with_logits](examples/test_aclnn_softmax_cross_entropy_with_logits.cpp) | Calls the SoftmaxCrossEntropyWithLogits operator through the [aclnnSoftmaxCrossEntropyWithLogits](docs/aclnnSoftmaxCrossEntropyWithLogits_en.md) interface. |
  | Graph mode | - | Calls the SoftmaxCrossEntropyWithLogits operator by constructing a graph through the [operator IR](op_graph/softmax_cross_entropy_with_logits_proto.h). |
