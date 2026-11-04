# ThresholdGradV2

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: completes the backward propagation gradient computation of the threshold forward operation.

- Computation formula:

  $$
      out_i =
      \begin{cases}
      gradients_i, \quad features_i > threshold_i\\
      0,  \quad features_i \leq threshold_i
      \end{cases}
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1430px">
  <colgroup>
    <col style="width: 127px">
    <col style="width: 120px">
    <col style="width: 273px">
    <col style="width: 292px">
    <col style="width: 152px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>gradients</td>
      <td>Input</td>
      <td>Upstream gradient, broadcast-compatible with features and threshold.</td>
      <td>FLOAT16, FLOAT, BF16, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>features</td>
      <td>Input</td>
      <td>Forward input, which determines the gating mask.</td>
      <td>FLOAT16, FLOAT, BF16, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>threshold</td>
      <td>Input</td>
      <td>Threshold tensor, broadcast-compatible with gradients and features.</td>
      <td>FLOAT16, FLOAT, BF16, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprops</td>
      <td>Output</td>
      <td>Backward gradient. Its dtype is consistent with gradients, and its shape is the shape after broadcasting the three.</td>
      <td>FLOAT16, FLOAT, BF16, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- The dtypes of gradients, features, threshold, and backprops are consistent.
- The shapes of gradients, features, and threshold must satisfy the broadcast relationship, and the shape of backprops is the shape after broadcasting.

## Invocation Description

| Invocation method   | Invocation sample                                                | Description                            |
| :--------- | :------------------------------------------------------ | :------------------------------ |
| GEIR invocation | [test_geir_threshold_grad_v2.cpp](./examples/test_geir_threshold_grad_v2.cpp) | Calls the threshold_grad_v2 operator in the form of a GE IR graph. |
