# ThresholdGradV2D

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

- Operator function: Performs the backpropagation gradient computation of the forward threshold operation. When threshold==0, it is equivalent to ReluGrad.

- Formula:

  $$
      out_i =
      \begin{cases}
      gradOutput_i, \quad self_i > threshold\\
      0,  \quad self_i \leq threshold
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
      <td>gradOutput</td>
      <td>Input</td>
      <td>Upstream gradient, broadcast-compatible with self.</td>
      <td>FLOAT16, FLOAT, BF16, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Forward input, which determines the gating mask.</td>
      <td>FLOAT16, FLOAT, BF16, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>threshold</td>
      <td>Attribute</td>
      <td>Threshold scalar, required. When threshold==0, it is equivalent to ReluGrad.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Backward gradient. The dtype is the same as self, and the shape is the shape after broadcasting.</td>
      <td>FLOAT16, FLOAT, BF16, INT32, INT8, UINT8</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- The dtypes of gradOutput, self, and out must be the same.

## Invocation Description

| Invocation method   | Invocation sample                                                | Description                            |
| :--------- | :------------------------------------------------------ | :------------------------------ |
| aclnn invocation | [test_aclnn_threshold_backward.cpp](./examples/test_aclnn_threshold_backward.cpp) | Invoke the threshold_grad_v2_d operator through the [aclnnThresholdBackward](./docs/aclnnThresholdBackward_en.md) API. |
