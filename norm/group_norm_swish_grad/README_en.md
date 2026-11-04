# GroupNormSwishGrad

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: Backward operation of [aclnnGroupNormSwish](../../norm/group_norm_swish/docs/aclnnGroupNormSwish_en.md).

- Formula:

$$
dYTemp_i = \hat{x_i} \cdot gamma + beta
$$

$$
dSwishTemp_i = swishScale \cdot dYTemp_i - \frac{swishScale \cdot dYTemp_i}{\exp(-swishScale \cdot dYTemp_i) + 1} + 1
$$

$$
dYNew_i = \frac{dSwishTemp_i}{\exp(-swishScale \cdot dYTemp_i) + 1} \cdot dy
$$

$$
dBeta = \sum_{i=1}^n dYNew_i
$$

$$
dGamma = \sum_{i=1}^n (dYNew_i \cdot \hat{x_i})
$$

$$
dx = rstd \cdot (dYNew * gamma - \hat{x} * (\sum_{i=1}^n gamma_i * dGamma) - (\sum_{i=1}^n gamma_i * dBeta))
$$

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>dy</td>
      <td>Input</td>
      <td>Gradient for backward computation.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Second output of forward computation, indicating the mean of each group after input is grouped.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Input</td>
      <td>Third output of forward computation, indicating the reciprocal of the standard deviation of each group after input is grouped.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Forward input x.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Scaling coefficient of each channel.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>Offset coefficient of each channel.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numGroups</td>
      <td>Attribute</td>
      <td>Indicates that the C dimension of input gradOut is divided into group groups.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dataFormatOptional</td>
      <td>Attribute</td>
      <td>Data format.</td>
      <td>Char*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>swishScale</td>
      <td>Attribute</td>
      <td>Coefficient in the Swish computation formula.</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dgammaIsRequire</td>
      <td>Attribute</td>
      <td>Indicates whether dgamma needs to be output.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dbetaIsRequire</td>
      <td>Attribute</td>
      <td>Indicates whether dbeta needs to be output.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dxOut</td>
      <td>Output</td>
      <td>Output gradient of x.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dgammaOut</td>
      <td>Output</td>
      <td>Output gradient of gamma.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dbetaOut</td>
      <td>Output</td>
      <td>Output gradient of beta.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_group_norm_swish_grad.cpp](examples/test_aclnn_group_norm_swish_grad.cpp) | Invoke the GroupNormSwishGrad operator through the [aclnnGroupNormSwishGrad](docs/aclnnGroupNormSwishGrad_en.md) API. |
| Graph mode | -  | Invoke the GroupNormSwishGrad operator through graph construction using the [operator IR](op_graph/group_norm_swish_grad_proto.h).         |
