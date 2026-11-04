# MaxPool3D

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    x     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description:
Provides the 3D max pooling operation for the input channels of the input signal.
- Formula:
  - Calculation formula of each element in the output tensor:

    $$
    out(N_i, C_j, d, h, w) = \max\limits_{{k\in[0,k_{D}-1],m\in[0,k_{H}-1],n\in[0,k_{W}-1]}}input(N_i,C_j,stride[0]\times d + k, stride[1]\times h + m, stride[2]\times w + n)
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
      <td>x</td>
      <td>Input</td>
      <td>Input tensor.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Attribute</td>
      <td>Window size of the max pooling.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attribute</td>
      <td>Stride of the window movement.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Attribute</td>
      <td>Specifies the padding mode.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>Attribute</td>
      <td>Number of layers padded on each edge.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilations</td>
      <td>Attribute</td>
      <td>Controls the stride of the elements in the window.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceilMode</td>
      <td>Attribute</td>
      <td>Rounding mode for computing the output shape.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_fromat</td>
      <td>Attribute</td>
      <td>Supported data format</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- **Value range restrictions:**
  - ksize: The array length must be 5, and the values corresponding to the N and C dimensions must be 1.
  - strides: The array length must be 5, and the values corresponding to the N and C dimensions must be 1.
  - padding: Only three modes are supported: "SAME", "VALID", and "CALCULATED".
  - pads: This parameter takes effect only when the padding mode is "CALCULATED".
  - dilations: The array length must be 5, and the values corresponding to the N and C dimensions must be 1.
  - ceilMode: A value of 0 means False with rounding down, and a non-zero value means True with rounding up. This parameter takes effect only when the padding mode is "CALCULATED".

## Invocation Description

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode interface | [test_max_pool_3d](examples/test_geir_max_pool_3d.cpp) | Calls the MaxPool3D operator by building a graph through the IR [MaxPool3D](./op_graph/max_pool3_d_proto.h). |
