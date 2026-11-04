# AvgPool3DGrad

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|×|
|Atlas training products|×|

## Function

- Operator function: Backpropagation of 3D average pooling, which computes the input gradient of the forward propagation of 3D average pooling.

- Formula:

$$
D_{out} = \left\lfloor \frac{D_{in} + 2 \times \text{pads}[0] - \text{ksize}[0]}{\text{strides}[0]} + 1 \right\rfloor
$$

$$
H_{out} = \left\lfloor \frac{H_{in} + 2 \times \text{pads}[1] - \text{ksize}[1]}{\text{strides}[1]} + 1 \right\rfloor
$$

$$
W_{out} = \left\lfloor \frac{W_{in} + 2 \times \text{pads}[2] - \text{ksize}[2]}{\text{strides}[2]} + 1 \right\rfloor
$$

  If the attribute ceil_mode is `True` and the condition **$(D_{out} - 1) \times \text{stride}[0] \geq D_{in} + \text{padding}[0]$** is satisfied, the last window is skipped, which reduces the dimension **$D_{out}$** by 1. This rule also applies to the dimensions **$W_{out}$** and **$H_{out}$**.

## Parameters

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
  <col style="width: 250px">
  <col style="width: 200px">
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
      <td>orig_input_shape</td>
      <td>Input</td>
      <td><ul><li>Value of the input shape of the forward propagation.</li><li>The input value is [N,C,Din,Hin,Win].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grads</td>
      <td>Input</td>
      <td><ul><li>Gradient of the backward input.</li><li>The input shape is [N,C,Dout,Hout,Wout] or [N,Dout,Hout,Wout,C].</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Attribute</td>
      <td>Size of the pooling window. The value must be greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attribute</td>
      <td>Stride of the pooling operation. The value must be greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>Attribute</td>
      <td>Number of layers padded with 0 along the D, H, and W directions of the input. The value must be greater than or equal to 0.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies whether the shape of the output inferred during forward average pooling is rounded up (True means rounding up).</li><li>Default value: false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>count_include_pad</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies whether to include the 0 values padded by pads when computing forward average pooling (True means including the padded 0 values).</li><li>Default value: true.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divisor_override</td>
      <td>Optional attribute</td>
      <td><ul><li>Divisor used for averaging. If it is specified, it is used as the divisor in the average computation. When the value is 0, this attribute does not take effect.</li><li>Default value: 0.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the data format of the input `grads`.</li><li>The value must be one of ["NDHWC","NCDHW"]. Default value: "NDHWC".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>Output parameter of the AvgPool3DGrad computation. Its shape must be consistent with the value of `orig_input_shape`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_avgpool3d_backward.cpp](examples/test_aclnn_avgpool3d_backward.cpp) | Invokes the AvgPool3DGrad operator through the [aclnnAvgPool3dBackward](docs/aclnnAvgPool3dBackward_en.md) interface. |
