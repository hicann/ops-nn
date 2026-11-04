# AvgPool3D

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
|Kirin X90 processor series products|√|
|Kirin 9030 processor series products|√|

## Function

- Operator function: performs a 3D average pooling operation on the input tensor with the window $kD * kH * kW$ and the stride $sD * sH * sW$, where $k$ is kernelSize, denoting the size of the pooling window, and $s$ is stride, denoting the stride of the pooling operation.

- Formula:
  The relationships between the input input($N,C,D_{in},H_{in},W_{in}$), the output out($N,C,D_{out},H_{out},W_{out}$), the pooling stride ($stride$), and the pooling window size kernelSize($kD,kH,kW$) are

$$
D_{out}=\lfloor \frac{D_{in}+2*padding[0]-kernelSize[0]}{stride[0]}+1 \rfloor
$$

$$
H_{out}=\lfloor \frac{H_{in}+2*padding[1]-kernelSize[1]}{stride[1]}+1 \rfloor
$$

$$
W_{out}=\lfloor \frac{W_{in}+2*padding[2]-kernelSize[2]}{stride[2]}+1 \rfloor
$$

$$
out(N_i,C_i,d,h,w)=\frac{1}{kD*kH*kW}\sum_{k=0}^{kD-1}\sum_{m=0}^{kH-1}\sum_{n=0}^{kW-1}input(N_i,C_i,stride[0]*d+k,stride[1]*h+m,stride[2]*w+n)
$$

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
      <td>x</td>
      <td>Input</td>
      <td>Denotes the tensor to be converted, corresponding to `input` in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Attribute</td>
      <td>Denotes the pooling window size, corresponding to `kernelSize` in the formula.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attribute</td>
      <td>Denotes the stride of the pooling operation, corresponding to `stride` in the formula.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>Attribute</td>
      <td>Denotes the number of layers of zero padding by pads along the D, H, and W directions of the input, corresponding to `padding` in the formula.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>Optional attribute</td>
      <td>Indicates that, when computing the output shape, the result is rounded down (False); otherwise it is rounded up.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>count_include_pad</td>
      <td>Optional attribute</td>
      <td>Indicates that the average computation includes zero padding (True); otherwise it does not include zero padding.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divisor_override</td>
      <td>Optional attribute</td>
      <td>If specified, it is used as the divisor in the average computation; when the value is 0, this attribute does not take effect.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Optional attribute</td>
      <td>Input data format; "NDHWC" is supported.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter for the AvgPool3D computation, corresponding to `out` in the formula. The data type and data format must be consistent with `x`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_avgpool3d.cpp](examples/test_aclnn_avgpool3d.cpp) | Invokes the AvgPool3d operator through the [aclnnAvgPool3d](docs/aclnnAvgPool3d_en.md) interface. |
