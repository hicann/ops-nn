# AdaptiveMaxPool3d

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

- Operator function: computes the size of each kernel based on the input output_size, performs the 3D max pooling operation on input x, and outputs the pooled value y and the indices.

- Computation formula:
  For an element at position $(l,m,n)$ on the DHW axes in the y tensor, the computation formula is:

  $$
  D^{l}_{left} = floor((l*D)/D_o)
  $$

  $$
  D^{l}_{right} = ceil(((l+1)*D)/D_o)
  $$

  $$
  H^{m}_{left} = floor((m*H)/H_o)
  $$

  $$
  H^{m}_{right} = ceil(((m+1)*H)/H_o)
  $$

  $$
  W^{n}_{left} = floor((n*W)/W_o)
  $$

  $$
  W^{n}_{right} = ceil(((n+1)*W)/W_o)
  $$

  $$
  y(N,C,l,m,n)=\underset {i \in [D^{l}_{left}, D^{l}_{right}],j\in [H^m_{left},H^m_{right}], k \in [W^n_{left},W^n_{right}] }{max} input(N,C,i,j,k)
  $$

  $$
  indices(N,C,l,m,n)=\underset {i \in [D^{l}_{left}, D^{l}_{right}],j\in [H^m_{left},H^m_{right}], k \in [W^n_{left},W^n_{right}] }{argmax} input(N,C,i,j,k)
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
      <td>Input parameter for the AdaptiveMaxPool3d computation. The product of the three dimensions of the D axis, H axis, and W axis cannot be greater than the maximum representable value of int32. Its data type is consistent with the output parameter `y`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>Attribute</td>
      <td>Denotes the spatial sizes of the output result in the D, H, and W dimensions. Its data type is consistent with the input parameter `x`.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter of the AdaptiveMaxPool3d computation. Its shape is consistent with the output parameter `indices`, and its data type is consistent with the input parameter `x`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Output</td>
      <td>Denotes the index positions of the `y` elements in the input `x`. Its shape is consistent with the output parameter `y`.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

 - Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraints

Shape description:

  - self.shape = (N, C, Din, Hin, Win)
  - outputSize = [Dout, Hout, Wout]
  - outputOut.shape = (N, C, Dout, Hout, Wout)
  - indicesOut.shape = (N, C, Dout, Hout, Wout)

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_adaptive_max_pool3d.cpp](examples/test_aclnn_adaptive_max_pool3d.cpp) | Calls the AdaptiveMaxPool3d operator through the [aclnnAdaptiveMaxPool3d](docs/aclnnAdaptiveMaxPool3d_en.md) interface. |
