# AdaptiveMaxPool2d

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|x|
|Atlas A2 products|x|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|×|
|Atlas training products|×|
|Kirin X90 processor series products|x|
|Kirin 9030 processor series products|x|

## Function

- Operator function: computes the size of each kernel according to the input output_size, performs the 3D max pooling operation on input x, and outputs the pooled value y and the index indices.

- Formula:

  For an element at position $(l,m,n)$ on the DHW axes in tensor y, the computation formula is:
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
  y(N,C,m,n)=\underset {j\in [H^m_{left},H^m_{right}), k \in [W^n_{left},W^n_{right}) }{max} input(N,C,j,k)
  $$

  $$
  indices(N,C,m,n)=\underset {j\in [H^m_{left},H^m_{right}), k \in [W^n_{left},W^n_{right}) }{argmax} input(N,C,j,k)
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
      <td><ul><li>Input parameter of the AdaptiveMaxPool2d computation. The product of the two dimensions on the H axis and W axis cannot be greater than the maximum value representable by int32. Its data type is consistent with that of output parameter y.</li><li>The shape is [N, C, Hin, Win] or [C, Hin, Win].</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>Attribute</td>
      <td><ul><li>Indicates the spatial size of the output result on the H and W dimensions. Its data type is consistent with that of input parameter x.</li><li>The value is [Hout, Wout].</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td><ul><li>Output parameter of the AdaptiveMaxPool2d computation. Its shape is consistent with that of output parameter indices, and its data type is consistent with that of input parameter x.</li><li>The shape is [N, C, Hout, Wout] or [C, Hout, Wout].</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Output</td>
      <td><ul><li>Indicates the index positions of the elements of y in input x. Its shape is consistent with that of output parameter y.</li><li>The shape is [N, C, Hout, Wout] or [C, Hout, Wout].</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_adaptive_max_pool2d.cpp](examples/test_aclnn_adaptive_max_pool2d.cpp) | Invokes the AdaptiveMaxPool2d operator through the [aclnnAdaptiveMaxPool2d](docs/aclnnAdaptiveMaxPool2d_en.md) interface. |
| Graph mode | [test_geir_adaptive_max_pool2d.cpp](examples/test_geir_adaptive_max_pool2d.cpp) | Invokes the AdaptiveMaxPool2d operator through the GE IR graph construction method. |
