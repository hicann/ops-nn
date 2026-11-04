# AvgPool

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|√|
|Atlas training products|√|

## Function

- Operator function: Performs 2D average pooling on the input tensor with a window of $kH * kW$ and a stride of $sH * sW$, where $k$ is kernelSize, indicating the size of the pooling window, and $s$ is stride, indicating the stride of the pooling operation.
- Formula:
  The relationship between the input input ($N,C,H,W$), the output out ($N,C,H_{out},W_{out}$), the pooling stride ($strides$), and the pooling window size ($kH,kW$) is

  $$
  H_{out}=\lfloor \frac{H_{in}+2*paddings[0]-kH}{strides[0]}+1 \rfloor
  $$

  $$
  W_{out}=\lfloor \frac{W_{in}+2*paddings[1]-kW}{strides[1]}+1 \rfloor
  $$

  $$
  out(N_i,C_i,h,w)=\frac{1}{kH*kW}\sum_{m=0}^{kH-1}\sum_{n=0}^{kW-1}input(N_i,C_i,strides[0]*h+m,strides[1]*w+n)
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
      <td>Input tensor, input in the formula</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor, output in the formula</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Attribute</td>
      <td>Represents the pooling window size, `kernelSize` in the formula.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attribute</td>
      <td>Represents the stride of the pooling operation, `stride` in the formula.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>Attribute</td>
      <td>Represents the number of zero-padding layers on the D and H directions of the input, `padding` in the formula.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Optional attribute</td>
      <td>Represents the input data type. The default is NHWC</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode interface | [test_geir_avg_pool](examples/arch35/test_geir_avg_pool.cpp) | Invokes the AvgPool operator through the IR-based [AvgPool](./op_graph/avg_pool_proto.h) graph construction method. |
