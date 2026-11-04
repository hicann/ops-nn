# AvgPoolV2

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

- Description: performs 2D average pooling on the input tensor with a window of $kH * kW$ and a stride of $sH * sW$, where $k$ is the kernelSize, which denotes the size of the pooling window, and $s$ is the stride, which denotes the stride of the pooling operation.
- Formula:
  The relationship between the input ($N,C,H,W$), the output out ($N,C,H_{out},W_{out}$), the pooling stride ($strides$), and the pooling window size ($kH,kW$) is as follows:

  Assume that `pads=[pad_top,pad_bottom,pad_left,pad_right]`. Then:

  $$
  H_{out}=\lfloor \frac{H_{in}+pad_{top}+pad_{bottom}-kH}{strides[0]}+1 \rfloor
  $$

  $$
  W_{out}=\lfloor \frac{W_{in}+pad_{left}+pad_{right}-kW}{strides[1]}+1 \rfloor
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
      <td>Denotes the pooling window size, `kernelSize` in the formula.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attribute</td>
      <td>Denotes the stride of the pooling operation, `stride` in the formula.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>Optional attribute</td>
      <td>Specifies the padding algorithm. The optional values are "VALID", "SAME" and "CALCULATED"</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>Optional attribute</td>
      <td>Denotes the number of layers of 0 padding on the D and H directions of the input, `padding` in the formula.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Optional attribute</td>
      <td>Denotes the data format of the input (such as NCHW/NHWC). The default value is NCHW</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>global_pooling</td>
      <td>Optional attribute</td>
      <td>Specifies whether to use global_pooling. The default value is false</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>Optional attribute</td>
      <td>Specifies whether the deduced shape of the output out is rounded up.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>exclusive</td>
      <td>Optional attribute</td>
      <td>Specifies whether to exclude padding points. The default value is true</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divisor_override</td>
      <td>Optional attribute</td>
      <td>Divisor used for computing the average.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode API | [test_geir_avg_pool_v2](examples/arch35/test_geir_avg_pool_v2.cpp) | Invokes the AvgPoolV2 operator by constructing the graph through the IR [AvgPool](./op_graph/avg_pool_v2_proto.h). |
| aclnn invocation  | [test_aclnn_avgpool2d.cpp](../avg_pool3_d/examples/test_aclnn_avgpool2d.cpp) | Invokes the AvgPoolV2 operator through the [aclnnAvgPool2d](../../pooling/avg_pool3_d/docs/aclnnAvgPool2d_en.md) API. |
