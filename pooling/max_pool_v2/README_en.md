# MaxPoolV2

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|<term>Ascend 950PR&950DT products</term>   |     √    |
| <term>Atlas A3 products</term>    |     x     |
| <term>Atlas A2 products</term>     |     x     |
| <term>Atlas 200I/500 A2 inference products</term>                     |     x     |
| <term>Atlas inference products</term>                             |     x     |
| <term>Atlas training products</term>                             |     x     |

## Function

- Description:
Performs the max pooling operation on a 4-dimensional input tensor. Both the NHWC and NCHW data formats are supported, and both the SAME and VALID padding modes are supported.
- Formula:
  - When padding="VALID", the derivation formula of the H and W dimensions in the shape of the out tensor:

    $$
    [H_{out}, W_{out}]=[\lfloor{\frac{H_{in} - k_h}{s_h}}\rfloor + 1,\lfloor{\frac{W_{in} - k_w}{s_w}}\rfloor + 1]
    $$

  - When padding="SAME", the derivation formula of the H and W dimensions in the shape of the out tensor:

    $$
    [H_{out}, W_{out}]=[\lceil{\frac{H_{in}}{s_h}}\rceil, \lceil{\frac{W_{in}}{s_w}}\rceil]
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
      <td>The input tensor with 4 dimensions.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Input</td>
      <td>The window size of max pooling, a list of length 4.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Input</td>
      <td>The step size of the window movement, a list of length 4.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Attribute</td>
      <td>The padding mode. The value is "SAME" or "VALID". The SAME mode indicates that the output size is the input size divided by the step size, rounded up. The VALID mode indicates that the output size is the input size minus the window size, divided by the step size, plus 1.</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Attribute</td>
      <td>The data format. Supported values: "NHWC" or "NCHW". The default value is "NHWC".</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output tensor.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- **Value range restrictions:**
  - x: the input tensor must have 4 dimensions.
  - ksize: a list of length 4, where ksize[0] = 1 or ksize[3] = 1, and ksize[1] * ksize[2] <= 255.
  - strides: a list of length 4, where strides[0] = 1 or strides[3] = 1, strides[1] <= 63, strides[0] >= 1, strides[2] <= 63, and strides[2] >= 1.
  - padding: the value must be "SAME" or "VALID".
  - data_format: the value must be "NHWC" or "NCHW". The default value is "NHWC".

## Invocation Description

| Invocation method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | [test_geir_max_pool_v2](examples/test_geir_max_pool_v2.cpp) | Invokes the MaxPoolV2 operator through the [operator IR](op_graph/max_pool_v2_proto.h) interface. |
