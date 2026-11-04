# MaxPoolExt2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>   |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    x     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function

- Description:
For a 4-dimensional input tensor, performs the max pooling operation. Both the NHWC and NCHW data formats are supported, and both the SAME and VALID padding modes are supported.
- Formula:
  - When padding="SAME", the derivation formulas for the H and W dimensions in the shape of the out tensor are:

    $$
    [H_{out}, W_{out}]=[\lceil{\frac{H_{in}}{s_h}}\rceil, \lceil{\frac{W_{in}}{s_w}}\rceil]
    $$

  - When padding="VALID", the derivation formulas for the H and W dimensions in the shape of the out tensor are:

    $$
    [H_{out}, W_{out}]=[\lceil{\frac{H_{in} - k_h + 1}{s_h}}\rceil, \lceil{\frac{W_{in} - k_w + 1}{s_w}}\rceil]
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
      <td>Input tensor with 4 dimensions, corresponding to x in the formula.</td>
      <td>FLOAT16, FLOAT, INT8, INT16, INT32, INT64, UINT8, UINT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Attribute</td>
      <td>Window size of max pooling, a list of length 4, corresponding to k_h and k_w in the formula.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attribute</td>
      <td>Strides of window movement, a list of length 4, corresponding to s_h and s_w in the formula.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Attribute</td>
      <td>Padding mode, with the value "SAME" or "VALID". The SAME mode means that the output size is the input size divided by the stride, rounded up; the VALID mode means that the output size is the input size minus the window size plus 1, divided by the stride, rounded up.</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Attribute</td>
      <td>Data format; "NHWC" or "NCHW" is supported, and the default value is "NHWC".</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor with the same dtype as x and 4 dimensions.</td>
      <td>FLOAT16, FLOAT, INT8, INT16, INT32, INT64, UINT8, UINT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- **Value range restrictions:**
  - x: the input tensor must have 4 dimensions.
  - ksize: a list of length 4, where ksize[0]=1 and ksize[3]=1 (NHWC format) or ksize[0]=1 and ksize[1]=1 (NCHW format), and ksize[1] and ksize[2] (or the corresponding spatial axes) must be greater than 0.
  - strides: a list of length 4, where strides[0]=1 and strides[3]=1 (NHWC format) or strides[0]=1 and strides[1]=1 (NCHW format), and strides[1] and strides[2] (or the corresponding spatial axes) must be greater than 0.
  - padding: the value must be "SAME" or "VALID".
  - data_format: the value must be "NHWC" or "NCHW", and the default value is "NHWC".
  - The dtype of output y is the same as the dtype of input x.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation method</th>
    <th>Invocation sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Graph-mode invocation</td>
    <td><a href="./examples/test_geir_max_pool_ext2.cpp">test_geir_max_pool_ext2</a></td>
    <td>See <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
