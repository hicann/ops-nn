# MaxPool3DWithArgmaxV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |
| Atlas 200I/500 A2 inference products | × |
| Atlas inference products | × |
| Atlas training products | × |
| Kirin X90 processor series products | √ |
| Kirin 9030 processor series products | √ |

## Function

- Operator function: For the input channels of the input signal, perform 3D max pooling, and output the pooled value y and the index argmax.
- Formula:

  * Formula for computing each element in the y tensor:

    $$
    out(N_i, C_j, d, h, w) = \max\limits_{{k\in[0,k_{D}-1],m\in[0,k_{H}-1],n\in[0,k_{W}-1]}}input(N_i,C_j,stride[0]\times d + k, stride[1]\times h + m, stride[2]\times w + n)
    $$

  * Shape derivation formula of the out tensor (ceilMode=false by default, that is, rounding down):

    $$
    [N, C, D_{out}, H_{out}, W_{out}]=[N,C,\lfloor{\frac{D_{in}+2 \times padding[0] - dilation[0] \times (kernelSize[0] - 1) - 1}{stride[0]}}\rfloor + 1,\lfloor{\frac{H_{in}+2 \times padding[1] - dilation[1] \times (kernelSize[1] - 1) - 1}{stride[1]}}\rfloor + 1, \lfloor{\frac{W_{in}+2 \times padding[2] - dilation[2] \times (kernelSize[2] - 1) - 1}{stride[2]}}\rfloor + 1]
    $$

  * Shape derivation formula of the out tensor (ceilMode=true by default, that is, rounding up):

    $$
    [N, C, D_{out}, H_{out}, W_{out}]=[N,C,\lceil{\frac{D_{in}+2 \times padding[0] - dilation[0] \times (kernelSize[0] - 1) - 1}{stride[0]}}\rceil + 1,\lceil{\frac{H_{in}+2 \times padding[1] - dilation[1] \times (kernelSize[1] - 1) - 1}{stride[1]}}\rceil + 1, \lceil{\frac{W_{in}+2 \times padding[2] - dilation[2] \times (kernelSize[2] - 1) - 1}{stride[2]}}\rceil + 1]
    $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1300px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
  <col style="width: 300px">
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
      <td>Input parameter for the MaxPool3DWithArgmaxV2 computation.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCDHW, NDHWC</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Attribute</td>
      <td>Window size of max pooling.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attribute</td>
      <td>Stride of window movement.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>Attribute</td>
      <td>Number of layers padded on each edge. The padded positions are filled with "negative infinity".</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>Optional attribute</td>
      <td>Stride of elements in the window.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>Optional attribute</td>
      <td>When this parameter is True, the output shape is calculated by rounding up. When this parameter is False, the output shape is calculated by rounding down.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Optional attribute</td>
      <td>Data format of the input. "NCDHW" and "NDHWC" are supported.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Pooled result. The data type and data format must be the same as those of `x`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCDHW, NDHWC</td>
    </tr>
    <tr>
      <td>argmax</td>
      <td>Output</td>
      <td>Tensor consisting of the index positions of the maximum values. The shape must be the same as that of `y`.</td>
      <td>INT32, INT64</td>
      <td>NCDHW, NDHWC</td>
    </tr>
  </tbody></table>

  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: The INT64 data type is not supported for `argmax`. The NDHWC input data layout is not supported.
  - Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported for input/output. The INT64 data type is not supported for `argmax`.

## Constraints

None.

## Invocation Description

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_max_pool3d_with_argmax.cpp](examples/test_aclnn_max_pool3d_with_argmax.cpp) | Invoke the MaxPool3DWithArgmaxV2 operator through the [aclnnMaxPool3dWithArgmax](docs/aclnnMaxPool3dWithArgmax_en.md) API. |
