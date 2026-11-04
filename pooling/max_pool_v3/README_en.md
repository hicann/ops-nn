# MaxPoolV3

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|<term>Ascend 950PR&950DT products</term>   |     √    |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    x     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- Description:
Performs max pooling on input tensors of 3 or 4 dimensions.
- Formula:
  - When ceilMode is False, the derivation formula of the H and W dimensions in the shape of the out tensor:

    $$
    [H_{out}, W_{out}]=[\lfloor{\frac{H_{in}+  padding\_size_{Htop} + padding\_size_{Hbottom} - {dilation\_size \times(k_h - 1) - 1}}{s_h}}\rfloor + 1,\lfloor{\frac{W_{in}+ padding\_size_{Wleft} + padding\_size_{Wright} - {dilation\_size \times(k_w - 1) - 1}}{s_w}}\rfloor + 1]
    $$

  - When ceilMode is True, the derivation formula of the H and W dimensions in the shape of the out tensor:

    $$
    [H_{out}, W_{out}]=[\lceil{\frac{H_{in}+  padding\_size_{Htop} + padding\_size_{Hbottom} - {dilation\_size \times(k_h - 1) - 1}}{s_h}}\rceil + 1,\lceil{\frac{W_{in}+ padding\_size_{Wleft} + padding\_size_{Wright} - {dilation\_size \times(k_w - 1) - 1}}{s_w}}\rceil + 1]
    $$

    - When the start position at the top-left corner of the sliding window is located on the bottom or right padding positions or out of bounds (no valid value can be obtained), discard the sliding window result, and the shape of the corresponding spatial axis needs to be reduced by 1 based on the preceding derivation formula:

      $$
      \begin{cases}
      H_{out}=H_{out} - 1& \text{if } (H_{out}-1)*s_h>=H_{in}+padding\_size_{Htop} \\
      W_{out}=W_{out} - 1& \text{if } (W_{out}-1)*s_w>=W_{in}+padding\_size_{Wleft}  \\
      \end{cases}\\
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
      <td>4D input tensor.</td>
      <td>FLOAT16, BF16, FLOAT32, DOUBLE, INT32, INT64, UINT8, INT16, INT8, UINT16, QINT8</td>
      <td>NHWC, NCHW</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Attribute</td>
      <td>Window size of max pooling, a list of length 4. ksize in the N and C dimensions must be 1, and ksize in the H and W dimensions must be greater than 0.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attribute</td>
      <td>Stride of the sliding window, a list of length 4. stride in the N and C dimensions must be 1.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>Attribute</td>
      <td>Specifies the padding mode. "SAME", "VALID", and "CALCULATED" are supported, and the default is "CALCULATED". "SAME" pads zeros so that the output shape equals ceil(input shape / stride); "VALID" does not pad zeros; "CALCULATED" uses pads to compute the output shape.</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>Attribute</td>
      <td>Padding at the start and end positions along the spatial axes, corresponding to padding_size in the formula. Length 4, specifying the padding amounts of [top, bottom, left, right]. It takes effect only when padding_mode is "CALCULATED". Default value: {0,0,0,0}.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Attribute</td>
      <td>Data format of inputs and outputs. "NHWC" and "NCHW" are supported, and the default is "NCHW".</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>global_pooling</td>
      <td>Attribute</td>
      <td>Whether to use global pooling. When True, ksize and pads are ignored, and both output H and W are 1. Default False.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>Attribute</td>
      <td>Rounding mode for computing the output shape. True means rounding up, and False means rounding down. When padding_mode is "SAME" or "VALID", only False is supported. Default False.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>4D output tensor, with the same data type and format as input x.</td>
      <td>FLOAT16, BF16, FLOAT32, DOUBLE, INT32, INT64, UINT8, INT16, INT8, UINT16, QINT8</td>
      <td>NHWC, NCHW</td>
    </tr>
  </tbody></table>

## Constraints

- **Value range restrictions:**
  - ksize: a list of length 4. ksize in the H and W dimensions must be greater than 0, and ksize in the N and C dimensions must be 1. The product of ksize in the H and W dimensions should be less than or equal to 255.
  - strides: a list of length 4. stride in the N and C dimensions must be 1, and stride in the H and W dimensions must be greater than 0.
  - padding_mode: the three modes "SAME", "VALID", and "CALCULATED" are supported.
  - pads: length 4, specifying the padding amounts by [top, bottom, left, right]. It takes effect only when padding_mode is "CALCULATED". pads should be greater than or equal to 0 and less than the corresponding kernel size.
  - global_pooling: when True, both output H and W are 1, and ksize and pads are ignored.
  - ceil_mode: when padding_mode is "SAME" or "VALID", only False is supported.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface | [test_aclnn_max_pool](examples/test_aclnn_max_pool.cpp) | Calls the MaxPoolV3 operator through the [aclnnMaxPool](docs/aclnnMaxPool_en.md) interface. |
