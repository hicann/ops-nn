# AvgPoolV2Grad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
| <term>Ascend 950PR&950DT products</term>                     |    √     |
| <term>Atlas A3 products</term>    |    x     |
| <term>Atlas A2 products</term>     |    x     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    x     |
| <term>Atlas inference products</term>                             |    x     |
| <term>Atlas training products</term>                             |    x     |

## Function

- Description:
Backpropagation of 2D average pooling, which computes the input gradient of the forward propagation of 2D average pooling.

- Formula:
  - For the output gradient of forward average pooling, compute the gradient of the original input:

    $$
    out\_grad(i,j) = \frac{1}{divisor} \sum_{k \in pooling\_window} input\_grad(k)
    $$

    divisor is the pooling window size (if exclusive is true, the padding area is not included; if divisor_override is specified, that value is used).

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
      <td>orig_input_shape</td>
      <td>Input</td>
      <td>Shape of the original input, describing the input dimensions [N,C,H,W] or [N,H,W,C] of the forward AvgPoolV2.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_grad</td>
      <td>Input</td>
      <td>Gradient tensor of the average pooling output.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Attribute</td>
      <td>Size of the pooling window, with length 4 and values greater than 0.</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attribute</td>
      <td>Stride of the sliding window, with length 4 and values greater than 0.</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>Attribute</td>
      <td>Type of the padding algorithm. "VALID", "SAME", and "CALCULATED" are supported. Default: "CALCULATED".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>Attribute</td>
      <td>Padding size of the input feature map. Default: {0,0,0,0}.</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Attribute</td>
      <td>Data format. "NCHW" and "NHWC" are supported. Default: "NCHW".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>global_pooling</td>
      <td>Attribute</td>
      <td>Specifies whether to use global pooling. If it is true, ksize and pads are ignored. Default: false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>Attribute</td>
      <td>Specifies whether the shape of the inferred output out is rounded up. Default: false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>exclusive</td>
      <td>Attribute</td>
      <td>Specifies whether to exclude padding points when computing average pooling. Default: true.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divisor_override</td>
      <td>Attribute</td>
      <td>Specifies the override value of the divisor. If it is 0, the pooling area size is used. Default: 0.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out_grad</td>
      <td>Output</td>
      <td>Output gradient tensor, with the same shape as orig_input_shape.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- **Value range restrictions:**
  - orig_input_shape: must be a 1D tensor of length 4, describing [N,C,H,W] or [N,H,W,C].
  - input_grad: 4D tensor, supporting FLOAT16, FLOAT, and BFLOAT16 types.
  - ksize: length 4, must be greater than 0, specifying the pooling window size.
  - strides: length 4, must be greater than 0, specifying the sliding stride.
  - padding_mode: only the three modes "VALID", "SAME", and "CALCULATED" are supported.
  - pads: length 4, must be greater than 0, and takes effect only when padding_mode is "CALCULATED". It must not exceed 1/2 of the value at the corresponding position of ksize.
  - data_format: "NCHW" and "NHWC" are supported.
  - global_pooling: if it is true, ksize and pads are ignored.
  - ceil_mode: if it is true, ceil is used to compute the output shape; otherwise, floor is used.
  - exclusive: if it is true, the padding area is excluded when computing the mean.
  - divisor_override: if it is not 0, it is used as the divisor; otherwise, the pooling area size is used.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_avg_pool_v2_grad](examples/arch35/test_aclnn_avg_pool_v2_grad.cpp) | Invokes the AvgPoolV2Grad operator through the [aclnnAvgPool2dBackward](../../pooling/avg_pool3_d_grad/docs/aclnnAvgPool2dBackward_en.md) interface. |
