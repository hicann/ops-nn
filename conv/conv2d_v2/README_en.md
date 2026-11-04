# Conv2DV2

English | [简体中文](./README.md)

## Applicable Products

<table>
<tr>
<th style="text-align:left">Product</th>
<th style="text-align:center; width:100px">Support</th>
</tr>
<tr>
<td><term>Ascend 950PR&950DT products</term></td>
<td style="text-align:center">√</td>
</tr>
<tr>
<td><term>Atlas A3 products</term></td>
<td style="text-align:center">×</td>
</tr>
<tr>
<td><term>Atlas A2 products</term></td>
<td style="text-align:center">×</td>
</tr>
</table>

## Function

- Operator function: implements the 2D convolution function.
- Formula:

  - Assume that the shape of input (`x`) is $(N, C_{\text{in}}, H, W)$, the shape of (`filter`) is $(C_{\text{out}}, C_{\text{in}}, K_h, K_w)$, and the shape of output (`y`) is $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$
  - The output is expressed as:

  $$
  \text{y}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{filter}(C_{\text{out}_j}, k) \star \text{x}(N_i, k)
  $$

  Here, $\star$ denotes the convolution computation, which supports dilated convolution (`dilations` > 1) and grouped convolution (`groups` > 1). $N$ denotes the `batch size`, $C$ denotes the number of channels, and $H$ and $W$ denote the height and width respectively. The formulas for computing the corresponding output dimensions are as follows:

  $$
  H_{\text{out}} = (H + \text{pad\_top} + \text{pad\_bottom} - (\text{dilation\_h} \times (K_h - 1) + 1)) / \text{stride\_h} + 1 \\
    W_{\text{out}} = (W + \text{pad\_left} + \text{pad\_right} - (\text{dilation\_w} \times (K_w - 1) + 1)) / \text{stride\_w} + 1
  $$

## Parameters

<table>
<tr>
<th style="width:100px">Parameter name</th>
<th style="width:180px">Input/Output/Attribute</th>
<th style="width:420px">Description</th>
<th style="width:420px">Data type</th>
<th style="width:200px">Data format</th>
</tr>
<tr>
<td>x</td>
<td>Input</td>
<td>Input tensor x in the formula.</td>
<td>FLOAT16, FLOAT, BFLOAT16, HIFLOAT8</td>
<td>NCHW, NHWC</td>
</tr>
<tr>
<td>filter</td>
<td>Input</td>
<td>Convolution weight tensor filter in the formula.</td>
<td>FLOAT16, FLOAT, BFLOAT16, HIFLOAT8</td>
<td>NCHW, HWCN</td>
</tr>
<tr>
<td>bias</td>
<td>Optional input</td>
<td>Convolution bias tensor bias.</td>
<td>FLOAT16, FLOAT, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>offset_w</td>
<td>Optional input</td>
<td>Quantization offset tensor offset_w (unused).</td>
<td>INT8</td>
<td>-</td>
</tr>
<tr>
<td>y</td>
<td>Output</td>
<td>Output tensor y in the formula.</td>
<td>FLOAT16, FLOAT, BFLOAT16, HIFLOAT8</td>
<td>NCHW, NHWC</td>
</tr>
<tr>
<td>strides</td>
<td>Attribute</td>
<td>Convolution scan strides, including stride_h and stride_w.</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>pads</td>
<td>Optional attribute</td>
<td>Padding of the input, including pad_top, pad_bottom, pad_left, and pad_right.</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>dilations</td>
<td>Optional attribute</td>
<td>Spacing between the elements in the convolution kernel, including dilation_h and dilation_w.</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>groups</td>
<td>Optional attribute</td>
<td>Number of block connections from input channels to output channels; it must satisfy groups × the in_channels dimension of filter = the in_channels dimension of x. The supported range is [1, 65535].</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>data_format</td>
<td>Optional attribute</td>
<td>Input data format; "NCHW" and "NHWC" are supported.</td>
<td>STRING</td>
<td>-</td>
</tr>
<tr>
<td>offset_x</td>
<td>Optional attribute</td>
<td>Offset in the quantization algorithm (unused).</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>pad_mode</td>
<td>Optional attribute</td>
<td>Mode for automatic pads filling; "SPECIFIC", "SAME", "VALID", "SAME_UPPER", and "SAME_LOWER" are supported.</td>
<td>STRING</td>
<td>-</td>
</tr>
<tr>
<td>enable_hf32</td>
<td>Optional attribute</td>
<td>Indicates whether to enable hf32 computation; true and false are supported. The hf32 computation can be correctly enabled by setting this parameter to true only when `x`, `filter`, `bias`, and `y` are all of the `FLOAT` type.</td>
<td>BOOL</td>
<td>-</td>
</tr>
</table>

## Constraints

- Ascend 950PR&950DT products:

  - When the data type of `x` is `HIFLOAT8`, the data type of `filter` must be consistent with `x`, and the formats of `x` and `filter` both support only `NCHW`.
  - The size of each dimension of each group of `tensor` in `x`, `filter`, `bias`, and `y` should be within the range [1, 1000000].
  - The values of `strides` and `dilations` should be within the range [1, 1000000].
  - The values of `pads` should be within the range [0, 1000000].
  - The supported data type and format combinations are as follows:

  <table>
  <tr>
  <th style="text-align:center; width:80px">Tensor</th>
  <th style="text-align:center; width:150px">x</th>
  <th style="text-align:center; width:150px">filter</th>
  <th style="text-align:center; width:100px">bias</th>
  <th style="text-align:center; width:150px">y</th>
  </tr>
  <tr>
  <td rowspan="4" style="text-align:center">Data type</td>
  <td style="text-align:center">FLOAT16</td>
  <td style="text-align:center">FLOAT16</td>
  <td style="text-align:center">FLOAT16</td>
  <td style="text-align:center">FLOAT16</td>
  </tr>
  <tr>
  <td style="text-align:center">BFLOAT16</td>
  <td style="text-align:center">BFLOAT16</td>
  <td style="text-align:center">BFLOAT16</td>
  <td style="text-align:center">BFLOAT16</td>
  </tr>
  <tr>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  </tr>
  <tr>
  <td style="text-align:center">HIFLOAT8</td>
  <td style="text-align:center">HIFLOAT8</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">HIFLOAT8</td>
  </tr>
  <tr>
  <td rowspan="2" style="text-align:center">Data format</td>
  <td style="text-align:center">NCHW</td>
  <td style="text-align:center">NCHW</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">NCHW</td>
  </tr>
  <tr>
  <td style="text-align:center">NHWC</td>
  <td style="text-align:center">HWCN</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">NHWC</td>
  </tr>
  </table>
- If any parameter exceeds the above ranges, the correctness of the operator cannot be guaranteed.
- Due to hardware resource limitations, the operator may fail to execute for some parameter value combinations. Analyze and troubleshoot the problem according to the log information. If the problem cannot be solved, click [Link](https://www.hiascend.com/support) to obtain technical support.

## Invocation Description

| Invocation method  | Invocation sample                                                   | Description                                                                                              |
| --------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| aclnn invocation | [test_aclnn_conv2d_v2](./examples/test_aclnn_conv2d_v2.cpp) | Invokes the Conv2DV2 operator through the [aclnnConvolution](../convolution_forward/docs/aclnnConvolution_en.md) interface. |
