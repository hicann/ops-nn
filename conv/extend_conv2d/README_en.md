# ExtendConv2D

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>| × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: Implements 2D convolution.

- Formula:

  - Assume the shape of the input (`x`) is $(N, C_{\text{in}}, H, W)$, the shape of (`filter`) is $(C_{\text{out}}, C_{\text{in}}, K_h, K_w)$, and the shape of the output (`y`) is $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$.

  - The output is expressed as:

  $$
    \text{y}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{filter}(C_{\text{out}_j}, k) \star \text{x}(N_i, k)\\
    \text{y0}(N_i, C_{\text{out}_j}) = \text{out}(N_i, C_{\text{out}_j}) \times \text{scale0}(C_{\text{out}_j})\\
    \text{y1}(N_i, C_{\text{out}_j}) = \text{out}(N_i, C_{\text{out}_j}) \times \text{scale1}(C_{\text{out}_j})
  $$
  If relu is enabled:
  $$
    \text{y0} = \text{relu}(\text{y0})\\
    \text{y1} = \text{relu}(\text{y1})
  $$

  Here, $\star$ denotes the convolution computation, and dilated convolution (`dilations` > 1) and grouped convolution (`groups` > 1) are supported. $N$ denotes the `batch size`, $C$ denotes the number of channels, and $H$ and $W$ denote the height and width respectively. The formulas for the corresponding output dimensions are as follows:

  $$
    H_{\text{out}} = (H + \text{pad\_top} + \text{pad\_bottom} - (\text{dilation\_h} \times (K_h - 1) + 1)) / \text{stride\_h} + 1 \\
    W_{\text{out}} = (W + \text{pad\_left} + \text{pad\_right} - (\text{dilation\_w} \times (K_w - 1) + 1)) / \text{stride\_w} + 1
  $$

## Parameters

<table>
<tr>
<th style="width:100px">Parameter name</th>
<th style="width:180px">Input / Output / Attribute</th>
<th style="width:420px">Description</th>
<th style="width:420px">Data type</th>
<th style="width:200px">Data format</th>
</tr>
<tr>
<td>x</td>
<td>Input</td>
<td>Input tensor x in the formula.</td>
<td>HIFLOAT8, INT8, FLOAT8_E4M3FN, FLOAT16</td>
<td>NCHW, NHWC</td>
</tr>
<tr>
<td>filter</td>
<td>Input</td>
<td>Convolution weight tensor filter in the formula.</td>
<td>HIFLOAT8, INT8, FLOAT8_E4M3FN, FLOAT16</td>
<td>NCHW, HWCN</td>
</tr>
<tr>
<td>bias</td>
<td>Optional input</td>
<td>Convolution bias tensor bias.</td>
<td>FLOAT, INT32, FLOAT16</td>
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
<td>scale0</td>
<td>Optional input</td>
<td>Quantization parameter scale0 of output 0.</td>
<td>INT64, UINT64</td>
<td>ND</td>
</tr>
<tr>
<td>relu_weight0</td>
<td>Optional input</td>
<td>relu slope parameter relu_weight0 corresponding to output 0.</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>clip_value0</td>
<td>Optional input</td>
<td>Clip value clip_value0 corresponding to output 0 (unused).</td>
<td>FLOAT16, HIFLOAT8, INT8, FLOAT8_E4M3FN, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>scale1</td>
<td>Optional input</td>
<td>Quantization parameter scale1 of output 1.</td>
<td>INT64, UINT64</td>
<td>ND</td>
</tr>
<tr>
<td>relu_weight1</td>
<td>Optional input</td>
<td>relu slope parameter relu_weight1 corresponding to output 1.</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>clip_value1</td>
<td>Optional input</td>
<td>Clip value clip_value1 corresponding to output 1 (unused).</td>
<td>FLOAT16, HIFLOAT8, INT8, FLOAT8_E4M3FN, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>y0</td>
<td>Output</td>
<td>Output tensor y0 in the formula.</td>
<td>FLOAT16, FLOAT, BFLOAT16, HIFLOAT8, INT8, FLOAT8_E4M3FN</td>
<td>NCHW, NHWC</td>
</tr>
<tr>
<td>y1</td>
<td>Output</td>
<td>Output tensor y1 in the formula. The caller must always provide y1, and y1 is written with a valid result only when dual_output is true.</td>
<td>FLOAT16, FLOAT, BFLOAT16, HIFLOAT8, INT8, FLOAT8_E4M3FN</td>
<td>NCHW, NHWC</td>
</tr>
<tr>
<td>strides</td>
<td>Attribute</td>
<td>Convolution scan stride, including stride_h, stride_w.</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>pads</td>
<td>Optional attribute</td>
<td>Padding of the input, including pad_top, pad_bottom, pad_left, pad_right.</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>dilations</td>
<td>Optional attribute</td>
<td>Spacing between elements in the convolution kernel, including dilation_h, dilation_w.</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>groups</td>
<td>Optional attribute</td>
<td>Number of block connections from input channels to output channels. It must satisfy groups × the in_channels dimension of filter = the in_channels dimension of x, and the out_channels dimension of filter must be a multiple of groups. Support range: [1, 65535].</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>data_format</td>
<td>Optional attribute</td>
<td>Input data format. "NCHW" and "NHWC" are supported.</td>
<td>STRING</td>
<td>-</td>
</tr>
<tr>
<td>offset_x</td>
<td>Optional attribute</td>
<td>Offset offset_x in the quantization algorithm. Support range: [-128, 127]. When the type of `x` is `HIFLOAT8` or `FLOAT8_E4M3FN`, only 0 can be configured.</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>round_mode</td>
<td>Optional attribute</td>
<td>Rounding mode of the output. If the data type of the output is HIFLOAT8, this parameter must be 'round'. Default: 'rint'.</td>
<td>STRING</td>
<td>-</td>
</tr>
<tr>
<td>pad_mode</td>
<td>Optional attribute</td>
<td>Mode for automatic padding by pads. "SPECIFIC", "SAME", "VALID", "SAME_UPPER", and "SAME_LOWER" are supported.</td>
<td>STRING</td>
<td>-</td>
</tr>
<tr>
<td>enable_hf32</td>
<td>Optional attribute</td>
<td>Specifies whether to enable hf32 computation. true and false are supported. This parameter is not supported yet.</td>
<td>BOOL</td>
<td>-</td>
</tr>
<tr>
<td>enable_relu0</td>
<td>Optional attribute</td>
<td>Specifies whether to enable relu computation for output 0. true and false are supported.</td>
<td>BOOL</td>
<td>-</td>
</tr>
<tr>
<td>enable_relu1</td>
<td>Optional attribute</td>
<td>Specifies whether to enable relu computation for output 1. true and false are supported.</td>
<td>BOOL</td>
<td>-</td>
</tr>
<tr>
<td>dual_output</td>
<td>Optional attribute</td>
<td>Specifies whether to use dual outputs. true and false are supported.</td>
<td>BOOL</td>
<td>-</td>
</tr>
<tr>
<td>dtype0</td>
<td>Optional attribute</td>
<td>Data type of output y0. The supported list includes [-1 (default), 0 (FLOAT), 1 (FLOAT16), 2 (INT8), 27 (BFLOAT16), 34 (HIFLOAT8), 36 (FLOAT8_E4M3FN)].</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>dtype1</td>
<td>Optional attribute</td>
<td>Data type of output y1. The supported list includes [-1 (default), 0 (FLOAT), 1 (FLOAT16), 2 (INT8), 27 (BFLOAT16), 34 (HIFLOAT8), 36 (FLOAT8_E4M3FN)].</td>
<td>INT32</td>
<td>-</td>
</tr>
</table>

## Constraints

- Ascend 950PR&950DT products:
  - The data type of `x` must be the same as that of `filter`.
  - The dimension sizes of `bias` and `scale` should be consistent with the size of the `N` dimension of `filter`.
  - Each dimension size of each `tensor` in `x`, `filter`, `bias`, `scale0/1`, `relu_weight0/1`, `clip_value0/1`, and `y` should be within [1, 1000000].
  - The values of `strides` and `dilations` should be within [1, 1000000].
  - The values of `pads` should be within [0, 1000000].
  - The supported data type and format combinations are shown in the following table:
  <table>
  <tr>
  <th style="text-align:center; width:80px">Tensor</th>
  <th style="text-align:center; width:150px">x</th>
  <th style="text-align:center; width:150px">filter</th>
  <th style="text-align:center; width:100px">bias</th>
  <th style="text-align:center; width:150px">scale0/1</th>
  <th style="text-align:center; width:150px">clip_value0/1</th>
  <th style="text-align:center; width:150px">y0/1</th>
  </tr>
  <tr>
  <td rowspan="4" style="text-align:center">Data type</td>
  <td style="text-align:center">INT8</td>
  <td style="text-align:center">INT8</td>
  <td style="text-align:center">INT32</td>
  <td style="text-align:center">INT64/UINT64</td>
  <td style="text-align:center">INT8</td>
  <td style="text-align:center">FLOAT16/INT8</td>
  </tr>
  <tr>
  <td style="text-align:center">HIFLOAT8</td>
  <td style="text-align:center">HIFLOAT8</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">INT64/UINT64</td>
  <td style="text-align:center">HIFLOAT8</td>
  <td style="text-align:center">FLOAT/FLOAT16/BFLOAT16/HIFLOAT8</td>
  </tr>
  <tr>
  <td style="text-align:center">FLOAT16</td>
  <td style="text-align:center">FLOAT16</td>
  <td style="text-align:center">FLOAT16</td>
  <td style="text-align:center">INT64/UINT64</td>
  <td style="text-align:center">FLOAT16/INT8</td>
  <td style="text-align:center">FLOAT16/INT8</td>
  </tr>
  <tr>
  <td style="text-align:center">FLOAT8_E4M3FN</td>
  <td style="text-align:center">FLOAT8_E4M3FN</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">INT64/UINT64</td>
  <td style="text-align:center">FLOAT8_E4M3FN</td>
  <td style="text-align:center">FLOAT/FLOAT16/BFLOAT16/FLOAT8_E4M3FN</td>
  </tr>
  <tr>
  <td rowspan="2" style="text-align:center">Data format</td>
  <td style="text-align:center">NCHW</td>
  <td style="text-align:center">NCHW</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">NCHW</td>
  </tr>
  <tr>
  <td style="text-align:center">NHWC</td>
  <td style="text-align:center">HWCN</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">NHWC</td>
  </tr>
  </table>

- If any parameter exceeds the preceding ranges, the correctness of the operator cannot be guaranteed.
- Due to hardware resource limits, the operator may fail to execute for some parameter value combinations. Analyze and troubleshoot the problem based on the log information. If the problem cannot be solved, click [Link](https://www.hiascend.com/support) to obtain technical support.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_extend_conv2d](./examples/arch35/test_aclnn_extend_conv2d.cpp) | Invokes the extend_conv2d operator through the [aclnnQuantConvolution](../convolution_forward/docs/aclnnQuantConvolution_en.md) interface.
