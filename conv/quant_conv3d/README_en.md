# QuantConv3D

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
<tr>
<td><term>Atlas 200I/500 A2 inference products</term></td>
<td style="text-align:center">×</td>
</tr>
<tr>
<td><term>Atlas inference products</term></td>
<td style="text-align:center">×</td>
</tr>
<tr>
<td><term>Atlas training products</term></td>
<td style="text-align:center">×</td>
</tr>
</table>

## Function

- Operator function: Implements the 3D quantization convolution function.

- Formula:

  - Assume that the shape of the input (`x`) is $(N, C_{\text{in}}, D, H, W)$, the shape of (`filter`) is $(C_{\text{out}}, C_{\text{in}}, K_d, K_h, K_w)$, and the shape of the output (`y`) is $(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})$

  - The output is expressed as:

  $$
    \text{y}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \text{scale} \times \sum_{k = 0}^{C_{\text{in}} - 1} \text{filter}(C_{\text{out}_j}, k) \star \text{x}(N_i, k)
  $$

  Here, $\star$ denotes the convolution computation. Dilated convolution (`dilations` > 1) and grouped convolution (`groups` > 1) are supported. $N$ denotes the batch size, $C$ denotes the number of channels, and $D$, $H$, and $W$ denote the depth, height, and width respectively. The formulas for the corresponding output dimensions are as follows:

  $$
    D_{\text{out}} = (D + \text{pad\_head} + \text{pad\_tail} - (\text{dilation\_d} \times (K_d - 1) + 1)) / \text{stride\_d} + 1 \\
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
<td>FLOAT8_E4M3FN, INT8, HIFLOAT8</td>
<td>NCDHW</td>
</tr>
<tr>
<td>filter</td>
<td>Input</td>
<td>Convolution weight tensor filter in the formula.</td>
<td>FLOAT8_E4M3FN, INT8, HIFLOAT8</td>
<td>NCDHW</td>
</tr>
<tr>
<td>scale</td>
<td>Input</td>
<td>Scaling factor tensor scale.</td>
<td>INT64, UINT64</td>
<td>ND</td>
</tr>
<tr>
<td>bias</td>
<td>Optional input</td>
<td>Convolution bias tensor bias.</td>
<td>FLOAT, INT32</td>
<td>ND</td>
</tr>
<tr>
<td>offset</td>
<td>Optional input</td>
<td>Offset tensor offset (not used).</td>
<td>FLOAT</td>
<td>NCDHW</td>
</tr>
<tr>
<td>y</td>
<td>Output</td>
<td>Output tensor y in the formula.</td>
<td>FLOAT, FLOAT16, BFLOAT16, FLOAT8_E4M3FN, HIFLOAT8</td>
<td>NCDHW</td>
</tr>
<tr>
<td>dtype</td>
<td>Attribute</td>
<td>Specifies the data type of the output y. The supported list includes [0(FLOAT), 1(FLOAT16), 27(BFLOAT16), 34(HIFLOAT8), 36(FLOAT8_E4M3FN)].</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>strides</td>
<td>Attribute</td>
<td>Convolution scan strides. stride_d ∈ [1,1000000], stride_h, stride_w ∈ [1,63].</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>pads</td>
<td>Optional attribute</td>
<td>Padding of the input. pad_top, pad_bottom, pad_left, pad_right ∈ [0,255], pad_head, pad_tail ∈ [0,1000000].</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>dilations</td>
<td>Optional attribute</td>
<td>Spacing between the elements in the convolution kernel. dilation_h, dilation_w ∈ [1,255], dilation_d ∈ [1,1000000].</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>groups</td>
<td>Optional attribute</td>
<td>Number of block connections from input channels to output channels. It must satisfy groups × the in_channels dimension of filter = the in_channels dimension of x. The supported range is [1, 65535].</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>data_format</td>
<td>Optional attribute</td>
<td>Input data format. "NCDHW" is supported.</td>
<td>STRING</td>
<td>-</td>
</tr>
<tr>
<td>offset_x</td>
<td>Optional attribute</td>
<td>Offset in the quantization algorithm, used as the padding value of pad. The supported range is [-128, 127]. When the type of `x` is `HIFLOAT8` or `FLOAT8_E4M3FN`, only 0 can be configured.</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>round_mode</td>
<td>Optional attribute</td>
<td>Specifies the rounding mode of the output. If the data type of the output is HIFLOAT8, this parameter must be 'round'. The default value is 'rint'.</td>
<td>STRING</td>
<td>-</td>
</tr>
<tr>
<td>pad_mode</td>
<td>Optional attribute</td>
<td>Specifies the automatic padding mode of pads. "SPECIFIC", "SAME", "VALID", "SAME_UPPER", and "SAME_LOWER" are supported.</td>
<td>STRING</td>
<td>-</td>
</tr>

</table>

## Constraints

- Ascend 950PR&950DT products:
  - The data type of `x` must be the same as that of `filter`.
  - For the `filter` input, the sizes of `H` and `W` should be in the range [1, 511].
  - For `x`, `filter`, `bias`, `scale`, and `y`, the size of each dimension of each `tensor` should be in the range [1, 1000000].
  - The dimension sizes of `bias` and `scale` should be consistent with the size of the `N` dimension of `filter`.
  - The supported combinations of data types and formats are shown in the following table:

  <table>
  <tr>
  <th style="text-align:center; width:80px">Tensor</th>
  <th style="text-align:center; width:150px">x</th>
  <th style="text-align:center; width:150px">filter</th>
  <th style="text-align:center; width:150px">scale</th>
  <th style="text-align:center; width:100px">bias</th>
  <th style="text-align:center; width:150px">y</th>
  </tr>
  <tr>
  <td rowspan="3" style="text-align:center">Data type</td>
  <td style="text-align:center">INT8</td>
  <td style="text-align:center">INT8</td>
  <td style="text-align:center">INT64/UINT64</td>
  <td style="text-align:center">INT32</td>
  <td style="text-align:center">FLOAT16</td>
  </tr>
  <tr>
  <td style="text-align:center">HIFLOAT8</td>
  <td style="text-align:center">HIFLOAT8</td>
  <td style="text-align:center">INT64/UINT64</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT/FLOAT16/BFLOAT16/HIFLOAT8</td>
  </tr>
  <tr>
  <td style="text-align:center">FLOAT8_E4M3FN</td>
  <td style="text-align:center">FLOAT8_E4M3FN</td>
  <td style="text-align:center">INT64/UINT64</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT/FLOAT16/BFLOAT16/FLOAT8_E4M3FN</td>
  </tr>
  <tr>
  <td rowspan="2" style="text-align:center">Data format</td>
  <td style="text-align:center">NCDHW</td>
  <td style="text-align:center">NCDHW</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">NCDHW</td>
  </tr>
  </table>

- If any parameter exceeds the preceding ranges, the correctness of the operator cannot be guaranteed.
- Due to hardware resource limits, the operator fails to execute in some parameter value combination scenarios. Analyze and troubleshoot the problem based on the log information. If the problem cannot be solved, click [Link](https://www.hiascend.com/support) to obtain technical support.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_quant_conv3d](./examples/arch35/test_aclnn_quant_conv3d.cpp) | Calls the QuantConv3D operator through the [aclnnQuantConvolution](../convolution_forward/docs/aclnnQuantConvolution_en.md) interface.    |
