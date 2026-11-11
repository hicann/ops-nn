# Conv3DV2

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
<td style="text-align:center">√</td>
</tr>
<tr>
<td><term>Atlas A2 products</term></td>
<td style="text-align:center">√</td>
</tr>
</table>

## Function

- Description: Implements the 3D convolution function.

- Formula:

  - Assume that the shape of the input (`x`) is $(N, C_{\text{in}}, D, H, W)$, the shape of (`filter`) is $(C_{\text{out}}, C_{\text{in}}, K_d, K_h, K_w)$, and the shape of the output (`y`) is $(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})$

  - For inputs of the INT8 type, the output is expressed as:

  $$
    \text{y}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \text{scale} \times \sum_{k = 0}^{C_{\text{in}} - 1} \text{filter}(C_{\text{out}_j}, k) \star \text{x}(N_i, k)
  $$

  - For inputs of other data types, the output is expressed as:

  $$
    \text{y}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{filter}(C_{\text{out}_j}, k) \star \text{x}(N_i, k)
  $$

  Here, $\star$ denotes the convolution computation, and dilated convolution (`dilations` > 1) and grouped convolution (`groups` > 1) are supported. $N$ denotes `batch size`, $C$ denotes the number of channels, and $D$, $H$, and $W$ denote depth, height, and width respectively. The formulas for computing the corresponding output dimensions are as follows:

  $$
    D_{\text{out}} = (D + \text{pad\_head} + \text{pad\_tail} - (\text{dilation\_d} \times (K_d - 1) + 1)) / \text{stride\_d} + 1 \\
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
<td>FLOAT16, FLOAT, BFLOAT16, INT8, HIFLOAT8</td>
<td>NCDHW, NDHWC, NDC1HWC0</td>
</tr>
<tr>
<td>filter</td>
<td>Input</td>
<td>Convolution weight tensor filter in the formula.</td>
<td>FLOAT16, FLOAT, BFLOAT16, INT8, HIFLOAT8</td>
<td>NCDHW, DHWCN, FRACTAL_Z_3D</td>
</tr>
<tr>
<td>bias</td>
<td>Optional input</td>
<td>Convolution bias tensor bias.</td>
<td>FLOAT16, FLOAT, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>scale</td>
<td>Optional input</td>
<td>Scaling factor tensor scale.</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>offset</td>
<td>Optional input</td>
<td>Offset tensor offset (unused).</td>
<td>FLOAT</td>
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
<td>NCDHW, NDHWC, NDC1HWC0</td>
</tr>
<tr>
<td>strides</td>
<td>Attribute</td>
<td>Convolution scanning stride; stride_d ∈ [1,1000000], stride_h, stride_w ∈ [1,63].</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>pads</td>
<td>Optional attribute</td>
<td>Padding of the input; pad_h, pad_w ∈ [0,255], paddingD ∈ [0,1000000].</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>dilations</td>
<td>Optional attribute</td>
<td>Spacing between elements in the convolution kernel; dilation_h, dilation_w ∈ [1,255], dilation_d ∈ [1,1000000].</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>groups</td>
<td>Optional attribute</td>
<td>Number of block connections from input channels to output channels. It must satisfy groups × the in_channels dimension of filter = the in_channels dimension of x. Supported range: [1, 65535].</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>data_format</td>
<td>Optional attribute</td>
<td>Input data format. "NCDHW", "NDHWC", and "NDC1HWC0" are supported.</td>
<td>STRING</td>
<td>-</td>
</tr>
<tr>
<td>offset_x</td>
<td>Optional attribute</td>
<td>Offset offset_x in the quantization algorithm (unused).</td>
<td>INT32</td>
<td>-</td>
</tr>
<tr>
<td>pad_mode</td>
<td>Optional attribute</td>
<td>Mode of automatic pads filling. "SPECIFIC", "SAME", "VALID", "SAME_UPPER", and "SAME_LOWER" are supported.</td>
<td>STRING</td>
<td>-</td>
</tr>
<tr>
<td>enable_hf32</td>
<td>Optional attribute</td>
<td>Whether to enable hf32 computation. true and false are supported. hf32 computation can be correctly enabled by configuring this parameter to true only when `x`, `filter`, `bias`, and `y` are all of the `FLOAT` type.</td>
<td>BOOL</td>
<td>-</td>
</tr>
</table>

- Atlas A2 products and Atlas A3 products:
  - The `HIFLOAT8` data type is not supported.
  - The `scale` and `offset` parameters support only the `FLOAT` type.
  - When the input is of the `INT8` data type, `bias` is a required input, and `groups` supports only 1.
  - `filter` and `y` do not support the `DHWCN` data format.
  - The `pad_mode` attribute is not supported.
  - When the `Pointwise` branch conditions are not met, `x` supports `NDC1HWC0`, and `filter` supports `FRACTAL_Z_3D`.

## Constraints

- Atlas A2 products and Atlas A3 products:
  - The range of the `H` and `W` dimensions of `filter`: [1,511].
  - An empty `tensor` is not supported.
  - When `groups` is 1, all `dilation` values are 1, all `padding` values are 0, `filter` has no dimension of 1, the `D`*`H`*`W` of `x` is less than 65536, and `bias` is `FLOAT`, the `Pointwise` branch is entered, and the `NCDHW` format can be used.

  <table>
  <tr>
  <th style="text-align:center; width:80px">Tensor</th>
  <th style="text-align:center; width:120px">x</th>
  <th style="text-align:center; width:120px">filter</th>
  <th style="text-align:center; width:100px">bias</th>
  <th style="text-align:center; width:80px">scale</th>
  <th style="text-align:center; width:80px">offset</th>
  <th style="text-align:center; width:120px">y</th>
  </tr>
  <tr>
  <td rowspan="4" style="text-align:center">Data type</td>
  <td style="text-align:center">FLOAT16</td>
  <td style="text-align:center">FLOAT16</td>
  <td style="text-align:center">FLOAT16</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT16</td>
  </tr>
  <tr>
  <td style="text-align:center">BFLOAT16</td>
  <td style="text-align:center">BFLOAT16</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">BFLOAT16</td>
  </tr>
  <tr>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  </tr>
  <tr>
  <td style="text-align:center">INT8</td>
  <td style="text-align:center">INT8</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">BFLOAT16</td>
  </tr>
  <tr>
  <td style="text-align:center">Data format</td>
  <td style="text-align:center">NCDHW, NDC1HWC0</td>
  <td style="text-align:center">NCDHW, FRACTAL_Z_3D</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">NCDHW, NDC1HWC0</td>
  </tr>
  </table>

- Ascend 950PR&950DT products:
  - When the data type of `x` is `HIFLOAT8`, the data type of `filter` must be consistent with that of `x`, and the formats of both `x` and `filter` support only `NCDHW`.
  - For the `filter` input, the sizes of `H` and `W` should be within the range [1, 511].
  - For `x`, `filter`, `bias`, `scale`, and `y`, the size of each dimension of each `tensor` should be within the range [1, 1000000], except that the D-axis dimensions (the `d` of `x`, the `kernel_d` of `filter`, and the `d` of `y`) support up to 10000000.
  - When the data type of `x` is `INT8`, the data format of `x` supports only `NCDHW`, the data format of `filter` supports `NCDHW` and `FRACTAL_Z_3D`, and the data format of `output` supports only `NDHWC`.
  - The supported combinations of data types and formats are as follows:
  <table>
  <tr>
  <th style="text-align:center; width:80px">Tensor</th>
  <th style="text-align:center; width:150px">x</th>
  <th style="text-align:center; width:150px">filter</th>
  <th style="text-align:center; width:100px">bias</th>
  <th style="text-align:center; width:100px">scale</th>
  <th style="text-align:center; width:150px">y</th>
  </tr>
  <tr>
  <td rowspan="6" style="text-align:center">Data type</td>
  <td style="text-align:center">FLOAT16</td>
  <td style="text-align:center">FLOAT16</td>
  <td style="text-align:center">FLOAT16</td>
  <td style="text-align:center">/</td>
  <td style="text-align:center">FLOAT16</td>
  </tr>
  <tr>
  <td style="text-align:center">BFLOAT16</td>
  <td style="text-align:center">BFLOAT16</td>
  <td style="text-align:center">BFLOAT16</td>
  <td style="text-align:center">/</td>
  <td style="text-align:center">BFLOAT16</td>
  </tr>
  <tr>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">/</td>
  <td style="text-align:center">FLOAT</td>
  </tr>
  <tr>
  <td style="text-align:center">HIFLOAT8</td>
  <td style="text-align:center">HIFLOAT8</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">/</td>
  <td style="text-align:center">HIFLOAT8</td>
  </tr>
  <tr>
  <td style="text-align:center">INT8</td>
  <td style="text-align:center">INT8</td>
  <td style="text-align:center">FLOAT/FLOAT16</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">FLOAT16</td>
  </tr>
  <tr>
  <td style="text-align:center">INT8</td>
  <td style="text-align:center">INT8</td>
  <td style="text-align:center">FLOAT/BFLOAT16</td>
  <td style="text-align:center">FLOAT</td>
  <td style="text-align:center">BFLOAT16</td>
  </tr>
  <tr>
  <td rowspan="4" style="text-align:center">Data format</td>
  <td style="text-align:center">NCDHW</td>
  <td style="text-align:center">NCDHW</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">/</td>
  <td style="text-align:center">NCDHW</td>
  </tr>
  <tr>
  <td style="text-align:center">NDHWC</td>
  <td style="text-align:center">DHWCN</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">/</td>
  <td style="text-align:center">NDHWC</td>
  </tr>
  <tr>
  <td style="text-align:center">NCDHW</td>
  <td style="text-align:center">NCDHW</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">NDHWC</td>
  </tr>
  <tr>
  <td style="text-align:center">NCDHW</td>
  <td style="text-align:center">FRACTAL_Z_3D</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">ND</td>
  <td style="text-align:center">NDHWC</td>
  </tr>
  </table>

- If any parameter exceeds the preceding ranges, the correctness of the operator cannot be guaranteed.
- Due to hardware resource limitations, the operator may fail to execute in scenarios with certain parameter value combinations. Analyze and troubleshoot the problem based on the log information. If the problem cannot be resolved, click [Link](https://www.hiascend.com/support) to obtain technical support.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_conv3d_v2](./examples/test_aclnn_conv3d_v2.cpp) | Invokes the computation of the Conv3DV2 operator for non-`INT8` data types through the [aclnnConvolution](../convolution_forward/docs/aclnnConvolution_en.md) interface.    |
| aclnn invocation | [test_aclnn_quant_conv3d](../quant_conv3d/examples/arch35/test_aclnn_quant_conv3d.cpp) | Invokes the computation of the Conv3DV2 operator for the `INT8` data type through the [aclnnQuantConvolution](../convolution_forward/docs/aclnnQuantConvolution_en.md) interface.    |
