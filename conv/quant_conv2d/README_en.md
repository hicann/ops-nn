# QuantConv2D

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>| × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: Implements 2D quantized convolution.
- Formula:

  - Assume that the shape of the input (`x`) is $(N, C_{\text{in}}, H, W)$, the shape of (`filter`) is $(C_{\text{out}}, C_{\text{in}}, K_h, K_w)$, and the shape of the output (`y`) is $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$.
  - The output is represented as:

  $$
  \text{y}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \text{scale} \times \sum_{k = 0}^{C_{\text{in}} - 1} \text{filter}(C_{\text{out}_j}, k) \star \text{x}(N_i, k)
  $$

  where $\star$ denotes the convolution computation, and dilated convolution (`dilations` > 1) and grouped convolution (`groups` > 1) are supported. $N$ represents the batch size, $C$ represents the number of channels, and $H$ and $W$ represent the height and width respectively. The formulas for computing the corresponding output dimensions are as follows:

  $$
  H_{\text{out}} = (H + \text{pad\_top} + \text{pad\_bottom} - (\text{dilation\_h} \times (K_h - 1) + 1)) / \text{stride\_h} + 1 \\
    W_{\text{out}} = (W + \text{pad\_left} + \text{pad\_right} - (\text{dilation\_w} \times (K_w - 1) + 1)) / \text{stride\_w} + 1
  $$

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Input tensor x in the formula. | FLOAT8_E4M3FN, INT8, HIFLOAT8 | NCHW |
| filter | Input | Convolution weight tensor filter in the formula. | FLOAT8_E4M3FN, INT8, HIFLOAT8 | NCHW |
| scale | Input | Scaling factor tensor scale. | INT64, UINT64 | ND |
| bias | Optional input | Convolution bias tensor bias. | FLOAT, INT32 | ND |
| offset | Optional input | Offset tensor offset (unused). | FLOAT | NCHW |
| y | Output | Output tensor y in the formula. | FLOAT, FLOAT16, BFLOAT16, FLOAT8_E4M3FN, HIFLOAT8 | NCHW |
| dtype | Attribute | Data type of the output y. The supported list includes [0(FLOAT), 1(FLOAT16), 27(BFLOAT16), 34(HIFLOAT8), 36(FLOAT8_E4M3FN)]. | INT32 | - |
| strides | Attribute | Convolution scan strides, including stride_h and stride_w.| INT32 | - |
| pads | Required attribute | Padding of the input, including pad_top, pad_bottom, pad_left, and pad_right. | INT32 | - |
| dilations | Optional attribute | Spacing of the elements in the convolution kernel, including dilation_h and dilation_w.| INT32 | - |
| groups | Optional attribute | Number of block connections from the input channels to the output channels. It must satisfy groups × the in_channels dimension of filter = the in_channels dimension of x. Supported range: [1, 65535]. | INT32 | - |
| data_format | Optional attribute | Input data format. Only "NCHW" is supported. | STRING | - |
| offset_x | Optional attribute | Offset in the quantization algorithm, used as the padding value for pad. Supported range: [-128, 127]. When the type of `x` is `HIFLOAT8` or `FLOAT8_E4M3FN`, only 0 can be configured. | INT32 | - |
| round_mode | Optional attribute | Rounding mode of the output. If the data type of the output is HIFLOAT8, this parameter must be 'round'. Default: 'rint'. | STRING | - |

## Constraints

- Ascend 950PR&950DT products:

  - The data type of `x` must be the same as that of `filter`.
  - Each dimension of each group of `tensor` in `x`, `filter`, `bias`, `scale`, and `y` should be within the range [1, 1000000].
  - The values of `strides` and `dilations` should be within the range [1, 1000000].
  - The values of `pads` should be within the range [0, 1000000].
  - The dimension size of `bias` and `scale` should be the same as the size of the `N` dimension of `filter`.
  - The supported combinations of data types and formats are as follows:

  | x | filter | scale | bias | y |
  | :---: | :---: | :---: | :---: | :---: |
  | INT8 | INT8 | INT64/UINT64 | INT32 | FLOAT16 |
  | HIFLOAT8 | HIFLOAT8 | INT64/UINT64 | FLOAT | FLOAT/FLOAT16/BFLOAT16/HIFLOAT8 |
  | FLOAT8_E4M3FN | FLOAT8_E4M3FN | INT64/UINT64 | FLOAT | FLOAT/FLOAT16/BFLOAT16/FLOAT8_E4M3FN |
  | NCHW | NCHW | ND | ND | NCHW |

- If any parameter exceeds the preceding ranges, the correctness of the operator cannot be guaranteed.
- Due to hardware resource limitations, the operator may fail to execute in some parameter value combination scenarios. Analyze and troubleshoot the problem based on the log information. If the problem cannot be solved, click [Link](https://www.hiascend.com/support) to obtain technical support.

## Invocation Description

| Invocation method | Invocation sample | Description |
| :--- | :--- | :--- |
| Graph mode | [test_geir_quant_conv2d](./examples/arch35/test_geir_quant_conv2d.cpp) | Invoke the QuantConv2D operator by building a graph through the [operator IR](./graph/quant_conv2d_proto.h). |
