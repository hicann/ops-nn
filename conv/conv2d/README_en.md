# Conv2D

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

- Operator function: implements 2D convolution.

- Formula:

  - Assume that the shape of the input (`x`) is $(N, C_{\text{in}}, H, W)$, the shape of (`filter`) is $(C_{\text{out}}, C_{\text{in}}, K_h, K_w)$, and the shape of the output (`y`) is $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$

  - The output is expressed as:

  $$
    \text{y}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{filter}(C_{\text{out}_j}, k) \star \text{x}(N_i, k)
  $$

  Here $\star$ denotes the convolution computation, which supports dilated convolution (`dilations` > 1) and grouped convolution (`groups` > 1). $N$ denotes the `batch size`, $C$ denotes the number of channels, and $H$ and $W$ denote the height and width respectively. The output dimensions are computed as follows:

  $$
    H_{\text{out}} = (H + \text{pad\_top} + \text{pad\_bottom} - (\text{dilation\_h} \times (K_h - 1) + 1)) / \text{stride\_h} + 1 \\
    W_{\text{out}} = (W + \text{pad\_left} + \text{pad\_right} - (\text{dilation\_w} \times (K_w - 1) + 1)) / \text{stride\_w} + 1
  $$

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Input tensor x in the formula. | FLOAT16, FLOAT, INT8, BFLOAT16, HIFLOAT8 | NCHW, NHWC |
| filter | Input | Convolution weight tensor filter in the formula. | FLOAT16, FLOAT, INT8, BFLOAT16, HIFLOAT8 | NCHW, HWCN |
| bias | Optional input | Convolution bias tensor bias. | FLOAT16, FLOAT, BFLOAT16, INT32 | ND |
| offset_w | Optional input | Quantization offset tensor offset_w (unused). | INT8 | ND |
| y | Output | Output tensor y in the formula. | FLOAT16, FLOAT, INT32, BFLOAT16, HIFLOAT8 | NCHW, NHWC |
| strides | Attribute | Convolution scan stride. An integer list of length 4, where the n and in_channels dimensions must be 1; when format is "NHWC", the shape is [1, stride_h, stride_w, 1], and when format is "NCHW", the shape is [1, 1, stride_h, stride_w]. | INT32 | - |
| pads | Attribute | Padding of the input, including pad_top, pad_bottom, pad_left, and pad_right. | INT32 | - |
| dilations | Optional attribute | Spacing between elements in the convolution kernel. An integer list of length 4, where the n and in_channels dimensions must be 1; when format is "NHWC", the shape is [1, dilation_h, dilation_w, 1], and when format is "NCHW", the shape is [1, 1, dilation_h, dilation_w]. The default value is [1, 1, 1, 1]. | INT32 | - |
| groups | Optional attribute | Number of blocked connections from input channels to output channels. It must satisfy groups × the in_channels dimension of filter = the in_channels dimension of x, and both in_channels and out_channels must be divisible by groups. The default value is 1. | INT32 | - |
| data_format | Optional attribute | Input data format, supporting "NCHW" and "NHWC". The default value is "NHWC". | STRING | - |
| offset_x | Optional attribute | Offset offset_x in the quantization algorithm (unused). The default value is 0. The supported range is [-128, 127]. | INT32 | - |

## Constraints

- Ascend 950PR/Ascend 950DT:

  - The size of each dimension of each `tensor` in `x`, `filter`, `bias`, `offset_w`, and `y` should be within the range [1, 1000000].
  - The values of `strides` and `dilations` should be within the range [1, 1000000].
  - The values of `pads` should be within the range [0, 1000000].
  - The supported combinations of data types and formats are as follows:

  | x | filter | bias | offset_w | y |
  | :---: | :---: | :---: | :---: | :---: |
  | FLOAT16 | FLOAT16 | FLOAT16 | INT8 | FLOAT16 |
  | BFLOAT16 | BFLOAT16 | BFLOAT16 | INT8 | BFLOAT16 |
  | FLOAT | FLOAT | FLOAT | INT8 | FLOAT |
  | HIFLOAT8 | HIFLOAT8 | FLOAT | INT8 | HIFLOAT8 |
  | INT8 | INT8 | INT32 | INT8 | INT32 |
  | NCHW | NCHW | ND | ND | NCHW |
  | NHWC | HWCN | ND | ND | NHWC |

  `HIFLOAT8` supports only `NCHW` (`x`/`filter`/`y` are `NCHW`, and `bias`/`offset_w` are `ND`). The other data types support both Format rows in the table above.

- If any parameter exceeds the ranges above, the correctness of the operator cannot be guaranteed.
- Due to hardware resource limits, the operator fails to execute under some parameter value combinations. Analyze and troubleshoot the problem based on the log information. If the problem cannot be solved, click [Link](https://www.hiascend.com/support) to obtain technical support.

## Invocation Description

| Invocation method | Invocation sample | Description |
| :--- | :--- | :--- |
| Graph mode | [test_geir_conv2d](./examples/arch35/test_geir_conv2d.cpp) | Calls the Conv2D operator by constructing a graph through the [operator IR](./op_graph/conv2d_proto.h). |
