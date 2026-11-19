# DepthwiseConv2D

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: implements the 2D depthwise convolution. Each input channel is convolved with only its corresponding kernel, which is equivalent to Conv2D with `groups = number of channels`.

- Formula:

  - Assume that the shape of the input (`x`) is $(N, C_{\text{in}}, H, W)$, the shape of `filter` is $(C_{\text{in}} \times M, 1, K_h, K_w)$ (NCHW format) or $(K_h, K_w, C_{\text{in}}, M)$ (HWCN format), $M$ is the channel multiplier, and the shape of the output (`y`) is $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$.

  - The output is expressed as:

  $$
    C_{\text{out}} = C_{\text{in}} \times M
  $$

  $$
    H_{\text{out}} = (H + pad\_top + pad\_bottom - (\text{dilation\_h} \times (K_h - 1) + 1)) / \text{stride\_h} + 1 \\
    W_{\text{out}} = (W + pad\_left + pad\_right - (\text{dilation\_w} \times (K_w - 1) + 1)) / \text{stride\_w} + 1
  $$

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Input tensor x in the formula. | FLOAT16, FLOAT, BFLOAT16, HIFLOAT8 | NCHW, NHWC |
| filter | Input | Convolution weight tensor filter in the formula. | FLOAT16, FLOAT, BFLOAT16, HIFLOAT8 | NCHW, HWCN |
| bias | Optional input | Convolution bias tensor bias. | FLOAT16, FLOAT, BFLOAT16 | ND |
| offset_w | Optional input | Quantization offset tensor offset_w (not supported). | INT8 | ND |
| y | Output | Output tensor y in the formula. | FLOAT16, FLOAT, BFLOAT16, HIFLOAT8 | NCHW, NHWC |
| strides | Attribute | Convolution scan strides. A list of 4 integers. When format is "NHWC", the shape is [1, stride_h, stride_w, 1]; when format is "NCHW", the shape is [1, 1, stride_h, stride_w]. | INT32 | - |
| pads | Attribute | A list of 4 integers with the shape [pad_top, pad_bottom, pad_left, pad_right]. | INT32 | - |
| dilations | Optional attribute | Spacing between the elements in the convolution kernel. A list of 4 integers. When format is "NHWC", the shape is [1, dilation_h, dilation_w, 1]; when format is "NCHW", the shape is [1, 1, dilation_h, dilation_w]. Default: [1, 1, 1, 1]. | INT32 | - |
| data_format | Optional attribute | Input data format. "NCHW" and "NHWC" are supported. Default: "NHWC". | STRING | - |
| offset_x | Optional attribute | Offset in the quantization algorithm. Default: 0. Supported range: [-128, 127]. | INT32 | - |

## Constraints

- Ascend 950PR&950DT products:

  - `x` and `filter` support only 4D; the format of `x` supports NCHW and NHWC, and the format of `filter` supports NCHW and HWCN.
  - The data types of `x` and `filter` must be the same.
  - `kh` and `kw` of `filter` cannot be 0.
  - The supported data type and format combinations are as follows:

  | x | filter | bias | y |
  | :---: | :---: | :---: | :---: |
  | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 |
  | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 |
  | FLOAT | FLOAT | FLOAT | FLOAT |
  | HIFLOAT8 | HIFLOAT8 | FLOAT | HIFLOAT8 |
  | NCHW | NCHW | ND | NCHW |
  | NHWC | HWCN | ND | NHWC |

  `HIFLOAT8` supports only `NCHW` (`x`/`filter`/`y` are `NCHW`, and `bias` is `ND`). Other data types support both format rows in the table above.

- When `x` is dynamic shape, `filter` must be static shape; when `x` is static shape, `filter` can have a dimension of -1, and the corresponding output dimension is inferred as -1.
- If any parameter exceeds the above ranges, the correctness of the operator cannot be guaranteed.

## Invocation Description

| Invocation method | Invocation sample | Description |
| :--- | :--- | :--- |
| Graph mode | [test_geir_depthwise_conv2d](./examples/arch35/test_geir_depthwise_conv2d.cpp) | Invokes the DepthwiseConv2D operator by constructing a graph through the [operator IR](./op_graph/depthwise_conv2d_proto.h). |
