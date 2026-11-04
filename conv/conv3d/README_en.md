# Conv3D

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:-----|:-------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: Performs 3D convolution on a 5D input tensor. It supports dilated convolution (`dilations` > 1), grouped convolution (`groups` > 1), and bias (`bias`) accumulation, and computes `y = CONV(x, filter) + bias`.

- Formula:

  - Assume the shape of the input (`x`) is $(N, C_{\text{in}}, D, H, W)$ (NCDHW format) or $(N, D, H, W, C_{\text{in}})$ (NDHWC format), the shape of (`filter`) is $(C_{\text{out}}, C_{\text{in}}, K_d, K_h, K_w)$ (NCDHW format) or $(K_d, K_h, K_w, C_{\text{in}}, C_{\text{out}})$ (DHWCN format), and the shape of the output (`y`) is $(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})$ (NCDHW format) or $(N, D_{\text{out}}, H_{\text{out}}, W_{\text{out}}, C_{\text{out}})$ (NDHWC format).

  - The output is expressed as:

  $$
    \text{y}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{filter}(C_{\text{out}_j}, k) \star \text{x}(N_i, k)
  $$

  Where $\star$ denotes the 3D convolution computation. The output dimensions are computed as follows:

  $$
    D_{\text{out}} = (D + \text{pad\_head} + \text{pad\_tail} - \text{dilation\_d} \times (K_d - 1) - 1) / \text{stride\_d} + 1 \\
    H_{\text{out}} = (H + \text{pad\_top} + \text{pad\_bottom} - \text{dilation\_h} \times (K_h - 1) - 1) / \text{stride\_h} + 1 \\
    W_{\text{out}} = (W + \text{pad\_left} + \text{pad\_right} - \text{dilation\_w} \times (K_w - 1) - 1) / \text{stride\_w} + 1
  $$

- Output dtype inference: the dtype of output `y` is the same as that of input `x`; when `x` is INT8, the output is INT32.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
|:-------|:--------------|:-----|:--------|:--------|
| x | Input | Input feature tensor of the convolution, 5D. | FLOAT16, BFLOAT16, FLOAT, HIFLOAT8, INT8 | NCDHW, NDHWC |
| filter | Input | Convolution kernel weight tensor, 5D, with the same dtype as `x`. | FLOAT16, BFLOAT16, FLOAT, HIFLOAT8, INT8 | NCDHW, DHWCN |
| bias | Optional input | Convolution bias tensor, 1D, with shape `[C_out]`. | FLOAT16, FLOAT, BFLOAT16, INT32 | ND |
| offset_w | Optional input | Quantization offset tensor (reserved, currently unused). | INT8 | ND |
| y | Output | Output feature tensor of the convolution, 5D. | FLOAT16, BFLOAT16, FLOAT, HIFLOAT8, INT32 | NCDHW, NDHWC |
| strides | Attribute | Sliding window stride. It is a list of length 5, and the order is determined by `data_format`; only the D/H/W dimensions participate in output shape inference (the N/C dimensions do not participate and are conventionally set to 1). | ListInt, required | - |
| pads | Attribute | Padding of each dimension. Lengths 1/3/6 are supported: length 6 is explicit per dimension, in the order `[pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]`; length 3 is `[pad_d, pad_h, pad_w]`, broadcast to both sides of the corresponding dimensions; length 1 uses the same value for all dimensions. | ListInt, required | - |
| dilations | Attribute | Dilation rate of dilated convolution. It is a list of length 5, and the order is determined by `data_format`. The default is `[1, 1, 1, 1, 1]`. | ListInt | - |
| groups | Attribute | Number of groups in grouped convolution. `C_in` and `C_out` must be divisible by `groups`. The default is 1. When groups is 1 and `C_in` is divisible by the `C_in` of filter, the value takes effect implicitly as `C_in / C_in(filter)` (aligned with RT1.0; the number of groups is inferred in the actual execution stage). | Int | - |
| data_format | Attribute | Input/output data format. The value is `NCDHW` or `NDHWC`. The default is `NDHWC`. | String | - |
| offset_x | Attribute | Offset in the quantization algorithm (reserved, currently unused). The default is 0. | Int | - |

Supported dtype/format combinations:

| x | filter | bias | y |
|:--|:-------|:-----|:--|
| float16 + NCDHW | float16 + NCDHW | float16 + ND | float16 + NCDHW |
| bfloat16 + NCDHW | bfloat16 + NCDHW | bfloat16 + ND | bfloat16 + NCDHW |
| float + NCDHW | float + NCDHW | float + ND | float + NCDHW |
| hifloat8 + NCDHW | hifloat8 + NCDHW | float + ND | hifloat8 + NCDHW |
| float16 + NDHWC | float16 + DHWCN | float16 + ND | float16 + NDHWC |
| bfloat16 + NDHWC | bfloat16 + DHWCN | bfloat16 + ND | bfloat16 + NDHWC |
| float + NDHWC | float + DHWCN | float + ND | float + NDHWC |
| int8 + NCDHW | int8 + NCDHW | int32 + ND | int32 + NCDHW |
| int8 + NDHWC | int8 + DHWCN | int32 + ND | int32 + NDHWC |

## Constraints

- The shapes of `x`, `filter`, and `y` are all 5D (`x` supports dynamic shape `-1` and dynamic rank `-2`).
- The lengths of `strides` and `dilations` must be 5, and the D/H/W dimensions of `strides` must be greater than 0; the D dimension of `dilations` is in the range [1, 1000000], and the H/W dimensions are in the range [1, 255].
- Lengths 1/3/6 of `pads` are supported (length 3 is broadcast to both sides of the corresponding dimensions as `[pad_d, pad_h, pad_w]`, and length 1 uses the same value for all dimensions); for static shapes, the values must not be less than 0.
- `data_format` supports only `NCDHW` or `NDHWC`; the format of `x` supports only NCDHW/NDHWC, `filter` supports only NCDHW/NDHWC/DHWCN, and `y` supports only NCDHW/NDHWC.
- The D/H/W dimensions of `filter` must not be less than 1.
- The input size after padding must not be less than the kernel size (validated when both `x` and `filter` have static shapes).
- Zero tensor: when any dimension of `x` is 0 or the N/C dimensions of `filter` are 0, the output must be inferable as a zero tensor; otherwise the invocation is blocked.
- Channel constraints: for static shapes, `C_in` must be divisible by the `C_in` dimension of `filter`; when `groups` is explicitly set (not 1), it must equal `C_in / C_in(filter)`.
- When the frontend (TensorFlow) passes the `padding = SAME/VALID` attribute, `pads` is dynamically computed based on the input size (dynamic dimensions are recorded as -1 for SAME) or set to 0 (VALID).

## Invocation Description

| Invocation method | Invocation sample | Description |
|:--------|:--------|:-----|
| Graph-mode invocation | [test_geir_conv3d](examples/test_geir_conv3d.cpp) | Invokes the Conv3D operator through the operator IR-based [graph construction](op_graph/conv3d_proto.h) method. |
