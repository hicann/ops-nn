# SwigluMxQuant

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |
| <term>Kirin X90 processor series products</term> | × |
| <term>Kirin 9030 processor series products</term> | × |

## Function

- Operator function: This is a fused operator that implements the combined computation of the SwiGLU activation function and dynamic block quantization. It first computes the SwiGLU activation function on the input, then performs block-based dynamic quantization on the result, and outputs a low-precision FP4/FP8 tensor and the corresponding scale factors.

- Formula:

  **Stage 1: SwiGLU activation function**

  <p style="text-align: center">
  gate, hidden = split(x, dim=dim)
  </p>
  <p style="text-align: center">
  swish = sigmoid(gate) * gate
  </p>
  <p style="text-align: center">
  act = swish * hidden
  </p>

  **Stage 2: Dynamic block quantization**

  Dynamic quantization is performed block by block along the axis dimension with blocksize=32. When scale_alg=0 (OCP), the elements {V_i} in each block are quantized by the following formulas:

  <p style="text-align: center">
  shared_exp = floor(log2(max_i(|V_i|))) - emax
  </p>
  <p style="text-align: center">
  mxscale = 2^shared_exp
  </p>
  <p style="text-align: center">
  y_i = cast_to_dst_type(V_i / mxscale, round_mode)
  </p>

  Where emax is the exponent of the largest regular number of the target data type. The values are as follows:

  | Data type | emax |
  | :------: | :--: |
  | FLOAT4_E2M1 | 2 |
  | FLOAT4_E1M2 | 0 |
  | FLOAT8_E4M3FN | 8 |
  | FLOAT8_E5M2 | 15 |

  When scale_alg=1 (cuBLAS, only FP8 types), block scale factor derivation is used. For the formulas, refer to [aclnnSwigluMxQuant](./docs/aclnnSwigluMxQuant_en.md).

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 100px">
  <col style="width: 380px">
  <col style="width: 280px">
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
      <td>Input tensor. Its size along the dimension specified by activate_dim must be a multiple of 2.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>Optional input</td>
      <td>The shape must be 1-dimensional, and shape[0] must be greater than 0 and less than or equal to 256; each value of data must be an integer greater than or equal to 0, the sum of all values must be greater than 0, and the sum must be less than or equal to the total number of rows of x after the SwiGLU operation (that is, the total number of rows of the SwiGLU output y: when activate_dim is the last axis, it equals the product of the remaining axes of x excluding the last axis; when activate_dim is not the last axis, it equals half of the total number of rows of x).</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor after quantization. Its shape is the same as x, with the activate_dim dimension being half of that of x.</td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxscale</td>
      <td>Output</td>
      <td>Scale factor of each quantization block (32 elements per group). The shape along the axis dimension is the value of the corresponding axis of y divided by 32, rounded up, and aligned to an even number; the last dimension is fixed to 2 and stores the scales of two adjacent quantization blocks. When axis=-2 and group_index exists, the shape along the axis dimension is the value of the corresponding axis of y integer-divided by 64 plus group_num.</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activate_dim</td>
      <td>Attribute</td>
      <td>Split dimension of SwiGLU. The value range is [-1, -2, xDim-2, xDim-1] (xDim is the number of dimensions of input x, that is, the last two dimensions). When activate_dim is not the last dimension, swiglu_mode cannot be 1.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>Attribute</td>
      <td>Indicates whether the SwiGLU activation is applied to the first half or the second half of the input. When it is false, the second half is activated. This parameter does not take effect when swiglu_mode=1.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>swiglu_mode</td>
      <td>Attribute</td>
      <td>SwiGLU computation mode: 0=traditional SwiGLU, 1=odd-even interleaved variant, 2=front-back split clamp variant, 3=front-back split sigmoid-clamp variant. The value range is [0, 3]. When activate_dim is not the last dimension, swiglu_mode cannot be 1.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clamp_limit</td>
      <td>Attribute</td>
      <td>Clamp threshold of the SwiGLU variants (swiglu_mode=1/2/3). It must be greater than 0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>glu_alpha</td>
      <td>Attribute</td>
      <td>Sigmoid scale coefficient of the SwiGLU variants (swiglu_mode=1/2).</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>glu_bias</td>
      <td>Attribute</td>
      <td>Bias of the linear part of the SwiGLU variants (swiglu_mode=1/2).</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group_mode</td>
      <td>Attribute</td>
      <td>Takes effect when group_index exists. 0=count mode, 1=cumsum mode. Currently, only 0 is supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Attribute</td>
      <td>Quantization axis. Block-based quantization is performed along this dimension. The value range is [-1, -2, xDim-2, xDim-1] (xDim is the number of dimensions of input x, that is, the last two dimensions).</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Attribute</td>
      <td>Target quantization type: 40=FP4_E2M1, 41=FP4_E1M2, 36=FP8_E4M3FN, 35=FP8_E5M2.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Attribute</td>
      <td>Rounding mode, used for type conversion during quantization. The value is "rint", "floor", or "round".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>Attribute</td>
      <td>Scale algorithm: 0=OCP, 1=cuBLAS, 2=RNE (reserved). Currently, only 0 and 1 are supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_dtype_value</td>
      <td>Attribute</td>
      <td>Reserved parameter. It does not take effect in the current version.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- Input x supports 2-7 dimensional tensors. Its size along the dimension specified by activate_dim must be divisible by 2.
- activate_dim and axis must take the last two dimensions of input x. The value range is [-1, -2, xDim-2, xDim-1] (xDim is the number of dimensions of input x).
- When activate_dim is not the last dimension, swiglu_mode cannot be 1.
- When swiglu_mode is 2 or 3, axis must be -1.
- When activate_dim = -2 or axis = -2 and group_index exists, input x must be 2-dimensional.
- When dst_type is an FP4 type, the last dimension of the output shape must be divisible by 2.
- When dst_type is an FP4 type, scale_alg must be 0.
- When group_index exists, it must be 1-dimensional, and shape[0] must be greater than 0 and less than or equal to 256; each value of data must be an integer greater than or equal to 0, the sum of all values must be greater than 0, and the sum must be less than or equal to the total number of rows of x after the SwiGLU operation (that is, the total number of rows of the SwiGLU output y: when activate_dim is the last axis, it equals the product of the remaining axes of x excluding the last axis; when activate_dim is not the last axis, it equals half of the total number of rows of x).
- FP8 output types support only the "rint" rounding mode.
- Empty tensors are not supported: when any dimension in the shape of input x, optional input group_index, or outputs y and mxscale is 0 (the number of elements is 0), operator validation fails and an error is reported.
- Input x must be 2-7 dimensional. rank=0 scalar tensors are not supported.
- The caller allocates memory for outputs y/mxscale according to the shape inferred by the interface, and the operator writes the quantization result completely back to the output tensors. The parts of the output beyond the range covered by the sum of all elements of group_index are not cleared and contain garbage data.
- The memory spaces of inputs and outputs must not overlap. In-place modification (inplace) scenarios are not supported.
- The operator does not modify any input data during execution.

## Invocation Description

| Invocation method | Invocation sample | Description |
| :------- | :------- | :--- |
| aclnn invocation | [test_aclnn_swiglu_mx_quant](./examples/arch35/test_aclnn_swiglu_mx_quant.cpp) | Invokes the SwigluMxQuant operator through the [aclnnSwigluMxQuant](./docs/aclnnSwigluMxQuant_en.md) interface. |
