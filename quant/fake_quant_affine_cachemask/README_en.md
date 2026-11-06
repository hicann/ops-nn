# FakeQuantAffineCachemask

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|<term>Ascend 950PR&950DT products</term>|√|
|<term>Atlas A3 products</term>|√|
|<term>Atlas A2 products</term>|√|
|<term>Atlas 200I/500 A2 inference products</term>|×|
|<term>Atlas inference products</term>|×|
|<term>Atlas training products</term>|×|
|<term>Kirin X90 processor series products</term>|√|
|<term>Kirin 9030 processor series products</term>|√|

## Function

- Operator function: For the input data x, performs fake quantization on the input x along the specified axis using scale and zero_point, updates the value range of the fake quantization output according to quant_min and quant_max, and finally returns the result y and the corresponding position mask mask.
- Formula (consistent with the measured semantics of ATen `fake_quantize_per_*_affine_cachemask`, branched by the zero_point data type; in the implementation, zero_point is promoted to the fp32 domain for computation, and x/scale uses true division):

  When zero_point is INT32 (two-step):

  $$
  qval = Round(x / scale) + zeroPoint
  $$

  When zero_point is FLOAT/FLOAT16 (one-step):

  $$
  qval = Round(x / scale + zeroPoint)
  $$

  $$
  y = (Min(quantMax, Max(quantMin, qval)) - zeroPoint) * scale
  $$

  $$
  mask = (qval >= quantMin)   \&  (qval <= quantMax)
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
  <col style="width: 250px">
  <col style="width: 200px">
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
      <td>It is `x` in the formulas.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>It is `scale` in the formulas, representing the scaling coefficient of the fake quantization of the input. The shape supports 0-dim (scalar, only for the per-tensor interface and graph mode) or 1-dim; for the per-channel interface, the size is equal to the length of `x` on the axis dimension and the dtype is the same as `x`; for the per-tensor interface, it is a single element.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_point</td>
      <td>Input</td>
      <td>It is `zero_point` in the formulas, representing the zero-based parameter of the fake quantization of the input. The shape supports 0-dim (scalar, only for the per-tensor interface and graph mode) or 1-dim; for the per-channel interface, the size is equal to the length of `x` on the axis dimension. The aclnn per-channel interface supports INT32, FLOAT16 and FLOAT; the per-tensor interface supports only INT32; the GEIR graph mode additionally supports BFLOAT16 (see the product difference notes below for the product domains of floating-point zero_point and cross combinations with `x`).</td>
      <td>INT32 (per-tensor aclnn); INT32, FLOAT, FLOAT16 (per-channel aclnn); INT32, FLOAT, FLOAT16, BFLOAT16 (GEIR graph mode)</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Attribute</td>
      <td>Represents the computation dimension (this parameter exists only in the per-channel interface). The valid range is [-x.dim(), x.dim()-1]; when x is 0-dim (scalar), only 0 or -1 is supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_min</td>
      <td>Attribute</td>
      <td>Represents the minimum value of the input data after fake quantization. Must satisfy quant_min not greater than quant_max, with an absolute value less than 2^24.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_max</td>
      <td>Attribute</td>
      <td>Represents the maximum value of the input data after fake quantization. Must satisfy quant_min not greater than quant_max, with an absolute value less than 2^24.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>It is `y` in the formulas. The dimensions are consistent with those of the computation input `x`, and the dtype is the same as `x`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>Output</td>
      <td>It is `mask` in the formulas. The dimensions are consistent with those of the computation input `x`.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

Product difference notes (by product):

- <term>Ascend 950PR&950DT products</term>: The GEIR graph mode supports the BFLOAT16 type and cross combinations of floating-point zero_point with `x`.
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Support FLOAT, FLOAT16 and same-dtype floating-point zero_point (INT32 is also supported).
- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: The zero_point supports only INT32.
- aclnn interfaces: x/scale/y support only FLOAT and FLOAT16; zero_point supports INT32, FLOAT16 and FLOAT on the per-channel interface (the actually executable domain of floating-point forms is as described above), and only INT32 on the per-tensor interface.

## Constraints

- The dimension range of x is 0 to 8 dimensions (ND format). Dynamic shapes (dimension value -1) and dynamic rank (-2 placeholder, where static validation is skipped in the inference stage and the actual shape is determined by the runtime tiling) are supported.
- The dtype of scale is the same as that of x. The shape of scale/zero_point supports 0-dim (scalar, only for the per-tensor interface and graph mode) or 1-dim; for the per-channel interface, the size is equal to the length of x on the axis dimension, and for the per-tensor interface, it is a single element.
- quant_min must not be greater than quant_max, and the absolute values of both must be less than 2^24.
- The absolute value of a zero_point value should be less than 2^24 (the integer domain exactly representable in fp32). When exceeded, zero_point is rounded in fp32 during computation, and y/mask may deviate from the formula by one quantization level — this constraint is a precision recommendation: the interface does not reject such input, but the behavior is treated as implementation-defined.
- When x is 0-dim (scalar), axis supports only 0 or -1.
- Non-finite (NaN/±Inf) inputs: aligned with PyTorch semantics (see the special-value semantics section in the docs) — mask is always False; out is clamped to the quantization boundary, that is, out=(quant_min−zero_point)·scale for NaN/−Inf, and out=(quant_max−zero_point)·scale for +Inf.
- Empty tensor input (any input/output with 0 elements): the aclnn interface returns success directly without writing the output buffer (workspaceSize=0), and the caller should not read the output.
- <term>Atlas A2 products</term> (non-regbase path) support only axis=0 for direct graph-mode invocation; the aclnn entry is not subject to this restriction because it normalizes via an internal Transpose. To use axis≠0, use the aclnn interface or pre-transpose the input.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_fake_quant_per_channel_affine_cachemask.cpp](examples/test_aclnn_fake_quant_per_channel_affine_cachemask.cpp) | Invokes the FakeQuantAffineCachemask operator through the [aclnnFakeQuantPerChannelAffineCachemask](docs/aclnnFakeQuantPerChannelAffineCachemask_en.md) interface. |
| aclnn invocation  | [test_aclnn_fake_quant_per_tensor_affine_cachemask.cpp](examples/test_aclnn_fake_quant_per_tensor_affine_cachemask.cpp) | Invokes the FakeQuantAffineCachemask operator through the [aclnnFakeQuantPerTensorAffineCachemask](docs/aclnnFakeQuantPerTensorAffineCachemask_en.md) interface. |
| Graph-mode invocation | - | Builds the graph and invokes the FakeQuantAffineCachemask operator through the [operator IR](op_graph/fake_quant_affine_cachemask_proto.h).         |
