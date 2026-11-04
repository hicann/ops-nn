# WtsARQ

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

WtsARQ (Weights Adaptive Range Quantization) is used in the AMCT quantization-aware training (QAT) scenario. Based on the input weight `w` and its quantization range lower bound `w_min` and upper bound `w_max`, it adaptively computes the quantization scale, performs round-to-nearest (half-to-even) fake quantization of `w` to the int8 grid, dequantizes the result back to floating point, and outputs `y`. The operator is delivered in GE IR graph mode and has no aclnn interface. Frontends such as TensorFlow call it after being mapped to the same IR through framework plugins.

Computation process (perform range correction first, and then compute the scale):

1. Range correction: $w_{min} = min(w_{min}, 0)$, $w_{max} = max(w_{max}, 0)$
2. `offset_flag=false` (default, asymmetric abs-max): $scale = max(|w_{min}| / 128,\ w_{max} / 127)$
3. `offset_flag=true` (symmetric with offset): $scale = w_{max}/255 - w_{min}/255$, and compute $offset = -rint(w_{min}/scale) - 128$
4. Division-by-zero protection: when $scale < 1.1920929 \times 10^{-7}$ (float32 eps), take $scale = 1.0$
5. Fake quantization (`rint` is round-to-nearest half-to-even): $q = clamp(rint(w/scale) + offset,\ -128,\ 127) - offset$, and $offset = 0$ when `offset_flag=false`
6. Dequantization: $y = q \times scale$

## Parameters

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
<tr>
<th>Parameter name</th>
<th>Input/Output/Attribute</th>
<th>Description</th>
<th>Data type</th>
<th>Data format</th>
</tr>
</thead>
<tbody>
<tr>
<td>w</td>
<td>Input</td>
<td>The weight tensor to be fake-quantized. It is the computation object of the range correction and the quantization formulas.</td>
<td>float16, float32</td>
<td>ND</td>
</tr>
<tr>
<td>w_min</td>
<td>Input</td>
<td>The lower bound of the weight range. It participates in the computation with restricted broadcast: w_min = min(w_min, 0) is executed first.</td>
<td>float16, float32</td>
<td>ND</td>
</tr>
<tr>
<td>w_max</td>
<td>Input</td>
<td>The upper bound of the weight range. It participates in the computation with restricted broadcast: w_max = max(w_max, 0) is executed first.</td>
<td>float16, float32</td>
<td>ND</td>
</tr>
<tr>
<td>y</td>
<td>Output</td>
<td>The weight after fake quantization and dequantization. Its shape and dtype are the same as w.</td>
<td>float16, float32</td>
<td>ND</td>
</tr>
<tr>
<td>num_bits</td>
<td>Optional attribute</td>
<td>The quantization bit width. Currently only 8 is supported. The default value is 8.</td>
<td>int64</td>
<td>-</td>
</tr>
<tr>
<td>offset_flag</td>
<td>Optional attribute</td>
<td>Specifies whether to use the symmetric quantization branch with offset. The default value is false. When it is true, the scale is computed as w_max/255 - w_min/255.</td>
<td>bool</td>
<td>-</td>
</tr>
</tbody>
</table>

## Constraints

- Both the inputs and the outputs are in ND format. Non-ND storage formats are rejected in the tiling stage.
- The dtypes of `w`, `w_min`, `w_max`, and `y` must be consistent, and only float16 or float32 is supported.
- The rank of `w` supports 0~8. rank=0 (scalar) is processed as a single-element tensor.
- The ranks of `w_min` and `w_max` must equal the rank of `w`, and their shapes must be exactly the same. Each dimension must equal the corresponding dimension of `w` or be 1 (restricted broadcast).
- The total number of elements of `w` must not exceed 2^31. For an empty tensor (with a zero-length dimension), the output is also empty, and no data is written back.
- In dynamic shape scenarios, InferShape supports unknown dimension `-1` and unknown rank `-2`. Illicit dimension relationships that can be statically determined are rejected, and the tiling stage requires all dimensions to be concrete.
- `num_bits` supports only 8, and other values are rejected.

## Invocation Description

| Invocation method | Invocation sample | Description |
|----------|----------|------|
| Graph-mode invocation | [test_geir_wts_arq](./examples/test_geir_wts_arq.cpp) | Invoke the WtsARQ operator by constructing a graph through the [operator IR](./op_graph/wts_arq_proto.h). |
