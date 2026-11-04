# AscendRequant

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | × |

## Function

AscendRequant is the dequantization-requantization operator in the INT8 static quantization inference pipeline. It requantizes the INT32 accumulation results produced by the upstream Conv/MatMul with INT8 weights back to INT8 according to the dequantization scaling factor, as the input of the next layer of INT8 computation. `req_scale` is a hardware-affinity dequantization scaling factor of the UINT64 type (DeqScale encoding), applied to `x` following the numpy broadcast rules; the computation result is rounded by round-to-nearest-even and saturated to the INT8 range `[-128, 127]`.

Formula:

$$
y = \text{SaturateInt8}(\text{RoundToNearestEven}(x \times scale))
$$

Here `scale` is decoded from `req_scale` (UINT64 DeqScale encoding), and `SaturateInt8` saturates the result to `[-128, 127]`. When `relu_flag=true`, `y = max(y, 0)` is executed to truncate negative values to zero.

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
<td>x</td>
<td>Input</td>
<td>INT32 accumulation result to be requantized, corresponding to x in the formula.</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>req_scale</td>
<td>Input</td>
<td>UINT64 hardware-affinity dequantization scaling factor (DeqScale encoding), the source of scale in the formula.</td>
<td>UINT64</td>
<td>ND</td>
</tr>
<tr>
<td>y</td>
<td>Output</td>
<td>Requantized INT8 result, with the same shape as x, corresponding to y in the formula.</td>
<td>INT8</td>
<td>ND</td>
</tr>
<tr>
<td>relu_flag</td>
<td>Optional attribute</td>
<td>Whether to fuse ReLU into the requantization result. The default value is false; when true, negative values are truncated to zero.</td>
<td>BOOL</td>
<td>-</td>
</tr>
</tbody>
</table>

## Constraints

- Fixed data type combination: `x` is INT32, `req_scale` is UINT64, and `y` is INT8. Other data type combinations are not supported.
- The data format supports only ND.
- The dimension range of `x` is 1-8; the dimension range of `req_scale` is 0-8, and `rank(req_scale)` must be less than or equal to `rank(x)`.
- The shape of `req_scale` must be broadcastable to the shape of `x` following the numpy broadcast rules, and broadcasting is handled inside the Kernel; the output shape equals the shape of `x`.
- Inputs must be contiguous tensors, and non-contiguous tensors are not supported.

## Invocation Description

| Invocation method | Invocation sample | Description |
|----------|----------|------|
| Graph-mode invocation | [test_geir_ascend_requant](./examples/test_geir_ascend_requant.cpp) | Calls the AscendRequant operator by constructing a graph through the [operator IR](./op_graph/ascend_requant_proto.h). |
