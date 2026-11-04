# ActULQClampMaxGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Used in quantization awareness training of ULQ (Ultra-Low-bit Quantization) to compute the backward gradient of the clamp upper bound truncation. The operator performs an all-axis reduction sum on the input tensor and outputs a scalar gradient value.
- Formula:

  $$
  clamp\_max\_grad = \sum_{\text{all axes}}(y\_grad \times (x\_clamped\_loss + |clamp\_max\_mask|))
  $$

  where:

  - $y\_grad$: Gradient tensor from the subsequent layers
  - $clamp\_max\_mask$: Clamp upper bound mask (Boolean or floating point)
  - $x\_clamped\_loss$: Loss value tensor after clamp
  - $clamp\_max\_grad$: Output scalar gradient (0-dimensional tensor)

## Parameters

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 400px">
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
  </tr></thead>
<tbody>
  <tr>
    <td>y_grad</td>
    <td>Input</td>
    <td>Gradient tensor from the subsequent layers. Required parameter.</td>
    <td>FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>clamp_max_mask</td>
    <td>Input</td>
    <td>Clamp upper bound mask tensor, with the same shape as y_grad. Required parameter.</td>
    <td>FLOAT16, FLOAT32, BOOL</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x_clamped_loss</td>
    <td>Input</td>
    <td>Loss value tensor after clamp, with the same shape as y_grad. Required parameter.</td>
    <td>FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>clamp_max_grad</td>
    <td>Output</td>
    <td>Output scalar gradient (0-dimensional tensor), with the data type consistent with y_grad. Required parameter.</td>
    <td>FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

### Supported Data Type Combinations

| y_grad | clamp_max_mask | x_clamped_loss | clamp_max_grad |
|--------|----------------|----------------|----------------|
| FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 |
| FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |
| FLOAT16 | BOOL | FLOAT16 | FLOAT16 |
| FLOAT32 | BOOL | FLOAT32 | FLOAT32 |

## Constraints

- The shapes of the three input tensors must be exactly the same, and broadcast is not supported.
- The dimension (rank) range of the input tensors is 0 to 8: 0-dimensional (scalar) inputs are supported, and tensors of 9 dimensions or above are not supported (an error is reported when rank>8).
- The output clamp_max_grad is a 0-dimensional scalar obtained through the all-axis reduction (with the shape `()`).
- When clamp_max_mask is of the BOOL type, y_grad and x_clamped_loss must be of the same floating-point type (both FLOAT16 or both FLOAT32).
- When clamp_max_mask is of a floating-point type, the three inputs must be of the same floating-point type.
- The input tensors must be stored contiguously.
- Dynamic formats are not supported.
- Dynamic shapes and dynamic ranks are supported.

## Invocation Description

| Invocation method   | Invocation sample            | Description                          |
|-----------|--------------------|----------------------------------------------|
| GE graph mode | [test_geir_act_ulq_clamp_max_grad](examples/test_geir_act_ulq_clamp_max_grad.cpp) | Calls the ActULQClampMaxGrad operator by building a graph through the [operator IR](op_graph/act_ulq_clamp_max_grad_proto.h). |
