# ActULQClampMinGrad

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

- Description: Used in ULQ (Ultra-Low-bit Quantization) quantization-aware training to compute the backward gradient of the clamp lower-bound truncation. The operator performs an all-axis reduction sum on the input tensor and outputs a scalar gradient value.
- Formula:

  $$
  clamp\_min\_grad = \sum_{\text{all axes}}(y\_grad \times ((1 - clamp\_min\_mask) - x\_clamped\_loss))
  $$

  where:

  - $y\_grad$: the gradient tensor from the subsequent layer
  - $clamp\_min\_mask$: the clamp lower-bound mask (Boolean or floating point, with the value 0 or 1)
  - $x\_clamped\_loss$: the loss value tensor after clamp
  - $clamp\_min\_grad$: the output scalar gradient (0-dimensional tensor)

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
    <td>Gradient tensor from the subsequent layer. Required parameter.</td>
    <td>FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>clamp_min_mask</td>
    <td>Input</td>
    <td>Clamp lower-bound mask tensor, with the same shape as y_grad. Required parameter.</td>
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
    <td>clamp_min_grad</td>
    <td>Output</td>
    <td>Output scalar gradient (0-dimensional tensor), with the data type consistent with that of y_grad. Required parameter.</td>
    <td>FLOAT16, FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

### Supported data type combinations

| y_grad | clamp_min_mask | x_clamped_loss | clamp_min_grad |
|--------|----------------|----------------|----------------|
| FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 |
| FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |
| FLOAT16 | BOOL | FLOAT16 | FLOAT16 |
| FLOAT32 | BOOL | FLOAT32 | FLOAT32 |

## Constraints

- The shapes of the three input tensors must be exactly the same; broadcasting is not supported.
- When clamp_min_mask is of the BOOL type, y_grad and x_clamped_loss must be of the same floating-point type (both FLOAT16 or both FLOAT32).
- When clamp_min_mask is of a floating-point type, the three inputs must be of the same floating-point type.
- The input tensors must be stored contiguously.
- Dynamic format is not supported.
- Dynamic shape and dynamic rank are supported.

## Invocation Description

| Invocation method   | Invocation sample            | Description                          |
|-----------|--------------------|----------------------------------------------|
| GE graph mode | [test_geir_act_ulq_clamp_min_grad](examples/test_geir_act_ulq_clamp_min_grad.cpp) | Invokes the ActULQClampMinGrad operator by building a graph through the [operator IR](op_graph/act_ulq_clamp_min_grad_proto.h). |
