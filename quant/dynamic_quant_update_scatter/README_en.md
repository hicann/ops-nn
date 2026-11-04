# DynamicQuantUpdateScatter

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|×|
|Atlas A2 products|√|
|Kirin X90 processor series products|√|
|Kirin 9030 processor series products|√|

## Function

- Operator function: Fuses DynamicQuant + scatter + scatter into the DynamicQuantUpdateScatter operator to improve performance.

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>var</td>
      <td>Input/Output</td>
      <td>Tensor to be updated.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var_scale</td>
      <td>Input/Output</td>
      <td>Quantization scale factor, a tensor to be updated.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Indicates the update positions.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>Indicates the update data.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scales</td>
      <td>Input</td>
      <td>Represents the smoothScales of DynamicQuant.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Attribute</td>
      <td>Scatter axis. Negative values are supported (normalized by the number of dimensions of var). After normalization, it must be an inner axis (it cannot be dimension 0 or the last dimension).</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduce</td>
      <td>Attribute</td>
      <td>Update mode. update is supported; none and the empty string are accepted for compatibility, and their semantics are the same as update.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraints

1. The number of dimensions of indices can only be 1 or 2. If it is 2, the size of its second dimension must be 2.
2. The number of dimensions of updates is the same as that of var; the size of its first dimension equals the size of the first dimension of indices and is not greater than the size of the first dimension of var; the size of its axis dimension is not greater than the size of the axis dimension of var; except for the first dimension and the axis dimension, the sizes of all other dimensions must be exactly the same as those of var.
3. The number of dimensions of var_scale is the same as that of var. The sizes of all dimensions except the last one must be the same as those of var, and the size of the last dimension must be 1 (that is, each quantized row corresponds to one scale, and the number of elements of var_scale equals the number of elements of var divided by the size of the last dimension of var).
4. smooth_scales is 1-dimensional and its size is the same as var[-1]. Its data type must be the same as the data type of updates.
5. reduce supports "update". For compatibility with historical invocations, "none" and the empty string also perform the update operation.
6. The tail axis requires 32B alignment: the product of the dimensions of var and updates after the axis dimension (with the tail axis merged) counted in the number of INT8 elements must be a multiple of 32, and the two must be equal.
7. The scatter data segments mapped by indices must not overlap. If they overlap, multi-core concurrency causes different results across multiple executions.
8. axis supports negative values, normalized as "axis + number of dimensions of var". After normalization, it must be an inner axis (it cannot be dimension 0 or the last dimension). Therefore, the number of dimensions of var must be greater than or equal to 3.
9. The size of each dimension of each input shape must be a positive number (empty tensors are not supported).

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_dynamic_quant_update_scatter](./examples/test_geir_dynamic_quant_update_scatter.cpp)   | Invokes the DynamicQuantUpdateScatter operator through the operator IR-based [graph construction](./op_graph/dynamic_quant_update_scatter_proto.h) method. |
