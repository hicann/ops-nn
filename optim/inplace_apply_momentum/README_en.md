# InplaceApplyMomentum

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>| × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

> Note: InplaceApplyMomentum currently has the AICore information library configured only for the <term>Ascend 950PR&950DT products</term>. Other products are not adapted yet.

## Function

- Operator function: updates the variable "var" according to the momentum scheme. If use_nesterov=True is set, Nesterov momentum is used.

- Formula:

  $$
  accum = accum \times momentum + grad
  $$

  - If use_nesterov = True:

    $$
    var = var - (grad \times lr + accum \times momentum \times lr)
    $$

  - If use_nesterov = False:

    $$
    var = var - lr \times accum
    $$

## Parameters

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 300px">
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
    <td>var</td>
    <td>Input</td>
    <td>Parameter tensor to be updated. It should come from Variable.</td>
    <td>FLOAT16, FLOAT, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>Input</td>
    <td>Gradient accumulation value. It should come from Variable. Its shape and dtype must be consistent with var.</td>
    <td>FLOAT16, FLOAT, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>Input</td>
    <td>Learning rate, a scalar (0-dimensional or with 1 element).</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>Input</td>
    <td>Gradient tensor. Its shape and dtype must be consistent with var.</td>
    <td>FLOAT16, FLOAT, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>momentum</td>
    <td>Input</td>
    <td>Momentum coefficient, a scalar (0-dimensional or with 1 element).</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_nesterov</td>
    <td>Attribute</td>
    <td><ul><li>Specifies whether to use Nesterov momentum.</li><li>The default value is False.</li></ul></td>
    <td>Bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>Attribute</td>
    <td><ul><li>Specifies whether to use a lock mechanism to protect the update operation.</li><li>The default value is False.</li></ul></td>
    <td>Bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>var</td>
    <td>Output</td>
    <td>Updated parameter tensor, sharing Device memory with input var.</td>
    <td>FLOAT16, FLOAT, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>Output</td>
    <td>Updated gradient accumulation value, sharing Device memory with input accum.</td>
    <td>FLOAT16, FLOAT, BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraints

- var, accum, and grad must have the same data type and shape.
- lr and momentum are scalar tensors (0-dimensional or with 1 element), and their data type is consistent with var.
- Intermediate computation of FP16/BF16 inputs is uniformly promoted to FP32 for execution and restored to the original data type before output.

## Invocation Description

| Invocation method | Invocation sample | Description |
|----------|----------|------|
| Graph-mode invocation | [test_geir_inplace_apply_momentum](./examples/arch35/test_geir_inplace_apply_momentum.cpp) | Invokes the InplaceApplyMomentum operator through the [operator IR](./op_graph/inplace_apply_momentum_proto.h) graph construction method. |
| Graph-mode invocation (dynamic shape) | [test_geir_inplace_apply_momentum_dynamic](./examples/arch35/test_geir_inplace_apply_momentum_dynamic.cpp) | Constructs the graph with a dynamic shape (`-2`) and verifies the computation correctness under multiple actual shapes. |
