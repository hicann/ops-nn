# InplaceApplyAdagrad

English | [简体中文](./README.md)

<!-- codespell:ignore accum adagrad -->

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

> This operator registers the `ascend950`, `ascend350`, and `ascend960dt` compute units. No implementation is provided for other products.

## Function

- Operator function: Performs a single-step Adagrad optimizer update. The operator updates the model parameter `var` and the accumulated squared gradient `accum` with an adaptive learning rate. The `var` and `accum` outputs share device memory with their same-named inputs, implementing in-place updates. The computation semantics align with the CANNDEV `ApplyAdagradD` interface.

- Formula:
  $$
  \begin{aligned}
  accum_{new} &=
  \begin{cases}
  accum + grad^2, & \text{update\_slots} = true \\
  accum, & \text{update\_slots} = false
  \end{cases} \\
  var_{new} &= var - lr \times \frac{grad}{\sqrt{accum_{new}}}
  \end{aligned}
  $$

  where `lr` is the learning rate and `update_slots` controls whether `accum` is updated.

## Parameters

<table style="table-layout: fixed; width: 1430px">
  <colgroup>
    <col style="width: 140px">
    <col style="width: 160px">
    <col style="width: 420px">
    <col style="width: 260px">
    <col style="width: 120px">
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
      <td>var</td>
      <td>Input / Output (in-place)</td>
      <td>The parameter tensor to be updated, expected to come from a Variable. Corresponds to var in the formula. The var output shares device memory with the var input.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>Input / Output (in-place)</td>
      <td>The accumulated squared gradient, expected to come from a Variable. Corresponds to accum in the formula. Its shape/dtype must match var. The accum output shares device memory with the accum input.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>Input</td>
      <td>The learning rate. Corresponds to lr in the formula. Supports a rank-0 scalar or a single-element tensor with shape {1}; its dtype must match var.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>The current gradient tensor. Corresponds to grad in the formula. Its shape/dtype must match var.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>update_slots</td>
      <td>Attribute</td>
      <td>Specifies whether to update accum; the default value is true. When set to false, accum remains unchanged and the accum output equals the accum input.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>Attribute</td>
      <td>Specifies whether to lock during the update; the default value is false. The current implementation does not introduce additional lock semantics.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>var (output)</td>
      <td>Output</td>
      <td>The updated var tensor, sharing device memory with the var input (in-place).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum (output)</td>
      <td>Output</td>
      <td>The updated accum tensor, sharing device memory with the accum input (in-place).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- The input tensors `var`, `accum`, and `grad` must have the same shape, dtype, and format. The `var` and `accum` outputs have the same shape/dtype as their inputs.
- `lr` must be a rank-0 scalar or a single-element tensor with shape `{1}`.
- Ranks 0 to 8 are supported; empty tensors are supported.
- FP16/BF16 inputs are computed in the FP32 domain and written back in the destination dtype.
- `use_locking` is provided only for compatibility and does not change the computation result.

## Invocation Description

| Invocation method | Sample code | Description |
|:------------------|:------------|:------------|
| Graph mode | [test_geir_inplace_apply_adagrad.cpp](examples/test_geir_inplace_apply_adagrad.cpp) | Calls the InplaceApplyAdagrad operator by building a graph through the [operator IR](op_graph/inplace_apply_adagrad_proto.h). |
| TensorFlow graph mode | [inplace_apply_adagrad_tf_plugin.cpp](framework/inplace_apply_adagrad_tf_plugin.cpp) | Maps TensorFlow `ApplyAdagrad` / `ResourceApplyAdagrad` to InplaceApplyAdagrad on Ascend950; falls back to the built-in ApplyAdagrad on other products. |
