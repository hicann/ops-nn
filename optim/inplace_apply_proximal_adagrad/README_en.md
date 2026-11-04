# InplaceApplyProximalAdagrad

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term>     |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: InplaceApplyProximalAdagrad is an optimizer operator that combines the Adagrad adaptive learning rate with the FOBOS (Forward-Backward Splitting) Proximal algorithm, and its functionality is aligned with `tf.raw_ops.ApplyProximalAdagrad`. It adaptively adjusts the learning rate based on the gradient square accumulator, and updates the model parameters in place through soft thresholding (L1 regularization) and scaling (L2 regularization).
- Formula:

  $$
  \begin{aligned}
  \text{accum}_t &= \text{accum}_{t-1} + \text{grad}_t^2 \\
  \eta_t &= \frac{\text{lr}}{\sqrt{\text{accum}_t}} \\
  \text{prox}_t &= \text{var}_{t-1} - \eta_t \cdot \text{grad}_t \\
  \text{var}_t &= \frac{\text{sign}(\text{prox}_t)}{1 + \eta_t \cdot \text{l2}} \cdot \max\!\left(|\text{prox}_t| - \eta_t \cdot \text{l1},\ 0\right)
  \end{aligned}
  $$

  When L1 = 0, it is simplified to:

  $$
  \text{var}_t = \frac{\text{prox}_t}{1 + \eta_t \cdot \text{l2}}
  $$

- Notes:
  - `var` (parameter) and `accum` (gradient square accumulator) are both Ref Tensors, and they are **updated in place** after the operator executes.
  - `lr`, `l1`, and `l2` are 0-D scalar Tensors, which require `lr > 0`, `l1 ≥ 0`, and `l2 ≥ 0` respectively.
  - The computation is element-wise and independent, is naturally deterministic, and has no cross-element/cross-core dependencies.
  - `float16` / `bfloat16` inputs are Cast to `float32` inside the operator for intermediate computation and then Cast back to `float16` / `bfloat16` outputs, consistent with the implementation conventions of PyTorch / TensorFlow.
  - On the `bfloat16` path, the scalar reads of `lr` / `l1` / `l2` and the construction of `accum` tail padding 1.0 are implemented through the LocalTensor detour + Vector Cast, working around the limitation of the `bisheng` compiler backend on bf16 type conversion in the scalar path ("not support bf16 type cast").
  - Unlike ApplyProximalAdagrad V1 (1 output var), V2 explicitly exposes 2 output ports (var + accum), both sharing storage with the corresponding inputs through inplace aliases.

## Parameters

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 150px">
<col style="width: 150px">
<col style="width: 420px">
<col style="width: 200px">
<col style="width: 140px">
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
    <td>var in the formula, the model parameters to be updated (Ref Tensor, updated in place). The shape is the same as accum/grad.</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>Input</td>
    <td>accum in the formula, the gradient square accumulator (Ref Tensor, updated in place). The shape is the same as var/grad, and all elements are required to be non-negative.</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>Input</td>
    <td>lr in the formula, the learning rate. A 0-D or 1-element 1-D Tensor, requiring lr &gt; 0.</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>l1</td>
    <td>Input</td>
    <td>l1 in the formula, the L1 regularization strength. A 0-D or 1-element 1-D Tensor, requiring l1 ≥ 0.</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>l2</td>
    <td>Input</td>
    <td>l2 in the formula, the L2 regularization strength. A 0-D or 1-element 1-D Tensor, requiring l2 ≥ 0.</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>Input</td>
    <td>grad in the formula, the gradient tensor of the current step. The shape is the same as var/accum.</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>var (output)</td>
    <td>Output</td>
    <td>The updated parameters, sharing storage with input var (inplace update).</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum (output)</td>
    <td>Output</td>
    <td>The updated gradient square accumulator, sharing storage with input accum (inplace update).</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>Attribute</td>
    <td>Specifies whether to use a lock to protect the update operations of var and accum. The default is False.</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
</tbody></table>

## Constraints

- Supports the three data types `float32` / `float16` / `bfloat16`; the data types of all tensors (var / accum / lr / l1 / l2 / grad) must be strictly the same.
- The shapes of `var`, `accum`, and `grad` must be exactly the same, and all three are contiguously laid out ND Tensors.
- `lr`, `l1`, and `l2` must be 0-D or 1-element 1-D scalar Tensors.
- The caller must ensure `accum ≥ 0`, `lr > 0`, `l1 ≥ 0`, and `l2 ≥ 0`; the operator does not perform runtime value range validation.
- When `accum + grad^2 == 0`, `rsqrt` outputs Inf/NaN. This behavior is consistent with the native implementations of PyTorch / TensorFlow, and upstream callers must work around it.
- `var` and `accum` are both Ref Tensors: both are written back in place through explicit output ports, and the caller must treat them as storage that the operator can modify.

## Invocation Description

| Invocation method | Sample code | Description |
| -------- | -------- | ---- |
| Graph-mode invocation | [test_geir_inplace_apply_proximal_adagrad.cpp](examples/test_geir_inplace_apply_proximal_adagrad.cpp) | Builds and runs the InplaceApplyProximalAdagrad operator in GE IR graph mode. The sample chains 6 `Data`/scalar inputs in the order of `var → accum → lr → l1 → l2 → grad`, and the output ports declare two inplace outputs, `var` and `accum`. The executable parameter `fp16` / `bf16` switches the data type. |
