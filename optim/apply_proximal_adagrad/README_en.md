# ApplyProximalAdagrad

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function

- Operator function: ApplyProximalAdagrad is an optimizer operator that combines the Adagrad adaptive learning rate with the FOBOS (Forward-Backward Splitting) Proximal proximal algorithm. Its function is aligned with `tf.raw_ops.ApplyProximalAdagrad`. It adaptively adjusts the learning rate based on the gradient square accumulator, and updates the model parameters in place through soft thresholding (L1 regularization) and scaling (L2 regularization).
- Formula:

  $$
  \begin{aligned}
  \text{accum}_t &= \text{accum}_{t-1} + \text{grad}_t^2 \\
  \eta_t &= \frac{\text{lr}}{\sqrt{\text{accum}_t}} \\
  \text{prox}_t &= \text{var}_{t-1} - \eta_t \cdot \text{grad}_t \\
  \text{var}_t &= \frac{\text{sign}(\text{prox}_t)}{1 + \eta_t \cdot \text{l2}} \cdot \max\!\left(|\text{prox}_t| - \eta_t \cdot \text{l1},\ 0\right)
  \end{aligned}
  $$

  When L1 = 0, the formula is simplified to:

  $$
  \text{var}_t = \frac{\text{prox}_t}{1 + \eta_t \cdot \text{l2}}
  $$

- Description:
  - `var` (parameter) and `accum` (gradient square accumulator) are both Ref Tensors, and the operator performs an **in-place update** on them after execution.
  - `lr`, `l1`, and `l2` are 0-D scalar tensors, and `lr > 0`, `l1 ≥ 0`, and `l2 ≥ 0` are required respectively.
  - The computation is element-wise independent, naturally deterministic, and has no cross-element or cross-core dependency.
  - `float16` / `bfloat16` inputs are cast to `float32` inside the operator to complete the intermediate computation, and then cast back to `float16` / `bfloat16` outputs, consistent with the implementation conventions of PyTorch / TensorFlow.
  - On the `bfloat16` path, the reading of the `lr` / `l1` / `l2` scalars and the construction of `accum` tail padding 1.0 are both implemented through LocalTensor borrowing + Vector Cast, which avoids the limitation of the `bisheng` compiler backend on bf16 type conversion in the scalar path ("not support bf16 type cast").

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
    <td>var in the formula, the model parameters to be updated (Ref Tensor, updated in place). The shape is consistent with accum/grad.</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>Input</td>
    <td>accum in the formula, the gradient square accumulator (Ref Tensor, updated in place). The shape is consistent with var/grad, and all elements are required to be non-negative.</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>Input</td>
    <td>lr in the formula, the learning rate. A 0-D or 1-element 1-D tensor, and lr &gt; 0 is required.</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>l1</td>
    <td>Input</td>
    <td>l1 in the formula, the L1 regularization strength. A 0-D or 1-element 1-D tensor, and l1 ≥ 0 is required.</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>l2</td>
    <td>Input</td>
    <td>l2 in the formula, the L2 regularization strength. A 0-D or 1-element 1-D tensor, and l2 ≥ 0 is required.</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>Input</td>
    <td>grad in the formula, the gradient tensor of the current step. The shape is consistent with var/accum.</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>var (output)</td>
    <td>Output</td>
    <td>The updated parameters, which share storage with the input var (inplace update). The operator exposes only this one output port.</td>
    <td>FLOAT32, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraints

- Three data types, `float32` / `float16` / `bfloat16`, are supported; the data types of all tensors (var / accum / lr / l1 / l2 / grad) must be strictly consistent.
- The shapes of `var`, `accum`, and `grad` must be exactly the same, and all of them must be ND tensors with contiguous layout.
- `lr`, `l1`, and `l2` must be 0-D or 1-element 1-D scalar tensors.
- The caller must guarantee `accum ≥ 0`, `lr > 0`, `l1 ≥ 0`, and `l2 ≥ 0`; the operator does not perform runtime value range validation internally.
- When `accum + grad^2 == 0`, `rsqrt` outputs Inf/NaN. This behavior is consistent with the native implementations of PyTorch / TensorFlow, and the upstream caller needs to avoid it.
- `var` and `accum` are both Ref Tensors: `var` is written back in place through the explicit output port, and `accum` is updated in place directly through the input ref port. The caller must treat them as storage that can be modified by the operator.

## Invocation Description

| Invocation method | Sample code | Description |
| -------- | -------- | ---- |
| Graph mode | [test_geir_apply_proximal_adagrad.cpp](examples/test_geir_apply_proximal_adagrad.cpp) | Builds and runs the ApplyProximalAdagrad operator in GE IR graph mode. The sample connects the 6 `Data`/scalar inputs in the order of `var → accum → lr → l1 → l2 → grad`, declares only `var` on the output port, and updates `accum` in place through the input ref port. The executable parameter `fp16` / `bf16` switches the data type. |
