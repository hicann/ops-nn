# InplaceApplyAdagradDA

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:---|:---:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- **Operator function**: Perform a single-step parameter update of the Adagrad Dual Averaging optimizer. The operator combines the adaptive learning rate mechanism of Adagrad with the sparsity-inducing property of Dual Averaging, and updates the model parameters and gradient accumulators **in place** based on the current gradient (inplace semantics). The computation semantics are aligned with the `tf.raw_ops.ApplyAdagradDA` API in TensorFlow.

- **Formula**:

  $$
  \begin{aligned}
  g_{out}   &= g_{acc} + grad \\
  gg_{out}  &= gg_{acc} + grad^2 \\
  T         &= float(global\_step) \\
  tmp       &= sign(g_{out}) \cdot \max(|g_{out}| - l1 \cdot T,\ 0) \quad &\text{if } l1 > 0 \\
  tmp       &= g_{out} \quad &\text{if } l1 \le 0 \\
  var_{out} &= \frac{-lr \cdot tmp}{l2 \cdot T \cdot lr + \sqrt{gg_{out}}}
  \end{aligned}
  $$

  Here `lr` is the learning rate, `l1` is the L1 regularization coefficient, `l2` is the L2 regularization coefficient, and `global_step` is the number of training steps. Sparse truncation is performed only when `l1 > 0`. For the numerical boundary behaviors (NaN/Inf, value ranges, and empty tensors), see [Constraints](#constraints).

  > **Note**: `var_out` is the **new parameter value** recomputed directly from the gradient accumulators by the Dual Averaging method, and its computation does not depend on the `var` input (the TF `ApplyAdagradDA` and the built-in CANN `ApplyAdagradDA` (including the deprecated alias `ApplyAdagradDAD`) have the same semantics). Do not interpret it as the "update amount to be applied" — combined with the inplace write-back of this operator, executing `var = var - var_out` once more downstream would cause double application.

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
    <td>Weight parameter to be updated, corresponding to var in the formula. The kernel updates it in place, and the single-output view of GE IR shares device memory with input var.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gradient_accumulator</td>
    <td>Input</td>
    <td>Gradient accumulator, corresponding to g_acc in the formula. The shape/dtype must be the same as those of var. The kernel explicitly writes back to the input GM address (inplace update).</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gradient_squared_accumulator</td>
    <td>Input</td>
    <td>Squared gradient accumulator, corresponding to gg_acc in the formula. The shape/dtype must be the same as those of var. The kernel explicitly writes back to the input GM address (inplace update).</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>Input</td>
    <td>Current gradient tensor, corresponding to grad in the formula. The shape/dtype must be the same as those of var.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>Input</td>
    <td>Learning rate, corresponding to lr in the formula. A 1-element scalar tensor with shape={1}, and its dtype must be the same as that of var.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>l1</td>
    <td>Input</td>
    <td>L1 regularization coefficient, corresponding to l1 in the formula. A 1-element scalar tensor with shape={1}, and its dtype must be the same as that of var.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>l2</td>
    <td>Input</td>
    <td>L2 regularization coefficient, corresponding to l2 in the formula. A 1-element scalar tensor with shape={1}, and its dtype must be the same as that of var.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>global_step</td>
    <td>Input</td>
    <td>Number of training steps, corresponding to global_step in the formula. A 1-element scalar tensor with shape={1}. Regardless of whether the other inputs are FLOAT or FLOAT16, INT32 and INT64 are supported.</td>
    <td>INT32, INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>var (output)</td>
    <td>Output</td>
    <td>Updated var tensor, which shares device memory with input var (inplace).</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gradient_accumulator (output)</td>
    <td>Output</td>
    <td>Updated gradient accumulator tensor, which shares device memory with input gradient_accumulator (inplace).</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gradient_squared_accumulator (output)</td>
    <td>Output</td>
    <td>Updated squared gradient accumulator tensor, which shares device memory with input gradient_squared_accumulator (inplace).</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>Attribute</td>
    <td>Indicates whether to lock during the update. Default value: false. When it is true, the value is ignored and the computation proceeds normally (aligned with the built-in CANN TBE implementation and TensorFlow — the lock is a concurrency protection semantic, and there is no concurrent update scenario in single-operator execution).</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
</tbody></table>

## Constraints

- The data types of var/gradient_accumulator/gradient_squared_accumulator/grad/lr/l1/l2 must be the same.
- The caller must ensure that gradient_squared_accumulator ≥ 0, lr > 0, l2 > 0, and global_step > 0. The operator does not perform runtime value range validation, and violating values are computed according to the formula (for example, a negative gradient_squared_accumulator produces NaN through sqrt, and lr=0 makes var_out 0). The behavior is consistent with the native TensorFlow implementation and must be avoided by upstream callers.
- When the inputs contain NaN/Inf, the operator does not validate or reject them, and processes them according to the computation formula: when an element of a tensor input participating in the computation (gradient_accumulator/gradient_squared_accumulator/grad) is NaN/Inf, the output element at the corresponding position is NaN/Inf (element-wise propagation); when a scalar input (lr/l1/l2/global_step) is NaN/Inf, all output elements involved in the computation are affected through broadcasting; the value of the var input does not participate in the computation, and its NaN/Inf does not affect the output. The propagation behavior is consistent with the native TensorFlow ApplyAdagradDA and must be avoided by upstream callers.
- 0D to 8D ND inputs are supported.
- An empty tensor (containing a 0-length dimension) is a valid input: the output passes through the empty shape, and no computation is performed.
- Non-contiguous tensors: the operator registration framework automatically makes them contiguous (AutoContiguous). Non-contiguous inputs are converted by the framework and then computed in the contiguous ND layout; the operator itself does not provide native support for the original strides.

## Error Behavior Description

The operator has no independent named error codes. For invalid inputs, it always returns ge::GRAPH_FAILED and outputs logs with OP_LOGE (the log characteristics of each scenario are shown in the following table). The error codes received by the caller at the framework layer are generated by the GE wrapper and vary with the framework version, and are not part of the contract of this operator.

| Violation scenario | Rejection stage | Log characteristics |
|---|---|---|
| A required input is missing | GE graph construction parameter check (E80001); empty-check fallback in Host inference | required input[i] is missing; input name=... |
| var rank > 8 | Host InferShape / Tiling | var rank %zu exceeds maximum supported rank %zu |
| The shape of a tensor input is inconsistent with var | Host InferShape / Tiling | input[i] shape must exactly match var shape; ... |
| The shape of a scalar input is not [1] | Host InferShape / Tiling | input[i] must have scalar shape [1]; ... |
| dtype out of domain or inconsistent with var / global_step is not an integer type | Host InferDataType / Tiling | unsupported dtype ... / dtype must match input var dtype ... |
| format is not ND | Tiling | input[i] format ... is unsupported; legal format=ND |

Note: for dynamic shape graphs (including the `-1` unknown dimension flag), the shape relationships are not determined during graph construction, and are validated by tiling using the actual bound shapes.

## Invocation Description

| Invocation method | Sample code | Description |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| GE graph mode | [test_geir_inplace_apply_adagrad_da](./examples/arch35/test_geir_inplace_apply_adagrad_da.cpp) | Invoke the InplaceApplyAdagradDA operator through graph construction using the [operator IR](./op_graph/inplace_apply_adagrad_da_proto.h). |
| TensorFlow frontend | - | `ApplyAdagradDA`/`ResourceApplyAdagradDA` in the TF graph are mapped and integrated through [framework/apply_adagrad_da_tf_plugin.cpp](./framework/apply_adagrad_da_tf_plugin.cpp) (committed support; end-to-end verification will be enabled after the plugin registration name is unified with the operator registration name). |
