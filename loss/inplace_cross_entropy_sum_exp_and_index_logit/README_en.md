# InplaceCrossEntropySumExpAndIndexLogit

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:----------------------------|:-----------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>| × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: a fusion operator for local CrossEntropy computation in the vocab parallel (Tensor Parallel) scenario (in-place update version). For the local logits of a TP rank after splitting along the vocab dimension, it completes in one pass — after `all_reduce(MAX)` and before the `all_reduce(SUM)` of `predicted_logits` / `sum_exp_logits` — the logits shift (subtracting the global maximum), the target out-of-range mask determination, the local target offset computation, the gather of the logit corresponding to the target, the `exp` operation, and the local summation along the vocab dimension. The difference from [CrossEntropySumExpAndIndexLogit](../cross_entropy_sum_exp_and_index_logit/README.md) is that `exp_logits` is no longer returned as a separate output; instead, it overwrites the input parameter `vocab_parallel_logits` in place, reducing the overhead of intermediate tensor materialization. It targets the training scenarios of large models with extremely large vocabularies.

- Computation formula: let the local vocab shard of the current rank be `vocab_parallel_logits`, the global maximum logit be `global_logits_max`, and the vocab shard range be `[vocab_start_index, vocab_end_index)`. For each token `i` and each local vocab position `j`:

  **1. target mask:**

  $$
  target\_mask[i] = \begin{cases} 1, & target[i] < vocab\_start\_index \text{ or } target[i] \geq vocab\_end\_index \\ 0, & \text{otherwise} \end{cases}
  $$

  **2. local target offset:**

  $$
  target\_offset[i] = \begin{cases} 0, & target\_mask[i] = 1 \\ target[i] - vocab\_start\_index, & \text{otherwise} \end{cases}
  $$

  **3. predicted logit gather (logits shift):**

  $$
  predicted\_logits[i] = \begin{cases} 0, & target\_mask[i] = 1 \\ vocab\_parallel\_logits[i, target\_offset[i]] - global\_logits\_max[i], & \text{otherwise} \end{cases}
  $$

  **4. exp computation (written back to vocab_parallel_logits in place):**

  $$
  vocab\_parallel\_logits[i, j] \leftarrow \exp\big(vocab\_parallel\_logits[i, j] - global\_logits\_max[i]\big)
  $$

  **5. local summation:**

  $$
  sum\_exp\_logits[i] = \sum_{j=0}^{V\_local-1} vocab\_parallel\_logits[i, j]
  $$

- Special processing description: the intermediate computation forcibly uses FLOAT (a `BFLOAT16` input is automatically promoted in precision), the exp operand has already been subtracted by the global maximum to suppress overflow, and underflow naturally returns 0 as FLOAT. This operator involves no division, so divide-by-zero protection is not needed. The accumulation of `sum_exp_logits` along the V_local dimension is completed within a single core (no cross-core reduction and no AtomicAdd), and the accumulation order is fixed, so multiple invocations with the same input produce consistent results. It is a deterministic implementation.

## Parameters

<table style="table-layout: fixed; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">Parameter name</th>
      <th style="white-space: nowrap">Input/Output/Attribute</th>
      <th style="white-space: nowrap">Description</th>
      <th style="white-space: nowrap">Data type</th>
      <th style="white-space: nowrap">Data format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>vocab_parallel_logits</td>
      <td>Input / Output</td>
      <td>Local vocab shard logits of the current TP rank, corresponding to vocab_parallel_logits in the formula. After the call, it is overwritten in place by the exp result. Non-contiguous tensors are supported; the row stride (stride[0] for 2D, stride[1] for 3D) must be >= V_local.</td>
      <td>FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>Input</td>
      <td>Global vocab index, corresponding to target in the formula. The value is a non-negative integer. Non-contiguous tensors are supported.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>global_logits_max</td>
      <td>Input</td>
      <td>Global maximum logit obtained after all_reduce(MAX), corresponding to global_logits_max in the formula. The data type must be consistent with vocab_parallel_logits. Non-contiguous tensors are supported.</td>
      <td>Same as vocab_parallel_logits</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vocab_start_index</td>
      <td>Attribute</td>
      <td>Start index (global) of the vocab shard of the current rank, corresponding to vocab_start_index in the formula. It must satisfy vocab_end_index &gt; vocab_start_index.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>vocab_end_index</td>
      <td>Attribute</td>
      <td>End index (global) of the vocab shard of the current rank, corresponding to vocab_end_index in the formula. It must satisfy vocab_end_index - vocab_start_index == vocab_parallel_logits.size(-1).</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>predicted_logits</td>
      <td>Output</td>
      <td>Result of the logit corresponding to the target minus the global maximum, corresponding to predicted_logits in the formula. When the target is not within the shard of the current rank, the corresponding position is 0.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_exp_logits</td>
      <td>Output</td>
      <td>Sum of the local exp(logits - global_max) along the last dimension, corresponding to sum_exp_logits in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target_offset</td>
      <td>Output</td>
      <td>target - vocab_start_index, corresponding to target_offset in the formula. When the target is not within the shard of the current rank, it is set to 0.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target_mask</td>
      <td>Output</td>
      <td>Vocab out-of-range mask: 1 indicates that the target is not within the shard of the current rank, and 0 indicates that it is within the shard, corresponding to target_mask in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- `vocab_parallel_logits` supports only the 2D `[N, V_local]` or the 3D `[S, B, V_local]` (in which case `N = S * B`), and `shape[:-1]` must be completely consistent with `target.shape` and `global_logits_max.shape`.
- V_local alignment constraint: when the input is BFLOAT16, V_local must be a multiple of 16; when the input is FLOAT, V_local must be a multiple of 8 (to ensure 32-byte UB alignment).
- For non-contiguous `vocab_parallel_logits`, the row stride (stride[0] for 2D, stride[1] for 3D) must be >= V_local, and as_strided views with overlapping rows are forbidden.
- The operator returns 4 outputs (predicted_logits, sum_exp_logits, target_offset, and target_mask) and does not include exp_logits; exp_logits is obtained from `vocab_parallel_logits` after the call, and its dtype follows the input.

### Specification constraints

| Specification item | Specification | Description |
|:--- |:--- |:--- |
| N | 1~32K | prod(target.shape), that is, the total number of tokens after flattening. |
| V_local | 16~200K | Length of the local vocab shard of the current rank; it must be a multiple of 16 for BFLOAT16 and a multiple of 8 for FLOAT. |

### Typical values

| Specification item | Typical value |
|:--- |:--- |
| N | 1024/2048/4096 |
| V_local | 37888 (recommended) |

## Invocation Description

| Invocation method | Sample code | Description |
|:--- |:--- |:--- |
| aclnn interface | - | Calls the operator through the [aclnnInplaceCrossEntropySumExpAndIndexLogit](docs/aclnnInplaceCrossEntropySumExpAndIndexLogit.md) interface. |
| GE graph mode | - | Calls the operator by constructing a graph through the [operator IR](op_graph/inplace_cross_entropy_sum_exp_and_index_logit_proto.h). |
| PyTorch API | - | Calls the operator through the [torch.ops.cann_ops_nn.inplace_cross_entropy_sum_exp_and_index_logit](docs/torchapi_inplace_cross_entropy_sum_exp_and_index_logit.md) interface. |
