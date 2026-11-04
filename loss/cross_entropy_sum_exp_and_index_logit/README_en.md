# CrossEntropySumExpAndIndexLogit

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

- Operator function: a CrossEntropy local computation fusion operator for the vocab parallel (Tensor Parallel) scenario. For the local logits of the TP rank after splitting along the vocab dimension, this operator completes in one pass, after `all_reduce(MAX)` and before the `all_reduce(SUM)` of `predicted_logits` / `sum_exp_logits`, the logits shift (subtracting the global maximum), the target out-of-bounds mask determination, the local target offset computation, the gather of the logit corresponding to the target, and the `exp` computation with the local sum along the vocab dimension. This reduces the overhead of small operator launches, GM reads and writes, and intermediate tensor materialization. The operator targets large model training scenarios with extremely large vocabularies.

- Formula: let the local vocab shard of the current rank be `vocab_parallel_logits`, the global maximum logit be `global_logits_max`, and the vocab shard range be `[vocab_start_index, vocab_end_index)`. For each token `i` and each local vocab position `j`:

  **1. Target mask:**

  $$
  target\_mask[i] = \begin{cases} 1, & target[i] < vocab\_start\_index \text{ or } target[i] \geq vocab\_end\_index \\ 0, & \text{otherwise} \end{cases}
  $$

  **2. Local target offset:**

  $$
  target\_offset[i] = \begin{cases} 0, & target\_mask[i] = 1 \\ target[i] - vocab\_start\_index, & target\_mask[i] = 0 \end{cases}
  $$

  **3. Predicted logit gather (logits shift):**

  $$
  predicted\_logits[i] = \begin{cases} 0, & target\_mask[i] = 1 \\ vocab\_parallel\_logits[i, target\_offset[i]] - global\_logits\_max[i], & target\_mask[i] = 0 \end{cases}
  $$

  **4. Exponential computation:**

  $$
  exp\_logits[i, j] = \exp\big(vocab\_parallel\_logits[i, j] - global\_logits\_max[i]\big)
  $$

  **5. Local sum:**

  $$
  sum\_exp\_logits[i] = \sum_{j=0}^{V\_local-1} exp\_logits[i, j]
  $$

- Special processing notes: intermediate computation forcibly uses FLOAT (the `BFLOAT16` input is automatically promoted in precision). The exp operand has already subtracted the global maximum to suppress overflow, and underflow naturally returns 0 as FLOAT. This operator does not involve division and requires no division-by-zero protection. The accumulation of `sum_exp_logits` along the V_local dimension is completed within a single core (no cross-core reduction and no AtomicAdd), with a fixed accumulation order. Multiple calls with the same input produce consistent results, making it a deterministic implementation.

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
      <td>Input</td>
      <td>The local vocab shard logits of the current TP rank, corresponding to vocab_parallel_logits in the formula. Non-contiguous tensors are supported.</td>
      <td>FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>Input</td>
      <td>The global vocab index, corresponding to target in the formula. The value is a non-negative integer. Non-contiguous tensors are supported.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>global_logits_max</td>
      <td>Input</td>
      <td>The global maximum logit obtained after all_reduce(MAX), corresponding to global_logits_max in the formula. The data type must be consistent with that of vocab_parallel_logits. Non-contiguous tensors are supported.</td>
      <td>Same as vocab_parallel_logits</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vocab_start_index</td>
      <td>Attribute</td>
      <td>The start index (global) of the vocab shard of the current rank, corresponding to vocab_start_index in the formula. It must satisfy vocab_end_index &gt; vocab_start_index.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>vocab_end_index</td>
      <td>Attribute</td>
      <td>The end index (global) of the vocab shard of the current rank, corresponding to vocab_end_index in the formula. It must satisfy vocab_end_index - vocab_start_index == vocab_parallel_logits.size(-1).</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>predicted_logits</td>
      <td>Output</td>
      <td>The logit corresponding to the target minus the global maximum, corresponding to predicted_logits in the formula. When the target is not in the shard of the current rank, the corresponding position is 0.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_exp_logits</td>
      <td>Output</td>
      <td>The local sum of exp(logits - global_max) along the last dimension, corresponding to sum_exp_logits in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>exp_logits</td>
      <td>Output</td>
      <td>exp(vocab_parallel_logits - global_logits_max), corresponding to exp_logits in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target_offset</td>
      <td>Output</td>
      <td>target - vocab_start_index, corresponding to target_offset in the formula. When the target is not in the shard of the current rank, the value is set to 0.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target_mask</td>
      <td>Output</td>
      <td>The vocab out-of-bounds mask. 1 indicates that the target is not in the shard of the current rank, and 0 indicates that it is in the shard, corresponding to target_mask in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- `vocab_parallel_logits` supports only two dimensions `[N, V_local]` or three dimensions `[S, B, V_local]` (in this case, `N = S * B`). `shape[:-1]` must be completely consistent with `target.shape` and `global_logits_max.shape`.
- V_local alignment constraint: when the input is BFLOAT16, V_local must be a multiple of 16. When the input is FLOAT, V_local must be a multiple of 8 (to ensure 32-byte UB alignment).

### Specification Constraints

| Specification Item | Specification | Specification Description |
|:--- |:--- |:--- |
| N | 1~32K | prod(target.shape), that is, the total number of tokens after flattening. |
| V_local | 16~200K | The length of the local vocab shard of the current rank. BFLOAT16 requires a multiple of 16, and FLOAT requires a multiple of 8. |

### Typical Values

| Specification Item | Typical Value |
|:--- |:--- |
| N | 1024/2048/4096 |
| V_local | 37888 (recommended) |

## Invocation Description

| Invocation method | Sample Code | Description |
|:--- |:--- |:--- |
| aclnn API | [test_aclnn_cross_entropy_sum_exp_and_index_logit.cpp](examples/arch35/test_aclnn_cross_entropy_sum_exp_and_index_logit.cpp) | Invokes the operator through the [aclnnCrossEntropySumExpAndIndexLogit](docs/aclnnCrossEntropySumExpAndIndexLogit_en.md) API. |
| GE graph mode | [test_geir_cross_entropy_sum_exp_and_index_logit.cpp](examples/arch35/test_geir_cross_entropy_sum_exp_and_index_logit.cpp) | Invokes the operator through graph construction using the [operator IR](op_graph/cross_entropy_sum_exp_and_index_logit_proto.h). |
| PyTorch API | - | Invokes the operator through the [torch.ops.cann_ops_nn.cross_entropy_sum_exp_and_index_logit](docs/torchapi_cross_entropy_sum_exp_and_index_logit_en.md) API. |
