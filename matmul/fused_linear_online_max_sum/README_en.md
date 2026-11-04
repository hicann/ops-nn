# FusedLinearOnlineMaxSum

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|  <term>Ascend 950PR&950DT products</term>                   |    √     |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>   |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: the implementation is functionally equivalent to the matmul and fused\_vocab\_parallel\_cross\_entropy of Megatron. It supports splitting the device along the vocabulary\_size dimension and fusing matmul with celoss. During the process, it is split according to communication into [FusedLinearOnlineMaxSum](./) and [FusedCrossEntropyLossWithMaxSum](../../loss/fused_cross_entropy_loss_with_max_sum).
- Formula:
  1. Perform matrix multiplication between $input$ and $weight^T$ to obtain:

     $$
     vocab\_parallel\_logits\_out = input @ weight^T
     $$
  2. Compute the maximum value of each row of $vocab\_parallel\_logits\_out$:

     $$
     logits\_max\_local = max(vocab\_parallel\_logits\_out, dim=-1)
     $$
  3. Compute the difference between $vocab\_parallel\_logits\_out$ and $logits\_max\_local$:

     $$
     sub\_res[b][n] = vocab\_parallel\_logits\_out[b][n] - logits\_max\_local[b]
     $$
  4. Compute the sum of each row of $sub\_res$ after the exponential operation

     $$
     sum\_exp\_logits\_local = sum(exp(sub\_res), dim=-1)
     $$
  5. Compute the mask where $target$ is less than $vocab\_start\_index$ or $target$ is greater than or equal to $vocab\_end\_index$

     $$
     target\_mask = (target < vocab\_start\_index) | (target >= vocab\_end\_index)
     $$
  6. Compute $masked\_target$

     $$
     masked\_target[b] =
     \begin{cases}
     0 & \text{target\_mask[b]=true}\\
     target[b] - vocab\_start\_index & \text{target\_mask[b]=false}
     \end{cases}
     $$
  7. Compute $predicted\_logits\_local$

     $$
     predicted\_logits\_local[b] =
     \begin{cases}
     0 & \text{target\_mask[b]=true}\\
     sub\_res[b][masked\_target[b]] & \text{target\_mask[b]=false}
     \end{cases}
     $$
  8. Compute $target\_mask$

     $$
     align\_num = (input.size(0) + 7) / 8 * 8\\
     mask\_bit[p] = \begin{cases}
     uint8(target\_mask[p]) & \text{p < input.size(0)}\\
     1 & \text{input.size(0) <= p < align\_num}
     \end{cases} \\
     target\_mask[k] = 0b(mask\_bit[8*k:8*k+8])
     $$

  Here, $0 \le b \lt input.size(0), 0 \le n \lt weight.size(0), 0 \le p \lt align\_num, 0 \le k \lt align\_num / 8$.

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
      <td>input</td>
      <td>Input</td>
      <td><ul><li>The left matrix of the matmul computation, the input in the formula.</li><li>input.size(1) must be less than or equal to 65534.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td><ul><li>The right matrix of the matmul computation, the weight in the formula.</li><li>The data type is consistent with that of input.</li><li>weight.size(0) must be greater than 0, and weight.size(1) must be consistent with input.size(1).</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>Input</td>
      <td><ul><li>The target index, the target in the formula.</li><li>target.size(0) must be consistent with input.size(0).</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vocab_start_index</td>
      <td>Attribute</td>
      <td><ul><li>The start index allocated to the current device, the vocab_start_index in the formula.</li><li>The value range is [0, max(target) - 1].</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>vocab_end_index</td>
      <td>Attribute</td>
      <td><ul><li>The end index allocated to the current device (excluded from the shard), the vocab_end_index in the formula, that is, the numbers the current device is responsible for are in the left-closed and right-open interval [vocab_start_index, vocab_end_index).</li><li>The value range is [vocab_start_index, vocab_start_index + weight.size(0)].</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>vocab_parallel_logits_out_flag</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates whether vocab_parallel_logits_out is output.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>logits_max_local</td>
      <td>Output</td>
      <td><ul><li>The maximum value of each row after the matmul computation, the logits_max_local in the formula.</li><li>logits_max_local.size(0) must be consistent with input.size(0).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_exp_logits_local</td>
      <td>Output</td>
      <td><ul><li>The result of subtracting the maximum value of each row from the matmul computation result, computing exp, and accumulating within each row, the sum_exp_logits_local in the formula.</li><li>sum_exp_logits_local.size(0) must be consistent with input.size(0).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>predicted_logits_local</td>
      <td>Output</td>
      <td><ul><li>The result of subtracting the maximum value of each row from the matmul computation result and filtering with masked_target, the predicted_logits_local in the formula.</li><li>predicted_logits_local.size(0) must be consistent with input.size(0).</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target_mask</td>
      <td>Output</td>
      <td><ul><li>The mask used to filter the vocabulary, the target_mask in the formula.</li><li>The shape is [(input.size(0) + 7) / 8].</li></ul></td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>masked_target</td>
      <td>Output</td>
      <td><ul><li>The result of target filtered by target_mask, the masked_target in the formula.</li><li>The data type must be consistent with that of target.</li><li>masked_target.size(0) must be consistent with input.size(0).</li></ul></td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vocab_parallel_logits_out</td>
      <td>Optional output</td>
      <td><ul><li>The matmul computation result, the vocab_parallel_logits_out in the formula.</li><li>The data type must be consistent with that of input.</li><li>The shape is [input.size(0), weight.size(0)].</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_fused_linear_online_max_sum.cpp](examples/arch22/test_aclnn_fused_linear_online_max_sum.cpp) | Invokes the FusedLinearOnlineMaxSum operator through the [aclnnFusedLinearOnlineMaxSum](docs/aclnnFusedLinearOnlineMaxSum_en.md) API. |
