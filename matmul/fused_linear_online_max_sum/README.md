# FusedLinearOnlineMaxSum


##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|  <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>   |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 算子功能：功能等价Megatron的matmul与fused\_vocab\_parallel\_cross\_entropy的实现，支持vocabulary\_size维度切卡融合matmul与celoss，中间根据通信拆分为[FusedLinearOnlineMaxSum](./)和[FusedCrossEntropyLossWithMaxSum](../../loss/fused_cross_entropy_loss_with_max_sum)。
- 计算公式：
  1. $input$与$wight^T$做矩阵乘得到：

     $$
     vocab\_parallel\_logits\_out = input @ weight^T
     $$
     
  2. 计算$vocab\_parallel\_logits\_out$每行的最大值：

     $$
     logits\_max\_local = max(vocab\_parallel\_logits\_out, dim=-1)
     $$
     
  3. 计算$vocab\_parallel\_logits\_out$与$logits\_max\_local$的差值：

     $$
     sub\_res[b][n] = vocab\_parallel\_logits\_out[b][n] - logits\_max\_local[b]
     $$

  4. 计算$sub\_res$经过指数运算后每行的和

     $$
     sum\_exp\_logits\_local = sum(exp(sub\_res), dim=-1)
     $$

  5. 计算$target$小于$vocab\_start\_index$或$target$大于$vocab\_end\_index$的mask

     $$
     target\_mask = (target < vocab\_start\_index) | (target > vocab\_end\_index)
     $$

  6. 计算$masked\_target$

     $$
     masked\_target[b] =
     \begin{cases}
     0 & \text{target\_mask[b]=true}\\
     target[b] - vocab\_start\_index & \text{target\_mask[b]=false}
     \end{cases}
     $$

  7. 计算$predicted\_logits\_local$

     $$
     predicted\_logits\_local[b] =
     \begin{cases}
     0 & \text{target\_mask[b]=true}\\
     sub\_res[b][masked\_target[b]] & \text{target\_mask[b]=false}
     \end{cases}
     $$

  8. 计算$target\_mask$

     $$
     align\_num = (input.size(0) + 7) / 8 * 8\\
     mask\_bit[p] = \begin{cases}
     uint8(target\_mask[p]) & \text{p < input.size(0)}\\
     1 & \text{input.size(0) <= p < align\_num}
     \end{cases} \\
     target\_mask[k] = 0b(mask\_bit[8*k:8*k+8])
     $$

  其中$0 \le b \lt input.size(0), 0 \le n \lt weight.size(0), 0 \le p \lt align\_num, 0 \le k \lt align\_num / 8$。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input</td>
      <td>输入</td>
      <td><ul><li>表示matmul计算的左矩阵，公式中的input。</li><li>input.size(1)需要小于等于65534。</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td><ul><li>表示matmul计算的右矩阵，公式中的weight。</li><li>数据类型与input保持一致。</li><li>weight.size(0)需要大于0，weight.size(1)需要与input.size(1)一致。</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>输入</td>
      <td><ul><li>表示目标索引，公式中的target。</li><li>target.size(0)需要与input.size(0)一致。</li></ul></td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vocab_start_index</td>
      <td>属性</td>
      <td><ul><li>表示分到本卡上的开始索引，公式中的vocab_start_index。</li><li>取值范围为[0, max(target) - 1]。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>vocab_end_index</td>
      <td>属性</td>
      <td><ul><li>表示分到本卡上的结束索引，公式中的vocab_end_index。</li><li>取值范围为[vocab_start_index, min(vocab_start_index + weight.size(0) - 1, max(target) - 1)]。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>vocab_parallel_logits_out_flag</td>
      <td>可选属性</td>
      <td><ul><li>表示vocab_parallel_logits_out是否输出。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>logits_max_local</td>
      <td>输出</td>
      <td><ul><li>表示matmul计算后各行的最大值，公式中的logits_max_local。</li><li>logits_max_local.size(0)需要与input.size(0)一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_exp_logits_local</td>
      <td>输出</td>
      <td><ul><li>表示matmul计算结果与其各行的最大值作差后经过exp计算后各行内累加的结果，公式中的sum_exp_logits_local。</li><li>sum_exp_logits_local.size(0)需要与input.size(0)一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>predicted_logits_local</td>
      <td>输出</td>
      <td><ul><li>表示matmul计算结果与其各行的最大值作差后经过masked_target筛选后的结果，公式中的predicted_logits_local。</li><li>predicted_logits_local.size(0)需要与input.size(0)一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target_mask</td>
      <td>输出</td>
      <td><ul><li>表示用于筛选词表的mask，公式中的target_mask。</li><li>shape为[(input.size(0) + 7) / 8]。</li></ul></td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>masked_target</td>
      <td>输出</td>
      <td><ul><li>表示target经过target_mask过滤后的结果，公式中的masked_target。</li><li>数据类型需要与target一致。</li><li>masked_target.size(0)需要与input.size(0)一致。</li></ul></td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vocab_parallel_logits_out</td>
      <td>可选输出</td>
      <td><ul><li>表示matmul计算结果，公式中的vocab_parallel_logits_out。</li><li>数据类型需要input一致。</li><li>shape为[input.size(0), weight.size(0)]。</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_fused_linear_online_max_sum.cpp](examples/test_aclnn_fused_linear_online_max_sum.cpp) | 通过[aclnnFusedLinearOnlineMaxSum](docs/aclnnFusedLinearOnlineMaxSum.md)接口方式调用FusedLinearOnlineMaxSum算子。 |