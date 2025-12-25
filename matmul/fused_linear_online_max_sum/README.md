# FusedLinearOnlineMaxSum


##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|

## 功能说明

- 算子功能：功能等价Megatron的matmul与fused\_vocab\_parallel\_cross\_entropy的实现，支持vocabulary\_size维度切卡融合matmul与celoss。
- 计算公式：
  1. $input$与$wight^T$做矩阵乘得到：

     $$
     vocabParallelLogitsOutOptional = input @ weight^T
     $$
     
  2. 计算$vocabParallelLogitsOutOptional$每行的最大值：

     $$
     logitsMaxLocalOut = max(vocabParallelLogitsOutOptional, dim=-1)
     $$
     
  3. 计算$vocabParallelLogitsOutOptional$与$logitsMaxLocalOut$的差值：

     $$
     subRes[b][n] = vocabParallelLogitsOutOptional[b][n] - logitsMaxLocalOut[b]
     $$

  4. 计算$subRes$经过指数运算后每行的和

     $$
     sumExpLogitsLocalOut = sum(exp(subRes), dim=-1)
     $$

  5. 计算$target$小于$vocabStartIndex$或$target$大于$vocabEndIndex$的mask

     $$
     targetMask = (target < vocabStartIndex) | (target > vocabEndIndex)
     $$

  6. 计算$maskedTargetOut$

     $$
     maskedTargetOut[b] =
     \begin{cases}
     0 & \text{targetMask[b]=true}\\
     target[b] - vocabStartIndex & \text{targetMask[b]=false}
     \end{cases}
     $$

  7. 计算$predictedLogitsLocalOut$

     $$
     predictedLogitsLocalOut[b] =
     \begin{cases}
     0 & \text{targetMask[b]=true}\\
     subRes[b][maskedTargetOut[b]] & \text{targetMask[b]=false}
     \end{cases}
     $$

  8. 计算$targetMaskOut$

     $$
     alignNum = (input.size(0) + 7) / 8 * 8\\
     maskBit[p] = \begin{cases}
     uint8(targetMask[p]) & \text{p < input.size(0)}\\
     1 & \text{input.size(0) <= p < alignNum}
     \end{cases} \\
     targetMaskOut[k] = 0b(maskBit[8*k:8*k+8])
     $$

  其中$0 \le b \lt input.size(0), 0 \le n \lt weight.size(0), 0 \le p \lt alignNum, 0 \le k \lt alignNum / 8$。

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
      <td>表示matmul计算的左矩阵，公式中的input，input.size(1)需要小于等于65534。</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>表示matmul计算的右矩阵，公式中的weight，数据类型与input保持一致，weight.size(0)需要大于0，weight.size(1)需要与input.size(1)一致。</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>输入</td>
      <td>表示目标索引，公式中的target，target.size(0)需要与input.size(0)一致。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vocab_start_index</td>
      <td>属性</td>
      <td>表示分到本卡上的开始索引，公式中的vocabStartIndex，取值范围为[0, max(target) - 1]。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>vocab_end_index</td>
      <td>属性</td>
      <td>表示分到本卡上的结束索引，公式中的vocabEndIndex，取值范围为[vocab_start_index, min(vocab_start_index + weight.size(0) - 1, max(target) - 1)]</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>vocab_parallel_logits_out_flag</td>
      <td>属性</td>
      <td>表示vocab_parallel_logits_out是否输出，默认值为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>logits_max_local</td>
      <td>输出</td>
      <td>表示matmul计算后各行的最大值，公式中的logitsMaxLocalOut，logits_max_local.size(0)需要与input.size(0)一致。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_exp_logits_local</td>
      <td>输出</td>
      <td>表示matmul计算结果与其各行的最大值作差后经过exp计算后各行内累加的结果，公式中的sumExpLogitsLocalOut，sum_exp_logits_local.size(0)需要与input.size(0)一致。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>predicted_logits_local</td>
      <td>输出</td>
      <td>表示matmul计算结果与其各行的最大值作差后经过maskedTargetOut筛选后的结果，公式中的predictedLogitsLocalOut，predicted_logits_local.size(0)需要与input.size(0)一致。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target_mask</td>
      <td>输出</td>
      <td>表示用于筛选词表的mask，公式中的targetMaskOut，shape为[(input.size(0) + 7) / 8]。</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>masked_target</td>
      <td>输出</td>
      <td>表示target经过targetMaskOut过滤后的结果，公式中的maskedTargetOut，数据类型需要与target一致，masked_target.size(0)需要与input.size(0)一致。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vocab_parallel_logits_out</td>
      <td>输出</td>
      <td>表示matmul计算结果，可选输出，公式中的vocabParallelLogitsOutOptional，数据类型需要input一致，shape为[input.size(0), weight.size(0)]。</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_fused_linear_online_max_sum.cpp](examples/test_aclnn_fused_linear_online_max_sum.cpp) | 通过[aclnnFusedLinearOnlineMaxSum](docs/aclnnFusedLinearOnlineMaxSum.md)接口方式调用ApplyTopKTopPWithSorted算子。 |