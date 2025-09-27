# ApplyTopKTopPWithSorted

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
|Atlas 200I/500 A2推理产品|×|

## 功能说明

- 算子功能：对原始输入logits进行top-k和top-p采样过滤。
- 计算公式：
  - 对输入logits按最后一轴进行升序排序，得到对应的排序结果sortedValue和sortedIndices。
  $$sortedValue, sortedIndices = sort(logits, dim=-1, descend=false, stable=true)$$
  - 计算保留的阈值（第k大的值）。
  $$topKValue[b][v] = sortedValue[b][sortedValue.size(1) - k[b]]$$
  - 生成top-k需要过滤的mask。
  $$topKMask = sortedValue < topKValue$$
  - 通过topKMask将小于阈值的部分置为-inf。
  $$
  sortedValue[b][v] = 
  \begin{cases}
  -inf & \text{topKMask[b][v]=true}\\
  sortedValue[b][v] & \text{topKMask[b][v]=false}
  \end{cases}
  $$
  - 通过softmax将经过top-k过滤后的数据按最后一轴转换为概率分布。
  $$probsValue = softmax(sortedValue, dim=-1)$$
  - 按最后一轴计算累计概率（从最小的概率开始累加）。
  $$probsSum = cumsum(probsValue, dim=-1)$$
  - 生成top-p的mask，累计概率小于等于1-p的位置需要过滤掉，并保证每个batch至少保留一个元素。
  $$topPMask[b][v] = probsSum[b][v] <= 1-p[b]$$
  $$topPMask[b][-1] = false$$
  - 通过topPMask将小于阈值的部分置为-inf。
  $$
  sortedValue[b][v] = 
  \begin{cases}
  -inf & \text{topPMask[b][v]=true}\\
  sortedValue[b][v] & \text{topPMask[b][v]=false}
  \end{cases}
  $$
  - 将过滤后的结果按sortedIndices还原到原始顺序。
  $$out[b][v] = sortedValue[b][sortedIndices[b][v]]$$
  其中$0 \le b \lt logits.size(0), 0 \le v \lt logits.size(1)$。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1050px"><colgroup>
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 500px">
  <col style="width: 250px">
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
      <td>logits</td>
      <td>输入</td>
      <td>表示需要处理的数据，公式中的logits。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>p</td>
      <td>输入</td>
      <td>表示top-p的阈值，公式中的p。数据类型需要与`logits`一致，shape需要与`logits.size(0)`一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>输入</td>
      <td>表示top-k的阈值，公式中的k，shape需要与`logits.size(0)`一致。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示过滤后的数据，公式中的out。数据类型、shape需要与`logits`一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明
无。


## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_apply_top_k_top_p.cpp](examples/test_aclnn_apply_top_k_top_p.cpp) | 通过[aclnnApplyTopKTopP](docs/aclnnApplyTopKTopP.md)接口方式调用ApplyTopKTopPWithSorted算子。 |

