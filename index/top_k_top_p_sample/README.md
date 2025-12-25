# TopKTopPSample

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：根据输入词频logits、topK/topP采样参数、随机采样权重分布q，进行topK-topP-sample采样计算，输出每个batch的最大词频logitsSelectIdx，以及topK-topP采样后的词频分布logitsTopKPSelect。

  算子包含三个可单独使能，但上下游处理关系保持不变的采样算法（从原始输入到最终输出）：TopK采样、TopP采样、指数采样（本文档中Sample所指）。它们可以构成八种计算场景。如下表所示：
  | 计算场景 | TopK采样 | TopP采样 | 指数分布采样 |备注|
  | :-------:| :------:|:-------:|:-------:|:-------:|
  |Softmax-Argmax采样|×|×|×|对输入logits按每个batch，取SoftMax后取最大结果|
  |topK采样|√|×|×|对输入logits按每个batch，取前topK[batch]个最大结果|
  |topP采样|×|√|×|对输入logits按每个batch从大到小排序，取累加值大于等于topP[batch]值的前n个结果进行采样|
  |Sample采样|×|×|√|对输入logits按每个batch，进行Softmax后与q进行除法取最大结果|
  |topK-topP采样|√|√|×|对输入logits按每个batch，先进行topK采样，再进行topP采样后取最大结果|
  |topK-Sample采样|√|×|√|对输入logits按每个batch，先进行topK采样，再进行Sample采样后取最大结果|
  |topP-Sample采样|×|√|√|对输入logits按每个batch，先进行topP采样，再进行Sample采样后取最大结果|
  |topK-topP-Sample采样|√|√|√|对输入logits按每个batch，先进行topK采样，再进行topP采样，最后进行Sample采样后取最大结果|
- 计算公式：

  输入logits为大小为[batch, voc_size]的词频表，其中每个batch对应一条输入序列，而voc_size则是约定每个batch的统一长度。<br>
logits中的每一行logits[batch][:]根据相应的topK[batch]、topP[batch]、q[batch, :]，执行不同的计算场景。<br>
下述公式中使用b和v来分别表示batch和voc_size方向上的索引。

  TopK采样

  1. 按分段长度v采用分段topk归并排序，用{s-1}块的topK对当前{s}块的输入进行预筛选，渐进更新单batch的topK，减少冗余数据和计算。
  2. topK[batch]对应当前batch采样的k值，有效范围为1≤topK[batch]≤min(voc_size[batch], 1024)，如果top[k]超出有效范围，则视为跳过当前batch的topK采样阶段，也同样会则跳过当前batch的排序，将输入logits[batch]直接传入下一模块。

  * 对当前batch分割为若干子段，滚动计算topKValue[b]：

  $$
  topKValue[b] = {Max(topK[b])}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ topKValue[b]\left \{s-1 \right \}  \cup \left \{ logits[b][v] \ge topKMin[b][s-1] \right \} \right \}\\
  Card(topKValue[b])=topK[b]
  $$

  其中：

  $$
  topKMin[b][s] = Min(topKValue[b]\left \{  s \right \})
  $$

  v表示预设的滚动topK时固定的分段长度：

  $$
  v=8*1024
  $$

  * 生成需要过滤的mask

  $$
  sortedValue[b] = sort(topKValue[b], descendant)
  $$

  $$
  topKMask[b] = sortedValue[b]<Min(topKValue[b])
  $$

  * 将小于阈值的部分通过mask置为-Inf

  $$
  sortedValue[b][v]=
  \begin{cases}
  -Inf & \text{topKMask[b][v]=true} \\
  sortedValue[b][v] & \text{topKMask[b][v]=false} &
  \end{cases}
  $$

  * 通过softmax将经过topK过滤后的logits按最后一轴转换为概率分布

  $$
  probsValue[b]=sortedValue[b].softmax (dim=-1)
  $$

  * 按最后一轴计算累积概率（从最小的概率开始累加）

  $$
  probsSum[b]=probsValue[b].cumsum (dim=-1)
  $$

  TopP采样

  * 如果前序topK采样已有排序输出结果，则根据topK采样输出计算累积词频，并根据topP截断采样：

    $$
    topPMask[b] = probsSum[b][*] < topP[b]
    $$

  * 如果topK采样被跳过，则先对输入logits[b]进行softmax处理：

  $$
  logitsValue[b] = logits[b].softmax(dim=-1)
  $$

  * 尝试使用topKGuess，对logits进行滚动排序，获取计算topP的mask：

  $$
  topPValue[b] = {Max(topKGuess)}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ topPValue[b]\left \{s-1 \right \}  \cup \left \{ logitsValue[b][v] \ge topKMin[b][s-1] \right \} \right \}
  $$

  * 如果在访问到logitsValue[b]的第1e4个元素之前，下条件得到满足，则视为topKGuess成功，

  $$
  \sum^{topKGuess}(topPValue[b]) \ge topP[b]\\
  topPMask[b][Index(topPValue[b])] = false
  $$

  * 如果topKGuess失败，则对当前序logitsValue[b]进行全排序和cumsum，按topP[b]截断采样：

  $$
  sortedLogits[b] = sort(logitsValue[b], descendant) \\
  probsSum[b]=sortedLogits[b].cumsum (dim=-1) \\
  topPMask[b] = (probsSum[b] - sortedLogits[b])>topP[b] 
  $$

  * 将需要过滤的位置设置为-Inf，得到sortedValue[b][v]：

    $$
    sortedValue[b][v] = \begin{cases} -Inf& \text{topPMask[b][v]=true}\\sortedValue[b][v]& \text{topPMask[b][v]=false}\end{cases}
    $$

    取过滤后sortedValue[b][v]每行中前topK个元素，查找这些元素在输入中的原始索引，整合为`logitsIdx`:

    $$
    logitsIdx[b][v] = Index(sortedValue[b][v] \in logits)
    $$

  指数采样（Sample）

  * 如果`isNeedLogits=true`，则根据`logitsIdx`，选取采样后结果输出到`logitsTopKPSelect`：

  $$
  logitsTopKPSelect[b][logitsIdx[b][v]]=sortedValue[b][v]
  $$

  * 对`logitsSort`进行指数分布采样：

    $$
    probs = softmax(logitsSort)
    $$

    $$
    probsOpt = \frac{probs}{q + eps}
    $$

  * 从`probsOpt`中取出每个batch的最大元素，从`logitsIdx`中gather相应元素的输入索引，作为输出`logitsSelectIdx`：
    
    $$
    logitsSelectIdx[b] = logitsIdx[b][argmax(probsOpt[b][:])]
    $$

  其中0≤b<sortedValue.size(0)，0≤v<sortedValue.size(1)。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 312px">
    <col style="width: 213px">
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
        <td>待采样的输入词频，对应公式中的`logits`。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>topK</td>
        <td>输入</td>
        <td>表示每个batch采样的k值</td>
        <td>INT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>topP</td>
        <td>输入</td>
        <td>表示每个batch采样的p值。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>-</td>
      </tr>
      <tr>
        <td>q</td>
        <td>输入</td>
        <td>表示topK-topP采样输出的指数采样矩阵，维度与尺寸需要与`logits`保持一致。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>eps</td>
        <td>输入</td>
        <td>在softmax和权重采样中防止除零，对应公式中的`eps`，建议设置为1e-8。</td>
        <td>FLOAT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>isNeedLogits</td>
        <td>输入</td>
        <td>控制logitsTopKPselect的输出条件，建议设置为0。</td>
        <td>UINT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>topKGuess</td>
        <td>输入</td>
        <td>表示每个batch在尝试topP部分遍历采样logits时的候选logits大小，对应公式中的`topKGuess`。</td>
        <td>UINT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>logitsSelectIdx</td>
        <td>输出</td>
        <td>表示经过topK-topP-sample计算流程后，每个batch中词频最大元素max(probsOpt[batch, :])在输入logits中的位置索引。对应公式中的`logitsSelectIdx`。</td>
        <td>INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>logitsTopKPSelect</td>
        <td>输出</td>
        <td>表示经过topK-topP计算流程后，输入logits中剩余未被过滤的logits。对应公式中的`logitsTopKPSelect`。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody></table>


## 约束说明
  * 输入值域限制：
    * 对于所有参数，它们的尺寸必须满足，batch>0，0<vocSize<=2^20。
  * 输入shape限制：
    * logits、q、logitsTopKPselect的尺寸和维度必须完全一致，目前仅支持两维。
    * logits、topK、topP、logitsSelectIdx除最后一维以外的所有维度必须顺序和大小完全一致。目前logits只能是2维，topK、topP、logitsSelectIdx必须是1维非空Tensor。logits、topK、topP不允许空Tensor作为输入，如需跳过相应模块，需按相应规则设置输入。
  * 其他限制：  
    * 如果需要单独跳过topK模块，请传入[batch, 1]大小的Tensor，并使每个元素均为无效值。
    * 如果1024<topK[batch]<vocSize[batch]，则视为选择当前batch的全部有效元素并跳过topK采样。
    * 如果需要单独跳过topP模块，请传入[batch, 1]大小的Tensor，并使每个元素均≥1。
    * 如果需要单独跳过sample模块，传入`q=nullptr`即可；如需使用sample模块，则必须传入尺寸为[batch, vocSize]的Tensor。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_top_k_top_p_sample](examples/test_aclnn_top_k_top_p_sample.cpp) | 通过[aclnnTopKTopPSample](docs/aclnnTopKTopPSample.md)接口方式调用TopKTopPSample算子。 |
| 图模式 | [test_aclnn_top_k_top_p_sample](examples/test_aclnn_top_k_top_p_sample.cpp)  | 通过[算子IR](op_graph/top_k_top_p_sample_proto.h)构图方式调用TopKTopPSample算子。         |
