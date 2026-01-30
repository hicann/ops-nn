# TopKTopPSampleV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 算子功能：根据输入词频logits、topK/topP/minP采样参数、随机采样权重分布q，进行topK-topP-minP-sample采样计算。当输入isNeedSampleResult为false时，输出每个batch的最大词频logitsSelectIdx，以及topK-topP-minP采样后的词频分布logitsTopKPSelect；当输入isNeedSampleResult为true时，输出topK-topP-minP采样后的中间计算结果logitsIdx和logitsSortMasked，其中logitsSortMasked为词频logits经过topK-topP-minP采样计算后的中间结果，logitsIdx为logitsSortMasked在logits中对应的索引。

  算子包含四个可单独使能，但上下游处理关系保持不变的采样算法（从原始输入到最终输出）：TopK采样、TopP采样、MinP采样、指数采样（本文档中Sample所指）。目前支持以下计算场景。如下表所示：
  | 计算场景 | TopK采样 | TopP采样 | minP采样 | 指数分布采样 | 输出中间计算结果 |备注|
  | :-------:| :------:|:-------:|:-------:|:-------:|:-------:|:-------:|
  |Softmax-Argmax采样|×|×|×|×|×|对输入logits按每个batch，取SoftMax后取最大结果|
  |topK采样|√|×|×|×|×|对输入logits按每个batch，取前topK[batch]个最大结果|
  |topP采样|×|√|×|×|×|对输入logits按每个batch从大到小排序，取累加值大于等于topP[batch]值的前n个结果进行采样|
  |Sample采样|×|×|×|√|×|对输入logits按每个batch，进行Softmax后与q进行除法取最大结果|
  |topK-topP采样|√|√|×|×|×|对输入logits按每个batch，先进行topK采样，再进行topP采样后取最大结果|
  |topK-Sample采样|√|×|×|√|×|对输入logits按每个batch，先进行topK采样，再进行Sample采样后取最大结果|
  |topP-Sample采样|×|√|×|√|×|对输入logits按每个batch，先进行topP采样，再进行Sample采样后取最大结果|
  |topK-topP-Sample采样|√|√|×|√|×|对输入logits按每个batch，先进行topK采样，再进行topP采样，最后进行Sample采样后取最大结果|
  |topK-topP-minP采样-中间结果|√|√|√|×|√|对输入logits按每个batch，先进行topK采样，再进行topP采样，最后进行minP采样，输出中间计算结果|
  |topK-minP采样-中间结果|√|×|√|×|√|对输入logits按每个batch，先进行topK采样，再进行minP采样，输出中间计算结果|
  |topK-topP采样-中间结果|√|√|×|×|√|对输入logits按每个batch，先进行topK采样，再进行minP采样，输出中间计算结果|
  |topK采样-中间结果|√|×|×|×|√|对输入logits按每个batch，进行topK采样，输出中间计算结果|
- 计算公式：

  输入logits为大小为[batch, voc_size]的词频表，其中每个batch对应一条输入序列，而voc_size则是约定每个batch的统一长度。<br>
logits中的每一行logits[batch][:]根据相应的topK[batch]、topP[batch]、q[batch, :]，执行不同的计算场景。<br>
下述公式中使用b和v来分别表示batch和voc_size方向上的索引。

  TopK采样

  1. 按分段长度v采用分段topk归并排序，用{s-1}块的topK对当前{s}块的输入进行预筛选，渐进更新单batch的topK，减少冗余数据和计算。
  2. topK[batch]对应当前batch采样的k值，有效范围为1≤topK[batch]≤min(voc_size[batch], ks_max)，如果top[k]超出有效范围，则视为跳过当前batch的topK采样阶段，也同样会则跳过当前batch的排序，将输入logits[batch]直接传入下一模块。

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
  v = 8 * \text{ks\_max}
  $$
  ks_max有效取值范围[1,1024]，默认为1024，并且需要向上对齐到8的整数倍。
  * 生成需要过滤的mask

  $$
  sortedValue[b] = sort(topKValue[b], descendant)
  $$

  $$
  topKMask = sortedValue \geq topKValue
  $$

  * 将小于阈值的部分通过mask置为defLogit:

  $$
  sortedValue[b][v]=
  \begin{cases}
  defLogit & \text{topKMask[b][v] = false} \\
  sortedValue[b][v] & \text{topKMask[b][v] = true} &
  \end{cases}
  $$

  * 其中defLogit取决于入参约束属性input_is_logits，该属性控制输入Logits和输出logits_top_kp_select的归一化：
  $$
    \text{defLogit} = 
    \begin{cases} 
    -inf, & \text{inputIsLogits} = \text{true} \\
    0, & \text{inputIsLogits} = \text{false}
    \end{cases}
  $$

  TopP采样
  * 根据入参约束属性inputIsLogits，如果该属性为True，则对排序后结果进行归一化：
    $$
    \text{logit\_sortProb} = 
    \begin{cases}
    \text{softmax}(\text{logits\_sort}), & \text{inputIsLogits} = \text{True} \\
    \text{logits\_sort}, & \text{inputIsLogits} = \text{False}
    \end{cases}
    $$

  * 根据输入`top_p[b]`的数值，本模块的处理策略如下：

    | 参数类型 | ≤0 | 有效域 | 无效域 |
    | :-------:| :------:|:-------:|:-------:|
    |`top_p[b]`|保留1个最大词频token|0<top_p<1,执行topP采样|top_p≥1,跳过topP采样|

  * 如果执行常规topP采样，且如果前序topK环节已有排序输出结果，则根据topK采样输出计算累积词频，并根据top_p截断采样：
    $$
    topPMask[b] =
    \begin{cases}
    0, & \sum_{\text{topKMask}[b]}^{} \text{logits\_sortProb}[b][*] > p[b] \\
    1, & \sum_{\text{topKMask}[b]}^{} \text{logits\_sortProb}[b][*] \leq p[b]
    \end{cases}
    $$
  * 如果执行常规topP采样，但前序topK环节被跳过，则计算top-p的mask:
    $$
    topPMask[b] =
    \begin{cases}
    topKMask[b][0:GuessK], & \sum_{\text{GuessK}}^{} probValue[b][*] \ge p[b] \\
    probSum[b][v] \le 1 - p[b], & \text{others}
    \end{cases}
    $$
  * 将需要过滤的位置设置为默认无效值defLogit，得到logits_sort，记为sortedValue[b][v]:
  $$
  sortedValue[b][v] =
  \begin{cases}
  defLogit & \quad \text{topPMask}[b][v] = \text{false} \\
  logit\_sortProb[b][v] & \quad \text{topPMask}[b][v] = \text{true}
  \end{cases}
  $$
  * 取过滤后sortedValue[b][v]每行中前topK个元素，查找这些元素在输入中的原始索引，整合为logits_idx:
  $$
  logitsIdx[b][v] = Index(sortedValue[b][v] \in Logits)
  $$
  * 使用截断后的sortedValue作为logitsSortMasked：
  $$
  logitsSortMasked[b,:] = sortedValue[b]
  $$
  minP采样
  * 如果min_ps[b]∈(0, 1)，则执行min_p采样：
    $$
    \text{logitsMax}[b] = \text{Max}(\text{logitsSortMasked}[b])
    $$
    $$
    \text{minPThd} = \text{logitsMax}[b] * \text{minPs}[b]
    $$
    $$
    \text{minPMask}[b] = 
    \begin{cases} 
    0, & \text{logitsSortMasked}[b] < \text{minPThd} \\
    1, & \text{logitsSortMasked}[b] \geq \text{minPThd}
    \end{cases}
    $$
    $$
    \text{logitsSortMasked}[b,:] = 
    \begin{cases} 
    \text{defLogit}, & \text{minPMask}[b] = 0 \\
    \text{logitsSortMasked}[b,:], & \text{minPMask}[b] = 1
    \end{cases}
    $$
  * 其他情况：
    $$
    \text{logitsSortMasked}[b, :] = 
    \begin{cases}
        \text{logitsSortMasked}[b, :], & \text{if } minPs[b] \leq 0 \\
        \max(\text{logitsSortMasked}[b, :]), & \text{if } minPs[b] \geq 1
    \end{cases}
    $$
    min_ps[b]≥1时，每个batch仅取1个最大token，其余位置填充defLogit。

  可选输出
  * 如果​入参属性IsNeedLogits=True，则使用topK-topP-minP联合采样后的logitsIndexMasked，进行`logits_top_kp_select`输出。
    $$
    \text{logitsIndex}[b][v] = \text{Index}(\text{logitsSortMasked}[b][v] \in \text{Logits})
    $$
    $$
    \text{logitsIndexMasked}[b,:] = \text{logitsIndex}[b,:] * \text{topKMask}[b] * \text{topPMask}[b] * \text{minPMask}[b]
    $$
    其中，topK、topP、minP采样环节如果被跳过，则相应mask为全1。
  * 接下来使用logitsIndexMasked对输入Logits进行Select，过滤输入Logits中的高频token作为`logits_top_kp_select`输出：
    $$
    \text{logitsTopKpSelect}[b][v] = 
    \begin{cases} 
    \text{logits}[b][v], & \text{if } logitsIndexMasked[b,v] = \text{True} \\
    \text{defLogit}, & \text{if } logitsIndexMasked[b,v] = \text{False}
    \end{cases}
    $$

  后继处理
  * 此阶段输入为前序对前序topK-topP-minP采样的联合结果logitsSortMasked。
  * 此处输入须要确保logitsSortMasked∈(0,1)，根据输入Logits的实际情况，配置入参约束属性inputIsLogits，即：
    $$
    \text{inputIsLogits} = 
    \begin{cases}
    True, & \text{Logits} \notin [0,1] \\
    False, & \text{Logits} \in [0,1]
    \end{cases}
    $$
    使得
    $$
    \text{probs}[b] = \text{logitsSortMasked}[b, :]
    $$
    接下来有三种模式：None，QSample，输出中间结果，通过入参约束属性isNeedSampleResult和是否输入q加以控制。
  * None:
  * isNeedSampleResult为false，且不输入q时为该模式。该模式下直接对每个batch通过Argmax取最大元素和索引，并通过gatherOut输出。
    $$
    \text{logitsSelectIdx}[b] = \text{LogitsIdx}[b]\left[\text{ArgMax}(\text{probs}[b][:])\right]
    $$
  * QSample：
  * isNeedSampleResult为false，且输入q时为该模式。该模式先对probs进行指数分布采样：
    $$
    qCnt = \text{Sum}(\text{MinPMask} == 1)
    $$
    $$
    \text{probsOpt}[b] = \frac{\text{probs}[b]}{q[b, :qCnt] + \text{eps}}
    $$
  * 再进行Argmax-GatherOut输出结果：
    $$
    \text{logitsSelectIdx}[b] = \text{LogitsIdx}[b][\text{ArgMax}(\text{probsOpt}[b][:])]
    $$
  * 输出中间结果:
  * isNeedSampleResult为true时，为该模式。此时会输出经过采样后的logitsSortMasked及其在输入中的原始索引logitsIdx：

    $$
    \text{logitsSortMasked}[b, v] = 
    \begin{cases}
        \text{logitsSortMasked}[b, v], & \text{if } \text{minPMask}[b, v] = 1 \\
        0, & \text{if } \text{minPMask}[b, v] = 0
    \end{cases}
    $$

    $$
    logitsIdx[b][v] = Index(logitsSortMasked[b][v])
    $$

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
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
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
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
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
        <td>minPs</td>
        <td>输入</td>
        <td>表示每个batch采样的minP值。</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>-</td>
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
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>topKGuess</td>
        <td>输入</td>
        <td>表示每个batch在尝试topP部分遍历采样logits时的候选logits大小，对应公式中的`GuessK`。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ksMax</td>
        <td>输入</td>
        <td>表示每个batch在topK采样时最大topK值，必须为正整数。</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputIsLogits</td>
        <td>输入</td>
        <td>表示输入的logits是否未进行归一化，默认为true。</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>isNeedSampleResult</td>
        <td>输入</td>
        <td>表示是否输出中间计算结果，默认为false。</td>
        <td>BOOL</td>
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
      <tr>
        <td>logitsIdx</td>
        <td>输出</td>
        <td>表示经过topK-topP-minP计算流程后，每个batch的中间采样结果在输入logits中的位置索引。</td>
        <td>INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>logitsSortMasked</td>
        <td>输出</td>
        <td>表示经过topK-topP-minP计算流程后，每个batch的中间采样结果。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody></table>


## 约束说明
  * 输入值域限制：
    * 对于所有参数，它们的尺寸必须满足，batch>0，0<vocSize<=2^20。
    * ksMax的值域为[1, 1024]。
  * 输入shape限制：
    * logits、q、logitsTopKPselect、logitsIdx、logitsSortMasked的尺寸和维度必须完全一致，目前仅支持两维。
    * llogits、topK、topP、minPs、logitsSelectIdx、logitsIdx、logitsSortMasked除最后一维以外的所有维度必须顺序和大小完全一致。目前logits只能是2维，topK、topP、logitsSelectIdx必须是1维非空Tensor。logits、topK、topP不允许空Tensor作为输入，如需跳过相应模块，需按相应规则设置输入。
  * 其他限制：  
    * 如果需要单独跳过topK模块，请传入[batch, 1]大小的Tensor，并使每个元素均为无效值。
    * 如果min(ksMaxAligned, 1024)<topK[batch]<vocSize[batch]，则视为选择当前batch的全部有效元素并跳过topK采样。其中ksMaxAligned为ksMax向上对齐到8的整数倍。
    * 如果需要单独跳过topP模块，请传入[batch, 1]大小的Tensor，并使每个元素均≥1。
    * 如果需要单独跳过minP模块，请传入`minPs=nullptr`或者传入[batch, 1]大小的Tensor，并使每个元素均≤0
    * 如果需要单独跳过sample模块，传入`q=nullptr`即可；如需使用sample模块，则必须传入尺寸为[batch, vocSize]的Tensor。
    * 如果需要输出中间结果，isNeedSampleResult设为true，并且传入`q=nullptr`，此时logitsSelectIdx不输出。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_top_k_top_p_sample_v2](examples/test_aclnn_top_k_top_p_sample_v2.cpp) | 通过[aclnnTopKTopPSampleV2](docs/aclnnTopKTopPSampleV2.md)接口方式调用TopKTopPSampleV2算子。 |