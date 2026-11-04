# TopKTopPSampleV2

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |     √     |
| <term>Atlas A3 products</term>    |    √    |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×   |

## Function

- Operator function: Performs topK-topP-minP-sample sampling computation based on the input token frequency logits, the topK/topP/minP sampling parameters, and the random sampling weight distribution q. When the input isNeedSampleResult is false, the operator outputs the maximum token frequency logitsSelectIdx of each batch and the token frequency distribution logitsTopKPSelect after topK-topP-minP sampling. When the input isNeedSampleResult is true, the operator outputs the intermediate computation results logitsIdx and logitsSortMasked after topK-topP-minP sampling. Here, logitsSortMasked is the intermediate result of the token frequency logits after topK-topP-minP sampling computation, and logitsIdx is the index of logitsSortMasked in logits.

  The operator contains four sampling algorithms (from the raw input to the final output) that can be enabled separately while the upstream/downstream processing relationship remains unchanged: TopK sampling, TopP sampling, MinP sampling, and exponential sampling (Sample in this document). Currently, the following computation scenarios are supported, as shown in the following table:

  | Computation scenario | TopK sampling | TopP sampling | minP sampling | Exponential distribution sampling | Output intermediate computation results |Remarks|
  | :-------:| :------:|:-------:|:-------:|:-------:|:-------:|:-------:|
  |Softmax-Argmax sampling|×|×|×|×|×|For the input logits, takes SoftMax for each batch and then takes the maximum result|
  |topK sampling|√|×|×|×|×|For the input logits, takes the first topK[batch] maximum results for each batch|
  |topP sampling|×|√|×|×|×|For the input logits, sorts each batch in descending order, takes the first n results whose cumulative value is greater than or equal to topP[batch], and performs sampling|
  |Sample sampling|×|×|×|√|×|For the input logits, performs Softmax for each batch, then divides the result by q and takes the maximum result|
  |topK-topP sampling|√|√|×|×|×|For the input logits, performs topK sampling for each batch first, then performs topP sampling, and takes the maximum result|
  |topK-Sample sampling|√|×|×|√|×|For the input logits, performs topK sampling for each batch first, then performs Sample sampling, and takes the maximum result|
  |topP-Sample sampling|×|√|×|√|×|For the input logits, performs topP sampling for each batch first, then performs Sample sampling, and takes the maximum result|
  |topK-topP-Sample sampling|√|√|×|√|×|For the input logits, performs topK sampling for each batch first, then performs topP sampling, finally performs Sample sampling, and takes the maximum result|
  |topK-topP-minP sampling-intermediate results|√|√|√|×|√|For the input logits, performs topK sampling for each batch first, then performs topP sampling, finally performs minP sampling, and outputs intermediate computation results|
  |topK-minP sampling-intermediate results|√|×|√|×|√|For the input logits, performs topK sampling for each batch first, then performs minP sampling, and outputs intermediate computation results|
  |topK-topP sampling-intermediate results|√|√|×|×|√|For the input logits, performs topK sampling for each batch first, then performs minP sampling, and outputs intermediate computation results|
  |topK sampling-intermediate results|√|×|×|×|√|For the input logits, performs topK sampling for each batch, and outputs intermediate computation results|

- Formula:

  The input logits is a token frequency table of size [batch, voc_size], where each batch corresponds to one input sequence, and voc_size is the agreed unified length of each batch. <br>
Each row logits[batch][:] in logits executes a different computation scenario according to the corresponding topK[batch], topP[batch], and q[batch, :]. <br>
The following formulas use b and v to denote the indices along the batch and voc_size directions respectively.

  TopK sampling

  1. Uses segmented topk merge sorting with the segment length v, pre-filters the input of the current {s} segment with the topK of the {s-1} segment, progressively updates the topK of a single batch, and reduces redundant data and computation.
  2. topK[batch] corresponds to the k value sampled for the current batch. The valid range is 1≤topK[batch]≤min(voc_size[batch], ks_max). If top[k] exceeds the valid range, the topK sampling stage of the current batch is regarded as skipped, the sorting of the current batch is also skipped, and the input logits[batch] is passed directly to the next module.

  * According to the value of the input `topK[b]`, the processing strategy of this module is as follows:

    | Parameter type | ≤0 | Valid range | Invalid range |
    | :-------:| :------:|:-------:|:-------:|
    |`topK[b]`|Skips topK sampling|1≤topK≤min(ksMax, 1024, vocSize), executes topK sampling|topK>min(ksMax, 1024, vocSize), skips topK sampling, and selects all valid elements|

  * Splits the current batch into several sub-segments and computes topKValue[b] in a rolling manner:

  $$
  topKValue[b] = {Max(topK[b])}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ topKValue[b]\left \{s-1 \right \}  \cup \left \{ logits[b][v] \ge topKMin[b][s-1] \right \} \right \}\\
  Card(topKValue[b])=topK[b]
  $$

  where:

  $$
  topKMin[b][s] = Min(topKValue[b]\left \{  s \right \})
  $$

  v denotes the fixed segment length preset for rolling topK:

  $$
  v = 8 * \text{ks\_max}
  $$
  The valid value range of ks_max is [1,1024], the default value is 1024, and the value must be aligned upward to an integer multiple of 8.

  * Generates the mask that needs filtering

  $$
  sortedValue[b] = sort(topKValue[b], descendant)
  $$

  $$
  topKMask = sortedValue \geq topKValue
  $$

  * Sets the part smaller than the threshold to defLogit through the mask:

  $$
  sortedValue[b][v]=
  \begin{cases}
  defLogit & \text{topKMask[b][v] = false} \\
  sortedValue[b][v] & \text{topKMask[b][v] = true} &
  \end{cases}
  $$

  * Here, defLogit depends on the input parameter constraint attribute input_is_logits. This attribute controls the normalization of the input Logits and the output logits_top_kp_select:
  $$
    \text{defLogit} =
    \begin{cases}
    -inf, & \text{inputIsLogits} = \text{true} \\
    0, & \text{inputIsLogits} = \text{false}
    \end{cases}
  $$

  TopP sampling

  * According to the input parameter constraint attribute inputIsLogits, if this attribute is True, normalizes the sorted result:
    $$
    \text{logit\_sortProb} =
    \begin{cases}
    \text{safe\_softmax}(\text{logits\_sort}), & \text{inputIsLogits} = \text{True} \\
    \text{logits\_sort}, & \text{inputIsLogits} = \text{False}
    \end{cases}
    $$

  * According to the value of the input `top_p[b]`, the processing strategy of this module is as follows:

    | Parameter type | ≤0 | Valid range | Invalid range |
    | :-------:| :------:|:-------:|:-------:|
    |`top_p[b]`|Retains 1 maximum token frequency token|0<top_p<1, executes topP sampling|top_p≥1, skips topP sampling|

  * If regular topP sampling is executed and the preceding topK stage already has sorted output results, computes the cumulative token frequency based on the topK sampling output and truncates the sampling according to top_p:
    $$
    topPMask[b] =
    \begin{cases}
    0, & \sum_{\text{topKMask}[b]}^{} \text{logits\_sortProb}[b][*] > p[b] \\
    1, & \sum_{\text{topKMask}[b]}^{} \text{logits\_sortProb}[b][*] \leq p[b]
    \end{cases}
    $$
  * If regular topP sampling is executed but the preceding topK stage is skipped, computes the top-p mask:
    $$
    topPMask[b] =
    \begin{cases}
    topKMask[b][0:GuessK], & if \space \sum_{\text{GuessK}}^{} probValue[b][*] \ge p[b] \\
    probSum[b][v] \le 1 - p[b], & if \space \text{others}
    \end{cases}
    $$
  * Sets the positions that need filtering to the default invalid value defLogit to obtain logits_sort, denoted as sortedValue[b][v]:
  $$
  sortedValue[b][v] =
  \begin{cases}
  defLogit & \quad \text{topPMask}[b][v] = \text{false} \\
  logit\_sortProb[b][v] & \quad \text{topPMask}[b][v] = \text{true}
  \end{cases}
  $$
  * Takes the first topK elements in each row of the filtered sortedValue[b][v], finds the original indices of these elements in the input, and integrates them into logits_idx:
  $$
  logitsIdx[b][v] = Index(sortedValue[b][v] \in Logits)
  $$
  * Uses the truncated sortedValue as logitsSortMasked:
  $$
  logitsSortMasked[b,:] = sortedValue[b]
  $$
  minP sampling
  * According to the value of the input `minPs[b]`, the processing strategy of this module is as follows:

    | Parameter type | ≤0 | Valid range | Invalid range |
    | :-------:| :------:|:-------:|:-------:|
    |`minPs[b]`|Skips minP sampling and keeps the preceding result unchanged|0<minPs<1, executes minP sampling|minPs≥1, retains only 1 maximum token|

  * If min_ps[b]∈(0, 1), executes min_p sampling:
    $$
    \text{logitsMax}[b] = \text{Max}(\text{logitsSortMasked}[b])
    $$

    $$
    \text{minPThd} = \text{logitsMax}[b] * \text{minPs}[b]
    $$

    $$
    \text{minPMask}[b] =
    \begin{cases}
    0, & \text{logitsSortMasked}[b][:] < \text{minPThd} \\
    1, & \text{logitsSortMasked}[b][:] \geq \text{minPThd}
    \end{cases}
    $$

    Through the gather operation, makes the minP sampling result and indices densely arranged:

    $$
    minPCnt = Sum(minPMask[b]) \\
    minPFiltIdx = Index(minPMask[b]=1)
    $$

    Pads the remaining elements:
    $$
    \text{logitsMinpFiltered}[b,:minPCnt] = logitsSortMasked[b][minPFiltIdx] \\
    \text{logitsMinpFiltered}[b,minPCnt:] = defLogit
    $$

  * Other cases:
    $$
    \text{logitsMinpFiltered}[b, :] =
    \begin{cases}
        \text{logitsSortMasked}[b, :], & \text{if } minPs[b] \leq 0 \\
        \max(\text{logitsSortMasked}[b, :]), & \text{if } minPs[b] \geq 1
    \end{cases}
    $$
    When min_ps[b]≥1, each batch takes only 1 maximum token, and the remaining positions are padded with defLogit.

  Optional output

  * If the input parameter attribute IsNeedLogits=True, uses logitsIndexMasked after the joint topK-topP-minP sampling for the `logits_top_kp_select` output.

    $$
    \text{logitsIndex}[b][v] = \text{Index}(\text{logitsMinpFiltered}[b][v] \in \text{Logits})
    $$
    $$
    \text{logitsIndexMasked}[b,:] = \text{logitsIndex}[b,:] * \text{topKMask}[b] * \text{topPMask}[b] * \text{minPMask}[b]
    $$
    Here, if the topK, topP, or minP sampling stage is skipped, the corresponding mask is all 1.

  * Then uses logitsIndexMasked to perform Select on the input Logits and filters the high-frequency tokens in the input Logits as the `logits_top_kp_select` output:
    $$
    \text{logitsTopKpSelect}[b][v] =
    \begin{cases}
    \text{logits}[b][v], & \text{if } logitsIndexMasked[b,v] = \text{True} \\
    \text{defLogit}, & \text{if } logitsIndexMasked[b,v] = \text{False}
    \end{cases}
    $$

  Follow-up processing

  * The input of this stage is logitsMinpFiltered, the joint result of the preceding `topK-topP-minP` sampling. <br>If none of the preceding sampling stages is enabled, directly uses the normalization result of the input logits:

    $$
    logitsMinpFiltered =
    \begin{cases}
    logits, & if \space inputIsLogits=False \\
    safe\_softmax(logits), & if \space inputIsLogits=True
    \end{cases}
    $$

  * Here you must ensure logitsMinpFiltered∈(0,1). According to the actual situation of the input Logits, configure the input parameter constraint attribute inputIsLogits, that is:
    $$
    \text{inputIsLogits} =
    \begin{cases}
    True, & \text{Logits} \notin [0,1] \\
    False, & \text{Logits} \in [0,1]
    \end{cases}
    $$

    so that

    $$
    \text{probs}[b] = \text{logitsMinpFiltered}[b, :]
    $$

    Next there are three modes: None, QSample, and outputting intermediate results. The modes are controlled by the input parameter constraint attribute isNeedSampleResult and whether q is input.
  * None:
  * This mode applies when isNeedSampleResult is false and q is not input. In this mode, the operator directly uses Argmax for each batch to take the maximum element and its index, and outputs them through gatherOut.
    $$
    \text{logitsSelectIdx}[b] = \text{LogitsIdx}[b]\left[\text{ArgMax}(\text{probs}[b][:])\right]
    $$
  * QSample:
  * This mode applies when isNeedSampleResult is false and q is input. In this mode, exponential distribution sampling is first performed on probs:
    $$
    qCnt = \text{Sum}(\text{MinPMask} == 1)
    $$
    $$
    \text{probsOpt}[b] = \frac{\text{probs}[b]}{q[b, :qCnt] + \text{eps}}
    $$
  * Then Argmax-GatherOut outputs the result:
    $$
    \text{logitsSelectIdx}[b] = \text{LogitsIdx}[b][\text{ArgMax}(\text{probsOpt}[b][:])]
    $$
  * Output intermediate results:
  * This mode applies when isNeedSampleResult is true. In this case, the operator outputs logitsMinpFiltered after sampling and its original index logitsIdx in the input:

    $$
    \text{logitsMinpFiltered}[b, v] =
    \begin{cases}
        \text{logitsMinpFiltered}[b, v], & \text{if } \text{minPMask}[b, v] = 1 \\
        0, & \text{if } \text{minPMask}[b, v] = 0
    \end{cases}
    $$

    $$
    logitsIdx[b][v] = Index(logitsMinpFiltered[b][v])
    $$

## Parameters

  <table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 312px">
    <col style="width: 213px">
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
        <td>logits</td>
        <td>Input</td>
        <td>The input token frequency to be sampled, corresponding to `logits` in the formula.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>topK</td>
        <td>Input</td>
        <td>Indicates the k value sampled for each batch</td>
        <td>INT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>topP</td>
        <td>Input</td>
        <td>Indicates the p value sampled for each batch.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>q</td>
        <td>Input</td>
        <td>Indicates the exponential sampling matrix output by topK-topP sampling. Its dimensions and size must be consistent with `logits`.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>minPs</td>
        <td>Input</td>
        <td>Indicates the minP value sampled for each batch.</td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>eps</td>
        <td>Input</td>
        <td>Prevents division by zero in softmax and weight sampling, corresponding to `eps` in the formula. It is recommended to set it to 1e-8.</td>
        <td>FLOAT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>isNeedLogits</td>
        <td>Input</td>
        <td>Controls the output condition of logitsTopKPselect. It is recommended to set it to 0.</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>topKGuess</td>
        <td>Input</td>
        <td>Indicates the size of the candidate logits when each batch traverses and samples logits in the topP part, corresponding to `GuessK` in the formula.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ksMax</td>
        <td>Input</td>
        <td>Indicates the maximum topK value for each batch in topK sampling. It must be a positive integer.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputIsLogits</td>
        <td>Input</td>
        <td>Indicates whether the input logits have not been normalized. The default value is true.</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>isNeedSampleResult</td>
        <td>Input</td>
        <td>Indicates whether to output intermediate computation results. The default value is false.</td>
        <td>BOOL</td>
        <td>-</td>
      </tr>
      <tr>
        <td>logitsSelectIdx</td>
        <td>Output</td>
        <td>Indicates the position index in the input logits of the maximum token frequency element max(probsOpt[batch, :]) of each batch after the topK-topP-sample computation flow. It corresponds to `logitsSelectIdx` in the formula.</td>
        <td>INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>logitsTopKPSelect</td>
        <td>Output</td>
        <td>Indicates the remaining unfiltered logits in the input logits after the topK-topP computation flow. It corresponds to `logitsTopKPSelect` in the formula.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>logitsIdx</td>
        <td>Output</td>
        <td>Indicates the position index in the input logits of the intermediate sampling result of each batch after the topK-topP-minP computation flow.</td>
        <td>INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>logitsSortMasked</td>
        <td>Output</td>
        <td>Indicates the intermediate sampling result of each batch after the topK-topP-minP computation flow.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody></table>

## Constraints

  * Input value range restrictions:
    * For logits, you must ensure that its elements, after the $exp(*)$ operation, produce a result no less than the minimum valid value of FLOAT32. Otherwise, the corresponding Safe_softmax result will be 0, which may lead to unexpected operator output.
    * For all parameters, their sizes must satisfy batch>0 and 0<vocSize<=2^20.
    * The value range of ksMax is [1, 1024].
  * Input shape restrictions:
    * The size and dimensions of logits, q, logitsTopKPselect, logitsIdx, and logitsSortMasked must be exactly the same. Currently only two dimensions are supported.
    * All dimensions except the last one of llogits, topK, topP, minPs, logitsSelectIdx, logitsIdx, and logitsSortMasked must be exactly the same in order and size. Currently logits can only be 2-dimensional, and topK, topP, and logitsSelectIdx must be 1-dimensional non-empty tensors. Empty tensors are not allowed as inputs of logits, topK, and topP. To skip the corresponding module, set the inputs according to the corresponding rules.
  * Q-Sample constraints:
    * The indices of the input tensor q are always aligned element by element to the $\text{probs}[b]$ input during the follow-up processing of the corresponding batch.
    * The $\text{probs}[b]$ input must be densely arranged, that is, valid elements are concentrated at the front of the current batch.
    * If at least one of the preceding topK or topP sampling stages of the current batch is enabled, the elements of $\text{probs}[b]$ will also be arranged in descending order.
  * Other restrictions:
    * To skip the topK module alone, pass in a tensor of size [batch, 1] and make each element an invalid value.
    * If min(ksMaxAligned, 1024)<topK[batch]<vocSize[batch], the operator regards this as selecting all valid elements of the current batch and skips topK sampling. Here ksMaxAligned is ksMax aligned upward to an integer multiple of 8.
    * To skip the topP module alone, pass in a tensor of size [batch, 1] and make each element ≥1.
    * To skip the minP module alone, pass in `minPs=nullptr` or pass in a tensor of size [batch, 1] and make each element ≤0
    * To skip the sample module alone, simply pass in `q=nullptr`. To use the sample module, you must pass in a tensor of size [batch, vocSize].
    * To output intermediate results, set isNeedSampleResult to true and pass in `q=nullptr`. In this case, you must still pass in a non-empty `logitsSelectIdx` output tensor (INT64, whose shape must be consistent with the first n-1 dimensions of `logits`). The operator does not write valid index values, and the caller must not read the tensor content.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_top_k_top_p_sample_v2](examples/test_aclnn_top_k_top_p_sample_v2.cpp) | Calls the TopKTopPSampleV2 operator through the [aclnnTopKTopPSampleV2](docs/aclnnTopKTopPSampleV2_en.md) interface. |
