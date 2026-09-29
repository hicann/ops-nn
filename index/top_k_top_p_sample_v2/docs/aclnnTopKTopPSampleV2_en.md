# aclnnTopKTopPSampleV2

[📄 View Source Code](https://gitcode.com/cann/ops-nn/tree/master/index/top_k_top_p_sample_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                         |    ×  |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×   |

## Function

- API function:
  Performs topK-topP-minP-sample sampling calculation based on the input word frequency logits, topK/topP/minP sampling parameters, and random sampling weight distribution q. If isNeedSampleResult is set to false, the maximum word frequency logitsSelectIdx of each batch and the word frequency distribution logitsTopKPSelect after topK-topP-minP sampling are output. If isNeedSampleResult is set to true, the intermediate calculation results logitsIdx and logitsSortMasked after topK-topP-minP sampling are output. logitsSortMasked is the intermediate result of the word frequency logits after topK-topP-minP sampling calculation, and logitsIdx is the index of logitsSortMasked in logits.

  The operator contains four sampling algorithms (from the original input to the final output) that can be enabled separately but the upstream and downstream processing relationships remain unchanged: TopK sampling, TopP sampling, MinP sampling, and exponential sampling (as described in this document). Currently, the following computing scenarios are supported. as follows.
  
  | Compute Scenario| TopK Sampling| TopP Sampling| Min-P Sampling| Exponential Distribution Sampling| Output intermediate calculation results.|Remarks|
  | :-------:| :------:|:-------:|:-------:|:-------:|:-------:|:-------:|
  |Softmax-Argmax Sampling|×|×|×|×|×|For each batch of input logits, obtains the maximum result after SoftMax is performed.|
  |TopK sampling|√|×|×|×|×|For each batch of input logits, obtains the topK[batch] maximum results.|
  |TopP sampling|×|√|×|×|×|For each batch of input logits, sorts the input logits in descending order, and samples the first n results whose accumulated value is greater than or equal to topP[batch].|
  |Sample sampling|×|×|×|√|×|For each batch of input logits, performs Softmax, divides the result by q, and obtains the maximum result.|
  |TopK-TopP sampling|√|√|×|×|×|For each batch of input logits, performs topK sampling and then topP sampling, and obtains the maximum result.|
  |TopK-Sample sampling|√|×|×|√|×|For each batch of input logits, performs topK sampling and then Sample sampling, and obtains the maximum result.|
  |TopP-Sample sampling|×|√|×|√|×|For each batch of input logits, performs topP sampling and then Sample sampling, and obtains the maximum result.|
  |TopK-TopP-Sample sampling|√|√|×|√|×|For each batch of input logits, performs topK sampling, topP sampling, and then Sample sampling, and obtains the maximum result.|
  |TopK-topP-minP sampling - intermediate result|√|√|√|×|√|Performs topK sampling on the input logits for each batch, performs topP sampling, performs minP sampling, and outputs the intermediate calculation result.|
  |TopK-minP sampling - intermediate result|√|×|√|×|√|Performs topK sampling on the input logits for each batch, performs minP sampling, and outputs the intermediate calculation result.|
  |TopK-topP sampling - intermediate result|√|√|×|×|√|Performs topK sampling on the input logits for each batch, performs minP sampling, and outputs the intermediate calculation result.|
  |TopK sampling - intermediate result|√|×|×|×|√|Performs topK sampling on the input logits for each batch, and outputs the intermediate calculation result.|

- Formula:
The input logits are a word frequency table with a size of [batch, voc_size], where each batch corresponds to one input sequence, and voc_size is the uniform length of each batch.<br>
Each row logits[batch][:] in logits performs different calculation scenarios based on the corresponding topK[batch], topP[batch], minP[batch, :], and q[batch, :].<br>
In the following formulas, b and v are used to represent the indices in the batch and voc_size directions, respectively.

  TopK sampling

  1. Computes the topK for each segment based on the segment length v, applies merge sort to the topK results, pre-filters the input of the current {s} segments using the topK of the previous {s-1} segments, and gradually updates the topK of a single batch to reduce data redundancy and computation.
  2. topK[batch] corresponds to the k value sampled in the current batch. The valid range is 1 ≤ topK[batch] ≤ min(voc_size[batch], ks_max). If top[k] is out of the valid range, the topK sampling phase of the current batch is skipped, and the current batch is also skipped. In this case, the input logits[batch] is directly transferred to the next module.

  * Divides the current batch into several sub-segments and calculates topKValue[b] in rolling mode.

  $$
  topKValue[b] = {Max(topK[b])}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ topKValue[b]\left \{s-1 \right \}  \cup \left \{ logits[b][v] \ge topKMin[b][s-1] \right \} \right \}\\
  Card(topKValue[b])=topK[b]
  $$

  Where:

  $$
  topKMin[b][s] = Min(topKValue[b]\left \{  s \right \})
  $$

  v indicates the preset fixed segment length during the rolling topK operation.

  $$
  v = 8 * \text{ks\_max}
  $$
  The value range of `ks_max` is [1, 1024] with a default value of `1024`, and it must be rounded up to a multiple of 8.

  * Generates the mask to be filtered.

  $$
  sortedValue[b] = sort(topKValue[b], descendant)
  $$

  $$
  topKMask = sortedValue \geq topKValue
  $$

  * Set the part that is less than the threshold to defLogit using the mask.

  $$
  sortedValue[b][v]=
  \begin{cases}
  defLogit & \text{topKMask[b][v] = false} \\
  sortedValue[b][v] & \text{topKMask[b][v] = true} &
  \end{cases}
  $$

  * defLogit depends on the input_is_logits attribute of the input parameter. This attribute controls the normalization of the input logits and output logits_top_kp_select.
  $$
    \text{defLogit} = 
    \begin{cases} 
    -inf, & \text{inputIsLogits} = \text{true} \\
    0, & \text{inputIsLogits} = \text{false}
    \end{cases}
  $$

  TopP sampling
  
  * Based on the inputIsLogits attribute of the input parameter, if the attribute is True, the sorted result is normalized.
    $$
    \text{logit\_sortProb} = 
    \begin{cases}
    \text{softmax}(\text{logits\_sort}), & \text{inputIsLogits} = \text{True} \\
    \text{logits\_sort}, & \text{inputIsLogits} = \text{False}
    \end{cases}
    $$

  * The processing strategy of this module varies depending on the value of the input `top_p[b]`:

    | Parameter Type| ≤0 | Valid Range| Invalid Range|
    | :-------:| :------:|:-------:|:-------:|
    |`top_p[b]`|Retain one token with the maximum logit value.|If `0< top_p <1`, perform top-P sampling.|If `top_p ≥ 1`, skip top-P sampling.|

  * If regular top-P sampling is performed and the preceding top-K stage has already produced sorted results, the cumulative probabilities are computed based on the top-K output, and sampling is truncated according to `top_p`.
    $$
    topPMask[b] =
    \begin{cases}
    0, & \sum_{\text{topKMask}[b]}^{} \text{logits\_sortProb}[b][*] > p[b] \\
    1, & \sum_{\text{topKMask}[b]}^{} \text{logits\_sortProb}[b][*] \leq p[b]
    \end{cases}
    $$
  * If regular top-P sampling is performed but the preceding top-K stage is skipped, the top-P mask is computed.
    $$
    topPMask[b] =
    \begin{cases}
    topKMask[b][0:GuessK], & \sum_{\text{GuessK}}^{} probValue[b][*] \ge p[b] \\
    probSum[b][v] \le 1 - p[b], & \text{others}
    \end{cases}
    $$
  * The positions to be filtered are set to the default invalid value defLogit, and logits_sort is obtained and recorded as sortedValue[b][v].
  $$
  sortedValue[b][v] =
  \begin{cases}
  defLogit & \quad \text{topPMask}[b][v] = \text{false} \\
  logit\_sortProb[b][v] & \quad \text{topPMask}[b][v] = \text{true}
  \end{cases}
  $$
  * Obtain the first topK elements in each row of sortedValue[b][v] after filtering, find the original indices of these elements in the input, and integrate them into logits_idx.
  $$
  logitsIdx[b][v] = Index(sortedValue[b][v] \in Logits)
  $$
  * Use the truncated sortedValue as logitsSortMasked.
  $$
  logitsSortMasked[b,:] = sortedValue[b]
  $$
  minP sampling
  * If `min_ps[b] ∈ (0, 1)`, perform min-P sampling.
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
  * In other cases:
    $$
    \text{logitsSortMasked}[b, :] = 
    \begin{cases}
        \text{logitsSortMasked}[b, :], & \text{if } minPs[b] \leq 0 \\
        \max(\text{logitsSortMasked}[b, :]), & \text{if } minPs[b] \geq 1
    \end{cases}
    $$
    When `min_ps[b] ≥ 1`, only 1 token with the maximum logit value is retained for each batch, and all other positions are filled with `defLogit`.

  Optional output

  * If the input parameter IsNeedLogits is set to True, logitsIndexMasked generated after topK-topP-minP joint sampling is used for `logits_top_kp_select` output.
    $$
    \text{logitsIndex}[b][v] = \text{Index}(\text{logitsSortMasked}[b][v] \in \text{Logits})
    $$
    $$
    \text{logitsIndexMasked}[b,:] = \text{logitsIndex}[b,:] * \text{topKMask}[b] * \text{topPMask}[b] * \text{minPMask}[b]
    $$
    If the top-K, top-P, or min-P sampling stage is skipped, its corresponding mask is set to all ones.
  * The logitsIndexMasked is used to select the input logits and filter out high-frequency tokens in the input logits as the `logits_top_kp_select` output.
    $$
    \text{logitsTopKpSelect}[b][v] = 
    \begin{cases} 
    \text{logits}[b][v], & \text{if } logitsIndexMasked[b,v] = \text{True} \\
    \text{defLogit}, & \text{if } logitsIndexMasked[b,v] = \text{False}
    \end{cases}
    $$

  Subsequent processing

  * The input to this stage is `logitsSortMasked`, which is the combined result of the preceding top-K, top-P, and min-P sampling stages.
  * Ensure that logitsSortMasked∈(0,1) is used as the input. The inputIsLogits attribute is configured based on the actual input logits. That is:
    $$
    \text{inputIsLogits} = 
    \begin{cases}
    True, & \text{Logits} \notin [0,1] \\
    False, & \text{Logits} \in [0,1]
    \end{cases}
    $$
    Ensure that
    $$
    \text{probs}[b] = \text{logitsSortMasked}[b, :]
    $$
    There are three modes: None, QSample, and output intermediate results. The modes are controlled by the input parameter constraints isNeedSampleResult and whether q is input.
  * None:
  * This mode is used when isNeedSampleResult is false and q is not input. In this mode, the maximum element and index are obtained for each batch by using Argmax, and the result is output through gatherOut.
    $$
    \text{logitsSelectIdx}[b] = \text{LogitsIdx}[b]\left[\text{ArgMax}(\text{probs}[b][:])\right]
    $$
  * QSample:
  * This mode is used when isNeedSampleResult is false and q is input. In this mode, exponential distribution sampling is performed on probs first.
    $$
    qCnt = \text{Sum}(\text{MinPMask} == 1)
    $$
    $$
    \text{probsOpt}[b] = \frac{\text{probs}[b]}{q[b, :qCnt] + \text{eps}}
    $$
  * Then, generate the output through `Argmax` and `GatherOut`:
    $$
    \text{logitsSelectIdx}[b] = \text{LogitsIdx}[b][\text{ArgMax}(\text{probsOpt}[b][:])]
    $$
  * Intermediate result output:
  * This mode is used when isNeedSampleResult is true. In this case, the sampled logitsSortMasked and its original index logitsIdx in the input are output.

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

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnTopKTopPSampleV2GetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, `aclnnTopKTopPSampleV2` is called to perform computation.

```Cpp
aclnnStatus aclnnTopKTopPSampleV2GetWorkspaceSize(
  const aclTensor *logits, 
  const aclTensor *topK, 
  const aclTensor *topP, 
  const aclTensor *q,
  const aclTensor *minPs, 
  double           eps, 
  bool             isNeedLogits, 
  int64_t          topKGuess,
  int64_t          ksMax,
  bool             inputIsLogits,
  bool             isNeedSampleResult,
  const aclTensor *logitsSelectIdx, 
  const aclTensor *logitsTopKPSelect,
  const aclTensor *logitsIdx, 
  const aclTensor *logitsSortMasked, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)

```

```Cpp
aclnnStatus aclnnTopKTopPSampleV2(
  void           *workspace, 
  uint64_t        workspaceSize, 
  aclOpExecutor  *executor, 
  aclrtStream     stream)

```

## aclnnTopKTopPSampleV2GetWorkspaceSize

- **Parameter description**:

  <table style="undefined;table-layout: fixed; width: 1503px"><colgroup>
      <col style="width: 146px">
      <col style="width: 120px">
      <col style="width: 271px">
      <col style="width: 392px">
      <col style="width: 228px">
      <col style="width: 101px">
      <col style="width: 100px">
      <col style="width: 145px">
      </colgroup>
      <thead>
        <tr>
          <th>Name</th>
          <th>Input/Output</th>
          <th>Description</th>
          <th>Usage</th>
          <th>Data Type</th>
          <th>Data Format</th>
          <th>Dimension (Shape)</th>
          <th>Non-contiguous Tensor</th>
        </tr></thead>
      <tbody>
      <tr>
        <td>logits</td>
        <td>Input</td>
        <td>Input word frequency to be sampled. The word frequency index is fixed to the last dimension, corresponding logits in the formula.</td>
        <td><ul><li>Empty tensors are not supported.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>topK</td>
        <td>Input</td>
        <td>k value sampled for each batch. It corresponds to topK[b] in the formula.</td>
        <td><ul><li>Empty tensors are not supported. </li><li>The shape must be the same as the first n–1 dimensions of logits.</li></ul></td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>topP</td>
        <td>Input</td>
        <td>p value sampled for each batch. It corresponds to topP[b] in the formula.</td>
        <td><ul><li>Empty tensors are not supported. </li><li>The shape must be the same as the first n–1 dimensions of logits, and the data type must be the same as that of logits.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>q</td>
        <td>Input</td>
        <td>Exponential sampling matrix output by topK-topP-minP sampling. It corresponds to q in the formula.</td>
        <td><ul><li>The shape must be the same as that of `logits`.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>minPs</td>
        <td>Input</td>
        <td>Minimum P value sampled in each batch. It corresponds to `minPs[b]` in the formula.</td>
        <td><ul><li>The shape must be the same as the first n-1 dimensions of `logits`, and the data type must be the same as that of `logits`.</li></ul></td>
        <td>FLOAT16, BFLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>eps</td>
        <td>Input</td>
        <td>Coefficient for preventing division by zero in softmax and weight sampling. The recommended value is 1e-8.</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>isNeedLogits</td>
        <td>Input</td>
        <td>This parameter indicates the output condition of logitsTopKPSelect. You are advised to set this parameter to 0.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>topKGuess</td>
        <td>Input</td>
        <td>Size of candidate logits when each batch attempts to traverse and sample logits in the topP part.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ksMax</td>
        <td>Input</td>
        <td>Maximum topK value for each batch during topK sampling. The value must be a positive integer.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>inputIsLogits</td>
        <td>Input</td>
        <td>Whether the input logits are not normalized. The default value is true.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>isNeedSampleResult</td>
        <td>Input</td>
        <td>Whether to output intermediate calculation results. The default value is false.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>logitsSelectIdx</td>
        <td>Output</td>
        <td>Indicates the index of the element with the maximum word frequency in each batch in the input logits, that is, max(probsOpt[batch, :]) after the topK-topP-minP-sample calculation process.</td>
        <td><ul><li>Empty tensors are not supported. </li><li>The shape must be the same as the first n–1 dimensions of logits.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>logitsTopKPSelect</td>
        <td>Output</td>
        <td>Indicates the remaining logits that are not filtered out in the input logits after the topK-topP-minP calculation process.</td>
        <td><ul><li>Empty tensors are not supported. </li><li>The shape must be the same as that of `logits`.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>logitsIdx</td>
        <td>Output</td>
        <td>Index of the intermediate sampling result of each batch in the input logits after the topK-topP-minP calculation process.</td>
        <td><ul><li>Empty tensors are not supported. </li><li>The shape must be the same as that of `logits`.</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>logitsSortMasked</td>
        <td>Output</td>
        <td>Intermediate sampling result of each batch after the topK-topP-minP calculation process.</td>
        <td><ul><li>Empty tensors are not supported. </li><li>The shape must be the same as that of `logits`.</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Size of the workspace to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Operator executor, containing the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      </tbody>
      </table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md). 

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input logits, topK, or topP is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of logits, topK, topP, q, and minPs are not supported.</td>
    </tr>
    <tr>
      <td>The dimensions or sizes of logits and q are inconsistent.</td>
    </tr>
    <tr>
      <td>The dimensions of topK, topP, and minPs are inconsistent with the first n-1 dimensions of logits.</td>
    </tr>
    <tr>
      <td>The data types of logits, topP, and minPs are inconsistent.</td>
    </tr>
  </tbody></table>
  
## aclnnTopKTopPSampleV2

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnTopKTopPSampleV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - The default deterministic implementation of aclnnTopKTopPSampleV2 is used.
- For all sampling parameters, their sizes must meet the following requirements: batch > 0 and 0 < vocSize <= 2^20.
- topK only accepts non-negative values as valid inputs. If 0 or a negative value is passed, the sampling of the corresponding batch is skipped.
- The sizes and dimensions of logits, q, logitsTopKPselect, logitsIdx, and logitsSortMasked must be the same.
- The sizes and dimensions of logits, topK, topP, minPs, logitsSelectIdx, logitsIdx, and logitsSortMasked, except the last dimension, must be the same. Currently, **logits** can only be two-dimensional, and **topK**, **topP**, and **logitsSelectIdx** must be one-dimensional non-empty tensors. Empty tensors cannot be used as the input of **logits**, **topK**, or **topP**. If the corresponding module needs to be skipped, set the input as required.
- To skip the topK module separately, pass a tensor of size [batch, 1] and set each element to an invalid value.
- If min(ksMaxAligned, 1024)<topK[batch]<vocSize[batch], all valid elements in the current batch are selected and the topK sampling is skipped. ksMaxAligned is the value of ksMax rounded up to the nearest multiple of 8. The value range of ksMax is [1, 1024].
- To skip the topP module separately, pass a tensor of size [batch, 1] and set each element to a value greater than or equal to 1.
- To skip the minP module, pass `minPs=nullptr` or a tensor of size [batch, 1] and ensure that each element is less than or equal to 0.
- To skip the sample module separately, pass q=nullptr. To use the sample module, pass a tensor of size [batch, vocSize].
- If intermediate results are required, set isNeedSampleResult to true and pass `q=nullptr`. In this case, logitsSelectIdx is not output.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_top_k_top_p_sample_v2.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // (Boilerplate) Initialize resources.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy the data on the host to the memory on the device. 
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
    // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    std::vector<int64_t> logitsShape = {48, 131072};
    std::vector<int64_t> topKPShape = {48};
    long long vocShapeSize = GetShapeSize(logitsShape);
    long long batchShapeSize = GetShapeSize(topKPShape);

    void* logitsDeviceAddr = nullptr;
    void* topKDeviceAddr = nullptr;
    void* topPDeviceAddr = nullptr;
    void* qDeviceAddr = nullptr;
    void* minPsDeviceAddr = nullptr;
    void* logitsSelectedIdxDeviceAddr = nullptr;
    void* logitsTopKPSelectDeviceAddr = nullptr;
    void* logitsIdxDeviceAddr = nullptr;
    void* logitsSortMaskedDeviceAddr = nullptr;

    aclTensor* logits = nullptr;
    aclTensor* topK = nullptr;
    aclTensor* topP = nullptr;
    aclTensor* q = nullptr;
    aclTensor* minPs = nullptr;
    aclTensor* logitsSelectedIdx = nullptr;
    aclTensor* logitsTopKPSelect = nullptr;
    aclTensor* logitsIdx = nullptr;
    aclTensor* logitsSortMasked = nullptr;
    std::vector<int16_t> logitsHostData(48 * 131072, 1);
    std::vector<int32_t> topKHostData(48, 128);
    std::vector<int16_t> topPHostData(48, 1);
    std::vector<float> qHostData(48 * 131072, 1.0f);
    std::vector<int16_t> minPsHostData(48, 1);

    std::vector<int64_t> logitsSelectedIdxHostData(48, 0);
    std::vector<float> logitsTopKPSelectHostData(48 * 131072, 0);
    std::vector<int64_t> logitsIdxHostData(48 * 131072, 0);
    std::vector<float> logitsSortMaskedHostData(48 * 131072, 0);

    float eps = 1e-8;
    int64_t isNeedLogits = 0;
    int32_t topKGuess =32;
    int32_t ks_max = 1024;
    bool inputIsLogits = true;
    bool isNeedSampleResult = false;

    // Create a logitsaclTensor.
    ret = CreateAclTensor(logitsHostData, logitsShape, &logitsDeviceAddr, aclDataType::ACL_BF16, &logits);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a topKaclTensor.
    ret = CreateAclTensor(topKHostData, topKPShape, &topKDeviceAddr, aclDataType::ACL_INT32, &topK);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a topPaclTensor.
    ret = CreateAclTensor(topPHostData, topKPShape, &topPDeviceAddr, aclDataType::ACL_BF16, &topP);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a q aclTensor.
    ret = CreateAclTensor(qHostData, logitsShape, &qDeviceAddr, aclDataType::ACL_FLOAT, &q);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create minps aclTensor.
    ret = CreateAclTensor(minPsHostData, topKPShape, &minPsDeviceAddr, aclDataType::ACL_BF16, &minPs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create logitsSelected aclTensor.
    ret = CreateAclTensor(logitsSelectedIdxHostData, topKPShape, &logitsSelectedIdxDeviceAddr, aclDataType::ACL_INT64, &logitsSelectedIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a logitsTopKPSelect aclTensor.
    ret = CreateAclTensor(logitsTopKPSelectHostData, logitsShape, &logitsTopKPSelectDeviceAddr, aclDataType::ACL_FLOAT, &logitsTopKPSelect);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create logitsIdx aclTensor.
    ret = CreateAclTensor(logitsIdxHostData, logitsShape, &logitsIdxDeviceAddr, aclDataType::ACL_INT64, &logitsIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create logitsSortMasked aclTensor.
    ret = CreateAclTensor(logitsSortMaskedHostData, logitsShape, &logitsSortMaskedDeviceAddr, aclDataType::ACL_FLOAT, &logitsSortMasked);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first part of the aclnnTopKTopPSampleV2 API.
    ret = aclnnTopKTopPSampleV2GetWorkspaceSize(logits, topK, topP, q, minPs, eps, isNeedLogits, topKGuess, ks_max, inputIsLogits, 
      isNeedSampleResult, logitsSelectedIdx, logitsTopKPSelect, logitsIdx, logitsSortMasked, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTopKTopPSampleV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second segment of the aclnnTopKTopPSampleV2 API.
    ret = aclnnTopKTopPSampleV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTopKTopPSampleV2 failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(topKPShape);
    std::vector<int64_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), logitsSelectedIdxDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
    }

    // 6. Release the aclTensor. Modify the code based on the API definition.
    aclDestroyTensor(logits);
    aclDestroyTensor(topK);
    aclDestroyTensor(topP);
    aclDestroyTensor(q);
    aclDestroyTensor(logitsSelectedIdx);
    aclDestroyTensor(logitsTopKPSelect);
    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(logitsDeviceAddr);
    aclrtFree(topKDeviceAddr);
    aclrtFree(topPDeviceAddr);
    aclrtFree(qDeviceAddr);
    aclrtFree(logitsSelectedIdxDeviceAddr);
    aclrtFree(logitsTopKPSelectDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
    }
  ```
