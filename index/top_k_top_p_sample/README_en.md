# TopKTopPSample

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    ×  |
| <term>Atlas A3 products</term>    |    √    |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×   |

## Function

- Operator function: performs topK-topP-sample sampling computation based on the input word-frequency logits, topK/topP sampling parameters, and the random sampling weight distribution q, and outputs the maximum word-frequency logitsSelectIdx of each batch as well as the word-frequency distribution logitsTopKPSelect after topK-topP sampling.

  The operator contains three sampling algorithms that can be enabled independently while their upstream and downstream processing relationships remain unchanged (from the original input to the final output): TopK sampling, TopP sampling, and exponential sampling (Sample in this document). They can form eight computation scenarios, as shown in the following table:

  | Computation scenario | TopK sampling | TopP sampling | Exponential distribution sampling | Remarks |
  | :-------:| :------:|:-------:|:-------:|:-------:|
  |Softmax-Argmax sampling|×|×|×|For each batch of the input logits, applies SoftMax and then takes the maximum result|
  |topK sampling|√|×|×|For each batch of the input logits, takes the first topK[batch] maximum results|
  |topP sampling|×|√|×|Sorts each batch of the input logits in descending order, and samples the first n results whose accumulated value is greater than or equal to topP[batch]|
  |Sample sampling|×|×|√|For each batch of the input logits, applies Softmax and then divides by q and takes the maximum result|
  |topK-topP sampling|√|√|×|For each batch of the input logits, performs topK sampling first, then topP sampling, and takes the maximum result|
  |topK-Sample sampling|√|×|√|For each batch of the input logits, performs topK sampling first, then Sample sampling, and takes the maximum result|
  |topP-Sample sampling|×|√|√|For each batch of the input logits, performs topP sampling first, then Sample sampling, and takes the maximum result|
  |topK-topP-Sample sampling|√|√|√|For each batch of the input logits, performs topK sampling first, then topP sampling, and finally Sample sampling, and takes the maximum result|

- Formula:

  The input logits is a word-frequency table of size [batch, voc_size], where each batch corresponds to one input sequence and voc_size is the agreed unified length of each batch. <br>
  Each row logits[batch][:] in logits executes a different computation scenario according to the corresponding topK[batch], topP[batch], and q[batch, :]. <br>
  In the following formulas, b and v denote the indexes along the batch and voc_size directions respectively.

  TopK sampling

  1. Uses segment-wise topk merge sorting with the segment length v, pre-filters the input of the current {s} segment with the topK of the {s-1} segment, progressively updates the topK of a single batch, and reduces redundant data and computation.
  2. topK[batch] corresponds to the k value sampled for the current batch, with the valid range 1≤topK[batch]≤min(voc_size[batch], 1024). If top[k] exceeds the valid range, the topK sampling stage of the current batch is regarded as skipped, the sorting of the current batch is also skipped, and the input logits[batch] is passed directly to the next module.

  * Splits the current batch into several sub-segments and computes topKValue[b] in a rolling manner:

  $$
  topKValue[b] = {Max(topK[b])}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ topKValue[b]\left \{s-1 \right \}  \cup \left \{ logits[b][v] \ge topKMin[b][s-1] \right \} \right \}\\
  Card(topKValue[b])=topK[b]
  $$

  where:

  $$
  topKMin[b][s] = Min(topKValue[b]\left \{  s \right \})
  $$

  v denotes the preset fixed segment length used during rolling topK:

  $$
  v=8*1024
  $$

  * Generates the mask to be filtered:

  $$
  sortedValue[b] = sort(topKValue[b], descendant)
  $$

  $$
  topKMask[b] = sortedValue[b]<Min(topKValue[b])
  $$

  * Sets the parts below the threshold to -Inf through the mask:

  $$
  sortedValue[b][v]=
  \begin{cases}
  -Inf & \text{topKMask[b][v]=true} \\
  sortedValue[b][v] & \text{topKMask[b][v]=false} &
  \end{cases}
  $$

  * Converts the logits filtered by topK into a probability distribution along the last axis through safe\_softmax:

  $$
  probsValue[b]=sortedValue[b].safe\_softmax (dim=-1)
  $$

  * Computes the cumulative probability along the last axis (accumulating from the smallest probability):

  $$
  probsSum[b]=probsValue[b].cumsum (dim=-1)
  $$

  TopP sampling

  * If the preceding topK sampling already has sorted output results, computes the accumulated word frequency according to the output of the topK sampling and performs truncated sampling according to topP:

    $$
    topPMask[b] = probsSum[b][*] < topP[b]
    $$

  * If the topK sampling is skipped, first performs safe_softmax processing on the input logits[b]:

  $$
  logitsValue[b] = logits[b].safe\_softmax(dim=-1)
  $$

  * Attempts to use topKGuess to perform rolling sorting on the logits and obtains the mask for computing topP:

  $$
  topPValue[b] = {Max(topKGuess)}_{s=1}^{\left \lceil \frac{S}{v} \right \rceil }\left \{ topPValue[b]\left \{s-1 \right \}  \cup \left \{ logitsValue[b][v] \ge topKMin[b][s-1] \right \} \right \}
  $$

  * If the following condition is satisfied before the 1e4-th element of logitsValue[b] is accessed, topKGuess is regarded as successful:

  $$
  \sum^{topKGuess}(topPValue[b]) \ge topP[b]\\
  topPMask[b][Index(topPValue[b])] = false
  $$

  * If topKGuess fails, performs full sorting and cumsum on the current sequence logitsValue[b] and performs truncated sampling according to topP[b]:

  $$
  sortedLogits[b] = sort(logitsValue[b], descendant) \\
  probsSum[b]=sortedLogits[b].cumsum (dim=-1) \\
  topPMask[b] = (probsSum[b] - sortedLogits[b])>topP[b]
  $$

  * Sets the positions to be filtered to -Inf to obtain sortedValue[b][v]:

    $$
    sortedValue[b][v] = \begin{cases} -Inf& \text{topPMask[b][v]=true}\\sortedValue[b][v]& \text{topPMask[b][v]=false}\end{cases}
    $$

    Takes the first topK elements of each row of the filtered sortedValue[b][v], searches for the original indexes of these elements in the input, and integrates them into `logitsIdx`:

    $$
    logitsIdx[b][v] = Index(sortedValue[b][v] \in logits)
    $$

  Exponential sampling (Sample)

  * If `isNeedLogits=true`, selects the sampling results according to `logitsIdx` and outputs them to `logitsTopKPSelect`:

  $$
  logitsTopKPSelect[b][logitsIdx[b][v]]=sortedValue[b][v]
  $$

  * Performs exponential distribution sampling on `logitsSort`:

    $$
    probs = safe\_softmax(logitsSort)
    $$

    $$
    probsOpt = \frac{probs}{q + eps}
    $$

  * Takes the maximum element of each batch from `probsOpt` and gathers the input indexes of the corresponding elements from `logitsIdx` as the output `logitsSelectIdx`:

    $$
    logitsSelectIdx[b] = logitsIdx[b][argmax(probsOpt[b][:])]
    $$

  where 0≤b<sortedValue.size(0) and 0≤v<sortedValue.size(1).

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
        <td>Input word frequency to be sampled, corresponding to `logits` in the formula.</td>
        <td>FLOAT16, BFLOAT16</td>
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
        <td>FLOAT16, BFLOAT16</td>
        <td>-</td>
      </tr>
      <tr>
        <td>q</td>
        <td>Input</td>
        <td>Indicates the exponential sampling matrix of the topK-topP sampling output. Its dimensions and size must be consistent with `logits`.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>eps</td>
        <td>Input</td>
        <td>Prevents division by zero in softmax and weight sampling, corresponding to `eps` in the formula. The recommended value is 1e-8.</td>
        <td>FLOAT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>isNeedLogits</td>
        <td>Input</td>
        <td>Controls the output condition of logitsTopKPselect. The recommended value is 0.</td>
        <td>UINT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>topKGuess</td>
        <td>Input</td>
        <td>Indicates the candidate logits size when each batch attempts to traverse and sample the logits in the topP part, corresponding to `topKGuess` in the formula.</td>
        <td>UINT32</td>
        <td>-</td>
      </tr>
      <tr>
        <td>logitsSelectIdx</td>
        <td>Output</td>
        <td>Indicates the position index in the input logits of the element with the maximum word frequency max(probsOpt[batch, :]) in each batch after the topK-topP-sample computation flow, corresponding to `logitsSelectIdx` in the formula.</td>
        <td>INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>logitsTopKPSelect</td>
        <td>Output</td>
        <td>Indicates the remaining unfiltered logits in the input logits after the topK-topP computation flow, corresponding to `logitsTopKPSelect` in the formula.</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody></table>

## Constraints

  * Input value range restrictions:
    * For logits, its elements must ensure that the corresponding results after the $exp(*)$ operation are not less than the minimum valid value of FLOAT32. Otherwise, the corresponding Safe_softmax result will be 0, which may consequently lead to unexpected output of the operator.
    * For all parameters, their sizes must satisfy batch>0 and 0<vocSize<=2^20.
  * Input shape restrictions:
    * The sizes and dimensions of logits, q, and logitsTopKPselect must be exactly consistent. Currently only two dimensions are supported.
    * All dimensions except the last dimension of logits, topK, topP, and logitsSelectIdx must be exactly consistent in order and size. Currently logits can only be 2-dimensional, and topK, topP, and logitsSelectIdx must be 1-dimensional non-empty tensors. Empty tensors are not allowed as inputs for logits, topK, and topP. If you need to skip the corresponding module, set the input according to the corresponding rules.
  * Other restrictions:
    * If you need to skip the topK module alone, pass in a tensor of size [batch, 1] and make each element an invalid value.
    * If 1024<topK[batch]<vocSize[batch], all valid elements of the current batch are selected and the topK sampling is skipped.
    * If you need to skip the topP module alone, pass in a tensor of size [batch, 1] and make each element ≥1.
    * If you need to skip the sample module alone, pass `q=nullptr`; if you need to use the sample module, you must pass in a tensor of size [batch, vocSize].

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_top_k_top_p_sample](examples/test_aclnn_top_k_top_p_sample.cpp) | Invokes the TopKTopPSample operator through the [aclnnTopKTopPSample](docs/aclnnTopKTopPSample_en.md) interface. |
| Graph mode | [test_aclnn_top_k_top_p_sample](examples/test_aclnn_top_k_top_p_sample.cpp)  | Invokes the TopKTopPSample operator through the [operator IR](op_graph/top_k_top_p_sample_proto.h) graph construction method.         |
