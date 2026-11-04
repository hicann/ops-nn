# ApplyTopKTopPWithSorted

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    √  |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √    |
| <term>Atlas training products</term>                              |    ×   |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: Performs top-k and top-p sampling filtering on the inputs sorted_value and sorted_indices. p and k are optional inputs, and at least one of them must be passed. Three modes are supported: top-k-only filtering, top-p-only filtering, and combined top-k and top-p filtering.
- Formula:
  - Compute the reserved threshold (the k-th largest value).
  $$topKValue[b][v] = sortedValue[b][sortedValue.size(1) - k[b]]$$
  - Generate the mask to be filtered for top-k.
  $$topKMask = sortedValue < topKValue$$
  - Use topKMask to set the parts smaller than the threshold to -Inf.

  $$
  sortedValue[b][v] =
  \begin{cases}
  -Inf & \text{topKMask[b][v]=true}\\
  sortedValue[b][v] & \text{topKMask[b][v]=false}
  \end{cases}
  $$

  - Use softmax to convert the top-k-filtered data into a probability distribution along the last axis.
  $$probsValue = softmax(sortedValue, dim=-1)$$
  - Compute the cumulative probability along the last axis (accumulation starts from the smallest probability).
  $$probsSum = cumsum(probsValue, dim=-1)$$
  - Generate the mask for top-p. Positions whose cumulative probability is less than or equal to 1-p must be filtered out, and each batch must retain at least one element.
  $$topPMask[b][v] = probsSum[b][v] <= 1-p[b]$$
  $$topPMask[b][-1] = false$$
  - Use topPMask to set the parts smaller than the threshold to -Inf.

  $$
  sortedValue[b][v] =
  \begin{cases}
  -Inf & \text{topPMask[b][v]=true}\\
  sortedValue[b][v] & \text{topPMask[b][v]=false}
  \end{cases}
  $$

  - Restore the filtered results to the original order according to sortedIndices.
  $$out[b][v] = sortedValue[b][sortedIndices[b][v]]$$
  where $0 \le b \lt logits.size(0), 0 \le v \lt logits.size(1)$.

## Parameters

<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
  <col style="width: 250px">
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
      <td>sorted_value</td>
      <td>Input</td>
      <td>Denotes the values of the data to be processed, corresponding to `sortedValue` in the formula. Ascending order is required.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sorted_indices</td>
      <td>Input</td>
      <td>Denotes the indices of the data to be processed, corresponding to `sortedIndices` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>p</td>
      <td>Input</td>
      <td>Denotes the threshold of top-p, corresponding to p in the formula. It is an optional input, and at least one of p and k must be passed. The data type must be consistent with `sorted_value`, and the shape must be consistent with `sorted_value.size(0)`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>Input</td>
      <td>Denotes the threshold of top-k, corresponding to k in the formula. It is an optional input, and at least one of k and p must be passed. The shape must be consistent with `sorted_value.size(0)`.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>logits</td>
      <td>Input</td>
      <td>Denotes the original data before sorting, corresponding to logits in the formula. It is an optional input. The data type must be consistent with `sorted_value`, and the shape must be consistent with `sorted_value`. When it is passed in, the operator completes the filtering and restoration computation based on this input.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Denotes the filtered data, corresponding to out in the formula. The data type and shape must be consistent with `sorted_value`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas inference products</term>: The data type BFLOAT16 is not supported for the parameters `sorted_value`, `p`, and `out`.

- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraints

- p and k are optional inputs. They cannot both be empty, and at least one must be passed.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_apply_top_k_top_p.cpp](examples/test_aclnn_apply_top_k_top_p.cpp) | Calls the ApplyTopKTopPWithSorted operator through the [aclnnApplyTopKTopP](docs/aclnnApplyTopKTopP_en.md) interface. |
