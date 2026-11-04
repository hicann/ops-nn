# EmbeddingDenseBackward

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

  Implements the backward computation of [Embedding], accumulating the row of grad corresponding to the same index `indices` into out.

## Parameters

  <table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 183px">
  <col style="width: 120px">
  <col style="width: 265px">
  <col style="width: 197px">
  <col style="width: 114px">
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
      <td>grad</td>
      <td>Computation input</td>
      <td>Original gradient of the data. It is an aclTensor on the Device, supports 2 to 8 dimensions, and after merging axes, its shape is the same as the shape of indices after merging axes. Non-contiguous tensors are supported.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Computation input</td>
      <td>Index values corresponding to the grad input. It is an aclTensor on the Device, with a value range of [0, numWeights), supports 1 to 8 dimensions, and supports non-contiguous tensors.</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT32, INT64, INT16, INT8, UINT8, BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numWeights</td>
      <td>Computation input</td>
      <td>Size of the leading axis of the output tensor (uint64_t).</td>
      <td>UINT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>paddingIdx</td>
      <td>Computation input</td>
      <td>Fills the paddingIdx-th row of the output tensor with 0; if paddingIdx is a negative number, no processing is performed (uint64_t).</td>
      <td>UINT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleGradByFreq</td>
      <td>Computation input</td>
      <td>Specifies whether to scale the gradient according to the word occurrence frequency. true: scale by word frequency; false: no processing.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Computation output</td>
      <td>Result output of the gradient summation. It is an aclTensor on the Device with 2 dimensions. The size of the leading axis is numWeights, and the size of the trailing axis is the same as that of the trailing axis of grad. Only the ND format is supported.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas training products</term>: The data types of the grad and out parameters do not support BFLOAT16.

## Constraints

- <term>Atlas training products</term>:
  - For the scenario where scale is true, set the last dimension of grad to embeddingDim. If its size exceeds the specified range, the computation is intercepted and an error is reported. The reasonable ranges are as follows:
    - When indices is int32, the following condition must be met:
    $$
    embeddingDim < \frac{180192 - countsSize * 4}{36}
    $$
    - When indices is int64, the following condition must be met:
    $$
    embeddingDim < \frac{180192 - countsSize * 8}{20}
    $$
    - The formula for countsSize is as follows, where coreNum represents the number of AI processor cores:
    $$
    countsSize = numWeights / coreNum + numWeights \% coreNum
    $$
- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:
  - When the parameter shapes exceed the following limits, high precision of the output cannot be guaranteed. If deterministic computation is enabled, high performance cannot be guaranteed either.
    - After grad is merged into a 2-dimensional shape, the first dimension exceeds INT32_MAX(2147483647)
    - numWeights exceeds INT32_MAX(2147483647)
  - When the dimensions of indices after merging axes exceed INT32_INF(2139095040), high performance cannot be guaranteed.

  ## Invocation Description

  | Invocation method | Invocation sample                                                                   | Description                                                           |
  |--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
  | aclnn invocation | [test_aclnn_embedding_dense_grad](./examples/test_aclnn_embedding_dense_grad.cpp) | Invoke the EmbeddingDenseGrad operator through the [aclnnEmbeddingDenseBackward](./docs/aclnnEmbeddingDenseBackward_en.md) API. |
