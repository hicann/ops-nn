# EmbeddingDenseGradV2

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    √     |

## Function

- Operator function: Implements the backward computation of Embedding, accumulating the row of grad corresponding to the same index indices onto out.

## Parameters

  <table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
    <col style="width: 50px">
    <col style="width: 60px">
    <col style="width: 250px">
    <col style="width: 150px">
    <col style="width: 60px">
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
        <td>Input</td>
        <td>Denotes the original gradient of the data.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>sort_indices</td>
        <td>Input</td>
        <td>Denotes the index values corresponding to the grad input.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Denotes the result output of the gradient summation.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>num_weights</td>
        <td>Attribute</td>
        <td>Denotes the size of the first axis of the output tensor.</td>
        <td>Int</td>
        <td>-</td>
      </tr>
      <tr>
        <td>padding_idx</td>
        <td>Optional attribute</td>
        <td><ul><li>Fills the paddingIdx-th row of the output tensor with 0. If paddingIdx is negative, no processing is performed.</li><li>The default value is -1.</li></ul></td>
        <td>Int</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scale_grad_by_freq</td>
        <td>Optional attribute</td>
        <td><ul><li>Determines whether to scale the gradient according to the frequency of word occurrence.</li><li>The default value is false.</li></ul></td>
        <td>Bool</td>
        <td>-</td>
      </tr>
    </tbody></table>

- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: grad does not support BFLOAT16 or FLOAT16, and indices does not support INT64.

## Constraints

- Atlas A3 products and Atlas A2 products:
  - When a parameter shape exceeds the following limits, high precision of the output cannot be guaranteed, and high performance cannot be guaranteed either if deterministic computation is enabled.
    - After grad is folded into a 2D shape, the first dimension exceeds INT32_MAX(2147483647).
    - num_weights exceeds INT32_MAX(2147483647).
  - When the dimension of sort_indices after folding exceeds INT32_INF(2139095040), high performance cannot be guaranteed.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_embedding_dense_grad_v2](./examples/test_aclnn_embedding_dense_grad_v2.cpp) | Calls the EmbeddingDenseGradV2 operator through the [aclnnEmbeddingDenseBackward](./docs/aclnnEmbeddingDenseBackward_en.md) interface. |
