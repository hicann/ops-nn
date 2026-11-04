# EmbeddingHashTableApplyAdamW

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
| <term>Ascend 950PR&950DT products</term> |√|
| <term>Atlas A3 products</term>    |    ✗     |
| <term>Atlas A2 products</term> |    ✗     |
| <term>Atlas 200I/500 A2 inference products</term> |      ✗     |
| <term>Atlas inference products</term> |      ✗     |
| <term>Atlas training products</term> |      ✗     |

## Function

- Operator function: Queries whether a key exists in the hash table. If the key exists, the operator updates its value; if the key does not exist, the operator inserts the key.

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
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
      <td>table_handle</td>
      <td>Input</td>
      <td>Handle of the input hash table, which contains the table header address of the hash table and other information.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keys</td>
      <td>Input</td>
      <td>Key sequence to be inserted.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>m</td>
      <td>Input</td>
      <td>Corrected first-order matrix in the AdamW algorithm.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v</td>
      <td>Input</td>
      <td>Corrected second-order matrix in the AdamW algorithm.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta1_power</td>
      <td>Input</td>
      <td>Preceding power result of beta1.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta2_power</td>
      <td>Input</td>
      <td>Preceding power result of beta2.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>Input</td>
      <td>Learning rate.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight_decay</td>
      <td>Input</td>
      <td>Weight decay coefficient.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta1</td>
      <td>Input</td>
      <td>Coefficient used to compute the first-order matrix.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta2</td>
      <td>Input</td>
      <td>Coefficient used to compute the second-order matrix.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Input</td>
      <td>Division-by-zero protection, improving numerical stability.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>Input gradient.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max_grad_norm</td>
      <td>Input</td>
      <td>In the AMSGrad algorithm, limits the maximum norm of the gradient.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>embedding_dim</td>
      <td>Input attribute</td>
      <td>Bucket depth of the hash table.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bucket_size</td>
      <td>Input attribute</td>
      <td>Number of buckets in the hash table.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>amsgrad</td>
      <td>Input attribute</td>
      <td>Determines whether to use the AMSGrad algorithm.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maximize</td>
      <td>Input attribute</td>
      <td>Determines whether to maximize the parameter.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>m</td>
      <td>Output</td>
      <td>Corrected first-order matrix in the AdamW algorithm.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>v</td>
      <td>Output</td>
      <td>Corrected second-order matrix in the AdamW algorithm.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta1_power</td>
      <td>Output</td>
      <td>Updated power result of beta1.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta2_power</td>
      <td>Output</td>
      <td>Updated power result of beta2.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max_grad_norm</td>
      <td>Output</td>
      <td>Updated gradient parameter.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| None | None | None |
