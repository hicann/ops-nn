# InitEmbeddingHashTable

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | ✗ |
| <term>Atlas A2 products</term> | ✗ |
| <term>Atlas 200I/500 A2 inference products</term> | ✗ |
| <term>Atlas inference products</term> | ✗ |
| <term>Atlas training products</term> | ✗ |

## Function

- Operator function: initializes the hash table.

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
      <td>The input hash table handle, which contains the table head address of the hash table and other information.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sampled_values</td>
      <td>Input</td>
      <td>Random numbers generated externally, used to fill the value. It takes effect when initializer_mode is random.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bucket_size</td>
      <td>Input attribute</td>
      <td>The number of buckets in the hash table.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>embedding_dim</td>
      <td>Input attribute</td>
      <td>The bucket depth of the hash table.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>initializer_mode</td>
      <td>Input attribute</td>
      <td>Supports random (initialized with sampled_values) and constant (initialized with constant_value).</td>
      <td>STRING</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>constant_value</td>
      <td>Input attribute</td>
      <td>The initial fill value of the hash table. It takes effect when initializer_mode is constant.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| None | None | None |
