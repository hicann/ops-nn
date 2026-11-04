# EmbeddingHashTableEvict

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
| <term>Ascend 950PR&950DT products</term> |√|
| <term>Atlas A3 products</term>    |    ✗     |
| <term>Atlas A2 products</term> |    ✗     |
| <term>Atlas 200I/500 A2 inference products</term> |      ✗     |
| <term>Atlas inference products</term> |      ✗     |
| <term>Atlas training products</term> |      ✗     |

## Function

- Operator function: searches the hash table for the bucket corresponding to the input key and marks it as evicted, resets the counter, and re-initializes the value according to the specified mode.

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
      <td>The handle of the input hash table, which contains information such as the header address of the hash table.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keys</td>
      <td>Input</td>
      <td>The sequence of keys to be evicted.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sampled_values</td>
      <td>Input</td>
      <td>Externally generated random numbers, used to fill the value when init_mode is random.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>table_cap</td>
      <td>Input attribute</td>
      <td>The capacity of the hash table.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>embedding_dim</td>
      <td>Input attribute</td>
      <td>The bucket depth of the hash table.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>init_mode</td>
      <td>Input attribute</td>
      <td>The re-initialization mode of the value after eviction. The value can be random or constant, and the default value is constant.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>const_val</td>
      <td>Input attribute</td>
      <td>The fill value used when init_mode is constant. The default value is 0.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| None | None | None |
