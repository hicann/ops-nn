# EmbeddingHashTableImport

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

- Operator function: inserts the input keys and values into the hash table.

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
      <td>table_handles</td>
      <td>Input</td>
      <td>Handles of the input hash table, which contain the header addresses of the hash table and other information.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>embedding_dims</td>
      <td>Input</td>
      <td>Bucket depth of the hash table.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bucket_sizes</td>
      <td>Input</td>
      <td>Number of buckets in the hash table.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keys</td>
      <td>Input</td>
      <td>Sequence of keys to insert.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>counters</td>
      <td>Input</td>
      <td>Number of keys to insert.</td>
      <td>UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>filter_flags</td>
      <td>Input</td>
      <td>Admission flags.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>Input</td>
      <td>Sequence of values corresponding to the keys to insert.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| None | None | None |
