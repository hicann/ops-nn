# EmbeddingHashTableExport

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

- Operator function: Exports the entire table.

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
      <td>Handle of the input hash table, which contains the header address of the hash table and so on.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>table_sizes</td>
      <td>Input</td>
      <td>Number of tables.</td>
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
      <td>Output</td>
      <td>Sequence of inserted keys.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>counters</td>
      <td>Output</td>
      <td>Counter value of each exported key, corresponding to keys one-to-one by position.</td>
      <td>UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>filter_flags</td>
      <td>Output</td>
      <td>Admission flags.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>Output</td>
      <td>Sequence of values corresponding to the exported keys.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>export_mode</td>
      <td>Input Attribute</td>
      <td>Export mode; the default value is all (full export).</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>filtered_export_flag</td>
      <td>Input Attribute</td>
      <td>Whether to filter the export by admission flag; the default value is false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| None | None | None |
