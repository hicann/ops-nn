# EmbeddingHashTableLookupOrInsert

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

- Operator function: checks whether a key exists in the table according to the key value; if the key exists, the value is not inserted, and the value at the current position of the key is exported; if the key does not exist, the key is hashed, and the value is inserted after the position is found.

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
      <td>Handle of the input hash table, which contains the header address of the hash table and other information.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>keys</td>
      <td>Input</td>
      <td>Key sequence to query.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>values</td>
      <td>Output</td>
      <td>If a queried key already exists, the value sequence at the corresponding position that is returned.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bucket_size</td>
      <td>Input attribute</td>
      <td>Number of buckets in the hash table.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>embedding_dim</td>
      <td>Input attribute</td>
      <td>Depth of the buckets in the hash table.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>filter_mode</td>
      <td>Input attribute</td>
      <td>Admission filter mode, with the default no_filter (no filtering).</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>filter_freq</td>
      <td>Input attribute</td>
      <td>Admission frequency threshold, used when filter_mode takes effect, with the default 0.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>default_key_or_value</td>
      <td>Input attribute</td>
      <td>Whether to use the default key or value, with the default false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>default_key</td>
      <td>Input attribute</td>
      <td>Default key value, with the default 0.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>default_value</td>
      <td>Input attribute</td>
      <td>Default value, with the default 0.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>filter_key_flag</td>
      <td>Input attribute</td>
      <td>Whether to enable filter_key filtering, with the default false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>filter_key</td>
      <td>Input attribute</td>
      <td>Key value that needs to be filtered, with the default -1.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| None | None | None |
