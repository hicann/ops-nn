# Embedding

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    √  |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term>     |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×   |

## Function

- Operator function: Maps a data set into a vector space, thereby quantizing the data. The 2D weight tensor of embedding is weight (m+1 rows, n columns). For any input index tensor indices (such as 1 row and 3 columns), the output out is a tensor of 3 rows and n columns, as shown below:

$$
\begin{aligned}
    & weight = \begin{bmatrix}
        x_{01} & x_{02} & ... & x_{0n}\\
        x_{11} & x_{12} & ... & x_{1n}\\
        x_{21} & x_{22} & ... & x_{2n}\\
        ...\\
        x_{m1} & x_{m2} & ... & x_{mn}\\
    \end{bmatrix}\\
    & indices = \begin{bmatrix}
        0,2,m
    \end{bmatrix}\\
    & out = \begin{bmatrix}
        x_{01} & x_{02} & ... & x_{0n}\\
        x_{21} & x_{22} & ... & x_{2n}\\
        x_{m1} & x_{m2} & ... & x_{mn}\\
    \end{bmatrix}\\
\end{aligned}
$$

## Parameters

<table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 140px">
  <col style="width: 140px">
  <col style="width: 180px">
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
      <td>weight</td>
      <td>Input</td>
      <td>The original input</td>
      <td>BF16, FLOAT16, FLOAT, UINT8, INT8, UINT16, INT16, UINT, INT, INT64, UINT64, BOOL, DOUBLE, COMPLEX64, COMPLEX32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>The index values</td>
      <td>INT64, INT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>The result after indexing</td>
      <td>BF16, FLOAT16, FLOAT, UINT8, INT8, UINT16, INT16, UINT, INT, INT64, UINT64, BOOL, DOUBLE, COMPLEX64, COMPLEX32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method  | Sample code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn | [test_aclnn_embedding](./examples/test_aclnn_embedding.cpp)   | Calls the Embedding operator through the [aclnn interface](./docs/aclnnEmbedding_en.md). |
