# Embedding

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |    √  |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×   |

## 功能说明

- 算子功能：把数据集合映射到向量空间，进而将数据进行量化。embedding的二维权重张量为weight(m+1行，n列)，对于任意输入索引张量indices（如1行3列），输出out是一个3行n列的张量，如下所示：

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

## 参数说明

<table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 140px">
  <col style="width: 140px">
  <col style="width: 180px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>原始输入</td>
      <td>BF16、FLOAT16、FLOAT、UINT8、INT8、UINT16、INT16、UINT、INT、INT64、UINT64、BOOL、DOUBLE、COMPLEX64、COMPLEX32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>索引值</td>
      <td>INT64、INT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>索引之后的结果</td>
      <td>BF16、FLOAT16、FLOAT、UINT8、INT8、UINT16、INT16、UINT、INT、INT64、UINT64、BOOL、DOUBLE、COMPLEX64、COMPLEX32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn | [test_aclnn_embedding](./examples/test_aclnn_embedding.cpp)   | 通过[aclnn接口](./docs/aclnnEmbedding.md)方式调用Embedding算子。 |