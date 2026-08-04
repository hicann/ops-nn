# Index

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：按 `indices` 指定的下标对输入 `x` 进行多维高级索引（advanced indexing），取出对应坐标的数据得到输出 `y`。各输入与功能的关系如下：
  - `indexed_sizes`：一维 INT64 张量，长度等于 `x` 的维度数，逐维标记 `x` 的每一维是否参与索引：取值为 `1` 表示该维由索引选择（消耗一个 `indices` 输入），取值大于 `1` 表示该维整体保留。
  - `indexed_strides`：一维 INT64 张量，长度等于 `x` 的维度数，给出按索引取数时在 `x` 中每一维上的步长。
  - `indices`：动态输入，个数等于 `indexed_sizes` 中取值为 `1` 的维度数量，作为对应被索引维度的下标；多个 `indices` 之间按广播规则对齐。
  - `y`：按上述规则从 `x` 中取出的数据，数据类型与 `x` 一致。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>x</td>
      <td>输入</td>
      <td>输入数据。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT64、INT8、UINT8、BOOL、COMPLEX64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indexed_sizes</td>
      <td>输入</td>
      <td>索引个数。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indexed_strides</td>
      <td>输入</td>
      <td>索引步长。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>索引。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>根据索引取出后的数据。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT64、INT8、UINT8、BOOL、COMPLEX64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型不支持BFLOAT16、COMPLEX64。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持COMPLEX64。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_index](./examples/test_aclnn_index.cpp) | 通过[aclnnIndex](./docs/aclnnIndex.md)接口方式调用Index算子。 |
