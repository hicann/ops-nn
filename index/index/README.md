# Index

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：根据索引indices将输入x对应坐标的数据取出。

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
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT64、INT8、UINT8、BOOL</td>
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
      <td>y</td>
      <td>输出</td>
      <td>根据索引取出后的数据。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT64、INT8、UINT8、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_index](./examples/test_aclnn_index.cpp) | 通过[aclnnIndex](./docs/aclnnIndex.md)接口方式调用Index算子。 |