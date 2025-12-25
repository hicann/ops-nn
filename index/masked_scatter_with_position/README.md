# Maskedscatterwithposition

##  产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品     |    ×     |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件 |    ×     |

## 功能说明

- 算子功能：根据掩码(mask)张量进行模拟广播到x的shape，mask中为True位置，复制对应（updates）到输入(x)对应位置。其中position为mask广播前的前缀和张量。

- 示例：
  - x=[[0, 0, 0, 0, 0],[0, 0, 0, 0, 0]]
  - masked=[[0, 0, 0, 1, 1]]
  - position=[[0, 0, 0, 1, 2]]
  - updates=[[0, 1, 2, 3, 4] ,[5, 6, 7, 8, 9]]
  - maskedscatterwithposition(x, masked, position, updates)的结果是[[0, 0, 0, 0, 1],[0, 0, 0, 2, 3]]。
  注意整型、布尔都是有界的。

## 参数说明

<table style="undefined;table-layout: fixed; width: 920px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 290px">
  <col style="width: 260px">
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
      <td>输入，原始数据。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、UINT8、INT8、INT16、INT32、INT64、DOUBLE、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输入</td>
      <td>输入的掩码。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>position</td>
      <td>输入</td>
      <td>掩码的前缀和。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>输入</td>
      <td>待填入的数据。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、UINT8、INT8、INT16、INT32、INT64、DOUBLE、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>输出结果。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、UINT8、INT8、INT16、INT32、INT64、DOUBLE、BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_masked_scatter_with_position.cpp](./examples/test_aclnn_masked_scatter_with_position.cpp) | 通过[aclnnInplaceMaskedScatter](../masked_scatter/op_host/op_api/aclnn_masked_scatter.cpp)接口方式调用MaskedScatterWithPosition算子的L0接口。 |
