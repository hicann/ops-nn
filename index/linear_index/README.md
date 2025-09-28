# LinearIndex

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
|Atlas 200I/500 A2推理产品|×|

## 功能说明

- 算子功能：将多维数组 / 矩阵的元素位置映射为单一整数的索引方式。

## 算子规格

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>stride</td>
      <td>输入</td>
      <td>步长。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices_list</td>
      <td>输入</td>
      <td>索引列表。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value_size</td>
      <td>输入</td>
      <td>值的大小。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>输出</td>
      <td>待输出的索引。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无
