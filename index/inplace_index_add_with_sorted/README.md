# InplaceIndexAddWithSorted

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|

## 功能说明

- 算子功能：将alpha和value相乘，并按照sorted_indices中的顺序，累加到var张量的指定axis维度中。

- 计算公式：

  $$
  var[indices]+= alpha * updates
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 60px">
  <col style="width: 60px">
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 60px">
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
      <td>var</td>
      <td>输入/输出</td>
      <td>公式中的`var`，表示待被累加的张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>公式中的`updates`，表示需要加到var上的张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sorted_indices</td>
      <td>输入</td>
      <td>公式中的`indices`，表示经过排序后的索引。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pos</td>
      <td>输入</td>
      <td>表示排序后的索引在原索引序列中的位置。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>公式中的`alpha`，表示对value的缩放，与value相乘后再累加至var。</td>
      <td>FLOAT、INT32、INT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>属性</td>
      <td>表示指定var在哪一轴相加。</td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_index_add](../inplace_scatter_add/examples/test_aclnn_inplace_scatter_add.cpp) | 通过[aclnnIndexAdd](../inplace_scatter_add/docs/aclnnIndexAdd.md)接口方式调用InplaceIndexAddWithSorted算子。 |