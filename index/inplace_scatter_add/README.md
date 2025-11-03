# InplaceScatterAdd

##  产品支持情况

| 产品 | 是否支持 |
|---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
 

## 功能说明

- 算子功能：根据给定的indices，将updates中的值加到输入张量var的第一维度上。

- 计算公式：
  $$
  var[indices]  +=  updates
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 40px">
  <col style="width: 60px">
  <col style="width: 260px">
  <col style="width: 60px">
  <col style="width: 40px">
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
      <td>表示需要被更新的张量，公式中的var</td>
      <td>FLOAT、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>表示对var的第一维的索引，shape只支持1维，数据取值范围为[0, var.dim[0])</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>输入</td>
      <td>表示要被加到var上的值。</td>
      <td>FLOAT、INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

无
