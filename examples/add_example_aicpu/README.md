# AddExample

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|

## 功能说明

- 算子功能：完成加法计算。

- 计算公式：

$$
y = x1 + x2
$$

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
      <td>x1</td>
      <td>输入</td>
      <td>待进行add_example计算的入参，公式中的x1。</td>
      <td>FLOAT、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>待进行add_example计算的入参，公式中的x2。</td>
      <td>FLOAT、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行add_example计算的出参，公式中的y。</td>
      <td>FLOAT、INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_add_example](./examples/test_geir_add_example_aicpu.cpp)   | 通过[算子IR](./op_graph/add_example_aicpu_proto.h)构图方式调用AddExample算子。 |