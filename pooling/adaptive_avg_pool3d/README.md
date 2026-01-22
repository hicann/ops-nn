# AdaptiveAvgPool3d

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend 950PR/Ascend 950DT|√|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
|Atlas 200I/500 A2推理产品|×|
|Atlas 推理系列产品|√|
|Atlas 训练系列产品|×|

## 功能说明

- 算子功能：在指定三维输出shape信息的情况下，完成输入张量的3D自适应平均池化计算。


## 参数说明

<table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 200px">
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
      <td>待进行adaptiveavgpool3d计算的入参。表示待转换的目标张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_size</td>
      <td>属性</td>
      <td>待进行adaptiveavgpool3d计算的入参。表示输出在DHW维度上的shape大小。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行adaptiveavgpool3d计算的出参。表示转换后的结果张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无。


## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_adaptive_avg_pool3d.cpp](examples/test_aclnn_adaptive_avg_pool3d.cpp) | 通过[aclnnAdaptiveAvgPool3d](docs/aclnnAdaptiveAvgPool3d.md)接口方式调用AdaptiveAvgPool3d算子。 |
