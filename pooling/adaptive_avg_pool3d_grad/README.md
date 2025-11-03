# AdaptiveAvgPool3dGrad

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
 

## 功能说明

- 算子功能：AdaptiveAvgPool3d的反向计算。


## 参数说明

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
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
      <td>gradOutput</td>
      <td>输入</td>
      <td>待进行AdaptiveAvgPool3dGrad计算的入参，表示当前节点的梯度。数据类型、数据格式、shape的总维数与入参`self`的保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NCDHW、ND</td>
    </tr>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>待进行AdaptiveAvgPool3dGrad计算的入参。数据类型、数据格式、shape的总维数与入参`gradOutput`的保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NCDHW、ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行AdaptiveAvgPool3dGrad计算的出参。数据类型、shape与入参`self`的保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NCDHW、ND</td>
    </tr>
  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_adaptive_avg_pool3d_backward.cpp](examples/test_aclnn_adaptive_avg_pool3d_backward.cpp) | 通过[aclnnAdaptiveAvgPool3dBackward](docs/aclnnAdaptiveAvgPool3dBackward.md)接口方式调用AdaptiveAvgPool3d算子。 |
