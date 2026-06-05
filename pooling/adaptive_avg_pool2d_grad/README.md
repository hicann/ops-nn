# AdaptiveAvgPool2dGrad

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend 950PR/Ascend 950DT|√|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|×|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|×|
|Atlas 200I/500 A2推理产品|×|
|Atlas 推理系列产品|×|
|Atlas 训练系列产品|×|

## 功能说明

- 算子功能：AdaptiveAvgPool2d的反向计算。

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
      <td>input_grad</td>
      <td>输入</td>
      <td>待进行AdaptiveAvgPool2dGrad计算的入参，表示当前节点的梯度。数据类型、数据格式、shape的总维数与入参`orig_input_shape`保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>orig_input_shape</td>
      <td>输入</td>
      <td>待进行AdaptiveAvgPool2dGrad计算的入参。数据类型、数据格式、shape的总维数与入参`input_grad`保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_grad</td>
      <td>输出</td>
      <td>待进行AdaptiveAvgPool2dGrad计算的出参。数据类型、shape与入参`orig_input_shape`的保持一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_adaptive_avg_pool2d_backward.cpp](../adaptive_avg_pool2d_grad/examples/test_aclnn_adaptive_avg_pool2d_backward.cpp) | 通过[aclnnAdaptiveAvgPool2dBackward](../adaptive_avg_pool2d_grad/docs/aclnnAdaptiveAvgPool2dBackward.md)接口方式调用AdaptiveAvgPool2dGrad算子。 |
