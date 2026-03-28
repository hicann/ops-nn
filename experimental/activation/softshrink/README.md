# Softshrink

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|

## 功能说明

- 算子功能：对输入Tensor逐元素执行Softshrink激活函数。

- 计算公式：

$$
y_i = \begin{cases} x_i - \lambda, & \text{if } x_i > \lambda \\ x_i + \lambda, & \text{if } x_i < -\lambda \\ 0, & \text{otherwise} \end{cases}
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
      <td>input_x</td>
      <td>输入</td>
      <td>待进行Softshrink计算的输入Tensor。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lambd</td>
      <td>属性</td>
      <td>Softshrink公式中的lambda值，默认0.5，需 >= 0。</td>
      <td>float</td>
      <td>标量</td>
    </tr>
    <tr>
      <td>output_y</td>
      <td>输出</td>
      <td>Softshrink计算结果，shape与input_x相同。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- lambd属性值必须 >= 0。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|----------|---------|------|
| aclnn调用 | [test_aclnn_softshrink.cpp](./examples/test_aclnn_softshrink.cpp) | 通过aclnn接口方式调用Softshrink算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| yourealize | 个人开发者 | Softshrink | 2025/03 | Softshrink算子适配开源仓 |
