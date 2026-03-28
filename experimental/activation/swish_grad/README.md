# SwishGrad

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|

## 功能说明

- 算子功能：求swish函数梯度。

- 计算公式：

$$
y = sigmoid(scale*x) + x*sigmoid(scale*x)
$$
$$
sigmoid = sigmoid*(1 - sigmoid)
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
      <td>grad</td>
      <td>输入</td>
      <td>待进行SwishGrad计算的入参，公式中的grad。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND,FRACTAL_NZ,NC1HWC0</td>
    </tr>  
    <tr>  
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待进行SwishGrad计算的入参，公式中的x。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND,FRACTAL_NZ,NC1HWC0</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>待进行SwishGrad计算的入参，公式中的scale。</td>
      <td>fp32</td>
      <td>1</td>
    </tr>  
    <tr>
      <td>y</td>
      <td>输入</td>
      <td>待进行SwishGrad计算的入参, 暂时不参与运算。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND,FRACTAL_NZ,NC1HWC0</td>
    </tr>
    <tr>
      <td>grad_x</td>
      <td>输出</td>
      <td>待进行SwishGrad计算的出参，公式中的输出。</td>
      <td>fp16、fp32、bf16</td>
      <td>ND,FRACTAL_NZ,NC1HWC0</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_swish_grad.cpp](./examples/test_aclnn_swish_grad.cpp) | 通过[test_aclnn_swish_grad](./docs/aclnnSwishBackward.md)接口方式调用SwishGrad算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| ilovescrapy | 个人开发者 | SwishGrad | 2025/12/23 | SwishGrad算子适配开源仓 |
