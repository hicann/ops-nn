# LogsigmoidV2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | √ |

## 功能说明

- 算子功能：计算输入张量的 LogSigmoid 值，即对每个元素计算 \( \log(\frac{1}{1 + e^{-x}}) \)。

- 计算公式：

$$
{out}_i = \log\left(\frac{1}{1 + e^{-x_i}}\right)
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
      <td>x</td>
      <td>输入</td>
      <td>待进行 LogsigmoidV2 计算的输入张量。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>  
    <tr>
      <td>z</td>
      <td>输出</td>
      <td>LogsigmoidV2 计算后的输出张量。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_logsigmoid_v2.cpp](./examples/test_aclnn_logsigmoid_v2.cpp) | 通过 [test_aclnn_logsigmoid_v2](./docs/test_aclnn_logsigmoid_v2.md) 接口方式调用 LogsigmoidV2 算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| Li Wen | 个人开发者 | 2025/11/25 | LogsigmoidV2 算子适配开源仓 |
