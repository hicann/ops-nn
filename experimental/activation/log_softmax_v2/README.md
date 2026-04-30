# LogSoftmaxV2

## 产品支持情况

| 产品                                         | 是否支持 |
| -------------------------------------------- | :------: |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 |    √    |

## 功能说明

- 算子功能：实现张量的对数归一化计算，在指定维度上计算输入元素的对数Softmax值。
- 计算公式：

$$
out_i = input_i - \log\left(\sum_j \exp(input_j)\right)
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
      <td>axes</td>
      <td>输入</td>
      <td>指定的归约轴</td>
      <td>ListInt</td>
      <td>/</td>
    </tr>  
    <tr>  
    <tr>
      <td>input</td>
      <td>输入</td>
      <td>待进行归约的tensor</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>  
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>归约的输出tensor</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式  | 调用样例                                                 | 说明                                                                              |
| --------- | -------------------------------------------------------- | --------------------------------------------------------------------------------- |
| aclnn调用 | [test_aclnn_s_where.cpp](./examples/test_aclnn_s_where.cpp) | 通过[test_aclnn_s_where](./docs/test_aclnn_s_where.md)接口方式调用LogSoftmaxV2算子。 |

## 贡献说明

| 贡献者   | 贡献方     | 贡献算子     | 贡献时间   | 贡献内容                   |
| -------- | ---------- | ------------ | ---------- | -------------------------- |
| 看不见我 | 个人开发者 | LogSoftmaxV2 | 2025/12/24 | LogSoftmaxV2算子适配开源仓 |
