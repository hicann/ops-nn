# EluGrad

## 产品支持情况

| 产品                                                               | 是否支持 |
| ------------------------------------------------------------------ | :------: |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件 |    √    |

## 功能说明

- 算子功能：求elu函数梯度。
- 计算公式：

$$
y = (min(activations,0) + 1) * grads
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>grads</td>
      <td>输入</td>
      <td>对应Elu操作的反向传播梯度</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activations</td>
      <td>输入</td>
      <td>与"grads"具有相同的类型、格式和形状。对应Elu操作的输出值。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量</td>
      <td>同grads</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口调用 | [test_aclnn_elu_grad.cpp](./examples/test_aclnn_elu_grad.cpp)  | 通过自动生成的aclnn接口调用EluGrad算子。         |

## 约束说明

无

## 贡献说明

| 贡献者      | 贡献方     | 贡献算子  | 贡献时间   | 贡献内容                |
| ----------- | ---------- | --------- | ---------- | ----------------------- |
| ilovescrapy | 个人开发者 | EluGrad | 2026/3/23 | EluGrad算子适配开源仓 |
