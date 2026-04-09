# ReluGrad

## 产品支持情况


| 产品                                                               | 是否支持 |
| ------------------------------------------------------------------ | :------: |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件 |    √    |

## 功能说明

- 算子功能：计算relu的梯度。
- 计算公式：

$$
y=\begin{cases} 1, & x\ge 0,\\0, & x < 0\end{cases}
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
      <td>gradients</td>
      <td>输入</td>
      <td>待进行relugrad计算的入参，公式中的x。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>features</td>
      <td>输入</td>
      <td>待进行relugrad计算的入参。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprops</td>
      <td>输出</td>
      <td>待进行relugrad计算的出参，公式中的y。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明


| 调用方式  | 调用样例                                                    | 说明                                                                   |
| --------- | ----------------------------------------------------------- | ---------------------------------------------------------------------- |
| aclnn调用 | [test_aclnn_threshold_backward](./examples/test_aclnn_threshold_backward.cpp) | 通过[aclnnThresholdBackward](./docs/aclnnThresholdBackward.md)接口方式调用ReluGrad算子。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| ilovescrapy | 个人开发者 | ReluGrad | 2025/12/26 | ReluGrad算子适配开源仓 |