# ThresholdV2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :------: |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

- 算子功能：对输入张量进行阈值判断，大于阈值的元素保持原值，小于等于阈值的元素替换为指定值。
- 计算公式：

$$
y_i = \begin{cases} x_i, & x_i > \text{threshold} \\ \text{value}, & x_i \leq \text{threshold} \end{cases}
$$

- 计算示例：

$$
\text{Input} = [1.0, 2.0, 0.5, 3.0, -1.0, 4.0], \text{threshold}=1.5, \text{value}=0.0
$$

$$
\text{Output} = [0.0, 2.0, 0.0, 3.0, 0.0, 4.0]
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 140px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 240px">
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
      <td>self</td>
      <td>输入</td>
      <td>输入张量，公式中的 $x$。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>threshold</td>
      <td>属性</td>
      <td>阈值，用于判断条件 $x > \text{threshold}$。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>value</td>
      <td>属性</td>
      <td>替换值，当 $x \leq \text{threshold}$ 时，输出为 value。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出张量，公式中的 $y$，与 self 的 shape 和数据类型一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入张量 self 与输出张量 out 的 shape 必须一致。
- threshold 和 value 为编译期/运行期常量，通过属性传入。
- 仅支持 Ascend950 芯片。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| -------- | -------- | ---- |
| aclnn调用 | [test_aclnn_threshold_v2](./examples/arch35/test_aclnn_threshold_v2.cpp) | 通过 `aclnnThresholdV2GetWorkspaceSize` 和 `aclnnThresholdV2` 接口方式调用。 |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ------ | ------ | -------- | -------- | -------- |
| liuyi | 个人开发者 | ThresholdV2 | 2026/04/28 | ThresholdV2算子适配开源仓 |
