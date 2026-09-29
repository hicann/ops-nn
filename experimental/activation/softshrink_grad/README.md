# SoftShrinkGrad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| <term>Atlas A3系列产品</term>| √ |
| <term>Atlas A2系列产品</term> | √ |

## 功能说明

- 算子功能：根据 Softshrink 正向输入与阈值的关系，选择透传上游梯度或输出 0。

- 计算公式：

  $$
  grad\_input_i =
  \begin{cases}
  grad\_output_i, & self_i > lambd \\
  grad\_output_i, & self_i < -lambd \\
  0, & -lambd \le self_i \le lambd
  \end{cases}
  $$

  当 `self` 等于 `lambd` 或 `-lambd` 时，输出为 0。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 310px">
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
      <td>grad_output</td>
      <td>输入</td>
      <td>上游回传梯度张量。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>Softshrink 正向输入张量。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lambd</td>
      <td>属性</td>
      <td>非负阈值，默认值为 0.5。</td>
      <td>Scalar</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>输出</td>
      <td>输入梯度，shape 为 grad_output 与 self 的广播结果。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `grad_output`、`self` 和 `grad_input` 的数据类型必须属于 FLOAT16、FLOAT、BFLOAT16。
- `grad_output` 与 `self` 必须满足广播规则，`grad_input` 的 shape 必须等于广播结果。
- `lambd` 必须大于或等于 0。
- 内部 `SoftShrinkGrad` Kernel 接收 shape 和 dtype 一致的连续 ND Tensor；ACLNN 层负责连续化、广播和必要的数据类型转换。
- 支持空 Tensor；空 Tensor 在 ACLNN 层直接返回，不启动 AICore Kernel。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- | ---- | ---- |
| ACLNN 调用 | [test_aclnn_softshrink_grad.cpp](./examples/test_aclnn_softshrink_grad.cpp) | 通过 `aclnnSoftshrinkBackwardGetWorkspaceSize` 和 `aclnnSoftshrinkBackward` 两段式接口调用 SoftShrinkGrad 算子。 |

运行调用样例：

```bash
bash build.sh --run_example softshrink_grad eager cust --vendor_name=custom --experimental
```

## 参考资源

- [aclnnSoftshrinkBackward 接口文档](./docs/aclnnSoftshrinkBackward.md)
- [需求 Issue #4197](https://gitcode.com/cann/ops-nn/issues/4197)

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| Delicate02 | 个人开发者 | SoftShrinkGrad | 2026/07/21 | SoftShrinkGrad 算子适配开源仓 |
