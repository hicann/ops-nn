# HardSwishGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term> |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
| <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
| <term>Atlas 推理系列产品</term>    |     ×    |
| <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：HardSwish激活函数的反向传播，完成张量self的梯度计算。
- 计算公式：

  $$
  out_{i} = gradOutput_{i} \times gradSelf_{i}
  $$

  其中 gradSelf 的计算公式为：

  $$
  gradSelf_{i} = \begin{cases}
  0, & self_{i} \le -3, \\
  self_{i} / 3 + 0.5, & -3 < self_{i} < 3, \\
  1, & self_{i} \ge 3
  \end{cases}
  $$

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>表示HardSwish激活函数正向输出的梯度，公式中的gradOutput。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>表示用于计算激活函数的张量，公式中的self。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>输出</td>
      <td>表示HardSwish正向输入的梯度值，即对输入进行求导后的结果。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 确定性计算：aclnnHardswishBackward默认确定性实现。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_hard_swish_backward](examples/test_aclnn_hard_swish_backward.cpp) | 通过[aclnnHardswishBackward](docs/aclnnHardswishBackward.md)接口方式调用HardSwishGrad算子。 |
