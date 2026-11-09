# SoftsignGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     √    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：完成Softsign激活函数的反向梯度计算。给定上游梯度gradients和前向输入特征features，按元素计算输出梯度。
- 计算公式：

  $$
  output_i = \frac{gradients_i}{(1 + |features_i|)^2}
  $$

  其中：
  - gradients: 上游梯度张量
  - features: Softsign前向函数的输入特征张量
  - output: 计算得到的梯度输出张量

## 参数说明

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">参数名</th>
      <th style="white-space: nowrap">输入/输出</th>
      <th style="white-space: nowrap">描述</th>
      <th style="white-space: nowrap">数据类型</th>
      <th style="white-space: nowrap">数据格式</th>
      <th style="white-space: nowrap">非连续Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>gradients</td>
      <td>输入</td>
      <td>上游梯度张量，对应公式中的gradients。支持0-8维tensor，支持动态shape，支持空tensor。shape需要与features的shape完全相同。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>×</td>
    </tr>
    <tr>
      <td>features</td>
      <td>输入</td>
      <td>Softsign前向函数的输入特征张量，对应公式中的features。支持0-8维tensor，支持动态shape，支持空tensor。shape需要与gradients的shape完全相同。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>×</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>计算得到的梯度输出张量，对应公式中的output。shape与输入相同，数据类型与输入一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>×</td>
    </tr>
  </tbody></table>

## 约束说明

- gradients和features的数据类型必须一致，且仅支持FLOAT、FLOAT16、BFLOAT16。
- gradients和features的shape必须完全相同，不支持广播。
- 非连续tensor（带stride的转置、切片视图等）不在支持范围内。当前版本不保证对非连续输入进行显式拦截或自动转换，调用方应确保传入连续稠密布局的tensor。
- 仅支持0-8维tensor，维度数、format（仅ND）、dtype 越界时在图编译/tiling 阶段受控拒绝（不执行 kernel）。
- FP16/BF16输入在内部会提升至FP32精度计算，结果再转换回原始精度。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式  | [test_geir_softsign_grad.cpp](examples/test_geir_softsign_grad.cpp) | 通过图模式方式调用[SoftsignGrad算子](op_graph/softsign_grad_proto.h)。 |
