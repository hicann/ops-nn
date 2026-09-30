# INTrainingUpdateGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     √    |
|  <term>Atlas推理系列产品</term>    |     √    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：InstanceNorm训练反向传播的第一阶段（对空间维归约、保留per-(N, C)的部分和）。先用输入的`mean`/`variance`将`x`归一化，再与`dy`相乘并沿空间维求和得到`res_gamma`；`dy`直接沿空间维求和得到`res_beta`。**不对N归约**（对N的归约由下游独立算子`INTrainingUpdateGradGammaBeta`完成）。
- 计算公式（`ε`为编译期常量`1e-6`）：

  $$
  x\_norm = (x - mean) \cdot \frac{1}{\sqrt{variance + \varepsilon}}
  $$

  $$
  res\_gamma = \sum_{spatial}(dy \cdot x\_norm), \quad res\_beta = \sum_{spatial}(dy)
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
      <td>dy</td>
      <td>输入</td>
      <td>表示反向传回的梯度，含完整空间维，对应公式中的`dy`。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>NDC1HWC0</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>表示正向算子的输入，含完整空间维，对应公式中的`x`。数据类型与入参`dy`一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>NDC1HWC0</td>
    </tr>
    <tr>
      <td>variance</td>
      <td>输入</td>
      <td>表示每个instance的方差，对应公式中的`variance`。空间维为1。</td>
      <td>FLOAT32</td>
      <td>NDC1HWC0</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>表示每个instance的均值，对应公式中的`mean`。空间维为1。</td>
      <td>FLOAT32</td>
      <td>NDC1HWC0</td>
    </tr>
    <tr>
      <td>res_gamma</td>
      <td>输出</td>
      <td>表示对`gamma`梯度的部分和，对应公式中的`res_gamma`。空间维为1，shape与入参`variance`一致。</td>
      <td>FLOAT32</td>
      <td>NDC1HWC0</td>
    </tr>
    <tr>
      <td>res_beta</td>
      <td>输出</td>
      <td>表示对`beta`梯度的部分和，对应公式中的`res_beta`。空间维为1，shape与入参`variance`一致。</td>
      <td>FLOAT32</td>
      <td>NDC1HWC0</td>
    </tr>
  </tbody></table>

## 约束说明

- 仅支持图模式调用。
- 数据格式仅支持NDC1HWC0（6维），所有输入与输出一致。
- 入参`x`的shape必须与`dy`完全一致。
- 入参`variance`、`mean`与输出`res_gamma`、`res_beta`的shape须为(N, 1, C1, 1, 1, C0)，即由`dy`的N、C1、C0确定、归约轴D/H/W折叠为1。
- 入参`dy`、`x`的数据类型须一致，同为FLOAT16或同为FLOAT32；`variance`、`mean`与两个输出恒为FLOAT32。
- 归约轴（D、H、W）中存在0时，`res_gamma`、`res_beta`的所有元素均为0。
- N、C1、C0维度需大于0；为0时不支持，算子拒绝执行。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_in_training_update_grad](examples/test_geir_in_training_update_grad.cpp) | 通过[算子IR](op_graph/in_training_update_grad_proto.h)构图方式调用INTrainingUpdateGrad算子。 |
