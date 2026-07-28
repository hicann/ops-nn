# L2NormalizeGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     √    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：前向L2Normalize（`y = x/max(sqrt(sum(x^2, dim)), eps)`）的反向算子。三输入`x`（前向输入）、`y`（前向输出，即归一化后的x）、`dy`（上游梯度），单输出`dx`（对x的梯度）。

- 计算公式（沿属性`dim`指定的单个归一化轴归约，keepdims广播回全形状）：

  $$
  n = \max\left(\sqrt{\sum_{dim} x^2}, \ eps\right)
  $$

  $$
  s = \sum_{dim} (y \cdot dy)
  $$

  $$
  dx = \frac{dy - y \cdot s}{n}
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
      <td>x</td>
      <td>输入</td>
      <td>表示正向算子的输入，对应公式中的`x`。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输入</td>
      <td>表示正向算子的输出，即归一化后的`x`，对应公式中的`y`。shape与数据类型与入参`x`一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>表示反向传回的梯度，对应公式中的`dy`。shape与数据类型与入参`x`一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td>表示归一化轴，长度为1的数组，取值范围为[-`x`.dim(), `x`.dim()-1]，默认值为1。</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td>表示分母的下限，默认值为1e-4。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td>表示对`x`的梯度，对应公式中的`dx`。shape与数据类型与入参`x`一致。</td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `x`为空Tensor时，`dx`同为空Tensor。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_l2_normalize_grad](examples/test_geir_l2_normalize_grad.cpp) | 通过[算子IR](op_graph/l2_normalize_grad_proto.h)构图方式调用L2NormalizeGrad算子。 |
