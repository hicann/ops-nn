# Dequantize

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>   |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：对量化后的整数张量进行反量化，将其转换回浮点数。

- 计算公式：

  支持三种反量化模式，由mode属性指定：

  **MIN_COMBINED模式：**

  $$
  y = \frac{(x + bias) \times (max\_range - min\_range)}{2^n - 1} + min\_range
  $$

  其中，$n$为输入数据类型位数，$bias = 2^{n-1}$（有符号类型）或$0$（无符号类型）。

  **MIN_FIRST模式：**

  对于INT8/UINT8：

  $$
  offset = min\_range + \frac{round(-min\_range \times (2^n - 1) / (max\_range - min\_range)) \times (max\_range - min\_range)}{2^n - 1}
  $$

  $$
  y = \frac{(x + bias) \times (max\_range - min\_range)}{2^n - 1} + min\_range - offset
  $$

  对于INT32：与MIN_COMBINED模式相同。

  **SCALED模式：**

  $$
  y = x \times \frac{max\_range}{2^{n-1} - 1}
  $$

  其中$n$为输入数据类型位数。

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
      <td>x</td>
      <td>输入</td>
      <td>量化后的输入张量</td>
      <td>INT8、UINT8、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min_range</td>
      <td>输入</td>
      <td>量化范围的最小值</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max_range</td>
      <td>输入</td>
      <td>量化范围的最大值</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>属性</td>
      <td>反量化模式，可选值：MIN_COMBINED、MIN_FIRST、SCALED</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
       <td>反量化后的输出张量，shape与输入x一致</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

1. min_range和max_range必须满足max_range > min_range。
2. 在SCALED模式下，max_range必须大于0。
3. min_range和max_range支持广播，其shape需要与输入x的shape兼容。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_dequantize.cpp](examples/test_geir_dequantize.cpp) | 通过[算子IR](op_graph/dequantize_proto.h)构图方式调用Dequantize算子。 |
