# FusedBiasLeakyRelu

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

- 接口功能：对输入张量x加上偏置bias后，应用带缩放因子的LeakyReLU激活函数。
- 计算公式：

  $$
  FusedBiasLeakyRelu(x, bias) =
  \begin{cases}
  (x + bias) \ast scale, \quad x + bias \geq 0\\
  (x + bias) \ast negative\_slope \ast scale, \quad x + bias < 0
  \end{cases}
  $$

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1300px"><colgroup>
  <col style="width: 101px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 200px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
      <tr>
      <td>x</td>
      <td>输入</td>
      <td>表示FusedBiasLeakyRelu的输入数据张量。</td>
      <td>支持空Tensor。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>bias</td>
      <td>输入</td>
      <td>表示偏置张量，shape与x完全相同。</td>
      <td>数据类型和shape必须与x一致。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>negative_slope</td>
      <td>属性</td>
      <td>LeakyReLU负半轴斜率系数。</td>
      <td>float类型可选属性，默认值0.2。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>scale</td>
      <td>属性</td>
      <td>缩放因子。</td>
      <td>float类型可选属性，默认值1.414213562373（即√2）。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示FusedBiasLeakyRelu的输出张量。</td>
      <td>shape和dtype与x相同。</td>
      <td>FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- x、bias、y三者的数据类型（dtype）必须一致，仅支持FLOAT16和FLOAT。
- x和bias的shape必须完全相同，不支持广播。
- negative_slope和scale为float类型可选属性，带默认值（negative_slope=0.2，scale=1.414213562373）。
- 支持的数据格式为ND。
- 张量最高支持8维。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式(GEIR)调用 | [test_geir_fused_bias_leaky_relu](./examples/test_geir_fused_bias_leaky_relu.cpp) | 通过 GE IR 图模式（REG_OP 注册）调用 FusedBiasLeakyRelu 算子。 |
