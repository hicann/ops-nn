# Softsign

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

## 功能说明

- 算子功能：完成 Softsign 激活函数计算，对输入张量的每个元素逐元素计算 softsign 值。
- 计算公式：

  $$
  y = \frac{x}{1 + |x|}
  $$

  其中 $x$ 为输入张量，$y$ 为输出张量，输出值域为 $(-1, 1)$。

- 数值特性：公式天然数值稳定，分母 $1 + |x| \geq 1$，无除零风险，无需 epsilon 保护。

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
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
    <td>公式中的输入 x，任意形状张量。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>公式中的输出 y，shape 和 dtype 与输入 x 完全一致，值域 (-1, 1)。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- 输入维度限制：输入张量维度不超过 8 维。
- 数据类型限制：仅支持 FLOAT、FLOAT16、BFLOAT16，不支持 DOUBLE。
- FP16/BF16 精度说明：FP16 和 BF16 类型在内部通过 Cast 到 FP32 进行中间计算，再 Cast 回原始类型，以保证计算精度。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td> 图模式调用 </td>
    <td><a href="./examples/arch35/test_geir_softsign.cpp">test_geir_softsign</a></td>
    <td> 通过[算子IR](./op_graph/softsign_proto.h)构图方式调用Softsign算子。 </td>
  </tr>
</tbody></table>
