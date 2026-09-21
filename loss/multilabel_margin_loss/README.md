# MultilabelMarginLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>  |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     √    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：计算输入`x`（预测分数）与`target`（多标签下标）之间的多标签间隔损失（multilabel margin loss），并输出`is_target`目标位掩码。
- 计算公式：

  对每个样本`n`（类别数为`C`），设该样本的有效标签集合为`T`（`target`逐元素取值、遇首个`-1`终止；有效值即标签下标），当`reduction`为`'none'`时：

  $$
  \text{loss}[n]=\frac{1}{C}\sum_{j\in T}\sum_{i\notin T}\max\bigl(0,\ 1-(x[n,j]-x[n,i])\bigr)
  $$

  当`reduction`为`'mean'`或`'sum'`时，对逐样本损失再做均值或求和归约。

  同时输出`is_target`：`is_target[n,i] = 1`（当`i∈T`）否则`0`，为`target`派生的0/1掩码。

  其中：
  - `x`：输入预测张量，shape `[N,C]`（2D）或`[C]`（1D）。
  - `target`：真实多标签下标，与`x`同shape，每行有效标签后以`-1`填充。
  - `is_target`：目标位掩码，与`target`同shape。

## 参数说明

<table style="table-layout: auto; width: 100%">
<thead>
    <tr>
    <th style="white-space: nowrap">参数名</th>
    <th style="white-space: nowrap">输入/输出/属性</th>
    <th style="white-space: nowrap">描述</th>
    <th style="white-space: nowrap">数据类型</th>
    <th style="white-space: nowrap">数据格式</th>
    </tr>
</thead>
<tbody>
    <tr>
    <td>x</td>
    <td>输入</td>
    <td>公式中的`x`，预测分数张量。shape为`[N,C]`（2D）或`[C]`（1D）。空tensor仅支持`N`轴为空（`[0,C]`，要求`C≥1`，空进空出）；`C=0`不支持（含`1/C`无意义，报`161002`）。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>target</td>
    <td>输入</td>
    <td>公式中的多标签下标，与`x`同shape，元素取值`[-1, C-1]`，`-1`为填充/终止符。</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y</td>
    <td>输出</td>
    <td>损失输出；`reduction=none`+2D为`[N]`，否则为标量`[1]`。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32（与`x`一致）</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>is_target</td>
    <td>输出</td>
    <td>目标位0/1掩码，与`target`同shape，掩码值由`target`派生（0/1）。dtype支持INT32或与`x`一致。</td>
    <td>INT32，或与`x`一致（FLOAT16、BFLOAT16、FLOAT32）</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>reduction</td>
    <td>属性</td>
    <td>归约方式，支持`'none'`、`'mean'`、`'sum'`。</td>
    <td>STRING</td>
    <td>-</td>
    </tr>

</tbody></table>

## 约束说明

无。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_multilabel_margin_loss.cpp](examples/test_aclnn_multilabel_margin_loss.cpp) | 通过 [aclnnMultilabelMarginLoss](docs/aclnnMultilabelMarginLoss.md) 接口方式调用MultilabelMarginLoss算子。 |
| 图模式（GE）  | [test_geir_multilabel_margin_loss.cpp](examples/test_geir_multilabel_margin_loss.cpp) | 通过GE图（`is_target`输出INT32）方式调用。 |
