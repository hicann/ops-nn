# LpNormUpdate

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:-------:|
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

- 算子功能：计算Lp范数的"开p次根"步骤。输入x为LpNormReduce的输出（即 $\sum|x_i|^p$），输出y为Lp范数归一化因子。该结果通常作为分母用于输入除法归一化，或直接作为Lp范数值输出。
- 计算公式：

  $$
  y = \max\left(x^{\frac{1}{p}},\ \varepsilon_{\text{eff}}\right)
  $$

  根据 $p$ 值不同，采用不同计算路径：
  - $p \in \{0, 1, +\infty, -\infty\}$：$y = \max(x,\ \varepsilon_{\text{eff}})$（恒等）
  - $p = 2$：$y = \max\left(\sqrt{x},\ \varepsilon_{\text{eff}}\right)$
  - 其他有限 $p$：$y = \max\left(x^{1/p},\ \varepsilon_{\text{eff}}\right)$（含有限负p，与PyTorch对齐）

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:-------|:--------------|:-----|:--------|:--------|
| x | 输入 | 表示LpNormReduce的输出 $\sum{\vert x_i \vert}^p$，对应公式中x。输入值理论上非负。 | FLOAT16、FLOAT | ND |
| p | 属性 | <ul><li>表示范数阶数，对应公式中p。</li><li>默认值2。</li><li>取值范围为任意整数（含0与负值）或 $\pm\infty$（$\pm\infty$ 经INT_MAX/INT_MIN映射传入）。</li><li>p=2时走Sqrt快速路径。</li><li>p∈{0,1,±∞} 时走恒等路径。</li></ul> | INT | - |
| epsilon | 属性 | <ul><li>表示数值稳定常数，对应公式中 $\varepsilon$。</li><li>默认值1e-12。</li><li>取值范围 $\geq 0$。</li><li>FP16下 $\epsilon = 0$ 时不夹紧（保持0）；$0 < \epsilon \leq 10^{-7}$ 时自适应夹紧至 $10^{-7}$ 防止下溢。</li></ul> | FLOAT | - |
| y | 输出 | 表示Lp范数归一化因子 $\max(x^{1/p}, \varepsilon_{\text{eff}})$，对应公式中y。 | FLOAT16、FLOAT | ND |

## 约束说明

- 确定性说明：默认确定性实现。
- `x`为LpNormReduce的输出（$\sum|x_i|^p$），非原始输入；调用方需先执行LpNormReduceV2。
- `p`为整数（INT类型），取值范围为任意整数（含0与负值）或 $\pm\infty$（$\pm\infty$ 经INT_MAX/INT_MIN映射传入）；p=0与p∈{1,±∞} 走恒等路径；p=2时走Sqrt快速路径，避免Power精度损失；有限负p走Power路径，与PyTorch `torch.linalg.vector_norm`对齐。
- `epsilon`取值范围 $\geq 0$；FP16下`epsilon` $= 0$ 时不夹紧（保持0），$0 < \epsilon \leq 10^{-7}$ 时自适应夹紧至 $10^{-7}$ 防止下溢。
- 维度（Rank）约束：`x`与`y`的维度数不超过8；超过8维的输入在tiling阶段返回`GRAPH_FAILED`。
- `x`的输入值理论上非负（LpNormReduce输出 $\geq 0$）；对负值输入会在Sqrt/Power前夹紧至0兜底，避免产生NaN。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|:--------|:--------|:-----|
| GE图模式 | [test_geir_lp_norm_update](examples/test_geir_lp_norm_update.cpp) | 通过[算子IR](op_graph/lp_norm_update_proto.h) 构图方式调用LpNormUpdate算子。 |
