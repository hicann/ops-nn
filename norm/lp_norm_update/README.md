# LpNormUpdate

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:-------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：计算 Lp 范数的"开 p 次根"步骤。输入 x 为 LpNormReduce 的输出（即 $\sum|x_i|^p$），输出 y 为 Lp 范数归一化因子。该结果通常作为分母用于输入除法归一化，或直接作为 Lp 范数值输出。
- 计算公式：

  $$
  y = \max\left(x^{\frac{1}{p}},\ \varepsilon_{\text{eff}}\right)
  $$

  根据 $p$ 值不同，采用不同计算路径：
  - $p \in \{1, +\infty, -\infty\}$：$y = \max(x,\ \varepsilon_{\text{eff}})$（恒等）
  - $p = 2$：$y = \max\left(\sqrt{x},\ \varepsilon_{\text{eff}}\right)$
  - 其他 $p \neq 0$：$y = \max\left(x^{1/p},\ \varepsilon_{\text{eff}}\right)$（含有限负 p，与 PyTorch 对齐）

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:-------|:--------------|:-----|:--------|:--------|
| x | 输入 | 表示 LpNormReduce 的输出 $\sum{\vert x_i \vert}^p$，对应公式中 x。输入值理论上非负。 | FLOAT16、FLOAT | ND |
| p | 属性 | <ul><li>表示范数阶数，对应公式中 p。</li><li>默认值 2。</li><li>取值范围 $p \neq 0$ 或 $\pm\infty$（$\pm\infty$ 经 INT_MAX/INT_MIN 映射传入）。</li><li>p=2 时走 Sqrt 快速路径。</li><li>p∈{1,±∞} 时走恒等路径。</li></ul> | INT | - |
| epsilon | 属性 | <ul><li>表示数值稳定常数，对应公式中 $\varepsilon$。</li><li>默认值 1e-12。</li><li>取值范围 $\geq 0$。</li><li>FP16 下自适应夹紧至 1e-7 防止下溢。</li></ul> | FLOAT | - |
| y | 输出 | 表示 Lp 范数归一化因子 $\max(x^{1/p}, \varepsilon_{\text{eff}})$，对应公式中 y。 | FLOAT16、FLOAT | ND |

## 约束说明

- 确定性说明：默认确定性实现。
- `x` 为 LpNormReduce 的输出（$\sum|x_i|^p$），非原始输入；调用方需先执行 LpNormReduceV2。
- `p` 为整数（INT 类型），取值范围 $p \neq 0$ 或 $\pm\infty$（$\pm\infty$ 经 INT_MAX/INT_MIN 映射传入）；p=2 时走 Sqrt 快速路径，避免 Power 精度损失；有限负 p 走 Power 路径，与 PyTorch `torch.linalg.vector_norm` 对齐。
- `epsilon` 取值范围 $\geq 0$；FP16 下 `epsilon` $\leq 10^{-7}$ 时自适应夹紧至 $10^{-7}$ 防止下溢。
- 空 Tensor 支持：输入 `x` 为空 Tensor 时，输出 `y` 同样为空 Tensor（kernel 直接返回，不执行计算）。
- `x` 的输入值理论上非负（LpNormReduce 输出 $\geq 0$）；对负值输入会在 Sqrt/Power 前夹紧至 0 兜底，避免产生 NaN。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|:--------|:--------|:-----|
| GE 图模式 | [test_geir_lp_norm_update](examples/test_geir_lp_norm_update.cpp) | 通过[算子 IR](op_graph/lp_norm_update_proto.h) 构图方式调用 LpNormUpdate 算子。 |
