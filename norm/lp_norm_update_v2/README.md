# LpNormUpdateV2

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：LpNormUpdateV2是Lp范数两步计算的"更新"（Update）阶段算子。接收LpNormReduceV2的归约结果`x = Σ|x|^p`（非负实数），按范数阶数`p`选择对应的取根分支计算`f_p(x)`，并与数值稳定常数`epsilon`取最大值，得到更新结果`y = max(f_p(x), ε_eff)`。

- 计算公式：

  $$
  y = \max\left(f_p(x),\ \varepsilon_{\text{eff}}\right)
  $$

  其中$f_p(x)$按$p$值分3个分支：

  | 分支 | 触发条件 | 计算公式 |
  |------|---------|---------|
  | 恒等分支 | p ∈ {0, 1, ±inf} | f_p(x) = x |
  | sqrt分支 | p ≈ 2.0 | f_p(x) = √x |
  | 幂运算分支 | 其他p值 | f_p(x) = exp(ln(x)/p) |

  有效epsilon $\varepsilon_{\text{eff}}$ 受输入dtype影响：FP16下 $\varepsilon \leq 10^{-7}$ 时提升到 $10^{-7}$（防下溢），FP32/BF16直接使用 $\varepsilon$。

- 使用场景：作为`torch.linalg.vector_norm`在Ascend NPU上的关键子步骤，与LpNormReduceV2配合使用。典型场景包括训练阶段梯度裁剪（`clip_grad_norm_`）、权重归一化（WeightNorm）等。

  端到端等价关系：

  ```text
  LpNormReduceV2(x, p) → LpNormUpdateV2(S, p, epsilon=0) ≡ linalg_vector_norm(x, ord=p)
  ```

## 参数说明

<table style="table-layout: fixed"><colgroup>
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
    <td>LpNormReduceV2的归约结果Σ|input|^p，非负实数。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>p</td>
    <td>属性</td>
    <td><ul><li>范数阶数，默认值为2.0。</li><li>支持任意浮点值及±inf。</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>属性</td>
    <td><ul><li>数值稳定常数，默认值为1e-12。</li><li>FP16下ε≤1e-7时自动提升到1e-7。</li></ul></td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>更新结果max(f_p(x), ε_eff)，与x同shape同dtype。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

GE IR定义：

```cpp
REG_OP(LpNormUpdateV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(p, Float, 2.0)
    .ATTR(epsilon, Float, 1e-12)
    .OP_END_FACTORY_REG(LpNormUpdateV2)
```

## 约束说明

- 输入`x`必须为非负实数（来自LpNormReduceV2的归约结果Σ|input|^p ≥ 0），否则log分支会产生NaN。
- 输出`y`与输入`x`完全同shape同dtype，无广播（逐元素操作）。
- dtype升精度策略：
  - FP16：中间计算提升到FP32（FP16→FP32→FP16），对齐PyTorch `opmath_type<Half>=float`。
  - BF16：中间计算提升到FP32（BF16→FP32→BF16），输出用四舍五入还原。
  - FP32：直接在FP32精度计算。
- epsilon保护：FP16下ε≤1e-7时自动提升到1e-7（FP16最小正规数约6e-8，1e-12会下溢为0）；FP32/BF16直接使用传入值。
- p分支规则：
  - p ∈ {0, 1, ±inf}：走恒等分支（归约结果已是终值，无需取根）。
  - p ≈ 2.0：走sqrt分支（使用高精度sqrt指令）。
  - 其他p值：走log-exp分支（exp(ln(x)/p)），大正值经exp可能溢出。
- 确定性说明：默认确定性实现（逐元素操作，无归约/矩阵运算，天然确定性）。
- 支持空Tensor和非连续Tensor。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/arch35/test_geir_lp_norm_update_v2.cpp">test_geir_lp_norm_update_v2</a></td>
    <td>通过REG_OP(LpNormUpdateV2)入图，由GE调度执行。</td>
  </tr>
</tbody></table>
