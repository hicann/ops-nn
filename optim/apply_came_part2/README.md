# ApplyCamePart2

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

- 算子功能：CAME（Confidence-guided Adaptive Memory Efficient Stochastic Optimizer，置信度引导的自适应内存高效随机优化器）优化器 4-Part 拆分的第二部分（Part2）。在前一部分（ApplyCamePart1）已计算出 grad 的行/列/总和统计（sum_grad_r、sum_grad_c、sum_grad_rc）后，本算子对上一 step 的行/列二阶矩估计 r、c 做 EMA 更新，并用置信度归一化计算归一化更新方向 u，同时累加 u 的平方和 sum_square_u 供 Part3（裁剪 + 一阶矩 + 残差统计）继续消费。其中 r、c 为原地（in-place）更新。
- 计算公式（对齐 kernel 实现 `op_kernel/apply_came_part2.cpp` `ProcessU`/`ComputeR`/`ComputeC`，记 $N=\text{grad.shape}[0]$ 行数、$M=\text{grad.shape}[1]$ 列数）：

  $$
  r_{\text{out}} = \beta_2 \cdot r + (1 - \beta_2) \cdot \frac{\text{sum\_grad\_r}}{M}
  $$

  $$
  c_{\text{out}} = \beta_2 \cdot c + (1 - \beta_2) \cdot \frac{\text{sum\_grad\_c}}{N}
  $$

  $$
  \text{sum\_r\_val} = \text{sum\_r}\ (\text{缺省时 } \sum r),\quad \text{denom} = \beta_2 \cdot \frac{\text{sum\_r\_val}}{N} + (1 - \beta_2) \cdot \frac{\text{sum\_grad\_rc}}{M \cdot N}
  $$

  $$
  u = \frac{\text{grad}}{\sqrt{r_{\text{out}} \cdot c_{\text{out}} \,/\, \text{denom}}} \quad (r_{\text{out}} \text{ 按行广播、} c_{\text{out}} \text{ 按列广播})
  $$

  $$
  \text{sum\_square\_u} = \sum (u \odot u)
  $$

  其中 $r_{\text{out}}$、$c_{\text{out}}$ 为 EMA 更新后的行/列二阶矩，$u$ 的归一化分母 `denom` 为置信度归一化项（`beta2*sum_r/N` 反映二阶矩量级、`(1-beta2)*sum_grad_rc/(M*N)` 反映当前梯度量级）。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:-------|:--------------|:-----|:--------|:--------|
| grad | 输入 | 梯度，二维张量 [n, m]。 | FLOAT16、FLOAT、BF16 | ND |
| sum_grad_r | 输入 | grad 按行求和，shape [n]。 | FLOAT | ND |
| sum_grad_c | 输入 | grad 按列求和，shape [m]。 | FLOAT | ND |
| sum_grad_rc | 输入 | grad 全体元素之和，shape [1]。 | FLOAT | ND |
| r | 输入/输出 | 行二阶矩估计，shape [n]，原地更新（`inplace_with(r)`）。dtype 与 grad 一致。 | FLOAT16、FLOAT、BF16 | ND |
| c | 输入/输出 | 列二阶矩估计，shape [m]，原地更新（`inplace_with(c)`）。dtype 与 grad 一致。 | FLOAT16、FLOAT、BF16 | ND |
| beta2 | 输入 | 二阶矩衰减系数，标量 shape [1]。 | FLOAT | ND |
| sum_r | 输入（可选） | r 的全体元素之和，shape [1]。缺省时由算子内部计算 $\sum r$。 | FLOAT | ND |
| global_shape | 输入（可选） | 原始 [n, m]，shape [2]。缺省时取 grad 的 shape。 | INT64 | ND |
| u | 输出 | 归一化更新方向，shape = grad.shape，强制 FLOAT32（保精度）。 | FLOAT | ND |
| sum_square_u | 输出 | u 的平方和，shape = sum_grad_rc.shape，FLOAT32。 | FLOAT | ND |

## 约束说明

- grad 必须为二维张量（shape `[N, M]`），r/c/sum_grad_r/sum_grad_c/sum_grad_rc 必须为一维张量。
- grad.shape[0] == r.shape[0] == sum_grad_r.shape[0] == N，grad.shape[1] == c.shape[0] == sum_grad_c.shape[0] == M。
- grad/r/c 三者数据类型须一致，支持 FLOAT16、FLOAT32、BFLOAT16。
- u 和 sum_square_u 强制输出 FLOAT32；sum_grad_r/sum_grad_c/sum_grad_rc/beta2/sum_r 为 FLOAT32；global_shape 为 INT64。
- r 和 c 为原地更新，输出与输入共享内存地址。
- sum_r 为可选输入，缺省时算子内部计算 $\sum r$；global_shape 为可选输入，缺省时取 grad 的 shape。
- 支持空 Tensor（N=0 或 M=0 时返回空结果，sum_square_u=0），不报错。
- 支持含 size-1 维广播（N=1 或 M=1）。
- 全零梯度时输出为 0，sum_square_u 始终非负。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|:--------|:--------|:-----|
| 图模式调用 | [test_geir_apply_came_part2](examples/test_geir_apply_came_part2.cpp) | 通过[算子IR](op_graph/apply_came_part2_proto.h)构图方式调用 ApplyCamePart2 算子。 |
