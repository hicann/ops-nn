# GruBlockCell

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:-------:|
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：GRU（门控循环单元）单步 cell 前向计算，对标 `tf.raw_ops.GRUBlockCell`（reset_before 语义）。给定当前步输入 x、上一步隐状态 hPrev、合并权重与偏置，一次调用融合完成门控 GEMM、候选 GEMM、sigmoid/tanh 门控激活与隐状态更新。其中 r/u/c 为反向传播所需的中间量外显输出，h 为新隐状态。B=batch、I=inputSize、H=hiddenSize。
- 计算公式：

  $$
  ru\_bar = [x,\ h_{prev}] \cdot w_{ru} + b_{ru} \in \mathbb{R}^{B \times 2H}
  $$

  $$
  r = \sigma(ru\_bar[:,\ 0{:}H]), \qquad u = \sigma(ru\_bar[:,\ H{:}2H])
  $$

  $$
  c = \tanh\left([x,\ h_{prev} \odot r] \cdot w_c + b_c\right)
  $$

  $$
  h = u \odot (h_{prev} - c) + c
  $$

  其中 $\sigma$ 为 sigmoid，$\odot$ 为逐元素乘，$[x,\ h_{prev}]$ 表示沿 axis 1 拼接。
- 语义要点：
  - reset_before 变体：重置门 r **先**与 hPrev 逐元素相乘、**再**进候选 GEMM，区别于 PyTorch GRU 的 reset_after 语义（`torch.nn.functional.gru_cell` 与本算子不等价，不做映射）。
  - 门序 r|u：wRu 列序前 H 列对应 r 门、后 H 列对应 u 门。
  - 步内串行依赖：候选 GEMM 依赖门控 GEMM 的输出 r，单 kernel 内两段 Cube 计算串行执行。
  - 特殊值语义：NaN 逐元素传播；±Inf 经 sigmoid/tanh 饱和（$\sigma(\pm\infty) \in \{0,1\}$、$\tanh(\pm\infty) = \pm 1$），不产生新 NaN；全零输入时输出为确定值（r=u=0.5、c=0、h=0）。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:-------|:--------------|:-----|:--------|:--------|
| x | 输入 | 表示当前步输入，对应公式中 x，shape [B, I]。 | FLOAT | ND |
| hPrev | 输入 | 表示上一步隐状态，对应公式中 $h_{prev}$，shape [B, H]，dim0 与 x 一致。 | FLOAT | ND |
| wRu | 输入 | 表示 r/u 门合并权重，对应公式中 $w_{ru}$，shape [I+H, 2H]（行：x 块 + h 块；列：r 块 \| u 块）。 | FLOAT | ND |
| wC | 输入 | 表示候选合并权重，对应公式中 $w_c$，shape [I+H, H]。 | FLOAT | ND |
| bRu | 输入 | <ul><li>表示 r/u 门合并偏置，对应公式中 $b_{ru}$，shape [2H]。</li><li>须为 rank-1；[1, 2H] size-1 广播视图暂不支持，见约束说明。</li></ul> | FLOAT | ND |
| bC | 输入 | <ul><li>表示候选偏置，对应公式中 $b_c$，shape [H]。</li><li>须为 rank-1；[1, H] size-1 广播视图暂不支持，见约束说明。</li></ul> | FLOAT | ND |
| r | 输出 | 表示重置门激活（训练中间量），对应公式中 r，shape [B, H]。 | FLOAT | ND |
| u | 输出 | 表示更新门激活（训练中间量），对应公式中 u，shape [B, H]。 | FLOAT | ND |
| c | 输出 | 表示候选状态（训练中间量），对应公式中 c，shape [B, H]。 | FLOAT | ND |
| h | 输出 | 表示新隐状态，对应公式中 h，shape [B, H]。 | FLOAT | ND |

参数顺序与 [op_host/gru_block_cell_def.cpp](op_host/gru_block_cell_def.cpp) 的注册序一致（该文件为算子定义真值源，[op_graph/gru_block_cell_proto.h](op_graph/gru_block_cell_proto.h) 的 GE IR 原型与之逐项对应）。本算子无属性，对齐 TF REGISTER_OP 的属性面仅 dtype 约束 T: {float}。

## 约束说明

- 数据类型：仅 float32，6 输入与 4 输出 dtype 必须一致，无 fp16/bf16 路径。
- 数据格式：仅 ND。
- shape 范围：B∈[1, 2³¹−1]，I∈[1, 65535]，H∈[1, 65535]；x/hPrev/wRu/wC 为 rank-2，bRu/bC 为 rank-1。
- 空 Tensor 支持：不支持，B/I/H 均须 ≥1。
- 偏置广播视图限制：[1, 2H]/[1, H] size-1 广播视图当前被 tiling 阶段拒绝（GRAPH_FAILED）。依据（实测登记）：GEIR 在线图路对 rank-2 size-1 视图输入的投递存在框架层缺陷——放行后 pass1（c/h）输出数值损坏而 pass0（r/u）正常，且与偏置取值无关（零偏置仍错、wC=0 恢复），仅改输入声明形状即可复现/消失，原版 kernel 同样复现（非本实现引入）；放行会把干净报错变成静默错数，故维持 rank-1 要求，待框架侧修复后放开（届时 kernel 按 ND 连续存储平表读取，无需改动）。
- 片上容量约束：隐状态维实际支持域为 H≤8152（按 8 对齐后 Hp≤8152），H 超限在 host tiling 阶段即被拒绝（GRAPH_FAILED），不下发设备执行。边界由 L1 的 aH 全幅回灌槽（h⊙r，pass1 的 h 段 Mmad 需整幅 A、H 向不可分片）决定，容量口径取平台查询值。逼近上界时列片宽 nL0c 被迫收窄，吞吐显著下降（H=8152 约为 H=6144 的 1/8），属优雅降级非功能缺陷。
- 权重布局校验前移：x/hPrev rank-2、x.shape[0]==hPrev.shape[0]、wRu [I+H, 2H]、wC [I+H, H]、bRu/bC 元素数 2H/H 等约束在 InferShape 阶段校验（优于 TF 的 kernel 运行时校验），违规报 `shape_mismatch`。
- 精度：与独立 fp64 golden 比对，容差 rtol=1e-5/atol=1e-6。K=I+H 超过噪声标定域（K>2048）时存在 fp32 L0C 顺序累加的精度地板，实测 K=8160 处 max_abs 约 1.85e-6；该地板由 split-K 分组（组宽 16）+ AIV 侧 Neumaier 补偿合并压制，组宽是精度契约而非调优旋钮。
- 实现形态：fp32 唯一路径，tilingKey 恒 0 单档（静态 shape 统一 kernel）；GM 搬运无 32B 对齐要求，任意合法 shape 可执行。

## 调用说明

Ascend 950PR&950DT系列产品：

| 调用方式 | 调用样例 | 说明 |
|:--------|:--------|:-----|
| GE 图模式 | [test_geir_gru_block_cell](examples/arch35/test_geir_gru_block_cell.cpp)、[test_geir_gru_block_cell_dynamic](examples/arch35/test_geir_gru_block_cell_dynamic.cpp) | 通过[算子 IR](op_graph/gru_block_cell_proto.h) 以 `op::GruBlockCell` 构图方式调用 GruBlockCell 算子，经 `ge::Session` 编译执行。前者为静态 shape 通路（13 个 suite，含容量边界与特殊值），后者覆盖 `-1`/`-2` 两种未知维声明的动态定形通路。编译运行：`bash build.sh --run_example gru_block_cell graph --soc=ascend950`。 |
| TensorFlow 原生接口 | [gru_block_cell_tf_plugin](framework/gru_block_cell_tf_plugin.cpp) | `tf.raw_ops.GRUBlockCell` 经 TF 插件 1:1 自动映射为 GruBlockCell 算子；两侧输入输出顺序、shape、合并权重布局完全一致，无需权重重排与子图展开。在线执行依赖 TF-NPU 适配器 `npu_device`，且算子须在 `tf.function` 内调用。 |
| aclnn API | - | 本算子 GEIR-only 交付，不提供 aclnn 接口。 |
| PyTorch API | - | 不提供 torch_extension 接口；且 PyTorch GRU 为 reset_after 语义，与本算子不等价，不做映射。 |
