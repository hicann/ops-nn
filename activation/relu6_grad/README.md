# Relu6Grad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

> 本算子目前仅在 Ascend 950 (dav-3510 / Atlas 350 加速卡) 架构上实现。

## 功能说明

- **算子功能**：ReLU6 激活函数的反向梯度。对前向输入落在严格开区间
  `(0, 6)` 的位置透传上游梯度 `dy`，落在区间外（含端点 `x = 0` 与
  `x = 6`）的位置输出 0。
- **计算公式**：

$$
dx_i = \begin{cases}
dy_i, & 0 < x_i < 6 \\
0,    & \text{otherwise}
\end{cases}
$$

  严格开区间。`x = 0` 与 `x = 6` 均落入零臂。`x = NaN` 由于 IEEE-754
  比较语义返回 0；`x = ±inf` 同样落入零臂。当 mask 为 true 时 `dy` 按值
  透传（`dy = NaN/Inf` 不会被屏蔽）。
- **第三方对标**：等价于 TensorFlow `Relu6Grad` 与 PyTorch
  `hardtanh_backward(min=0, max=6)` 在边界 `x ∈ {0, 6}` 取 0 的约定。

## 参数说明

| 参数名 | 输入/输出 | 描述                                                 | 数据类型                              | 数据格式 |
| :----: | :-------: | :--------------------------------------------------- | :-----------------------------------: | :------: |
| gradients | 输入   | 上游梯度 `dy`。                                       | FLOAT16, FLOAT, BFLOAT16              | ND       |
| features  | 输入   | 前向输入 `x`（非前向输出）；与 `gradients` 须可广播。 | FLOAT16, FLOAT, BFLOAT16              | ND       |
| backprops | 输出   | 反向梯度 `dx`。shape 为 `gradients`、`features` 广播后的统一形状。 | FLOAT16, FLOAT, BFLOAT16 | ND       |

## 约束说明

- `gradients`、`features`、`backprops` 必须为**同一种 dtype**；不支持 mix-dtype。
- 不支持 INT32 等整型数据。
- 支持任意 NumPy 广播形态（含标量 `[1]`、单维 broadcast、跨 rank broadcast）。
- 支持动态 shape 与动态 rank。
- `features` 应当为 ReLU6 的**前向输入** `x`，而不是前向输出。该约定与
  `canndev/ops/built-in/tbe/impl/dynamic/relu6_grad.py` 一致。

## 实现方案

| 层 | 文件 | 说明 |
| --- | --- | --- |
| 计算图原型 | `op_graph/relu6_grad_proto.h` | `REG_OP(Relu6Grad)`，二输入一输出 |
| 算子定义 | `op_host/relu6_grad_def.cpp` | `OpDef::AddConfig("ascend950", ...)` |
| InferShape | `op_host/relu6_grad_infershape.cpp` | 复用 `Ops::Base::InferShape4Broadcast` |
| Tiling | `op_host/arch35/relu6_grad_tiling_arch35.{h,cpp}` | 按 dtype 分支调用 `Ops::Base::BroadcastBaseTiling<OpDag>` |
| DAG | `op_kernel/arch35/relu6_grad_dag.h` | fp32 通路原生计算；fp16/bf16 通路提升 fp32 中间精度 |
| Struct | `op_kernel/arch35/relu6_grad_struct.h` | `BRC_TEMP_SCH_MODE_KEY_DECL/SEL` |
| Kernel 入口 | `op_kernel/relu6_grad_apt.cpp` | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` + `BroadcastSch<schMode, OpDag>` |

### fp32 通路

```
In0(dy) ─CopyInBrc─+
                   +──────────────────────────────+
                                                  |
In1(x)  ─CopyInBrc─+                              |
                   +─Compare(GT, x, 0) ─ mask_gt0 |
                   +─Compare(LT, x, 6) ─ mask_lt6 |
                          mask = And(mask_gt0, mask_lt6)
                                                  |
   Const(0,fp32) ─Duplicate─ Zero ─────────────── ┴ Select(mask, dy, Zero) ─ CopyOut ─ Out0
```

`Vec::Compare<u8, fp32, GT/LT>` 输出位掩码，`Vec::And<u8>` 合成 `(0 < x < 6)`
区间 mask；`Vec::Select<u8, fp32, TENSOR_TENSOR>` 按 `mask ? dy : 0` 逐元素
选择，避免分支，硬件原生 vsel 实现。

### fp16 / bf16 通路

```
In0/In1 ─CopyInBrc─ Cast(->fp32) ─+
                                  +─ Compare ── And ─ Select ── Cast(->T,RINT) ─ CopyOut ─ Out0
              Const(0,fp32) ── Zero ─+
              Const(6,fp32) ── Six ──+
```

把 fp16 / bf16 整体提升到 fp32 做 cmp/and/sel，末端用 `CAST_MODE_RINT`
（round-to-nearest-even）回退到原 dtype。原因：
1. 与 DSL `relu6_grad.py` 在 fp16 vcmpsel 不可用时 fallback 到 fp32 的行为一致；
2. 上下界 0.0 和 6.0 在 fp16/bf16 中均能精确表达，但 cmp 走 fp32 路径更稳健，
   且 dy 是 NaN/Inf 时 fp32 中间不会被半精度范围截断。

## 调用说明

| 调用方式   | 样例代码                                                     | 说明                                                         |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式 | [test_geir_relu6_grad](examples/test_geir_relu6_grad.cpp) | 通过[算子IR](op_graph/relu6_grad_proto.h)构图方式调用 Relu6Grad 算子；覆盖 fp32/fp16/bf16 基础用例 + 边界 x=0/x=6、x=NaN/±Inf、广播等关键特殊值用例。 |
