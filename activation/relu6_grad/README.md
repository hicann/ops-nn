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

## 功能说明

- 算子功能：激活函数Relu6的反向，计算对应Relu6操作的反向传播梯度。对前向输入落在开区间`(0, 6)`的位置透传上游梯度，其余位置（含端点`features = 0`与`features = 6`）输出0。

- 计算公式：

  $$
  backprops_{i} =
  \begin{cases}
  gradients_{i}, & 0 < features_{i} < 6 \\
  0, & otherwise
  \end{cases}
  $$

  其中：
  - gradients：对应Relu6操作的反向传播梯度。
  - features：作为输入传递给对应Relu6操作的特征，或其输出，使用任一者产生相同结果。

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
      <td>gradients</td>
      <td>输入</td>
      <td>传递给对应Relu6操作的反向传播梯度，公式中的gradients。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>features</td>
      <td>输入</td>
      <td>作为输入传递给对应Relu6操作的特征，或其输出，使用任一者产生相同结果，公式中的features。其shape须与gradients一致，或为标量[1]。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprops</td>
      <td>输出</td>
      <td>公式中的输出张量，shape与gradients一致。</td>
      <td>FLOAT16、FLOAT、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- gradients、features、backprops的数据类型须保持一致，不支持混合数据类型。
- 不支持广播：features的shape须与gradients完全一致，或为标量[1]。
- 支持动态shape与动态rank。

## 实现方案

| 层 | 文件 | 说明 |
| --- | --- | --- |
| 计算图原型 | `op_graph/relu6_grad_proto.h` | `REG_OP(Relu6Grad)`，二输入一输出 |
| 算子定义 | `op_host/relu6_grad_def.cpp` | `OpDef::AddConfig("ascend950", ...)` |
| InferShape | `op_host/relu6_grad_infershape.cpp` | 复用`Ops::Base::InferShape4Broadcast` |
| Tiling | `op_host/arch35/relu6_grad_tiling_arch35.{h,cpp}` | 按dtype分支调用`Ops::Base::BroadcastBaseTiling<OpDag>` |
| DAG | `op_kernel/arch35/relu6_grad_dag.h` | fp32通路原生计算；fp16/bf16通路提升fp32中间精度 |
| Struct | `op_kernel/arch35/relu6_grad_struct.h` | `BRC_TEMP_SCH_MODE_KEY_DECL/SEL` |
| Kernel入口 | `op_kernel/relu6_grad_apt.cpp` | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` + `BroadcastSch<schMode, OpDag>` |

### fp32通路

```Cpp
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

```Cpp
`Vec::Compare<u8, fp32, GT/LT>`输出位掩码，`Vec::And<u8>`合成`(0 < x < 6)`
区间mask；`Vec::Select<u8, fp32, TENSOR_TENSOR>`按`mask ? dy : 0`逐元素
选择，避免分支，硬件原生vsel实现。

### fp16 / bf16通路

In0/In1 ─CopyInBrc─ Cast(->fp32) ─+
                                  +─ Compare ── And ─ Select ── Cast(->T,RINT) ─ CopyOut ─ Out0
              Const(0,fp32) ── Zero ─+
              Const(6,fp32) ── Six ──+
```

把fp16/bf16整体提升到fp32做cmp/and/sel，末端用`CAST_MODE_RINT`
（round-to-nearest-even）回退到原dtype。原因：

1. 与DSL `relu6_grad.py`在fp16vcmpsel不可用时fallback到fp32的行为一致；
2. 上下界0.0和6.0在fp16/bf16中均能精确表达，但cmp走fp32路径更稳健，
   且dy是NaN/Inf时fp32中间不会被半精度范围截断。

## 调用说明

| 调用方式   | 样例代码           | 说明                                            |
| ---------------- | --------------------------- |-----------------------------------------------|
| 图模式 | [test_geir_relu6_grad.cpp](examples/arch35/test_geir_relu6_grad.cpp)  | 通过[算子IR](op_graph/relu6_grad_proto.h)构图方式调用Relu6Grad算子。 |
