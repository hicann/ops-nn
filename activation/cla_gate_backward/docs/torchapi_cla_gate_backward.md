# cla_gate_backward

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：

  融合算子，实现CLA（Cross-Layer Attention）gate merge的反向梯度计算。输入上游梯度G = ∂L/∂O、两路原始 Attention输出O_g/O_l与两路 gate logits z_g/z_l，一次性计算并返回四个梯度：两路Attention输出梯度（dO_g、dO_l）与两路 gate logits 梯度（dz_g、dz_l）。本接口不做梯度量化：输出保持输入精度（BF16/FP16），不生成FP8梯度。反向算子不接收前向sigmoid中间结果，s_g/s_l由算子内重新计算。

- 计算公式：

  记本rank上token数为T，attention head数为N，value head dim为D。

  **阶段1：Sigmoid重算（算子内重算，FP32域）**

  - 两路head-wise Sigmoid门控及其导数项（z_g、z_l为两路gate logits）：

    $$
    s\_g = \sigma(z\_g), \qquad s\_l = \sigma(z\_l), \qquad \sigma(x) = \frac{1}{1 + e^{-x}}
    $$

  **阶段2：Attention输出梯度（逐元素链式法则，s_g/s_l沿D维广播）**

  $$
  dO\_g = G \odot s\_g, \qquad dO\_l = G \odot s\_l
  $$

  **阶段3：gate logits梯度（链式法则 + Sigmoid导数，sum_d仅沿head_dim D归约，不跨token或head）**

  $$
  dz\_g = \left( \sum_{d=0}^{D-1} G \odot O\_g \right) \odot s\_g \odot (1 - s\_g), \qquad
  dz\_l = \left( \sum_{d=0}^{D-1} G \odot O\_l \right) \odot s\_l \odot (1 - s\_l)
  $$

  - G⊙O_g/G⊙O_l的乘积与沿D的归约均在 FP32域完成（一次pass内完成，无额外内存往返），以降低归约累积误差。
  - 无除法/指数溢出路径；NaN/Inf 输入按IEEE语义传播（无特殊保护需求）。

## 函数原型

```python
cann_ops_nn.cla_gate_backward(
    grad_merged,
    global_attn,
    local_attn,
    global_gate_logits,
    local_gate_logits,
    *,
    input_attn_layout="TND",
) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `grad_merged` | Tensor | 必选 | 输出投影反传到gate merge的梯度，对应公式中的`G`。支持非连续Tensor，不支持空Tensor。 | `torch.float16`、`torch.bfloat16` | 3维 `[T, N, D]`，`D`仅支持128或256 |
| `global_attn` | Tensor | 必选 | 前向Global/CLA分支Attention输出，对应公式中的`O_g`。支持非连续Tensor，不支持空Tensor。 | `torch.float16`、`torch.bfloat16`，数据类型与 gradMerged 一致 | 3维 `[T, N, D]`，与 `grad_merged`完全一致 |
| `local_attn` | Tensor | 必选 | 前向Local/SWA分支Attention输出，对应公式中的`O_l`。支持非连续Tensor，不支持空Tensor。 | `torch.float16`、`torch.bfloat16`，数据类型与 gradMerged 一致 | 3维 `[T, N, D]`，与 `grad_merged`完全一致 |
| `global_gate_logits` | Tensor | 必选 | 前向Global gate logits，对应公式中的`z_g`。支持非连续Tensor，不支持空Tensor。 | `torch.float16`、`torch.bfloat16`，数据类型与 gradMerged 一致 | 2维 `[T, N]`，`T`、`N`与三路TND输入的前两维一致 |
| `local_gate_logits` | Tensor | 必选 | 前向Local gate logits，对应公式中的`z_l`。支持非连续Tensor，不支持空Tensor。 | `torch.float16`、`torch.bfloat16`，数据类型与 gradMerged 一致 | 2维 `[T, N]`，与 `global_gate_logits`完全一致 |
| `input_attn_layout` | str | 可选 | 输入tensor的排布格式字符串，当前仅支持 `"TND"`。默认值 `"TND"`。 | - | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| `grad_global_attn_out` | Tensor | 必选 | Global Attention分支梯度，对应公式中的`dO_g`。 | 与 `global_attn` 一致 | 与 `global_attn` 一致 |
| `grad_local_attn_out` | Tensor | 必选 | Local Attention分支梯度，对应公式中的`dO_l`。 | 与 `local_attn` 一致 | 与 `local_attn` 一致 |
| `grad_global_gate_logits_out` | Tensor | 必选 | Global gate logits梯度，对应公式中的`dz_g`。 | 与 `global_gate_logits` 一致 | 与 `global_gate_logits` 一致 |
| `grad_local_gate_logits_out` | Tensor | 必选 | Local gate logits梯度，对应公式中的`dz_l`。 | 与 `local_gate_logits` 一致 | 与 `local_gate_logits` 一致 |

## 约束说明

- 该接口支持训练场景下使用。
- 该接口支持单算子模式调用。
- 算子不支持空Tensor；支持非连续Tensor。
- `D`取值仅支持128或256。

## 确定性计算

- 默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  # 构造输入：三路 TND 输入 [T, N, D]，两路 gate logits [T, N]，dtype 必须一致
  grad_merged = torch.randn(8, 64, 256, dtype=torch.bfloat16).npu()
  global_attn = torch.randn(8, 64, 256, dtype=torch.bfloat16).npu()
  local_attn = torch.randn(8, 64, 256, dtype=torch.bfloat16).npu()
  global_gate_logits = torch.randn(8, 64, dtype=torch.bfloat16).npu()
  local_gate_logits = torch.randn(8, 64, dtype=torch.bfloat16).npu()

  grad_global_attn_out, grad_local_attn_out, grad_global_gate_logits_out, grad_local_gate_logits_out = \
      cann_ops_nn.cla_gate_backward(
          grad_merged, global_attn, local_attn,
          global_gate_logits, local_gate_logits,
          input_attn_layout="TND",
      )
  print(grad_global_attn_out.shape, grad_global_attn_out.dtype)
  print(grad_local_attn_out.shape, grad_local_attn_out.dtype)
  print(grad_global_gate_logits_out.shape, grad_global_gate_logits_out.dtype)
  print(grad_local_gate_logits_out.shape, grad_local_gate_logits_out.dtype)
  ```
