# ClaGateBackward

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------- | ------|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 算子功能：融合算子，实现CLA（Cross-Layer Attention）gate merge的反向梯度计算。输入上游梯度G = ∂L/∂O、两路原始 Attention输出O_g/O_l与两路 gate logits z_g/z_l，一次性计算并返回四个梯度：两路Attention输出梯度（dO_g、dO_l）与两路 gate logits 梯度（dz_g、dz_l）。本接口不做梯度量化：输出保持输入精度（BF16/FP16），不生成FP8梯度。反向算子不接收前向sigmoid中间结果，s_g/s_l由算子内重新计算。

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

## 参数说明

<table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 640px">
  <col style="width: 180px">
  <col style="width: 90px">
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
      <td>grad_merged</td>
      <td>输入</td>
      <td>输出投影反传到gate merge的梯度，即公式中的G。shape为[T, N, D]。支持非连续Tensor，不支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>global_attn</td>
      <td>输入</td>
      <td>前向Global/CLA分支Attention输出，即公式中的O_g。shape与grad_merged一致，为[T, N, D]。数据类型与grad_merged一致。支持非连续Tensor，不支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>local_attn</td>
      <td>输入</td>
      <td>前向Local/SWA分支Attention输出，即公式中的O_l。shape与grad_merged一致，为[T, N, D]。数据类型与grad_merged一致。支持非连续Tensor，不支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>global_gate_logits</td>
      <td>输入</td>
      <td>前向Global gate的logits，即公式中的z_g。shape为[T, N]，T、N与grad_merged一致。数据类型与grad_merged一致。支持非连续Tensor，不支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>local_gate_logits</td>
      <td>输入</td>
      <td>前向Local gate的logits，即公式中的z_l。shape与global_gate_logits一致，为[T, N]。数据类型与grad_merged一致。支持非连续Tensor，不支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_global_attn_out</td>
      <td>输出</td>
      <td>Global Attention分支梯度，即公式中的dO_g。shape为[T, N, D]。数据类型与global_attn一致。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_local_attn_out</td>
      <td>输出</td>
      <td>Local Attention分支梯度，即公式中的dO_l。shape为[T, N, D]。数据类型与local_attn一致。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_global_gate_logits_out</td>
      <td>输出</td>
      <td>Global gate logits梯度，即公式中的dz_g。shape为[T, N]。数据类型与global_gate_logits一致。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_local_gate_logits_out</td>
      <td>输出</td>
      <td>Local gate logits梯度，即公式中的dz_l。shape为[T, N]。数据类型与local_gate_logits一致。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_attn_layout</td>
      <td>可选属性</td>
      <td>输入tensor的排布格式字符串。当前仅支持"TND"。默认值为"TND"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- `grad_merged`、`global_attn`、`local_attn`支持三维`[T, N, D]`，其中`D`取值仅支持128或256。

## 调用说明

- <term>Ascend 950PR/Ascend 950DT</term>：

  | 调用方式 | 调用样例 | 说明 |
  |---------|---------|------|
  | aclnn API | [test_aclnn_cla_gate_backward](./examples/test_aclnn_cla_gate_backward.cpp) | 通过[aclnnClaGateBackward](./docs/aclnnClaGateBackward.md)接口方式调用ClaGateBackward算子。 |
  | PyTorch API | - | 通过[cla_gate_backward](./docs/torchapi_cla_gate_backward.md)接口调用ClaGateBackward算子。 |
