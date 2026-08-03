# FusedPatchMlp

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 接口功能：

  将 ChatTS-14B 中 `TimeSeriesEmbedding` 的多层 Patch Embedding MLP（多个 `Linear` + `GELU` 重复，最后一层无激活）融合为单个算子。时间序列切分为 patch 后，每个 patch 行经过一个 `num_layers` 层的 MLP 映射到 `hidden_size`，融合后一次下发即可完成，减少多次 kernel 启动开销，对长序列、多 patch 场景收益明显。

- 计算公式：

  给定输入 `x`（形状 `[..., patch_size]`，支持 2D~8D，前置维度展平为 `N` 行），权重 `weights`、偏置 `biases` 按层展平，层数为 `num_layers`：

  $$
  h_0 = x
  $$

  对第 `l` 层（`l = 0, 1, ..., num\_layers-1`）：

  $$
  z_l = h_l \cdot W_l^\top + b_l
  $$

  除最后一层外应用 GELU（tanh 近似）激活：

  $$
  h_{l+1} = \text{GELU}(z_l) = 0.5 \cdot z_l \cdot \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \cdot \left( z_l + 0.044715 z_l^3 \right) \right) \right)
  $$

  最后一层不做激活，直接输出：

  $$
  y = z_{num\_layers-1}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 181px">
  <col style="width: 144px">
  <col style="width: 273px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>公式中的输入 x，形状 [..., patch_size]，支持 2D~8D，最后一维为 patch_size，其余前置维度展平为 patch 行数 N。</td>
      <td>FLOAT16 / FLOAT / BF16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>输入</td>
      <td>各层权重（转置后）展平拼接，公式中的 W。</td>
      <td>FLOAT16 / FLOAT / BF16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>biases</td>
      <td>输入</td>
      <td>各层偏置展平拼接，形状 [num_layers * hidden_size]，公式中的 b。当 x 为 BF16 时 biases 需为 FLOAT。</td>
      <td>FLOAT16 / FLOAT / FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_layers</td>
      <td>必选属性</td>
      <td>MLP 的 Linear 层数。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的 y，前置维度与 x 一致，最后一维由 patch_size 变为 hidden_size。</td>
      <td>FLOAT16 / FLOAT / BF16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 支持 FLOAT16 / FLOAT(fp32) / BF16 三种数据类型（`x`/`weights`/`y` 同 dtype）。**当 `x` 为 BF16 时 `biases` 需为 FLOAT**（bf16 matmul 的 bias 表为 fp32，硬件不支持 bf16→float 的 bias 搬运）。
- `hidden_size` 需使 bias 张量 ≥ 64 字节（FLOAT16/BF16 下 ≥ 32，FLOAT 下 ≥ 16）。
- `patch_size` 需 ≥ 16（满足 Cube 单元 K 方向的分形基本块 `C0=16`）。
- `weights` 按层顺序展平：首层为 `[hidden_size, patch_size]` 的转置，其余层为 `[hidden_size, hidden_size]` 的转置。
- `biases` 长度为 `num_layers * hidden_size`，`hidden_size = biases_length / num_layers`。

## 调用说明

| 调用方式   | 调用样例                                                                | 说明                                                                                  |
| ---------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| aclnn调用  | [test_aclnn_fused_patch_mlp](./examples/test_aclnn_fused_patch_mlp.cpp) | 通过[aclnnFusedPatchMlp](./docs/aclnnFusedPatchMlp.md)接口方式调用FusedPatchMlp算子。 |
| 图模式调用 | -                                                                       | 通过[算子IR](./op_graph/fused_patch_mlp_proto.h)构图方式调用FusedPatchMlp算子。       |
