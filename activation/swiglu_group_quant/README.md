# SwigluGroupQuant

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

**支持说明**：本算子支持 quant_mode=0/1/2/3/4 五种量化模式，本文档聚焦 **quant_mode=3（HiFp8 Dynamic Quant，HiF8 动态量化）**，仅 Ascend 950PR/Ascend 950DT 支持。

## 功能说明

- 算子功能：SwigluGroupQuant 算子实现 SwiGLU 激活函数与分组动态量化（quant_mode=3，HiF8 Dynamic Quant）融合计算。将输入 x 沿最后一维切分为两半 x0、x1，对 x0 施加 SiLU 激活后与 x1 相乘，再根据可选的 weight 加权，最终对加权结果动态计算缩放因子并量化为 hifloat8，输出量化结果 y、缩放因子 y_scale 以及可选的原始激活值 y_origin。
- 算子支持范围：支持 MoE 场景（传入 group_index）和非 MoE 场景（group_index 传空），支持可选的 Clamp 处理、Weight 加权、output_origin 输出。
- 计算流程：
  - 步骤〇：GroupIndex 处理（可选）→ 计算 real_bs
  - 步骤一：输入切分（将 x 切分为 x0 和 x1）
  - 步骤二：Clamp 处理（可选，仅处理前 real_bs 行）
  - 步骤三：SwiGLU 激活（仅处理前 real_bs 行）
  - 步骤四：Weight 加权（可选，仅处理前 real_bs 行）
  - 步骤五：HiF8 动态量化（仅处理前 real_bs 行）

- 计算公式：
  - **GroupIndex 处理（可选）**：当提供 `group_index` 时，仅处理前 real_bs 行，$\text{real\_bs} = \min(\sum_{g=0}^{G-1} \text{group\_index}[g], N)$，其中 $G$ 为 MoE 专家分组数，$N$ 为输入第一维。
  - **输入切分**：$x_0 = x[:, :H], \quad x_1 = x[:, H:]$
  - **Clamp 处理（可选，clamp_limit > 0 时）**：$x_0 = \min(x_0, c), \quad x_1 = \text{clip}(x_1, -c, c)$，其中 $c$ 为 `clamp_limit`。
  - **SwiGLU 激活**：$y = \text{silu}(x_0) \times x_1 = \frac{x_0}{1 + e^{-x_0}} \times x_1$
  - **Weight 加权（可选）**：$y = y \times w$，其中 $w$ 为 weight；未提供时跳过。
  - **HiF8 动态量化（quant_mode=3）**：
    - 非分组（无 group_index）：$\text{amax} = \max(\text{absmax}(y), \epsilon), \quad s = \frac{\text{amax}}{M_{\text{hif8}}}, \quad y_{\text{quant}} = \text{hif8\_cast}(y / s)$，y_scale shape=[1]
    - 分组（有 group_index）：对每个 group $g$ 独立计算 $\text{amax}_g = \max(\text{absmax}(y_g), \epsilon), \quad s_g = \frac{\text{amax}_g}{M_{\text{hif8}}}, \quad y_{\text{quant},g} = \text{hif8\_cast}(y_g / s_g)$，y_scale shape=[G]

    其中 $M_{\text{hif8}}$ 为 `dst_type_max_finite`（默认 448.0），$\epsilon$ 为数值稳定性常数，`hif8_cast` 为 HiFloat8 类型转换函数。
  - **output_origin（可选）**：当 `output_origin=true` 时，额外输出 SwiGLU 加权后的原始激活值 y_origin（未量化），dtype 与 x 一致，shape 为 [N, H]。

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
      <td>SwiGLU 输入张量，最后一维必须为偶数（2H）。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>可选输入</td>
      <td>MoE 权重张量，shape 需与 x 第一维一致。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>可选输入</td>
      <td>GroupIndex 张量，动态核分配，shape=[G]。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>可选输入</td>
      <td>静态量化 scale 张量（仅 quant_mode=2 使用，quant_mode=3 传空）。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>属性</td>
      <td>输出数据类型，quant_mode=3 时固定 HiFloat8（27）。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>属性</td>
      <td>量化模式，本文档场景取值 3（HiF8 Dynamic Quant）。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>属性</td>
      <td>分组大小，quant_mode=3 时不使用，默认 0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_scale</td>
      <td>属性</td>
      <td>Scale 取整优化（仅 quant_mode=0 有效），quant_mode=3 时忽略。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clamp_limit</td>
      <td>属性</td>
      <td><ul><li>Clamp 阈值。</li><li>取值范围 ≥ 0.0。</li><li>clamp_limit=0 表示不启用 Clamp。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type_max_finite</td>
      <td>属性</td>
      <td>HiFloat8 最大有限值，用于计算 scale = amax / dst_type_max_finite，默认 448.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_origin</td>
      <td>属性</td>
      <td>是否输出 y_origin，默认 false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>量化输出张量，shape=[N, H]。</td>
      <td>HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_scale</td>
      <td>输出</td>
      <td>缩放因子张量，非分组 shape=[1]，分组 shape=[G]。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_origin</td>
      <td>可选输出</td>
      <td>SwiGLU 原始激活值，output_origin=true 时输出，dtype 与 x 一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 确定性计算：
  - 当提供 `group_index` 参数时：前 real_bs 行保证计算结果确定性，后 N-real_bs 行不保证确定性
  - 当未提供 `group_index` 参数时：所有行数据保证计算结果确定性

- 输入 shape 约束：
  - x 最后一维必须为偶数（$2H$）
  - y 最后一维为 $H$，与 x 最后一维的一半对应
  - x 最后一维上限 ≤ 8192

- 量化轴约束：
  - quant_mode=3 只支持 -1 轴量化（最后一维）

- 可选参数约束：
  - weight 的 shape 需与 x 的第一维一致
  - group_index 为 1 维 INT64 张量，shape=[G]
  - scale 仅 quant_mode=2 使用，quant_mode=3 时传空

- 数据类型约束：
  - x 支持 FLOAT、FLOAT16、BFLOAT16
  - weight 必须为 FLOAT 类型
  - group_index 必须为 INT64 类型
  - y 为 HIFLOAT8 类型
  - y_scale 为 FLOAT 类型
  - y_origin 与 x 数据类型一致

- Clamp 约束：
  - clamp_limit 必须 ≥ 0.0
  - clamp_limit=0 表示不启用 Clamp

- 属性有效性约束：
  - quant_mode=3 时，dst_type 固定 HiFloat8，block_size、round_scale 不生效
  - output_origin 所有模式均支持

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|--------|------|
| aclnn 调用 | [test_aclnn_swiglu_group_quant](./examples/arch35/test_aclnn_swiglu_group_quant.cpp) | 通过 [aclnnSwigluGroupQuant](./docs/aclnnSwigluGroupQuant.md) 接口方式调用 SwigluGroupQuant 算子。 |
| 图模式调用 | - | 通过 [算子 IR](./op_graph/swiglu_group_quant_proto.h) 构图方式调用 SwigluGroupQuant 算子。 |
