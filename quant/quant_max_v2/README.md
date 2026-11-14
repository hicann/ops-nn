[English](README_en.md)

# QuantMaxV2

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：输入 `x` 支持按 `group_list` 分组（不传 `group_list` 时整个 tensor 视为单组，即 Per-Tensor），对每组求 `|x|` 的最大值 `amax`，并输出量化结果 `y`。传入 `scale` 时为静态量化；不传时为动态量化，此时额外输出量化系数 `y_scale`。`amax` 为必选输出。

  相较 [GroupedQuantMax](../grouped_quant_max/README.md)，V2 支持"scale 系数可选、group_list 分组可选、动态量化输出 `y_scale`"三项能力，`amax` 升级为必选输出，并新增属性 `dst_type_max`。

  | `scale` | `group_list` | 模式 | 输出 |
| :----------------: | :----------------: |  ---------------- | ------------------------------------------------- |
| 不传（`nullptr`）  | 不传（`nullptr`）   | Dynamic Per-Tensor | `y` + `amax[1]`（必选） + `y_scale[1]`（必选） |
| 传               | 不传（`nullptr`）   | Static Per-Tensor | `y` + `amax[1]`（必选） + `y_scale`（不输出，传 `nullptr`） |
| 不传（`nullptr`）  | 传                  | Dynamic Per-Group | `y` + `amax[G]`（必选） + `y_scale[G]`（必选） |
| 传               | 传                  | Static Per-Group | `y` + `amax[G]`（必选） + `y_scale`（不输出，传 `nullptr`） |

- 计算公式：

  `group_list` 沿 dim-0 将 `x` 切分为 `numGroups`（记 `G`）组，`group_list[g]` 为前 `g+1` 组在 dim-0 上的累计结束位置（cumsum 语义）。对每个 group（`g = 0, 1, …, G-1`）：

  $$
  x_g = x\left[\,groupList[g-1] : groupList[g],\ \ldots\,\right] \qquad(\ g = 0\ \text{时下界为}\ 0\ )
  $$

  $$
  a_g = \max\left(|x_g|\right)
  $$

  **静态量化**（传入 `scale`，`scale[g]` 为常规系数 `s` 的倒数 `1/s`）：

  $$
  q_g = scale[g], \qquad y_g = \mathrm{Cast}\left(x_g \times q_g,\ dstType,\ roundMode\right)
  $$

  **动态量化**（不传 `scale`，`dstTypeMax` 为目标类型最大有限值）：

  $$
  q_g = \begin{cases}
  \dfrac{dstTypeMax}{a_g}, & a_g > 0 \\[4pt]
  0, & a_g = 0
  \end{cases}, \qquad
  y_g = \mathrm{Cast}\left(x_g \times q_g,\ dstType,\ roundMode\right)
  $$

  **输出**：

  $$
  amax[g] = a_g, \qquad
  y = \mathrm{Concat}\left(y_0,\ y_1,\ \ldots,\ y_{G-1}\right), \qquad
  yScale[g] = \dfrac{1}{q_g} \quad(\text{仅动态量化输出})
  $$

  其中，`y_scale` 满足 `x ≈ y × y_scale`（`y_scale[g] = amax[g] / dstTypeMax`；`a_g = 0` 时 `q_g = 0`，对应 `y_scale[g] = 0`）。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
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
      <td>x</td>
      <td>输入</td>
      <td>待量化的输入张量，对应公式中的 x。支持 2-8 维，支持空 Tensor。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>可选输入</td>
      <td>量化缩放系数 s 的倒数 <code>1/s</code>。<code>nullptr</code> 表示动态量化（系数按本组 amax 现算）；非空表示静态量化。shape 为 [numGroups]（Per-Tensor 为 [1]）。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_list</td>
      <td>可选输入</td>
      <td>x 在 0 轴上每个 group 的累计结束位置（cumsum 语义）。<code>nullptr</code> 表示 Per-Tensor；非空表示 Per-Group，shape 为 [numGroups]，num_groups 取值范围 1~384。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>量化结果，对应公式中的 y。shape 与 x 完全一致，dtype 由 dst_type 决定。</td>
      <td>HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>amax</td>
      <td>输出</td>
      <td>量化前分组的 x 绝对值最大值（必选输出）。shape 为 [numGroups]（Per-Tensor 为 [1]），dtype 与 x 一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_scale</td>
      <td>可选输出</td>
      <td>y 的量化系数（<code>y_scale = 1/q</code>，动态量化下为 <code>amax/dstTypeMax</code>）。仅动态量化输出，shape 为 [numGroups]（Per-Tensor 为 [1]）；静态量化传 <code>nullptr</code>。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>属性</td>
      <td>量化结果转换到目标类型时的舍入模式（必选）。dst_type=34 支持 "round"、"hybrid"；dst_type=35/36 仅支持 "rint"。</td>
      <td>string</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>属性</td>
      <td>y 的目标数据类型（必选）。34=HIFLOAT8、35=FLOAT8_E5M2、36=FLOAT8_E4M3FN。</td>
      <td>int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type_max</td>
      <td>属性</td>
      <td>目标类型的最大有限值，默认 15.0，取值范围 0~32768。仅动态量化参与系数计算。</td>
      <td>float</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- `round_mode` 与 `dst_type` 的匹配关系：`dst_type` 为 `35` 或 `36` 时必须为 `"rint"`；`dst_type` 为 `34` 时必须为 `"round"` 或 `"hybrid"`。
- `group_list` 的值要求：每个值在 `[0, x.shape[0]]` 范围内、非递减（允许相等，即空 group），且最后一个数必须等于 x 的 0 轴大小。PyTorch 接口（`cann_ops_nn.quant_max`）对该约束做校验并抛出异常，aclnn C 接口要求调用方自行保证。
- `dst_type_max` 仅对 `dst_type=34`（HIFLOAT8）生效；`dst_type` 为 `35`/`36` 时按类型最大值（57344/448）推导动态系数，传入值不参与计算（范围校验恒生效）。
- `dst_type_max` 取值需为 0~32768 内的有限值，否则第一段接口报错。
- 确定性计算：<term>Ascend 950PR&950DT系列产品</term> 上 aclnnQuantMaxV2 默认为确定性实现。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_quant_max_v2](./examples/arch35/test_aclnn_quant_max_v2.cpp) | 通过[aclnnQuantMaxV2](./docs/aclnnQuantMaxV2.md)接口方式调用QuantMaxV2算子。 |
| 图模式调用 | - | 通过[算子IR](./op_graph/quant_max_v2_proto.h)构图方式调用QuantMaxV2算子。 |
