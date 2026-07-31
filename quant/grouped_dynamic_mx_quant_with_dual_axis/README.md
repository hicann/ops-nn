# GroupedDynamicMxQuantWithDualAxis

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：根据 group_index 描述的分组，对二维输入 x 一次性完成最后一维和倒数第二维两个方向的动态 MX 量化，分别输出两个方向的 FP8 量化结果 y1、y2 及对应的 FLOAT8_E8M0 缩放因子 y1_scale、y2_scale。
- 计算说明：以 32 个元素为一个 MX block，每个 block 内共享一个 FLOAT8_E8M0 类型的 scale。对 -1 轴和 -2 轴分别独立计算 block 的 scale 和量化结果。

- 计算公式：

  对一个 MX block 内的输入数据：

  $$
  D^{b}=[d^{i}]_{i=1}^{k},\quad k=32
  $$

  计算 block 内元素的最大绝对值：

  $$
  Amax(D^{b})=\max_i(|d^{i}|)
  $$

  scale_alg=1 时，根据目标 FP8 类型最大可表示值计算 FP32 scale：

  $$
  S_{fp32}^{b}=\frac{Amax(D^{b})}{Amax(dst_dtype)}
  $$

  从块缩放因子$S_{fp32}^{b}$中提取无偏指数$E_{int}^{b}$和尾数$M_{fixp}^{b}$。

  为保证量化时不溢出，对指数进行向上取整，且在 FP8 可表示的范围内：

  $$
  E_{int}^{b} = \begin{cases}
    E_{int}^{b} + 1, & \text{如果} S_{fp32}^{b} \text{为正规数，且} E_{int}^{b} < 254 \text{且} M_{fixp}^{b} > 0 \\
    E_{int}^{b} + 1, & \text{如果} S_{fp32}^{b} \text{为非正规数，且} M_{fixp}^{b} > 0.5 \\
    E_{int}^{b}, & \text{否则}
  \end{cases}
  $$

  计算块缩放因子：

  $$
  S_{ue8m0}^{b}=2^{E_{int}^{b}}
  $$

  计算块转换因子：

  $$
  R_{fp32}^{b}=\frac{1}{fp32(S_{ue8m0}^{b})}
  $$

  对 block 内每个元素执行量化：

  $$
  p_i=cast\_to\_dst\_type(d^{i}\times R_{fp32}^{b},round_mode)
  $$

  输出该 block 对应的 scale 和量化结果：

  $$
  (S_{ue8m0}^{b},[p_i]_{i=1}^{k})
  $$

## 参数说明

<table><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待量化的输入 Tensor，对应公式中的输入数据 D。</td>
      <td><ul><li>shape 为 [M, N]。</li><li>N 需要大于 0 且能被 64 整除。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>输入</td>
      <td>量化分组索引，采用 cumsum 形式描述各 group 边界。</td>
      <td><ul><li>每个元素表示一个 group 的结束行索引。</li><li>索引值需要非负且非递减。</li><li>最后一个元素需要等于 M。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td>表示 x 量化 -1 轴后的对应结果。</td>
      <td><ul><li>shape 需要与 x 一致，为 [M, N]。</li><li>数据类型需要与 dst_dtype 一致。</li></ul></td>
      <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y1_scale</td>
      <td>输出</td>
      <td>表示 -1 轴每个分组对应的量化尺度。</td>
      <td><ul><li>shape 为 [M, ceil(N/64), 2]。</li><li>最后一维每 2 个 scale 成对存放；N 方向按 32 个元素生成一个 scale，并进行偶数 pad，pad 填充值为 0。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>表示 x 量化 -2 轴后的对应结果。</td>
      <td><ul><li>shape 需要与 x 一致，为 [M, N]。</li><li>数据类型需要与 dst_dtype 一致。</li></ul></td>
      <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2_scale</td>
      <td>输出</td>
      <td>表示 -2 轴每个分组对应的量化尺度。</td>
      <td><ul><li>shape 为 [floor(M/64)+G, N, 2]，其中 G 为 group_index 元素个数。</li><li>倒数第二维方向按 32 个元素生成一个 scale，并进行偶数 pad。</li><li>y2_scale 输出需要对每两行数据进行交织处理。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>可选属性</td>
      <td>表示量化转换时的舍入模式，对应公式中的 round_mode。</td>
      <td><ul><li>仅支持取值 "rint"，默认值为 "rint"。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>可选属性</td>
      <td>表示 y1_scale 和 y2_scale 的计算方法。</td>
      <td><ul><li>仅支持取值为 1，默认值为 1。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_dtype</td>
      <td>可选属性</td>
      <td>表示 y1 和 y2 的目标输出数据类型。</td>
      <td><ul><li>取值 35 表示 FLOAT8_E5M2，取值 36 表示 FLOAT8_E4M3FN，默认值为 36。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_dtype_value</td>
      <td>可选属性</td>
      <td>预留参数。</td>
      <td><ul><li>仅支持 0.0，默认值为 0.0。</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入 x 的最后一维 N 必须能被 64 整除。
- y1 和 y2 的 shape 需要与 x 一致，均为 [M, N]；y1 和 y2 的数据类型需要与 dst_dtype 指定的目标类型一致。
- y1_scale 的 shape 为 [M, ceil(N/64), 2]；y2_scale 的 shape 为 [floor(M/64)+G, N, 2]，其中 G 为 group_index 元素个数。
- group_index 的最后一个元素必须等于 x.shape[0]，采用 cumsum 模式，每个值表示对应 group 的行数累积值。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :------- | :------- | :--- |
| aclnn API | [test_aclnn_grouped_dynamic_mx_quant_with_dual_axis](./examples/arch35/test_aclnn_grouped_dynamic_mx_quant_with_dual_axis.cpp) | 通过 [aclnnGroupedDynamicMxQuantWithDualAxis](./docs/aclnnGroupedDynamicMxQuantWithDualAxis.md) 接口调用 GroupedDynamicMxQuantWithDualAxis 算子。 |
| PyTorch API | [torchapi_grouped_dynamic_mx_quant_with_dual_axis](./docs/torchapi_grouped_dynamic_mx_quant_with_dual_axis.md) | 通过 [grouped_dynamic_mx_quant_with_dual_axis](./docs/torchapi_grouped_dynamic_mx_quant_with_dual_axis.md) 接口调用 GroupedDynamicMxQuantWithDualAxis 算子。 |
