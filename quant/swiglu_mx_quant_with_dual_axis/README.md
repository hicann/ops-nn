# SwigluMxQuantWithDualAxis

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |
| <term>Kirin X90 处理器系列产品</term> | × |
| <term>Kirin 9030 处理器系列产品</term> | × |

## 功能说明

- 算子功能：融合算子，实现 SwiGLU 激活函数与双轴动态块量化的组合计算。先对输入计算 SwiGLU 激活函数，然后对结果同时在 -1 轴和 -2 轴进行基于块的动态量化，输出低精度的 FP4/FP8 张量和对应的缩放因子。

- 计算公式：

  **阶段 1：SwiGLU 激活函数**

  <p style="text-align: center">
  gate, hidden = split(x, axis=-1)
  </p>
  <p style="text-align: center">
  swish = sigmoid(gate) * gate
  </p>
  <p style="text-align: center">
  act = swish * hidden
  </p>

  其中，当 `activate_left=True` 时，左半部分为 hidden，右半部分为 gate；当 `activate_left=False` 时，右半部分为 hidden，左半部分为 gate。

  **阶段 2：双轴动态块量化**

  - **-1 轴量化（列方向）**：将 SwiGLU 结果在 -1 轴上按照 32 个数进行分组，一组 32 个数 $\{\{V_i\}_{i=1}^{32}\}$ 量化为 $\{mxscale1, \{P_i\}_{i=1}^{32}\}$

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale1 = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale1, round\_mode), \space i\space from\space 1\space to\space 32
    $$

  - **-2 轴量化（行方向）**：将 SwiGLU 结果在 -2 轴上按照 32 个数进行分组，一组 32 个数 $\{\{V_j\}_{j=1}^{32}\}$ 量化为 $\{mxscale2, \{P_j\}_{j=1}^{32}\}$

    $$
    shared\_exp = floor(log_2(max_j(|V_j|))) - emax
    $$

    $$
    mxscale2 = 2^{shared\_exp}
    $$

    $$
    P_j = cast\_to\_dst\_type(V_j/mxscale2, round\_mode), \space j\space from\space 1\space to\space 32
    $$

  - emax: 对应数据类型的最大正则数的指数位。

    |   DataType    | emax |
    | :-----------: | :--: |
    |  FLOAT4_E2M1  |  2   |
    |  FLOAT4_E1M2  |  0   |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |

## 参数说明

<table><colgroup>
  <col style="width: 120px">
  <col style="width: 100px">
  <col style="width: 380px">
  <col style="width: 280px">
  <col style="width: 100px">
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
      <td>输入张量，必须为 2 维，且最后一维必须为偶数（shape 为 [M, 2N]）</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>可选输入</td>
      <td>分组索引，用于控制分组量化边界。shape 为 [G]，采用 cumsum 模式，表示每个 group 的行数累积值。传入空指针时表示不分组。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>属性</td>
      <td>SwiGLU 激活侧选择。True 表示左半部分为 hidden，右半部分为 gate；False 表示右半部分为 hidden，左半部分为 gate。默认值为 True。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>属性</td>
      <td>舍入模式，用于量化时的类型转换。<br>当 dst_type 为 FLOAT4_E2M1/FLOAT4_E1M2 时，支持 "rint"、"floor"、"round"；<br>当 dst_type 为 FLOAT8_E4M3FN/FLOAT8_E5M2 时，仅支持 "rint"。默认值为 "rint"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>属性</td>
      <td>缩放算法：0=OCP（Microscaling Formats Specification），1=cuBLAS。默认值为 0。<br>FP4 输出类型仅支持 scale_alg=0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>属性</td>
      <td>目标量化类型：40=FLOAT4_E2M1，41=FLOAT4_E1M2，36=FLOAT8_E4M3FN，35=FLOAT8_E5M2。默认值为 36。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_dtype_value</td>
      <td>属性</td>
      <td>预留参数，scale_alg=2 且 dst_type=FLOAT4_E1M2 时生效。默认值为 0.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td>-1 轴量化后的输出张量，形状为 [M, N]（SwiGLU 输出的一半）</td>
      <td>FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mx_scale1</td>
      <td>输出</td>
      <td>-1 轴每个量化块对应的缩放因子。shape 为 [M, (ceil(N/32)+2-1)/2, 2]，需进行偶数 pad，pad 填充值为 0。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>-2 轴量化后的输出张量，形状为 [M, N]</td>
      <td>FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mx_scale2</td>
      <td>输出</td>
      <td>-2 轴每个量化块对应的缩放因子。当 groupIndex 存在时 shape 为 [floor(M/64) + G, N, 2]，当 groupIndex 不存在时 shape 为 [ceil(M/64), N, 2]，需进行偶数 pad，pad 填充值为 0。输出需要对每两行数据进行交织处理。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入 x 必须为 2 维张量，最后一维必须能被 2 整除（shape 为 [M, 2N]）。
- 当 dst_type 为 FP4 类型（FLOAT4_E2M1/FLOAT4_E1M2）时，最后一维 N 必须能被 4 整除。
- FP8 输出类型（FLOAT8_E4M3FN/FLOAT8_E5M2）仅支持 "rint" 舍入模式。
- FP4 输出类型仅支持 scale_alg=0（OCP）。
- 当 group_index 存在时，采用 cumsum 模式，每个值表示对应 group 的行数累积值。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :------- | :------- | :--- |
| aclnn调用 | [test_aclnn_swiglu_mx_quant_with_dual_axis](./examples/test_aclnn_swiglu_mx_quant_with_dual_axis.cpp) | 通过[aclnnSwigluMxQuantWithDualAxis](./docs/aclnnSwigluMxQuantWithDualAxis.md)接口方式调用SwigluMxQuantWithDualAxis算子。 |
| 图模式 | - | 通过算子IR构图方式调用SwigluMxQuantWithDualAxis算子。 |
