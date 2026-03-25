# SwigluMxQuant

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

- 算子功能：融合算子，实现 SwiGLU 激活函数与动态块量化的组合计算。先对输入计算 SwiGLU 激活函数，然后对结果进行基于块的动态量化，输出低精度的 FP4/FP8 张量和对应的缩放因子。

- 计算公式：

  **阶段 1：SwiGLU 激活函数**

  <p style="text-align: center">
  gate, hidden = split(x, dim=dim)
  </p>
  <p style="text-align: center">
  swish = sigmoid(gate) * gate
  </p>
  <p style="text-align: center">
  act = swish * hidden
  </p>

  **阶段 2：动态块量化**

  <p style="text-align: center">
  scale[block_idx] = max(abs(block))
  </p>
  <p style="text-align: center">
  y[block_idx] = Cast(block / scale[block_idx])
  </p>

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
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
      <td>输入张量，在 dim 维度上的尺寸必须是 2 的倍数</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>可选输入</td>
      <td>预留参数</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activate_dim</td>
      <td>属性</td>
      <td>SwiGLU 的分割维度，取值范围为[-1, -2]</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>属性</td>
      <td>预留参数，表示 SwiGLU 时激活左半部或右半部</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>swiglu_mode</td>
      <td>属性</td>
      <td>预留参数，表示 SwiGLU 时是否采用变体计算方式</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clamp_limit</td>
      <td>属性</td>
      <td>预留参数，变体 SwiGLU 时生效</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>glu_alpha</td>
      <td>属性</td>
      <td>预留参数，变体 SwiGLU 时生效</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>glu_bias</td>
      <td>属性</td>
      <td>预留参数，变体 SwiGLU 时生效</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group_mode</td>
      <td>属性</td>
      <td>group_index 存在时生效，表示 group_index 为 count 模式或 cumsum 模式</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>属性</td>
      <td>量化轴，沿此维度进行分块量化，取值范围为[-1, -2]</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>属性</td>
      <td>目标量化类型：40=FP4_E2M1, 41=FP4_E1M2, 36=FP8_E4M3FN, 35=FP8_E5M2</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>属性</td>
      <td>舍入模式，用于量化时的类型转换，取值为"rint"、"floor"、"round"</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>属性</td>
      <td>缩放算法：0=OCP，1=cuBLAS, 2=RNE</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_dtype_value</td>
      <td>属性</td>
      <td>预留参数，scale_alg=2 且 y 为 FP4_E1M2 时生效</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>量化后的输出张量，形状为 SwiGLU 输出的一半（沿 dim 维度）</td>
      <td>FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxscale</td>
      <td>输出</td>
      <td>每个量化块的缩放因子</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入 x 支持 2-7 维张量，在 activate_dim 指定维度上的尺寸必须能被 2 整除。
- activate_dim 和 axis 的值必须为 -1 或 -2。
- 当前实现限制：仅支持 `activate_dim = -1` 且 `axis = -1` 的场景。
- 当 dst_type 为 FP4 类型时，最后一维必须能被 4 整除。
- FP8 输出类型仅支持 "rint" 舍入模式。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :------- | :------- | :--- |
| aclnn调用 | [test_aclnn_swiglu_mx_quant](./examples/test_aclnn_swiglu_mx_quant.cpp) | 通过[aclnnSwigluMxQuant](./docs/aclnnSwigluMxQuant.md)接口方式调用SwigluMxQuant算子。 |

