# SwigluGroupQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/activation/swiglu_group_quant)

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

### 接口功能

SwigluGroupQuant算子实现SwiGLU激活函数与分组量化融合计算。支持四种量化模式：
- **quant_mode=0**: Block Quant（FP8块量化，固定128元素分组）
- **quant_mode=1**: MX Quant（FP8 MX量化，固定32元素分组）
- **quant_mode=2**: HiFp8 Static Quant（HiFp8静态量化）
- **quant_mode=3**: HiFp8 Dynamic Quant（HiFp8动态量化）


### 计算公式

#### 基础计算流程

```
步骤〇：GroupIndex处理（可选）→ 计算real_bs
步骤一：输入切分（仅处理前real_bs行）
步骤二：Clamp处理（可选，仅处理前real_bs行）
步骤三：SwiGLU激活（仅处理前real_bs行）
步骤四：Weight加权（可选，仅处理前real_bs行）
步骤五：量化计算（仅处理前real_bs行）
```

#### 步骤〇：GroupIndex处理（可选）

当提供 `group_index` 时，用于动态计算实际处理的token数量：

$$
\text{group\_sum} = \sum_{g=0}^{G-1} \text{group\_index}[g]
$$

$$
\text{real\_bs} = \min(\text{group\_sum}, N)
$$

其中：
- $G$ 为MoE专家分组数
- $N$ 为输入张量的第一维（预设batch size）
- 后续所有步骤仅处理前 $\text{real\_bs}$ 行数据

**MoE场景说明**：在MoE推理中，不同专家可能处理不同数量的token，group_index允许动态调整处理范围，避免处理空数据。

#### 步骤一：输入切分

输入张量 $\mathbf{x} \in \mathbb{R}^{N \times D}$ 沿最后一维切分为两部分：

$$
\mathbf{x}_0[n, d] = \mathbf{x}[n, d], \quad d \in [0, D/2)
$$

$$
\mathbf{x}_1[n, d] = \mathbf{x}[n, d + D/2], \quad d \in [0, D/2)
$$

#### 步骤二：Clamp处理（可选）

当 `clamp_limit > 0` 时，对输入进行限制：

$$
\mathbf{x}_0'[n, d] = \min(\mathbf{x}_0[n, d], c)
$$

$$
\mathbf{x}_1'[n, d] = \min(\max(\mathbf{x}_1[n, d], -c), c)
$$

其中 $c$ 为 `clamp_limit`。

**Clamp的作用**：
- $\mathbf{x}_0$（门控分支）限制为正值范围 $[0, c]$，防止sigmoid梯度消失
- $\mathbf{x}_1$（线性分支）限制为对称范围 $[-c, c]$，防止数值溢出

#### 步骤三：SwiGLU激活

SwiGLU激活函数定义（逐元素计算）：

$$
\mathbf{y}_{\text{swiglu}}[n, d] = \text{Swish}(\mathbf{x}_0'[n, d]) \cdot \mathbf{x}_1'[n, d]
$$

其中Swish函数：

$$
\text{Swish}(z) = z \cdot \sigma(z) = z \cdot \frac{1}{1 + e^{-z}}
$$

**完整计算步骤分解**：

$$
\begin{aligned}
t_1[n, d] &= -\mathbf{x}_0'[n, d] \quad \text{(neg)} \\
t_2[n, d] &= e^{t_1[n, d]} = e^{-\mathbf{x}_0'[n, d]} \quad \text{(exp)} \\
t_3[n, d] &= t_2[n, d] + 1 = 1 + e^{-\mathbf{x}_0'[n, d]} \quad \text{(add)} \\
t_4[n, d] &= \frac{\mathbf{x}_0'[n, d]}{t_3[n, d]} = \text{Swish}(\mathbf{x}_0'[n, d]) \quad \text{(div)} \\
\mathbf{y}_{\text{swiglu}}[n, d] &= t_4[n, d] \cdot \mathbf{x}_1'[n, d] \quad \text{(mul)}
\end{aligned}
$$

#### 步骤四：Weight加权（可选）

当提供 `weight` 时，对SwiGLU输出进行加权：

$$
\mathbf{y}_{\text{weighted}}[n, d] = \mathbf{y}_{\text{swiglu}}[n, d] \cdot w[n]
$$

其中 $w[n]$ 为第 $n$ 个token的weight值。

**MoE场景**：weight来自专家路由器的softmax输出，表示该token对当前专家的权重。

#### 步骤五：量化计算

---

#### quant_mode=0 (Block Quant)

**分组划分**：将输出沿最后一维按128元素为一组划分：

$$
\mathbf{y} = [\mathbf{g}_0, \mathbf{g}_1, \ldots, \mathbf{g}_K], \quad K = \lceil D/2 / 128 \rceil
$$

每个组 $\mathbf{g}_i \in \mathbb{R}^{N \times 128}$。

**非有限值屏蔽与绝对值计算**：

$$
\begin{aligned}
\mathbf{z}[n, j] &= \mathbf{y}_{\text{weighted}}[n, j] \cdot 0 \quad \text{(生成零张量)} \\
\mathbf{m}_{\text{finite}}[n, j] &= (\mathbf{z}[n, j] = \mathbf{z}[n, j]) \\
\mathbf{y}_{\text{abs}}[n, j] &=
\begin{cases}
|\mathbf{y}_{\text{weighted}}[n, j]|, & \mathbf{m}_{\text{finite}}[n, j] \\
0, & \text{otherwise}
\end{cases}
\end{aligned}
$$

**屏蔽原理**：NaN的特性是 `NaN != NaN`；同时 `Inf * 0` 也会得到NaN，因此该步骤在计算amax时屏蔽NaN和Inf。

**Scale计算**：

对于第 $i$ 个组（包含128个连续元素）：

$$
a_i = \max_{j=0}^{127} \mathbf{y}_{\text{abs}}[j]
$$

$$
\hat{a}_i = \max(a_i, 10^{-4})
$$

$$
s_i^{\text{raw}} = \frac{\hat{a}_i}{M_{\text{fp8}}}
$$

其中 $M_{\text{fp8}}$ 取值：
- FP8 E4M3FN：$M_{\text{fp8}} = 448.0$
- FP8 E5M2：$M_{\text{fp8}} = 57344.0$

**Scale输出与InvScale计算**：

当 `round_scale=false` 时：

$$
s_i = s_i^{\text{raw}}, \quad \text{InvScale}_i = \frac{M_{\text{fp8}}}{\hat{a}_i} = \frac{1}{s_i}
$$

当 `round_scale=true` 时，将scale向上取整到2的幂：

$$
e_i = \lceil \log_2(s_i^{\text{raw}}) \rceil
$$

$$
s_i = 2^{e_i}, \quad \text{InvScale}_i = 2^{-e_i}
$$

其中 $s_i$ 写入FLOAT32类型的scale输出。

**量化计算**：

$$
\mathbf{y}_{\text{scaled}}[n, j] = \mathbf{y}_{\text{weighted}}[n, j] \cdot \text{InvScale}_i, \quad j \in \text{group } i
$$

若 $\mathbf{y}_{\text{scaled}}[n,j]$ 为NaN或Inf，实现会使用原始 $\mathbf{y}_{\text{weighted}}[n,j]$ 作为FP8 cast输入：

$$
\mathbf{y}_{\text{cast\_in}}[n, j] =
\begin{cases}
\mathbf{y}_{\text{scaled}}[n, j], & \mathbf{y}_{\text{scaled}}[n, j] \text{ is finite} \\
\mathbf{y}_{\text{weighted}}[n, j], & \text{otherwise}
\end{cases}
$$

$$
\mathbf{y}_{\text{quant}}[n, j] = \text{cast\_fp8\_rint}(\mathbf{y}_{\text{cast\_in}}[n, j])
$$

其中 `cast_fp8_rint` 为FP32到FP8的类型转换，采用**RINT（就近舍入）**模式。

---

#### quant_mode=1 (MX Quant)

**MX量化原理**：采用**E8M0 Scale** + **FP8 Data**的组合。

**分组方式**：每**32元素**为一组：

$$
\mathbf{y} = [\mathbf{g}_0, \mathbf{g}_1, \ldots, \mathbf{g}_K], \quad \mathbf{g}_i \in \mathbb{R}^{32}
$$

**Amax计算**：

$$
a_i = \max_{j=0}^{31} |\mathbf{g}_i[j]|
$$

$$
\hat{a}_i = \max(a_i, 10^{-4})
$$

**原始Scale计算**：

$$
s_i^{\text{raw}} = \frac{\hat{a}_i}{M_{\text{fp8}}}
$$

其中 $M_{\text{fp8}}$ 取值：
- FP8 E4M3FN：$M_{\text{fp8}} = 448.0$
- FP8 E5M2：$M_{\text{fp8}} = 57344.0$

quant_mode=1仅支持 `round_scale=true`，将原始scale向上取整到2的幂：

$$
e_i = \lceil \log_2(s_i^{\text{raw}}) \rceil
$$

等价于基于FP32位模式计算：

$$
e_i = E(s_i^{\text{raw}}) - 127 + \mathbf{1}_{\text{mantissa}(s_i^{\text{raw}}) \ne 0}
$$

**E8M0 Scale编码**：

$$
s_i^{\text{e8m0}} = e_i + 127
$$

其中 $s_i^{\text{e8m0}}$ 写入FLOAT8_E8M0类型的scale输出，表示的实际scale值为 $2^{e_i}$。

**InvScale计算**：

$$
\text{InvScale}_i = 2^{-e_i}
$$

**量化计算**：

$$
\mathbf{y}_{\text{quant}}[j] = \text{cast\_fp8\_rint}\left(\mathbf{y}_{\text{weighted}}[j] \cdot \text{InvScale}_i\right), \quad j \in \text{group } i
$$

---

#### quant_mode=2 (HiFp8 Static Quant)

**静态量化说明**：使用预先提供的 `invScale` 对加权后的SwiGLU输出进行缩放量化。

**情况1：无GroupIndex**（groupIndex为空）：

$$
\mathbf{y}_{\text{quant}}[n, d] = \text{hif8\_cast}\left(\mathbf{y}_{\text{weighted}}[n, d] \cdot \text{invScale}[0]\right), \quad n \in [0, N), \quad d \in [0, D/2)
$$

其中 `hif8_cast` 为HiFloat8类型转换函数。

**情况2：有GroupIndex**（groupIndex非空）：

设 $G$ 为MoE专家分组数，$\text{groupIndex}[g]$ 表示第 $g$ 个专家处理的token数量。

计算每个group的起止索引：

$$
\text{start}^{(0)} = 0, \quad \text{end}^{(g)} = \sum_{k=0}^{g} \text{groupIndex}[k], \quad \text{start}^{(g)} = \text{end}^{(g-1)}
$$

对于第 $g$ 个group，使用对应的缩放因子 $\text{invScale}[g]$ 进行量化：

$$
\mathbf{y}_{\text{quant}}[n, d] = \text{hif8\_cast}\left(\mathbf{y}_{\text{weighted}}[n, d] \cdot \text{invScale}[g]\right), \quad n \in [\text{start}^{(g)}, \text{end}^{(g)}), \quad d \in [0, D/2)
$$

**MoE场景说明**：在MoE推理中，不同专家处理不同数量的token，groupIndex用于标识每个专家处理的token范围，invScale为每个专家预先计算的静态缩放因子。

---

#### quant_mode=3 (HiFp8 Dynamic Quant)

**动态量化说明**：根据加权后的SwiGLU输出动态计算缩放因子进行量化。

**情况1：无GroupIndex**（groupIndex为空）：

计算全局绝对值最大值：

$$
a_{\max} = \max\left(\max_{n \in [0, N), d \in [0, D/2)} |\mathbf{y}_{\text{weighted}}[n, d]|, \epsilon\right)
$$

其中 $\epsilon$ 为数值稳定性常数。

计算缩放因子：

$$
s = \frac{a_{\max}}{M_{\text{hif8}}}
$$

其中 $M_{\text{hif8}}$ 为 `dstTypeMax`，表示HiFloat8类型的最大有限值。

量化计算：

$$
\mathbf{y}_{\text{quant}}[n, d] = \text{hif8\_cast}\left(\frac{\mathbf{y}_{\text{weighted}}[n, d]}{s}\right), \quad n \in [0, N), \quad d \in [0, D/2)
$$

其中 `hif8_cast` 为HiFloat8类型转换函数。

**情况2：有GroupIndex**（groupIndex非空）：

设 $G$ 为MoE专家分组数，$\text{groupIndex}[g]$ 表示第 $g$ 个专家处理的token数量。

计算每个group的起止索引：

$$
\text{start}^{(0)} = 0, \quad \text{end}^{(g)} = \sum_{k=0}^{g} \text{groupIndex}[k], \quad \text{start}^{(g)} = \text{end}^{(g-1)}
$$

对于第 $g$ 个group，提取对应的数据：

$$
\mathbf{y}^{(g)} = \mathbf{y}_{\text{weighted}}[\text{start}^{(g)}:\text{end}^{(g)}, :]
$$

计算该group的绝对值最大值：

$$
a_{\max}^{(g)} = \max\left(\max_{n \in [\text{start}^{(g)}, \text{end}^{(g)}), d \in [0, D/2)} |\mathbf{y}^{(g)}[n, d]|, \epsilon\right)
$$

计算该group的缩放因子：

$$
s^{(g)} = \frac{a_{\max}^{(g)}}{M_{\text{hif8}}}
$$

对该group进行量化：

$$
\mathbf{y}_{\text{quant}}[n, d] = \text{hif8\_cast}\left(\frac{\mathbf{y}_{\text{weighted}}[n, d]}{s^{(g)}}\right), \quad n \in [\text{start}^{(g)}, \text{end}^{(g)}), \quad d \in [0, D/2)
$$

**MoE场景说明**：在MoE推理中，不同专家处理不同数量的token，groupIndex用于标识每个专家处理的token范围，每个group独立计算缩放因子以适应不同数据分布。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 100px">
  <col style="width: 420px">
  <col style="width: 240px">
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
      <td>SwiGLU输入。shape为[...,D]，D必须大于等于256，且能被256整除。不支持空Tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入（可选）</td>
      <td>MOE权重张量，用于SwiGLU输出的加权计算。不支持空Tensor。不为空时，数据类型为FLOAT32，元素个数需等于x除最后一维外的元素个数之积。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>输入（可选）</td>
      <td>count模式的group token数。不支持空Tensor。不为空时，数据类型为INT64，shape为[G]。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入（可选）</td>
      <td>quantMode=2时静态量化输入的invScale张量。仅quant_mode为2时使用。groupIndex存在的话，shape=[G]，不存在的话shape=[1]。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>属性</td>
      <td>目标量化类型。仅quantMode为0或1时，该参数生效。支持取值35、36、40、41，分别表示FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2。dstType为40或41时，quantMode必须为1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>属性</td>
      <td>量化模式。支持取值0、1、2、3。0表示Block FP8模式。1表示MX模式。2表示HIFP8静态量化模式。3表示HIFP8动态量化模式。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>属性</td>
      <td>量化块大小。0表示使用当前量化模式的默认block大小。quantMode为0时，支持0或128。quantMode为1时，支持0或32。quantMode为2或3时，该参数不生效，默认0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_scale</td>
      <td>属性</td>
      <td>是否将scale取整为2的幂。quantMode为1时，roundScale必须为true。quantMode为2或3时，该参数不生效。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clamp_limit</td>
      <td>属性</td>
      <td>SwiGLU计算前的clamp阈值。-1.0表示不启用clamp。启用clamp时，clampLimit必须大于0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type_max</td>
      <td>属性</td>
      <td>目标量化类型的最大有限值。仅quantMode为3时，该参数生效。默认值为15.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_origin</td>
      <td>属性</td>
      <td>是否输出量化前的SwiGLU结果。true表示支持输出原始激活值yOrigin，false表示不支持输出原始激活值yOrigin。quantMode为0或1时支持false。quantMode为2或3时true/false都支持。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>量化输出。quantMode为0或1时，数据类型需与dstType一致，dstType为35或36时，shape为[...,D/2]；dstType为40或41时，shape为[...,D/4]；dstType为27时，shape为[...,D/2]。quantMode为2或3时，数据类型默认为HIFLOAT8，shape为[...,D/2]。不支持空Tensor。</td>
      <td>HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_scale</td>
      <td>输出</td>
      <td>量化scale输出。quantMode为0时，shape为[...,ceil((D/2)/128)]，数据类型为FLOAT32。quantMode为1时，shape为[...,ceil(ceil((D/2)/32)/2),2]，数据类型为FLOAT8_E8M0。quantMode为2或3时，无groupIndex时shape为[1]，有groupIndex时shape为[G]，数据类型为FLOAT32。不支持空Tensor。</td>
      <td>FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_origin</td>
      <td>输出</td>
      <td>量化前的SwiGLU结果。shape为[...,D/2]。数据类型需与x一致。不支持空指针。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 确定性计算：aclnnSwigluGroupQuant默认确定性实现。
- quantMode为0时，仅支持FP8输出，blockSize支持0或128。
- quantMode为1时，支持FP8/FP4输出，blockSize支持0或32，roundScale必须为true。
- quantMode为2或3时，支持HIFP8量化输出，dstType, blockSize和roundScale不生效。输入x的维度为[T, D]或[B, S, D]，需满足以下规格约束：

  | 规格项 | 规格 | 规格说明 |
  | :--- | :--- | :--- |
  | B | 1~31 | - |
  | S | 0~128K | - |
  | D/2 | 512, 768, 1024, 1536, 1792, 2048, 2560, 4096 | - |
  | dstTypeMax | 15, 56, 224, 32768 | - |
- dstType为FLOAT4_E2M1或FLOAT4_E1M2时，必须使用quantMode=1。
- yScale的数据类型必须与quantMode匹配：Block FP8为FLOAT32，MX为FLOAT8_E8M0，HIFP8为FLOAT32。

## 调用说明

|调用方式|调用样例|说明|
|:-------|:-------|:---|
|aclnn调用|[test_aclnn_swiglu_group_quant](./examples/test_aclnn_swiglu_group_quant.cpp)|通过[aclnnSwigluGroupQuant](./docs/aclnnSwigluGroupQuant.md)接口调用SwigluGroupQuant算子。|
|图模式调用|-|通过[算子IR](./op_graph/swiglu_group_quant_proto.h)构图方式调用SwigluGroupQuant算子。|
