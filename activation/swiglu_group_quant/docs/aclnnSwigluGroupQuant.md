# aclnnSwigluGroupQuant

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

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnSwigluGroupQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSwigluGroupQuant”接口执行计算。

```Cpp
aclnnStatus aclnnSwigluGroupQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weightOptional,
  const aclTensor *groupIndexOptional,
  const aclTensor *scaleOptional,
  int64_t          dstType,
  int64_t          quantMode,
  int64_t          blockSize,
  bool             roundScale,
  double           clampLimit,
  double           dstTypeMax,
  bool             outputOrigin,
  const aclTensor *yOut,
  const aclTensor *yScaleOut,
  const aclTensor *yOriginOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSwigluGroupQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnSwigluGroupQuantGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1480px"><colgroup>
  <col style="width: 260px">
  <col style="width: 110px">
  <col style="width: 220px">
  <col style="width: 420px">
  <col style="width: 180px">
  <col style="width: 90px">
  <col style="width: 100px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>SwiGLU输入。</td>
      <td><ul><li>shape为[...,D]。</li><li>D必须大于等于256，且能被256整除。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>weightOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>MOE权重张量，用于SwiGLU输出的加权计算。</td>
      <td><ul><li>可选参数，不支持空Tensor。</li><li>不为空时，数据类型为FLOAT32，元素个数需等于x除最后一维外的元素个数之积。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>groupIndexOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>count模式的group token数。</td>
      <td><ul><li>可选参数，不支持空Tensor。</li><li>不为空时，数据类型为INT64，shape为[G]。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>scaleOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>quantMode=2时静态量化输入的invScale张量。</td>
      <td><ul><li>可选参数，仅quant_mode为2时使用。</li><li>groupIndex存在的话，shape=[G]，不存在的话shape=[1]。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dstType（int64_t）</td>
      <td>输入</td>
      <td>目标量化类型。</td>
      <td><ul><li>仅quantMode为0或1时，该参数生效。</li><li>支持取值35、36、40、41，分别表示FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2。</li><li>dstType为40或41时，quantMode必须为1。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode（int64_t）</td>
      <td>输入</td>
      <td>量化模式。</td>
      <td><ul><li>支持取值0、1、2、3。</li><li>0表示Block FP8模式。</li><li>1表示MX模式。</li><li>2表示HIFP8静态量化模式。</li><li>3表示HIFP8动态量化模式。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockSize（int64_t）</td>
      <td>输入</td>
      <td>量化块大小。</td>
      <td><ul><li>0表示使用当前量化模式的默认block大小。</li><li>quantMode为0时，支持0或128。</li><li>quantMode为1时，支持0或32。</li><li>quantMode为2或3时，该参数不生效，默认0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundScale（bool）</td>
      <td>输入</td>
      <td>是否将scale取整为2的幂。</td>
      <td><ul><li>quantMode为1时，roundScale必须为true。</li><li>quantMode为2或3时，该参数不生效。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clampLimit（double）</td>
      <td>输入</td>
      <td>SwiGLU计算前的clamp阈值。</td>
      <td><ul><li>-1.0表示不启用clamp。</li><li>启用clamp时，clampLimit必须大于0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax（double）</td>
      <td>输入</td>
      <td>目标量化类型的最大有限值。</td>
      <td><ul><li>仅quantMode为3时，该参数生效。</li><li>默认值为15.0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputOrigin（bool）</td>
      <td>输入</td>
      <td>是否输出量化前的SwiGLU结果。</td>
      <td><ul><li>true表示支持输出原始激活值yOrigin，false表示不支持输出原始激活值yOrigin。</li><li>quantMode为0或1时支持false。</li><li>quantMode为2或3时true/false都支持。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut（aclTensor*）</td>
      <td>输出</td>
      <td>量化输出。</td>
      <td><ul><li>quantMode为0或1时，数据类型需与dstType一致，dstType为35或36时，shape为[...,D/2]；dstType为40或41时，shape为[...,D/4]；dstType为27时，shape为[...,D/2]。</li><li>quantMode为2或3时，数据类型默认为HIFLOAT8，shape为[...,D/2]。</li><li>不支持空Tensor。</li></ul></td>
      <td>HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2</td>
      <td>ND</td>
      <td>2-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>yScaleOut（aclTensor*）</td>
      <td>输出</td>
      <td>量化scale输出。</td>
      <td><ul><li>quantMode为0时，shape为[...,ceil((D/2)/128)]，数据类型为FLOAT32。</li><li>quantMode为1时，shape为[...,ceil(ceil((D/2)/32)/2),2]，数据类型为FLOAT8_E8M0。</li><li>quantMode为3时，无groupIndex时shape为[1]，有groupIndex时shape为[G]，数据类型为FLOAT32。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>yOriginOut（aclTensor*）</td>
      <td>输出</td>
      <td>量化前的SwiGLU结果。</td>
      <td><ul><li>shape为[...,D/2]。</li><li>数据类型需与x一致。</li><li>不支持空指针。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x、yOut、yScaleOut、yOriginOut、workspaceSize或executor存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>输入或输出的数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>输入或输出的shape不满足约束。</td>
    </tr>
    <tr>
      <td>dstType、quantMode、blockSize、roundScale或clampLimit不符合当前支持的值。</td>
    </tr>
    <tr>
      <td>dstType、quantMode、blockSize、roundScale和yScaleOut数据类型组合不匹配。</td>
    </tr>
    <tr>
      <td>weightOptional或groupIndexOptional不满足可选输入约束。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>多个输入tensor之间的shape信息不匹配、输入属性不在取值范围（详见参数说明）。</td>
    </tr>
  </tbody></table>

## aclnnSwigluGroupQuant

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSwigluGroupQuantGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

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
- yScale的数据类型必须与quantMode匹配：quantMode=0或3时数据类型为FLOAT32，quantMode=1时数据类型为FLOAT8_E8M0。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_group_quant.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

bool CheckHardwareSupport()
{
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) {
        LOG_PRINT("Warning: Cannot get SOC name, skip hardware check\n");
        return true;
    }

    LOG_PRINT("Current SOC: %s\n", socName);
    if (strstr(socName, "Ascend950") != nullptr || strstr(socName, "ascend950") != nullptr) {
        return true;
    }

    LOG_PRINT("Warning: SwigluGroupQuant only supports Ascend950, current SOC '%s' is not supported. Skip test.\n",
              socName);
    return false;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
}

struct AclTensorResource {
    void* deviceAddr = nullptr;
    aclTensor* tensor = nullptr;
};

void DestroyAclTensorResource(AclTensorResource& resource)
{
    if (resource.tensor != nullptr) {
        aclDestroyTensor(resource.tensor);
        resource.tensor = nullptr;
    }
    if (resource.deviceAddr != nullptr) {
        aclrtFree(resource.deviceAddr);
        resource.deviceAddr = nullptr;
    }
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

struct SwigluGroupQuantCase {
    const char* name;
    int64_t quantMode;
    int64_t blockSize;
    bool roundScale;
    std::vector<int64_t> yScaleShape;
    aclDataType yScaleDataType;
    bool yScaleIsE8M0;
    int64_t dstType;
    double dstTypeMax;
    aclDataType yDataType;
    int64_t yElementSize;
    bool hasScaleInput;
    std::vector<int64_t> scaleShape;
};

int RunSwigluGroupQuantCase(const SwigluGroupQuantCase& testCase, aclrtStream stream)
{
    std::vector<int64_t> xShape = {2, 256};
    std::vector<int64_t> yShape = {2, 128};
    std::vector<int64_t> yOriginShape = {2, 128};

    std::vector<uint16_t> xHostData(GetShapeSize(xShape), 0);
    for (size_t i = 0; i < xHostData.size(); ++i) {
        xHostData[i] = static_cast<uint16_t>(i % 23);
    }
    std::vector<uint8_t> yHostData(GetShapeSize(yShape) * testCase.yElementSize, 0);
    std::vector<uint8_t> yScaleE8M0HostData(GetShapeSize(testCase.yScaleShape), 0);
    std::vector<float> yScaleFp32HostData(GetShapeSize(testCase.yScaleShape), 0.0f);
    std::vector<uint16_t> yOriginHostData(GetShapeSize(yOriginShape), 0);
    std::vector<float> scaleHostData;
    if (testCase.hasScaleInput) {
        scaleHostData.resize(GetShapeSize(testCase.scaleShape), 1.0f);
    }

    AclTensorResource xResource;
    AclTensorResource yResource;
    AclTensorResource yScaleResource;
    AclTensorResource yOriginResource;
    AclTensorResource scaleResource;

    auto ret = CreateAclTensor(xHostData, xShape, &xResource.deviceAddr, ACL_FLOAT16, &xResource.tensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yResource.deviceAddr, testCase.yDataType, &yResource.tensor);
    CHECK_RET(ret == ACL_SUCCESS, DestroyAclTensorResource(xResource); return ret);
    if (testCase.yScaleIsE8M0) {
        ret = CreateAclTensor(yScaleE8M0HostData, testCase.yScaleShape, &yScaleResource.deviceAddr,
                              testCase.yScaleDataType, &yScaleResource.tensor);
    } else {
        ret = CreateAclTensor(yScaleFp32HostData, testCase.yScaleShape, &yScaleResource.deviceAddr,
                              testCase.yScaleDataType, &yScaleResource.tensor);
    }
    CHECK_RET(ret == ACL_SUCCESS, DestroyAclTensorResource(xResource); DestroyAclTensorResource(yResource);
              DestroyAclTensorResource(yScaleResource); return ret);
    ret = CreateAclTensor(yOriginHostData, yOriginShape, &yOriginResource.deviceAddr, ACL_FLOAT16,
                          &yOriginResource.tensor);
    CHECK_RET(ret == ACL_SUCCESS, DestroyAclTensorResource(xResource); DestroyAclTensorResource(yResource);
              DestroyAclTensorResource(yScaleResource); DestroyAclTensorResource(yOriginResource); return ret);
    if (testCase.hasScaleInput) {
        ret = CreateAclTensor(scaleHostData, testCase.scaleShape, &scaleResource.deviceAddr,
                              ACL_FLOAT, &scaleResource.tensor);
        CHECK_RET(ret == ACL_SUCCESS, DestroyAclTensorResource(xResource); DestroyAclTensorResource(yResource);
                  DestroyAclTensorResource(yScaleResource); DestroyAclTensorResource(yOriginResource);
                  DestroyAclTensorResource(scaleResource); return ret);
    }

    double clampLimit = -1.0;
    bool outputOrigin = false;
    const aclTensor* scaleTensor = testCase.hasScaleInput ? scaleResource.tensor : nullptr;

    LOG_PRINT("Run %s: quant_mode=%ld\n", testCase.name, testCase.quantMode);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnSwigluGroupQuantGetWorkspaceSize(xResource.tensor, nullptr, nullptr, scaleTensor, testCase.dstType,
                                                testCase.quantMode, testCase.blockSize, testCase.roundScale, clampLimit,
                                                testCase.dstTypeMax, outputOrigin, yResource.tensor,
                                                yScaleResource.tensor, yOriginResource.tensor, &workspaceSize,
                                                &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              DestroyAclTensorResource(xResource); DestroyAclTensorResource(yResource);
              DestroyAclTensorResource(yScaleResource); DestroyAclTensorResource(yOriginResource);
              DestroyAclTensorResource(scaleResource); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  DestroyAclTensorResource(xResource); DestroyAclTensorResource(yResource);
                  DestroyAclTensorResource(yScaleResource); DestroyAclTensorResource(yOriginResource);
                  DestroyAclTensorResource(scaleResource); return ret);
    }

    ret = aclnnSwigluGroupQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupQuant failed. ERROR: %d\n", ret);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } DestroyAclTensorResource(xResource);
        DestroyAclTensorResource(yResource); DestroyAclTensorResource(yScaleResource);
        DestroyAclTensorResource(yOriginResource); DestroyAclTensorResource(scaleResource); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } DestroyAclTensorResource(xResource);
        DestroyAclTensorResource(yResource); DestroyAclTensorResource(yScaleResource);
        DestroyAclTensorResource(yOriginResource); DestroyAclTensorResource(scaleResource); return ret);

    std::vector<uint8_t> resultData(GetShapeSize(yShape) * testCase.yElementSize, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yResource.deviceAddr,
                      resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } DestroyAclTensorResource(xResource);
        DestroyAclTensorResource(yResource); DestroyAclTensorResource(yScaleResource);
        DestroyAclTensorResource(yOriginResource); DestroyAclTensorResource(scaleResource); return ret);
    LOG_PRINT("%s result[0] is: %d\n", testCase.name, resultData[0]);

    DestroyAclTensorResource(xResource);
    DestroyAclTensorResource(yResource);
    DestroyAclTensorResource(yScaleResource);
    DestroyAclTensorResource(yOriginResource);
    DestroyAclTensorResource(scaleResource);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    if (!CheckHardwareSupport()) {
        LOG_PRINT("\n=== Test SKIPPED (hardware not supported) ===\n");
        Finalize(deviceId, stream);
        return ACL_SUCCESS;
    }

    std::vector<SwigluGroupQuantCase> testCases = {
        {"block_fp8", 0, 0, false, {2, 1}, ACL_FLOAT, false, 36, 448.0, ACL_FLOAT8_E4M3FN, 1, false, {}},
        {"mx_fp8", 1, 0, true, {2, 2, 2}, ACL_FLOAT8_E8M0, true, 36, 448.0, ACL_FLOAT8_E4M3FN, 1, false, {}},
        {"hifp8_static", 2, 0, false, {1}, ACL_FLOAT, false, 27, 448.0, ACL_HIFLOAT8, 1, true, {1}},
        {"hifp8_dynamic", 3, 0, false, {1}, ACL_FLOAT, false, 27, 15.0, ACL_HIFLOAT8, 1, false, {}},
    };

    for (const auto& testCase : testCases) {
        ret = RunSwigluGroupQuantCase(testCase, stream);
        CHECK_RET(ret == ACL_SUCCESS, Finalize(deviceId, stream); return ret);
    }

    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}
```
