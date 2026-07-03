# SwigluGroupQuantGrad

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 算子功能：SwigluGroupQuantGrad算子实现SwiGLU激活函数分组量化的反向梯度计算，用于计算输入梯度`grad_x`和权重梯度`grad_weight`。
- 算子支持范围：支持MoE场景（传入groupIndex）和非MoE场景（groupIndex传空），支持可选的Clamp反向传播掩码，支持可选的Weight梯度计算。
- 计算流程：
  - 步骤〇：GroupIndex处理（可选）→ 计算trunc
  - 步骤一：输入切分（将x切分为x0和x1）
  - 步骤二：Clamp处理（可选）
  - 步骤三：SwiGLU反向传播计算
  - 步骤四：Weight梯度计算（可选）
  - 步骤五：梯度拼接输出

- MoE场景GroupIndex处理公式：  
  $$  
    \text{trunc} = \sum_{g=0}^{G-1} \text{groupIndex}[g]  
  $$  
  其中：$G$ 为MoE专家分组数，后续所有步骤仅处理前 $\text{trunc}$ 行数据。

- 输入切分公式：  
  $$  
    \mathbf{x}_0[t, h] = \mathbf{x}[t, h], \quad h \in [0, H)  
  $$  
  
  $$  
    \mathbf{x}_1[t, h] = \mathbf{x}[t, h + H], \quad h \in [0, H)  
  $$  

- Clamp处理公式（当clamp_limit > 0时）：  
  $$  
    \mathbf{x}_0'[t, h] = \min(\mathbf{x}_0[t, h], c)  
  $$  
  
  $$  
    \mathbf{x}_1'[t, h] = \min(\max(\mathbf{x}_1[t, h], -c), c)  
  $$  
  其中 $c$ 为 `clamp_limit`。

- SiLU梯度公式：  
  $$  
    \frac{d\text{SiLU}}{d\mathbf{x}_0'} = \sigma(\mathbf{x}_0') \cdot \left(1 + \mathbf{x}_0' \cdot (1 - \sigma(\mathbf{x}_0'))\right)  
  $$  
  其中：$\sigma(\mathbf{x}_0') = \frac{1}{1 + e^{-\mathbf{x}_0'}}$

- 输入梯度计算公式：  
  $$  
    \mathbf{grad}_{x_0}[t, h] = \mathbf{grad}_{y_0}[t, h] \cdot \mathbf{x}_1'[t, h] \cdot \frac{d\text{SiLU}}{d\mathbf{x}_0'}[t, h]  
  $$  
  
  $$  
    \mathbf{grad}_{x_1}[t, h] = \mathbf{grad}_{y_0}[t, h] \cdot \text{SiLU}(\mathbf{x}_0'[t, h])  
  $$  
  其中：如果提供了weight，则 $\mathbf{grad}_{y_0} = \mathbf{grad}_{\text{output}} \cdot \mathbf{weight}$；如果未提供weight，则 $\mathbf{grad}_{y_0} = \mathbf{grad}_{\text{output}}$

- Weight梯度计算公式（可选）：  
  $$  
    \mathbf{grad}_{\text{weight}}[t] = \sum_{h=0}^{H-1} \mathbf{grad}_{\text{output}}[t, h] \cdot \mathbf{y}_{\text{origin}}[t, h]  
  $$  
  其中：$\mathbf{y}_{\text{origin}}$ 为SwiGLU前向传播的原始激活值输出，沿最后一维（H维度）求和。

- Clamp反向传播掩码公式（当clamp_limit > 0时）：  
  $$  
    \mathbf{grad}_{x_0}[t, h] = \mathbf{grad}_{x_0}[t, h] \cdot \mathbb{I}(\mathbf{x}_0[t, h] < c)  
  $$  
  
  $$  
    \mathbf{grad}_{x_1}[t, h] = \mathbf{grad}_{x_1}[t, h] \cdot \mathbb{I}(-c < \mathbf{x}_1[t, h] < c)  
  $$  
  其中 $\mathbb{I}$ 为指示函数。

- 梯度拼接与GroupIndex处理公式：  
  $$  
    \mathbf{grad}_x[t, h] = \begin{cases}  
    \mathbf{grad}_{x_0}[t, h] & h \in [0, H) \\  
    \mathbf{grad}_{x_1}[t, h-H] & h \in [H, 2H)  
    \end{cases}  
  $$  
  
  $$  
    \mathbf{grad}_x[t, :] = \mathbf{grad}_x[t, :] \cdot \mathbb{I}(t < \text{trunc})  
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
      <td>gradY</td>
      <td>输入</td>
      <td>梯度输出张量，来自下游层的梯度。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>前向传播的输入张量。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weightOptional</td>
      <td>输入</td>
      <td>MoE权重张量。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>yOriginOptional</td>
      <td>输入</td>
      <td>SwiGLU前向传播的原始激活值输出。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>groupIndexOptional</td>
      <td>输入</td>
      <td>GroupIndex张量，动态核分配。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>clampLimit</td>
      <td>属性</td>
      <td><ul><li>Clamp阈值。</li><li>取值范围≥0.0。</li><li>clampLimit=0表示不启用Clamp反向传播掩码。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradXOut</td>
      <td>输出</td>
      <td>输入梯度张量。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gradWeightOutOptional</td>
      <td>输出</td>
      <td>权重梯度张量。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 确定性计算：
  - 当提供 `groupIndex` 参数时：前 trunc 行保证计算结果确定性，后 T-trunc 行保证确定性（填充0）
  - 当未提供 `groupIndex` 参数时：所有行数据保证计算结果确定性

- 输入shape约束：
  - x最后一维必须为偶数（$2H$）
  - gradY最后一维为 $H$，与x最后一维的一半对应
  - gradY与x的前n-1维shape必须一致

- 可选参数约束：
  - weight提供时，必须同时提供yOrigin才能计算gradWeight
  - weight的shape需与gradY的第一维一致
  - yOrigin的shape需与gradY一致

- 数据类型约束：
  - gradY、x、yOrigin、gradXOut数据类型必须一致（FLOAT、FLOAT16或BFLOAT16）
  - weight、gradWeightOutOptional必须为FLOAT类型
  - groupIndex必须为INT64类型

- Clamp约束：
  - clampLimit必须 ≥ 0.0
  - clampLimit=0表示不启用Clamp反向传播掩码

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_swiglu_group_quant_grad](./examples/arch35/test_aclnn_swiglu_group_quant_grad.cpp) | 通过[aclnnSwigluGroupQuantGrad](./docs/aclnnSwigluGroupQuantGrad.md)接口方式调用SwigluGroupQuantGrad算子。    |
| 图模式调用 | -   | 通过[算子IR](./op_graph/swiglu_group_quant_grad_proto.h)构图方式调用SwigluGroupQuantGrad算子。 |
