# MultiScaleDeformableAttentionGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：

  MultiScaleDeformableAttention正向算子功能主要通过采样位置（sample location）、注意力权重（attention weights）、映射后的value特征、多尺度特征起始索引位置、多尺度特征图的空间大小（便于将采样位置由归一化的值变成绝对位置）等参数来遍历不同尺寸特征图的不同采样点。而反向算子的功能为根据正向的输入对输出的贡献及初始梯度求出输入对应的梯度。
- 计算公式：

    设$b \in [0, bs)$、$q \in [0, num\_queries)$、$h \in [0, num\_heads)$、$\ell \in [0, num\_levels)$、$p \in [0, num\_points)$、$c \in [0, channels)$，分别为batch、查询、头、特征图、采样点、通道的索引。第$\ell$层特征图的高和宽为$H_\ell$、$W_\ell$，即$\mathrm{spatialShape}[\ell] = (H_\ell, W_\ell)$；第$\ell$层特征图像素$(y, x)$在$\mathrm{value}$的$num\_keys$维上的展平索引为$k_\ell(y, x) = \mathrm{levelStartIndex}[\ell] + y \cdot W_\ell + x$。采样点$\mathrm{location}[b, q, h, \ell, p] = (u, v)$（最后一维第0、1个元素分别对应$x$、$y$方向）映射到像素坐标$x = u \cdot W_\ell - 0.5$、$y = v \cdot H_\ell - 0.5$；记$x_0 = \lfloor x \rfloor$、$x_1 = x_0 + 1$、$y_0 = \lfloor y \rfloor$、$y_1 = y_0 + 1$，$\alpha_x = x - x_0$、$\alpha_y = y - y_0$，双线性插值权重$w_{00} = (1-\alpha_y)(1-\alpha_x)$、$w_{10} = (1-\alpha_y)\alpha_x$、$w_{01} = \alpha_y(1-\alpha_x)$、$w_{11} = \alpha_y\alpha_x$，并记四邻点特征向量$V_{y_i, x_j} := \mathrm{value}[b,\; k_\ell(y_i, x_j),\; h,\; :]$（$i, j \in \{0, 1\}$，越界邻点按0处理）。

    - 正向输出为：

      $$
      \mathrm{output}[b,\; q,\; h \times channels + c] =
      \sum_{\ell=0}^{num\_levels-1} \sum_{p=0}^{num\_points-1}
      \mathrm{attnWeight}[b, q, h, \ell, p] \cdot
      \sum_{i,j \in \{0,1\}} w_{ij} \cdot V_{y_i, x_j}[c]
      $$

    - 反向算子根据梯度$\mathrm{gradOutput}$（shape为$(bs, num\_queries, num\_heads \times channels)$，最后一维按$h \times channels + c$排布）计算$\mathrm{value}$、$\mathrm{location}$、$\mathrm{attnWeight}$三个输入的梯度。记$\mathrm{gradOutput}$中head$h$的切片为$G_{b,q,h,c} := \mathrm{gradOutput}[b,\; q,\; h \times channels + c]$。
      1. 按注意力权重展开：

         $$
         \tilde{G}_{b,q,h,\ell,p,c} = \mathrm{attnWeight}[b, q, h, \ell, p] \cdot G_{b,q,h,c}
         $$

      2. 计算$\mathrm{value}$的梯度$\mathrm{gradValue}$（shape与value一致）：对每个采样点$(b, q, h, \ell, p)$，将其展开梯度按双线性权重累加到四个邻点位置（对所有$q$、$p$累加，越界邻点不累加）：

         $$
         \begin{aligned}
         \mathrm{gradValue}[b,\; k_\ell(y_0, x_0),\; h,\; c] &+= w_{00} \cdot \tilde{G}_{b,q,h,\ell,p,c}, \\
         \mathrm{gradValue}[b,\; k_\ell(y_0, x_1),\; h,\; c] &+= w_{10} \cdot \tilde{G}_{b,q,h,\ell,p,c}, \\
         \mathrm{gradValue}[b,\; k_\ell(y_1, x_0),\; h,\; c] &+= w_{01} \cdot \tilde{G}_{b,q,h,\ell,p,c}, \\
         \mathrm{gradValue}[b,\; k_\ell(y_1, x_1),\; h,\; c] &+= w_{11} \cdot \tilde{G}_{b,q,h,\ell,p,c}
         \end{aligned}
         $$

      3. 计算$\mathrm{attnWeight}$的梯度$\mathrm{gradAttnWeight}$（shape与attnWeight一致），为输出梯度与采样得到的特征向量的内积：

         $$
         \mathrm{gradAttnWeight}[b, q, h, \ell, p] =
         \sum_{c} G_{b,q,h,c} \cdot
         \left( \sum_{i,j \in \{0,1\}} w_{ij} \cdot V_{y_i, x_j}[c] \right)
         $$

      4. 计算$\mathrm{location}$的梯度$\mathrm{gradLocation}$（shape与location一致）。采样点像素坐标$(x, y)$的梯度由双线性插值的一阶偏导得到：

         $$
         \nabla x_{b,q,h,\ell,p} =
         \sum_{c} \tilde{G}_{b,q,h,\ell,p,c} \,
         \big[ (V_{y_0, x_1}[c] - V_{y_0, x_0}[c])(1-\alpha_y)
             + (V_{y_1, x_1}[c] - V_{y_1, x_0}[c])\alpha_y \big]
         $$

         $$
         \nabla y_{b,q,h,\ell,p} =
         \sum_{c} \tilde{G}_{b,q,h,\ell,p,c} \,
         \big[ (V_{y_1, x_0}[c] - V_{y_0, x_0}[c])(1-\alpha_x)
             + (V_{y_1, x_1}[c] - V_{y_0, x_1}[c])\alpha_x \big]
         $$

      5. 由$x = u \cdot W_\ell - 0.5$、$y = v \cdot H_\ell - 0.5$，按链式法则缩放回归一化坐标的梯度，写入$\mathrm{gradLocation}$（最后一维第0、1个元素分别对应$x$、$y$方向，与location一致）：

         $$
         \mathrm{gradLocation}[b, q, h, \ell, p] = (\nabla u, \nabla v), \qquad
         \nabla u = W_\ell \cdot \nabla x_{b,q,h,\ell,p}, \qquad
         \nabla v = H_\ell \cdot \nabla y_{b,q,h,\ell,p}
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
      <td>value</td>
      <td>输入</td>
      <td>特征图的特征值，shape为(bs, num_keys, num_heads, channels)。对应公式中的`value`。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value_spatial_shapes</td>
      <td>输入</td>
      <td>存储每个尺度特征图的高和宽，shape为(num_levels, 2)。对应公式中的`spatialShape`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value_level_start_index</td>
      <td>输入</td>
      <td>每张特征图在value的num_keys维上的起始索引，shape为(num_levels)。对应公式中的`levelStartIndex`。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sampling_locations</td>
      <td>输入</td>
      <td>采样点位置tensor，shape为(bs, num_queries, num_heads, num_levels, num_points, 2)。对应公式中的`location`。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attention_weights</td>
      <td>输入</td>
      <td>采样点权重tensor，shape为(bs, num_queries, num_heads, num_levels, num_points)。对应公式中的`​attnWeight`。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_output</td>
      <td>输入</td>
      <td>正向输出output的上游梯度（反向算子的初始梯度），shape为(bs, num_queries, num_heads × channels)。对应公式中的`gradOutput`。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_value</td>
      <td>输出</td>
      <td>输入value对应的梯度，shape与value一致。对应公式中的`gradValue`。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_sampling_locations</td>
      <td>输出</td>
      <td>输入sampling_locations对应的梯度，shape与sampling_locations一致。对应公式中的`gradLocation`。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_attention_weights</td>
      <td>输出</td>
      <td>输入attention_weights对应的梯度，shape与attention_weights一致。对应公式中的`gradAttnWeight`。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 通道数channels%8 = 0，且channels<=256
- 查询的数量num_queries < 500000
- 特征图的数量num_levels <= 16
- 头的数量num_heads <= 16
- 采样点的数量num_points <= 16

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_multi_scale_deformable_attention_grad](./examples/test_aclnn_multi_scale_deformable_attention_grad.cpp) | 通过[aclnnMultiScaleDeformableAttentionGrad](./docs/aclnnMultiScaleDeformableAttentionGrad.md)接口方式调用aclnnMultiScaleDeformableAttentionGrad算子。    |
