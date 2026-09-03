# MultiScaleDeformableAttnFunction

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      √     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：

  通过采样位置（sample location）、注意力权重（attention weights）、映射后的value特征、多尺度特征起始索引位置、多尺度特征图的空间大小（便于将采样位置由归一化的值变成绝对位置）等参数来遍历不同尺寸特征图的不同采样点。

- 计算公式：

    设$b \in [0, bs)$、$q \in [0, num\_queries)$、$h \in [0, num\_heads)$、$\ell \in [0, num\_levels)$、$p \in [0, num\_points)$、$c \in [0, channels)$，分别为batch、查询、头、特征图、采样点、通道的索引。第$\ell$层特征图的高和宽为$H_\ell$、$W_\ell$，即$\mathrm{spatialShape}[\ell] = (H_\ell, W_\ell)$；$\mathrm{value}$的$num\_keys$维将所有层的特征图像素按层展平拼接（$num\_keys = \sum_{\ell=0}^{num\_levels-1} H_\ell W_\ell$），第$\ell$层的起始索引为$\mathrm{levelStartIndex}[\ell]$。

    1. 将采样点的归一化坐标$\mathrm{location}[b, q, h, \ell, p] = (u, v) \in [0,1]^2$（$u$、$v$分别为最后一维的第0、1个元素，对应$x$、$y$方向）映射到第$\ell$层特征图的像素坐标系：

       $$
       x = u \cdot W_\ell - 0.5, \qquad y = v \cdot H_\ell - 0.5
       $$

    2. 确定采样点落在哪四个整数网格点之间：

       $$
       x_0 = \lfloor x \rfloor,\quad x_1 = x_0 + 1,\qquad
       y_0 = \lfloor y \rfloor,\quad y_1 = y_0 + 1
       $$

    3. 计算采样点相对于左上角网格点的偏移，用于插值权重：

       $$
       \alpha_x = x - x_0, \qquad \alpha_y = y - y_0
       $$

    4. 计算双线性插值权重，四个邻点的和为1：

       $$
       \begin{aligned}
       w_{00} &= (1-\alpha_y)(1-\alpha_x), \\
       w_{10} &= (1-\alpha_y)\alpha_x, \\
       w_{01} &= \alpha_y(1-\alpha_x), \\
       w_{11} &= \alpha_y\alpha_x
       \end{aligned}
       $$

    5. 第$\ell$层特征图上的像素$(y, x)$展平后在$\mathrm{value}$的$num\_keys$维上的索引为：

       $$
       k_\ell(y, x) = \mathrm{levelStartIndex}[\ell] + y \cdot W_\ell + x
       $$

    6. 对$\mathrm{value}$做双线性采样，得到采样点对应的特征向量（长度为$channels$）：

       $$
       \begin{aligned}
       \operatorname{bilinear}(\mathrm{value};\,b,h,\ell,x,y) ={}&
       w_{00} \cdot \mathrm{value}[b,\; k_\ell(y_0, x_0),\; h,\; :] \\
       &+ w_{10} \cdot \mathrm{value}[b,\; k_\ell(y_0, x_1),\; h,\; :] \\
       &+ w_{01} \cdot \mathrm{value}[b,\; k_\ell(y_1, x_0),\; h,\; :] \\
       &+ w_{11} \cdot \mathrm{value}[b,\; k_\ell(y_1, x_1),\; h,\; :]
       \end{aligned}
       $$

    7. 所有层、所有采样点的双线性采样结果，以$\mathrm{attnWeight}$加权求和得到最终输出$\mathrm{output}$（shape为$(bs, num\_queries, num\_heads \times channels)$，最后一维按$h \times channels + c$排布）：

       $$
       \mathrm{output}[b,\; q,\; h \times channels + c] =
       \sum_{\ell=0}^{num\_levels-1} \sum_{p=0}^{num\_points-1}
       \mathrm{attnWeight}[b, q, h, \ell, p] \cdot
       \operatorname{bilinear}\!\left(\mathrm{value};\, b, h, \ell,\;
       x_{b,q,h,\ell,p},\, y_{b,q,h,\ell,p}\right)[c]
       $$

       其中$x_{b,q,h,\ell,p}$、$y_{b,q,h,\ell,p}$为采样点$\mathrm{location}[b, q, h, \ell, p]$经上述坐标映射得到的像素坐标。

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
      <td>FLOAT、FLOAT16、BFLOAT16</td>
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
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attention_weights</td>
      <td>输入</td>
      <td>采样点权重tensor，shape为(bs, num_queries, num_heads, num_levels, num_points)。对应公式中的`attnWeight`。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>算子计算输出，shape为(bs, num_queries, num_heads × channels)。对应公式中的`output`。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas 推理系列产品</term>：不支持BFLOAT16

## 约束说明

- <term>Atlas 推理系列产品</term>：
  - 通道数channels%32 = 0，且channels <= 256
  - 查询的数量32 <= num_queries< 500000
  - 特征图的数量num_levels <= 16
  - 头的数量num_heads = [2, 4, 8]
  - 采样点的数量num_points = [4, 8]
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：
  - 通道数channels%8 = 0，且channels <= 256
  - 查询的数量32 <= num_queries < 500000
  - 特征图的数量num_levels <= 16
  - 头的数量num_heads <= 16
  - 采样点的数量num_points <= 16

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_multi_scale_deformable_attn_function](./examples/test_aclnn_multi_scale_deformable_attn_function.cpp) | 通过[aclnnMultiScaleDeformableAttnFunction](./docs/aclnnMultiScaleDeformableAttnFunction.md)接口方式调用aclnnMultiScaleDeformableAttnFunction算子。    |
