# MultiScaleDeformableAttnFunction

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：
  通过采样位置（sample location）、注意力权重（attention weights）、映射后的value特征、多尺度特征起始索引位置、多尺度特征图的空间大小（便于将采样位置由归一化的值变成绝对位置）等参数来遍历不同尺寸特征图的不同采样点。

- 计算公式：

    将采样点的归一化坐标 $(u,v)\in[0,1]$ 映射到第 $\ell$ 层特征图的像素坐标系：
    $$
    x = u \cdot W_\ell - 0.5, \qquad y = v \cdot H_\ell - 0.5
    $$


    确定采样点落在哪四个整数网格点之间：
    $$
    x_0 = \lfloor x \rfloor,\quad x_1 = x_0 + 1,\qquad
    y_0 = \lfloor y \rfloor,\quad y_1 = y_0 + 1
    $$

    计算采样点相对于左上角网格点的偏移，用于插值权重：
    $$
    \alpha_x = x - x_0, \qquad \alpha_y = y - y_0
    $$

    计算双线性插值权重，四个邻点的和为1
    $$
    \begin{aligned}
    w_{00} &= (1-\alpha_y)(1-\alpha_x), \\
    w_{10} &= (1-\alpha_y)\alpha_x, \\
    w_{01} &= \alpha_y(1-\alpha_x), \\
    w_{11} &= \alpha_y\alpha_x
    \end{aligned}
    $$

    计算得到采样点对应的特征向量（长度为 $D$）:
    $$
    \operatorname{bilinear}(V;\,b,h,\ell,x,y) =
    w_{00} \, V_{b,\ell,y_0,x_0,h,:}
    + w_{10} \, V_{b,\ell,y_0,x_1,h,:}
    + w_{01} \, V_{b,\ell,y_1,x_0,h,:}
    + w_{11} \, V_{b,\ell,y_1,x_1,h,:}
    $$

    所有层、所有采样点的双线性采样结果，加权求和得到最终输出：
    $$
    O_{b,q,h,:} =
    \sum_{\ell=0}^{L-1} \sum_{p=0}^{N_p-1}
    A_{b,q,h,\ell,p} \cdot
    \operatorname{bilinear}\!\left(V;\,b,h,\ell,
    x_{b,q,h,\ell,p}, y_{b,q,h,\ell,p}\right)
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 181px">
  <col style="width: 144px">
  <col style="width: 273px">
  <col style="width: 256px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><b>u, v</b></td><td>输入</td><td>采样点的归一化坐标，范围 [0,1]</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>W<sub>ℓ</sub>, H<sub>ℓ</sub></b></td><td>输入</td><td>第 ℓ 层特征图的宽、高</td><td>INT32、INT64</td></tr>
    <tr><td><b>V(b,ℓ,y,x,h,:)</b></td><td>输入</td><td>Value 特征向量：第 b 个 batch、第 ℓ 层、坐标 (y,x)、head h</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>A(b,q,h,ℓ,p)</b></td><td>输入</td><td>注意力权重：query q 在第 ℓ 层第 p 个采样点上的权重</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>x, y</b></td><td>中间量</td><td>映射到像素坐标系下的采样点位置</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>x₀, x₁, y₀, y₁</b></td><td>中间量</td><td>四邻点整数坐标</td><td>INT32、INT64</td></tr>
    <tr><td><b>αx, αy</b></td><td>中间量</td><td>采样点相对于左上角网格点的偏移量</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>w₀₀, w₁₀, w₀₁, w₁₁</b></td><td>中间量</td><td>双线性插值权重，和为 1</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>bilinear(V;·)</b></td><td>中间量</td><td>插值得到的采样向量</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>O(b,q,h,:)</b></td><td>输出</td><td>最终输出向量：query q 在 head h 的结果</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>b</b></td><td>属性</td><td>batch 索引</td><td>INT32、INT64</td></tr>
    <tr><td><b>q</b></td><td>属性</td><td>query 索引</td><td>INT32、INT64</td></tr>
    <tr><td><b>h</b></td><td>属性</td><td>head 索引</td><td>INT32、INT64</td></tr>
    <tr><td><b>ℓ</b></td><td>属性</td><td>特征层索引</td><td>INT32、INT64</td></tr>
    <tr><td><b>p</b></td><td>属性</td><td>采样点索引</td><td>INT32、INT64</td></tr>
    <tr><td><b>D</b></td><td>属性</td><td>每个 head 的嵌入维度</td><td>INT32、INT64</td></tr>
    <tr><td><b>Nq(num_queries)</b></td><td>属性</td><td>每个 batch 的 query 数</td><td>INT32、INT64</td></tr>
    <tr><td><b>Nh(num_heads)</b></td><td>属性</td><td>注意力 head 数</td><td>INT32、INT64</td></tr>
    <tr><td><b>L(num_levels)</b></td><td>属性</td><td>特征层数</td><td>INT32、INT64</td></tr>
    <tr><td><b>Np(num_points)</b></td><td>属性</td><td>每层每个 query 的采样点数</td><td>INT32、INT64</td></tr>
  </tbody>
</table>

- Atlas推理系列产品：不支持BFLOAT16

## 约束说明
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  - 通道数channels%8 = 0，且channels <= 256
  - 查询的数量32 <= num_queries < 500000
  - 特征图的数量num_levels <= 16
  - 头的数量num_heads <= 16
  - 采样点的数量num_points <= 16

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_multi_scale_deformable_attn_function](./examples/test_aclnn_multi_scale_deformable_attn_function.cpp) | 通过[aclnnMultiScaleDeformableAttnFunction](./docs/aclnnMultiScaleDeformableAttnFunction.md)接口方式调用aclnnMultiScaleDeformableAttnFunction算子。    |