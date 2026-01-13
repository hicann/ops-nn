# MultiScaleDeformableAttentionGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：
  MultiScaleDeformableAttention正向算子功能主要通过采样位置（sample location）、注意力权重（attention weights）、映射后的value特征、多尺度特征起始索引位置、多尺度特征图的空间大小（便于将采样位置由归一化的值变成绝对位置）等参数来遍历不同尺寸特征图的不同采样点。而反向算子的功能为根据正向的输入对输出的贡献及初始梯度求出输入对应的梯度。
- 计算公式：

    给定输出的梯度：

    $$
    G^{\text{out}}_{b,q,h,:} \in \mathbb{R}^D
    $$

    展开注意力权重：

    $$
    \tilde{G}_{b,q,h,\ell,p,:} = A_{b,q,h,\ell,p} \cdot G^{\text{out}}_{b,q,h,:}
    $$


    计算 Value 的梯度，对每个邻点 (y,x)：

    $$
    \nabla V_{b,\ell,y,x,h,:} \; += \;
    \sum_q \tilde{G}_{b,q,h,\ell,p,:} \cdot w_{ij}
    $$

    其中 $w_{ij}\in\{w_{00},w_{10},w_{01},w_{11}\}$ 由采样位置决定。

    计算注意力权重的梯度

    $$
    \nabla A_{b,q,h,\ell,p} =
    \left\langle G^{\text{out}}_{b,q,h,:},
    \operatorname{bilinear}(V;\,b,h,\ell,x,y)\right\rangle
    $$

    计算采样位置的梯度，采样点坐标 $(x,y)$ 的梯度由双线性插值的偏导得到：

    $$
    \nabla x_{b,q,h,\ell,p} =
    \sum_d \tilde{G}_{b,q,h,\ell,p,d}\;
    \big[ (V_{y_0,x_1,h,d} - V_{y_0,x_0,h,d})(1-\alpha_y)
        + (V_{y_1,x_1,h,d} - V_{y_1,x_0,h,d})\alpha_y \big]
    $$

    $$
    \nabla y_{b,q,h,\ell,p} =
    \sum_d \tilde{G}_{b,q,h,\ell,p,d}\;
    \big[ (V_{y_1,x_0,h,d} - V_{y_0,x_0,h,d})(1-\alpha_x)
        + (V_{y_1,x_1,h,d} - V_{y_0,x_1,h,d})\alpha_x \big]
    $$

    缩放回归一化坐标：

    $$
    \nabla u = \frac{1}{W_\ell} \nabla x,\qquad
    \nabla v = \frac{1}{H_\ell} \nabla y
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
    <tr><td><b>G<sup>out</sup>(grad_output)</b></td><td>梯度输入</td><td>输出 O 的上游梯度，维度 D 的向量</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>ṠG = A · G<sup>out</sup></b></td><td>中间量</td><td>按注意力权重展开的上游梯度（逐层逐采样点）</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>V(b,ℓ,y,x,h,:)</b></td><td>输入</td><td>Value 特征向量（被插值采样的底图）</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>A(b,q,h,ℓ,p)</b></td><td>输入</td><td>注意力权重（每层每采样点的权重）</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>u, v</b></td><td>输入</td><td>采样点的归一化坐标（loc_w, loc_h）</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>x, y</b></td><td>中间量</td><td>像素坐标</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>x₀, x₁, y₀, y₁</b></td><td>中间量</td><td>四邻点整数坐标</td><td>INT32、INT64</td></tr>
    <tr><td><b>αx, αy</b></td><td>中间量</td><td>相对左上角的小数偏移</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>w₀₀, w₁₀, w₀₁, w₁₁</b></td><td>中间量</td><td>双线性插值权重（四权重和为 1）</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>∇V(grad_value)</b></td><td>梯度输出</td><td>对 Value 特征的梯度（累加到四邻点）</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>∇A(grad_attn_weight)</b></td><td>梯度输出</td><td>对注意力权重的梯度（与采样到的特征点内积）</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>∇x, ∇y</b></td><td>中间量</td><td>像素坐标处的梯度（由双线性偏导得到）</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>∇u, ∇v</b></td><td>中间量</td><td>缩放回归一化坐标的梯度</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>grad_sampling_loc_out</b></td><td>梯度输出</td><td>采样位置的最终梯度（按代码写入：先 u 再 v）</td><td>FLOAT、FLOAT16、BFLOAT16</td></tr>
    <tr><td><b>W<sub>ℓ</sub>, H<sub>ℓ</sub></b></td><td>属性</td><td>第 ℓ 层特征图的宽与高</td><td>INT32、INT64</td></tr>
    <tr><td><b>b, q, h, ℓ, p</b></td><td>属性</td><td>batch、query、head、层、采样点索引</td><td>INT32、INT64</td></tr>
    <tr><td><b>D</b></td><td>属性</td><td>每个 head 的嵌入维度</td><td>INT32、INT64</td></tr>
    <tr><td><b>Nq, Nh, L, Np</b></td><td>属性</td><td>query 数、head 数、层数、每层采样点数</td><td>INT32、INT64</td></tr>
</tbody>
</table>

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