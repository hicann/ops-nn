# Dilation2D

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：形态学膨胀（Morphological Dilation）操作的二维实现。对输入图像的每个通道独立执行滑窗操作：在每个窗口位置，取输入值与滤波器值之和的最大值作为输出。

- 计算公式：

$$
y[b, oh, ow, c] = \max_{fh, fw} (x[b, oh \times stride_h + fh \times rate_h, ow \times stride_w + fw \times rate_w, c] + filter[fh, fw, c])
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>4D输入张量，NHWC或NCHW格式。NHWC时filter shape为(fH,fW,C)，NCHW时为(C,fH,fW)。</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>filter</td>
      <td>输入</td>
      <td>3D滤波器张量，C维度与x的C一致。NHWC时shape为(fH,fW,C)，NCHW时为(C,fH,fW)。</td>
      <td>与x相同</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>4D输出张量，oH/oW由输入shape+属性推导。</td>
      <td>与x相同</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>4个正整数，滑窗步幅[sN, sH, sW, sC]。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rates</td>
      <td>属性</td>
      <td>4个正整数，膨胀率[rN, rH, rW, rC]。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>属性</td>
      <td>padding模式，"SAME"或"VALID"，默认"SAME"。</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>属性</td>
      <td>4个非负整数，显式padding值，默认{0,0,0,0}。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>属性</td>
      <td>输出尺寸计算使用ceil还是floor，默认false。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>属性</td>
      <td>数据格式，"NHWC"或"NCHW"，默认"NHWC"。</td>
      <td>String</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入x必须为4D张量，filter必须为3D张量。
- x和filter的dtype必须一致，仅支持FLOAT16。
- filter的通道数（C维度）必须与x的通道数一致。NHWC时filter shape为(fH,fW,C)，NCHW时为(C,fH,fW)。
- strides和rates必须为4个正整数。NHWC时strides[0]=strides[3]=1, rates[0]=rates[3]=1；NCHW时strides[0]=strides[1]=1, rates[0]=rates[1]=1。
- CALCULATED模式下pads必须小于有效滤波器尺寸，pads格式为[H_top, H_bottom, W_left, W_right]。
- 超出输入边界的位置视为负无穷，不参与max比较。
- 支持SAME、VALID和CALCULATED三种padding模式。
- 整数类型加法采用饱和截断（saturate），防止溢出wrap-around。
- 仅支持Ascend 950。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式接口 | [test_geir_dilation2_d](./examples/arch35/test_geir_dilation2_d.cpp) | 通过[算子IR](./op_graph/dilation2_d_proto.h)构图方式调用Dilation2D算子。 |
