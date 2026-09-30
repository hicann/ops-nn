# Conv2D

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term>| × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：实现2D卷积功能。

- 计算公式：

  - 假定输入（`x`）的shape是 $(N, C_{\text{in}}, H, W)$ ，（`filter`）的shape是 $(C_{\text{out}}, C_{\text{in}}, K_h, K_w)$，输出（`y`）的shape是 $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$

  - 输出表示为：

  $$
    \text{y}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{filter}(C_{\text{out}_j}, k) \star \text{x}(N_i, k)
  $$

  其中，$\star$ 表示卷积计算，支持空洞卷积（`dilations` > 1）、分组卷积（`groups` > 1）。$N$ 代表`batch size`，$C$ 代表通道数，$H$ 和 $W$ 分别代表高和宽，相应输出维度的计算公式如下：

  $$
    H_{\text{out}} = (H + \text{pad\_top} + \text{pad\_bottom} - (\text{dilation\_h} \times (K_h - 1) + 1)) / \text{stride\_h} + 1 \\
    W_{\text{out}} = (W + \text{pad\_left} + \text{pad\_right} - (\text{dilation\_w} \times (K_w - 1) + 1)) / \text{stride\_w} + 1
  $$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 公式中的输入张量x。 | FLOAT16、FLOAT、INT8、BFLOAT16、HIFLOAT8 | NCHW、NHWC |
| filter | 输入 | 公式中的卷积权重张量filter。 | FLOAT16、FLOAT、INT8、BFLOAT16、HIFLOAT8 | NCHW、HWCN |
| bias | 可选输入 | 卷积偏置张量bias。 | FLOAT16、FLOAT、BFLOAT16、INT32 | ND |
| offset_w | 可选输入 | 量化偏移张量offset_w（未使用）。 | INT8 | ND |
| y | 输出 | 公式中的输出张量y。 | FLOAT16、FLOAT、INT32、BFLOAT16、HIFLOAT8 | NCHW、NHWC |
| strides | 属性 | 卷积扫描步长。长度为4的整数列表，n和in_channels维必须为1；format为"NHWC"时形状为[1, stride_h, stride_w, 1]，format为"NCHW"时形状为[1, 1, stride_h, stride_w]。 | INT32 | - |
| pads | 属性 | 对输入的填充，包括pad_top, pad_bottom, pad_left, pad_right。 | INT32 | - |
| dilations | 可选属性 | 卷积核中元素的间隔。长度为4的整数列表，n和in_channels维必须为1；format为"NHWC"时形状为[1, dilation_h, dilation_w, 1]，format为"NCHW"时形状为[1, 1, dilation_h, dilation_w]。默认为[1, 1, 1, 1]。 | INT32 | - |
| groups | 可选属性 | 从输入通道到输出通道的块链接个数，必须满足groups × filter的in_channels维度 = x的in_channels维度，且in_channels、out_channels都能被groups整除。默认为1。 | INT32 | - |
| data_format | 可选属性 | 输入数据格式，支持"NCHW"、"NHWC"。默认为"NHWC"。 | STRING | - |
| offset_x | 可选属性 | 量化算法中的偏移offset_x（未使用）。默认为0。支持范围[-128, 127]。 | INT32 | - |

## 约束说明

- Ascend 950PR/Ascend 950DT：

  - `x`、`filter`、`bias`、`offset_w`、`y`中每一组`tensor`的每一维大小都应该在[1, 1000000]范围内。
  - `strides`、`dilations`的值应该在[1, 1000000]范围内。
  - `pads`的值应该在[0, 1000000]范围内。
  - 支持的数据类型和Format组合如下表：

  | x | filter | bias | offset_w | y |
  | :---: | :---: | :---: | :---: | :---: |
  | FLOAT16 | FLOAT16 | FLOAT16 | INT8 | FLOAT16 |
  | BFLOAT16 | BFLOAT16 | BFLOAT16 | INT8 | BFLOAT16 |
  | FLOAT | FLOAT | FLOAT | INT8 | FLOAT |
  | HIFLOAT8 | HIFLOAT8 | FLOAT | INT8 | HIFLOAT8 |
  | INT8 | INT8 | INT32 | INT8 | INT32 |
  | NCHW | NCHW | ND | ND | NCHW |
  | NHWC | HWCN | ND | ND | NHWC |

  `HIFLOAT8`仅支持`NCHW`（`x`/`filter`/`y`为`NCHW`，`bias`/`offset_w`为`ND`）。其余数据类型同时支持上表两行Format。

- 如果任何参数超出上述范围，算子的正确性无法保证。
- 由于硬件资源限制，算子在部分参数取值组合场景下会执行失败，请根据日志信息提示分析并排查问题。若无法解决，请单击 [Link](https://www.hiascend.com/support)获取技术支持。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| 图模式 | [test_geir_conv2d](./examples/arch35/test_geir_conv2d.cpp) | 通过[算子IR](./op_graph/conv2d_proto.h)构图方式调用Conv2D算子。 |
