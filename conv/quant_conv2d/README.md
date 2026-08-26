# QuantConv2D

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：实现2D量化卷积功能。
- 计算公式：

  - 假定输入（`x`）的shape是 $(N, C_{\text{in}}, H, W)$ ，（`filter`）的shape是 $(C_{\text{out}}, C_{\text{in}}, K_h, K_w)$，输出（`y`）的shape是 $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$
  - 输出表示为：

  $$
  \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \text{scale} \times \sum_{k = 0}^{C_{\text{in}} - 1} \text{filter}(C_{\text{out}_j}, k) \star \text{x}(N_i, k)
  $$

  其中，$\star$ 表示卷积计算，支持空洞卷积、分组卷积。$N$ 代表batch size，$C$ 代表通道数，$H$ 和 $W$ 分别代表高和宽，相应输出维度的计算公式如下：

  $$
  H_{\text{out}} = (H + \text{pad\_top} + \text{pad\_bottom} - (\text{dilation\_h} \times (K_h - 1) + 1)) / \text{stride\_h} + 1 \\
    W_{\text{out}} = (W + \text{pad\_left} + \text{pad\_right} - (\text{dilation\_w} \times (K_w - 1) + 1)) / \text{stride\_w} + 1
  $$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 公式中的输入张量x。 | FLOAT8_E4M3FN、INT8、HIFLOAT8 | NCHW |
| filter | 输入 | 公式中的卷积权重张量filter。 | FLOAT8_E4M3FN、INT8、HIFLOAT8 | NCHW |
| scale | 输入 | 缩放因子张量scale。 | INT64、UINT64 | ND |
| bias | 可选输入 | 卷积偏置张量bias。 | FLOAT、INT32 | ND |
| offset | 可选输入 | 偏移张量offset（未使用）。 | FLOAT | NCHW |
| y | 输出 | 公式中的输出张量y。 | FLOAT、FLOAT16、BFLOAT16、FLOAT8_E4M3FN、HIFLOAT8 | NCHW |
| dtype | 属性 | 表示输出y的数据类型。支持的列表包括 [0(FLOAT)，1(FLOAT16)，27(BFLOAT16)，34(HIFLOAT8)，36(FLOAT8_E4M3FN)]。 | INT32 | - |
| strides | 属性 | 卷积扫描步长，包括stride_h, stride_w。stride_n, stride_c大小必须为1。 | INT32 | - |
| pads | 可选属性 | 对输入的填充，包括pad_top, pad_bottom, pad_left, pad_right。 | INT32 | - |
| dilations | 可选属性 | 卷积核中元素的间隔，包括dilation_h, dilation_w。dilation_n, dilation_c大小必须为1。 | INT32 | - |
| groups | 可选属性 | 从输入通道到输出通道的块链接个数，必须满足groups × filter的in_channels维度 = x的in_channels维度。支持范围 [1, 65535]。 | INT32 | - |
| data_format | 可选属性 | 输入数据格式，仅支持"NCHW"。 | STRING | - |
| offset_x | 可选属性 | 量化算法中的偏移，用于pad的填充值。支持范围 [-128, 127]。当`x`类型为`HIFLOAT8`或`FLOAT8_E4M3FN`时，仅支持配置为0。 | INT32 | - |
| round_mode | 可选属性 | 舍入模式。如果输出的数据类型是HIFLOAT8，此时该参数必须为'round'。默认为'rint'。 | STRING | - |

## 约束说明

- Ascend 950PR/Ascend 950DT：

  - `x`的数据类型必须与`filter`一致。
  - `x`、`filter`、`bias`、`scale`、`y`中每一组`tensor`的每一维大小都应该在[1, 1000000]范围内。
  - `strides`、`dilations`的值应该在[1, 1000000]范围内。
  - `pads`的值应该在[0, 1000000]范围内。
  - `bias`和`scale`维度大小应该与`filter`的`N`维度大小一致。
  - 支持的数据类型和Format组合如下表：

  | x | filter | scale | bias | y |
  | :---: | :---: | :---: | :---: | :---: |
  | INT8 | INT8 | INT64/UINT64 | INT32 | FLOAT16 |
  | HIFLOAT8 | HIFLOAT8 | INT64/UINT64 | FLOAT | FLOAT/FLOAT16/BFLOAT16/HIFLOAT8 |
  | FLOAT8_E4M3FN | FLOAT8_E4M3FN | INT64/UINT64 | FLOAT | FLOAT/FLOAT16/BFLOAT16/FLOAT8_E4M3FN |
  | NCHW | NCHW | ND | ND | NCHW |

- 如果任何参数超出上述范围，算子的正确性无法保证。
- 由于硬件资源限制，算子在部分参数取值组合场景下会执行失败，请根据日志信息提示分析并排查问题。若无法解决，请单击 [Link](https://www.hiascend.com/support)获取技术支持。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| 图模式 | [test_geir_quant_conv2d](./examples/arch35/test_geir_quant_conv2d.cpp) | 通过[算子IR](./graph/quant_conv2d_proto.h)构图方式调用QuantConv2D算子。 |
