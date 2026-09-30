# Conv3D

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:-------:|
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：对形状为5D的输入张量做三维卷积，支持空洞卷积（`dilations` > 1）、分组卷积（`groups` > 1）和偏置（`bias`）累加，计算`y = CONV(x, filter) + bias`。

- 计算公式：

  - 假定输入（`x`）的shape是$(N, C_{\text{in}}, D, H, W)$（NCDHW格式）或$(N, D, H, W, C_{\text{in}})$（NDHWC格式），（`filter`）的shape是$(C_{\text{out}}, C_{\text{in}}, K_d, K_h, K_w)$（NCDHW格式）或$(K_d, K_h, K_w, C_{\text{in}}, C_{\text{out}})$（DHWCN格式），输出（`y`）的shape是$(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})$（NCDHW格式）或$(N, D_{\text{out}}, H_{\text{out}}, W_{\text{out}}, C_{\text{out}})$（NDHWC格式）。

  - 输出表示为：

  $$
    \text{y}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{filter}(C_{\text{out}_j}, k) \star \text{x}(N_i, k)
  $$

  其中，$\star$ 表示三维卷积计算，输出维度的计算公式如下：

  $$
    D_{\text{out}} = (D + \text{pad\_head} + \text{pad\_tail} - \text{dilation\_d} \times (K_d - 1) - 1) / \text{stride\_d} + 1 \\
    H_{\text{out}} = (H + \text{pad\_top} + \text{pad\_bottom} - \text{dilation\_h} \times (K_h - 1) - 1) / \text{stride\_h} + 1 \\
    W_{\text{out}} = (W + \text{pad\_left} + \text{pad\_right} - \text{dilation\_w} \times (K_w - 1) - 1) / \text{stride\_w} + 1
  $$

- 输出dtype推导：输出`y`的dtype与输入`x`一致；当`x`为INT8时输出为INT32。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:-------|:--------------|:-----|:--------|:--------|
| x | 输入 | 卷积输入特征张量，5D。 | FLOAT16、BFLOAT16、FLOAT、HIFLOAT8、INT8 | NCDHW、NDHWC |
| filter | 输入 | 卷积核权重张量，5D，与`x`的dtype一致。 | FLOAT16、BFLOAT16、FLOAT、HIFLOAT8、INT8 | NCDHW、DHWCN |
| bias | 可选输入 | 卷积偏置张量，1D，shape为`[C_out]`。 | FLOAT16、FLOAT、BFLOAT16、INT32 | ND |
| offset_w | 可选输入 | 量化偏移张量（预留，当前未使用）。 | INT8 | ND |
| y | 输出 | 卷积输出特征张量，5D。 | FLOAT16、BFLOAT16、FLOAT、HIFLOAT8、INT32 | NCDHW、NDHWC |
| strides | 属性 | 滑窗步长，长度为5的列表，顺序由`data_format`决定；仅D/H/W维参与输出shape推导（N/C维不参与，按惯例置1）。 | ListInt，必填 | - |
| pads | 属性 | 各维度padding，长度支持1/3/6：长度6逐维显式，顺序为`[pad_head, pad_tail, pad_top, pad_bottom, pad_left, pad_right]`；长度3为`[pad_d, pad_h, pad_w]`，广播到对应维两侧；长度1全维同值。 | ListInt，必填 | - |
| dilations | 属性 | 空洞卷积空洞率，长度为5的列表，顺序由`data_format`决定。默认`[1, 1, 1, 1, 1]`。 | ListInt | - |
| groups | 属性 | 分组卷积的分组数，`C_in`与`C_out`必须能被`groups`整除。默认1；groups为1且`C_in`可被filter的`C_in`整除时，按`C_in / C_in(filter)`隐式生效（对齐RT1.0，分组数在实际执行阶段推导）。 | Int | - |
| data_format | 属性 | 输入/输出数据格式，取值`NCDHW`或`NDHWC`。默认`NDHWC`。 | String | - |
| offset_x | 属性 | 量化算法中的偏移（预留，当前未使用）。默认0。 | Int | - |

dtype/format支持组合：

| x | filter | bias | y |
|:--|:-------|:-----|:--|
| float16 + NCDHW | float16 + NCDHW | float16 + ND | float16 + NCDHW |
| bfloat16 + NCDHW | bfloat16 + NCDHW | bfloat16 + ND | bfloat16 + NCDHW |
| float + NCDHW | float + NCDHW | float + ND | float + NCDHW |
| hifloat8 + NCDHW | hifloat8 + NCDHW | float + ND | hifloat8 + NCDHW |
| float16 + NDHWC | float16 + DHWCN | float16 + ND | float16 + NDHWC |
| bfloat16 + NDHWC | bfloat16 + DHWCN | bfloat16 + ND | bfloat16 + NDHWC |
| float + NDHWC | float + DHWCN | float + ND | float + NDHWC |
| int8 + NCDHW | int8 + NCDHW | int32 + ND | int32 + NCDHW |
| int8 + NDHWC | int8 + DHWCN | int32 + ND | int32 + NDHWC |

## 约束说明

- `x`、`filter`、`y`的shape均为5D（`x`支持动态shape`-1`与动态rank`-2`）。
- `strides`、`dilations`长度必须为5，且`strides`的D/H/W维必须大于0；`dilations`的D维取值范围为[1, 1000000]，H/W维取值范围为[1, 255]。
- `pads`长度支持1/3/6（长度3按`[pad_d, pad_h, pad_w]`广播到对应维两侧，长度1全维同值）；静态shape下取值必须不小于0。
- `data_format`仅支持`NCDHW`或`NDHWC`；`x`的format仅支持NCDHW/NDHWC，`filter`仅支持NCDHW/NDHWC/DHWCN，`y`仅支持NCDHW/NDHWC。
- `filter`的D/H/W维必须不小于1。
- padding后的输入尺寸必须不小于卷积核尺寸（`x`与`filter`均为静态shape时校验）。
- 零Tensor：`x`任一维为0或`filter`的N/C维为0时，输出必须能推导出零Tensor，否则拦截。
- 通道约束：静态shape下`C_in`必须能被`filter`的`C_in`维整除；`groups`显式设置（非1）时必须等于`C_in / C_in(filter)`。
- 前端（TensorFlow）传入`padding = SAME/VALID`属性时，`pads`按输入尺寸动态计算（动态维记-1，SAME）或置0（VALID）。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|:--------|:--------|:-----|
| 图模式调用 | [test_geir_conv3d](examples/test_geir_conv3d.cpp) | 通过[算子IR](op_graph/conv3d_proto.h)构图方式调用Conv3D算子。 |
