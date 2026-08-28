# BNInference

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | √ |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：对输入特征图执行推理态批归一化，输出`y`的shape、数据类型和数据格式均与`x`一致。
- 以下公式中，`i`表示元素索引，`c`表示该元素所在的通道，`momentum_0`表示`momentum`的首个元素。
- 完整批归一化语义下，`scale`和`offset`均缺失时，计算公式为：

  $$
  factor = momentum_0 == 0 ? 0 : 1 / momentum_0
  $$

  $$
  alpha_c = -factor \times mean_c
  $$

  $$
  beta_c = \frac{1}{\sqrt{factor \times variance_c+epsilon}}
  $$

  $$
  y_i = (x_i+alpha_c) \times beta_c
  $$

- 完整批归一化语义下，`scale`和`offset`均存在时，`momentum`的数值不参与计算，计算公式为：

  $$
  s_c = \sqrt{variance_c+epsilon}
  $$

  $$
  inv\_s_c = \frac{1}{s_c}
  $$

  $$
  beta_c = scale_c \times inv\_s_c
  $$

  $$
  alpha_c = \frac{offset_c}{scale_c} \times s_c-mean_c
  $$

  $$
  y_i = (x_i+alpha_c) \times beta_c
  $$

  可选输入的合法组合以及缺省值见[产品差异说明](#产品差异说明)。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| x | 输入 | 待归一化的特征图。 | FLOAT、FLOAT16、BFLOAT16 | NC1HWC0、NCHW、NHWC、NCDHW、NDHWC、ND |
| mean | 输入 | 第一个逐通道参数；具体数学含义由计算模式确定，见公式和[产品差异说明](#产品差异说明)。 | FLOAT、FLOAT16、BFLOAT16 | NC1HWC0、ND |
| variance | 输入 | 第二个逐通道参数；具体数学含义由计算模式确定，见公式和[产品差异说明](#产品差异说明)。 | FLOAT、FLOAT16、BFLOAT16 | NC1HWC0、ND |
| momentum | 输入 | 均值和方差的缩放参数；参与计算的条件见公式和[产品差异说明](#产品差异说明)。 | FLOAT、FLOAT16、BFLOAT16 | NC1HWC0、ND |
| scale | 可选输入 | 逐通道缩放参数，对应公式中的`scale`；缺省行为见[产品差异说明](#产品差异说明)。 | FLOAT、FLOAT16、BFLOAT16 | NC1HWC0、ND |
| offset | 可选输入 | 逐通道偏移参数，对应公式中的`offset`；缺省行为见[产品差异说明](#产品差异说明)。 | FLOAT、FLOAT16、BFLOAT16 | NC1HWC0、ND |
| epsilon | 可选属性 | 完整批归一化语义下加到`variance`上的数值，默认值为`1e-5`。 | FLOAT | - |
| use_global_stats | 可选属性 | 是否使用全局统计量，默认值为`true`；有效取值见[产品差异说明](#产品差异说明)。 | BOOL | - |
| mode | 可选属性 | 计算模式，默认值为`1`；各产品的取值语义见[产品差异说明](#产品差异说明)。 | INT | - |
| y | 输出 | 批归一化结果，shape、数据类型和数据格式均与`x`一致。 | FLOAT、FLOAT16、BFLOAT16 | NC1HWC0、NCHW、NHWC、NCDHW、NDHWC、ND |

### 产品差异说明

#### 数据类型

| 产品 | 支持的数据类型组合 |
| :--- | :--- |
| <term>Ascend 950PR/Ascend 950DT</term> | 支持下表列出的11种数据类型组合。 |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | 所有输入和输出使用相同的数据类型，支持FLOAT、FLOAT16、BFLOAT16。 |
| <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term> | 所有输入和输出使用相同的数据类型，支持FLOAT、FLOAT16。 |

<term>Ascend 950PR/Ascend 950DT</term>数据类型组合：

| x | mean、variance | momentum | scale、offset | y |
| :---: | :---: | :---: | :---: | :---: |
| FLOAT | FLOAT | FLOAT | FLOAT | FLOAT |
| FLOAT16 | FLOAT | FLOAT | FLOAT16 | FLOAT16 |
| BFLOAT16 | FLOAT | FLOAT | BFLOAT16 | BFLOAT16 |
| FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 |
| FLOAT16 | FLOAT16 | FLOAT | FLOAT16 | FLOAT16 |
| BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 |
| BFLOAT16 | BFLOAT16 | FLOAT | BFLOAT16 | BFLOAT16 |
| FLOAT16 | FLOAT | FLOAT | FLOAT | FLOAT16 |
| BFLOAT16 | FLOAT | FLOAT | FLOAT | BFLOAT16 |
| BFLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | BFLOAT16 |
| FLOAT | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT |

`scale`或`offset`缺失时，不要求缺失Tensor的数据类型；二者均存在时必须使用表中同一行的数据类型。

#### 数据格式与shape

| 产品 | 参数或场景 | 静态shape能力 | 动态shape能力 | shape、rank及格式组合限制 |
| :--- | :--- | :--- | :--- | :--- |
| <term>Ascend 950PR/Ascend 950DT</term> | `x`、`y` | NCHW→NCHW、NHWC→NHWC、NCDHW→NCDHW、NDHWC→NDHWC、ND→ND | NCHW→NCHW、NHWC→NHWC、NCDHW→NCDHW、NDHWC→NDHWC、ND→ND | NCHW、NHWC为4D，NCDHW、NDHWC为5D，ND为4D或5D。NCHW、NCDHW的通道轴为第1轴，NHWC、NDHWC的通道轴为末轴。ND按原始格式（origin format）确定通道轴；原始格式为ND时，通道轴为第1轴。支持空Tensor，shape和元素数使用64位整数，具体限制见[约束说明](#约束说明)。 |
| <term>Ascend 950PR/Ascend 950DT</term> | `mean`、`variance`、`momentum`、`scale`、`offset` | ND | ND | `mean`、`variance`以及存在的`scale`、`offset`的shape为[C]；`momentum`的shape为[]、[1]或[C]。 |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term> | `x`、`y` | NC1HWC0→NC1HWC0、ND→ND | NC1HWC0→NC1HWC0、ND→ND | `x`为逻辑4D或5D特征图；`y`与`x`的shape和格式一致。NC1HWC0为这些产品的私有数据格式。 |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term> | `mean`、`variance`、`momentum`、`scale`、`offset` | NC1HWC0、ND | NC1HWC0、ND | 各参数表示逐通道数据，逻辑shape为[C]。参数格式与`x`相同：`x`为NC1HWC0时均为NC1HWC0，`x`为ND时均为ND。 |

#### mode与可选输入

在<term>Ascend 950PR/Ascend 950DT</term>的完整批归一化语义下，仅有`scale`或仅有`offset`时，
`momentum`的数值不参与计算：

$$
rstd_c = \frac{1}{\sqrt{variance_c+epsilon}}
$$

$$
\begin{aligned}
\text{仅有scale时：}\quad &y_i = ((x_i-mean_c) \times rstd_c) \times scale_c \\
\text{仅有offset时：}\quad &y_i = (x_i-mean_c) \times rstd_c+offset_c
\end{aligned}
$$

在<term>Ascend 950PR/Ascend 950DT</term>上，`mode=0`时，`mean`和`variance`分别表示预折叠系数`alpha`和`beta`：

$$
base_i = (x_i+mean_c) \times variance_c
$$

| 产品 | mode语义 | scale、offset组合 | use_global_stats |
| :--- | :--- | :--- | :--- |
| <term>Ascend 950PR/Ascend 950DT</term> | `mode=0`使用预折叠语义，`momentum`和`epsilon`不参与计算；任意非0整数使用完整批归一化语义。 | 完整批归一化支持均缺失、仅`scale`、仅`offset`和均存在；缺失的`scale`按1、缺失的`offset`按0。预折叠语义支持均缺失、仅`scale`和均存在，不支持仅`offset`。预折叠结果依次为`base`、`base*scale`和`base*scale+offset`。 | `true`和`false`均可使用，取值不改变计算结果。 |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term> | 任意整数均使用完整批归一化语义。 | 仅支持`scale`和`offset`均缺失或均存在。 | `true`和`false`均可使用，取值不改变计算结果。 |

## 约束说明

- 参数表中的数据类型和数据格式为各支持产品能力的并集，调用时必须使用“产品差异说明”中的合法组合，不能将数据类型和数据格式任意组合。
- <term>Ascend 950PR/Ascend 950DT</term>支持空Tensor。`x`的任意维可以为0，输出为同shape的空Tensor；所有输入仍须满足参数约束。C=0时，逐通道参数的shape须为[0]。
- <term>Ascend 950PR/Ascend 950DT</term>上，shape和元素数按64位处理；实际可分配大小受运行环境可用内存限制。
- 跨产品迁移时，若完整批归一化调用显式设置了`mode=0`，在<term>Ascend 950PR/Ascend 950DT</term>上须将`mode`改为非0值。

## 调用说明

本算子不提供aclnn单算子接口，仅支持GE图模式调用。

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | [test_geir_bn_inference](./examples/arch35/test_geir_bn_inference.cpp) | 通过[算子IR](./op_graph/bn_inference_proto.h)构图并调用BNInference算子；样例展示<term>Ascend 950PR/Ascend 950DT</term>支持的数据格式和可选输入组合。 |
