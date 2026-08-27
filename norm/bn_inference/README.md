# BNInference

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | √ |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：对输入特征图执行推理态批归一化，输出`y`的shape、数据类型和数据格式均与`x`一致。
- `mode`为非0整数（默认值为1）时，执行完整推理态批归一化语义。
- 在上述完整BN模式下，`scale`和`offset`均缺失（4输入）时，计算公式为：

  $$
  factor = momentum_0 == 0 ? 0 : 1 / momentum_0
  $$

  $$
  alpha_c = -factor * mean_c
  $$

  $$
  beta_c = 1 / \sqrt{factor * variance_c + epsilon}
  $$

  $$
  y_i = (x_i + alpha_c) * beta_c
  $$

- 在完整BN模式下，`scale`和`offset`均存在（6输入）时，`momentum`的数值不参与计算，计算公式为：

  $$
  s_c = \sqrt{variance_c + epsilon},\quad inv\_s_c = 1 / s_c
  $$

  $$
  beta_c = scale_c * inv\_s_c
  $$

  $$
  alpha_c = (offset_c / scale_c) * s_c - mean_c
  $$

  $$
  y_i = (x_i + alpha_c) * beta_c
  $$

- 在完整BN模式下，<term>Ascend 950PR/Ascend 950DT</term>支持仅有`scale`或仅有`offset`的5输入组合。此时按标准推理态BatchNorm计算，缺失的`scale`按1、缺失的`offset`按0，`momentum`数值不参与计算。

- 在<term>Ascend 950PR/Ascend 950DT</term>上，`mode=0`时，`mean`和`variance`分别表示预折叠系数`alpha`和`beta`，`momentum`、`epsilon`和`use_global_stats`不参与数值计算：

  $$
  base_i = (x_i + mean_c) * variance_c
  $$

  - `scale/offset`均缺失：$y_i = base_i$。
  - 仅`scale`存在：$y_i = base_i * scale_c$。
  - `scale/offset`均存在：$y_i = base_i * scale_c + offset_c$。
  - 仅`offset`存在：接口报错。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 110px">
  <col style="width: 130px">
  <col style="width: 430px">
  <col style="width: 210px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待归一化的4D或5D特征图。在<term>Ascend 950PR/Ascend 950DT</term>上，NCHW、NHWC支持4D，NCDHW、NDHWC支持5D，ND支持4D或5D。NCHW/NCDHW的逻辑通道轴为第1轴，NHWC/NDHWC为末轴；storage format为ND时，origin format为NCHW/NCDHW取第1轴，为NHWC/NDHWC取末轴，origin format仍为ND时沿用第1轴。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td><term>Ascend 950PR/Ascend 950DT</term>：NCHW、NHWC、NCDHW、NDHWC、ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>shape为[C]，其中C与x的逻辑通道数相同，表示逐通道均值。在<term>Ascend 950PR/Ascend 950DT</term>的预折叠语义下表示预折叠系数alpha。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>variance</td>
      <td>输入</td>
      <td>shape为[C]，其中C与x的逻辑通道数相同，表示逐通道方差。在<term>Ascend 950PR/Ascend 950DT</term>的预折叠语义下表示预折叠系数beta。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>momentum</td>
      <td>输入</td>
      <td>动量参数。在<term>Ascend 950PR/Ascend 950DT</term>上，shape支持rank=0（shape为[]）、[1]或[C]。非空x在完整BN语义的4输入组合中只读取展平后的首元素并据此计算factor，因此该组合的momentum必须至少有1个元素；[C]的其余元素不参与计算。其他完整BN组合、预折叠语义及空Tensor场景均不读取其数值；C=0的空Tensor允许momentum为[0]。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>可选输入</td>
      <td>逐通道缩放参数，存在时shape为[C]。在<term>Ascend 950PR/Ascend 950DT</term>的预折叠语义下对预折叠结果执行逐通道缩放。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>可选输入</td>
      <td>逐通道偏移参数，存在时shape为[C]。在<term>Ascend 950PR/Ascend 950DT</term>的预折叠语义下必须与scale同时存在，并添加到缩放后的结果。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td>完整BN语义下加到variance上的数值，默认值为1e-5；在<term>Ascend 950PR/Ascend 950DT</term>的预折叠语义下不参与计算。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_global_stats</td>
      <td>可选属性</td>
      <td>兼容属性，默认值为true。在<term>Ascend 950PR/Ascend 950DT</term>上，其取值不改变y的计算。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>可选属性</td>
      <td>默认值为1。<term>Ascend 950PR/Ascend 950DT</term>上，mode=0选择预折叠语义，任意非0整数选择完整BN语义。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>归一化结果，shape、数据类型和数据格式均与x一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td><term>Ascend 950PR/Ascend 950DT</term>：NCHW、NHWC、NCDHW、NDHWC、ND</td>
    </tr>
  </tbody>
</table>

在<term>Ascend 950PR/Ascend 950DT</term>上，输入数据类型仅支持下表中的组合；`scale`和`offset`存在时使用“仿射参数”列的数据类型，缺失时不要求对应Tensor。

| x | mean/variance | momentum | 仿射参数 | y |
|:---:|:---:|:---:|:---:|:---:|
| FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |
| FLOAT16 | FLOAT32 | FLOAT32 | FLOAT16 | FLOAT16 |
| BFLOAT16 | FLOAT32 | FLOAT32 | BFLOAT16 | BFLOAT16 |
| FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 |
| FLOAT16 | FLOAT16 | FLOAT32 | FLOAT16 | FLOAT16 |
| BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 |
| BFLOAT16 | BFLOAT16 | FLOAT32 | BFLOAT16 | BFLOAT16 |
| FLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT16 |
| BFLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 | BFLOAT16 |
| BFLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | BFLOAT16 |
| FLOAT32 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 |

## 约束说明

- `mean`和`variance`的shape必须为[C]；存在的`scale`和`offset`的shape也必须为[C]。
- <term>Ascend 950PR/Ascend 950DT</term>上，`mode=0`时不支持仅有`offset`而没有`scale`；该约束在空Tensor路径上也会校验。
- 产品形态差异：<term>Ascend 950PR/Ascend 950DT</term>上，`mode=0`选择预折叠语义，任意非0整数选择完整BN语义；其他产品形态上，`mode`取值不改变完整BN语义。跨产品形态迁移显式使用`mode=0`的完整BN调用时，需将`mode`修改为非0值。
- <term>Ascend 950PR/Ascend 950DT</term>：`x`和`y`仅支持NCHW、NHWC、NCDHW、NDHWC、ND格式。storage format为ND时保留origin format语义：NCHW/NHWC用于4D，NCDHW/NDHWC用于5D。
- <term>Ascend 950PR/Ascend 950DT</term>支持空Tensor。`x`的任意维可以为0，输出为同shape的空Tensor；所有输入仍须满足参数约束。C=0时，逐通道参数的shape须为[0]。
- <term>Ascend 950PR/Ascend 950DT</term>上，shape和元素数按64位处理；实际可分配大小仍受运行环境可用内存限制。
- 本算子不提供aclnn单算子接口，仅支持GE图模式调用。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|:---|:---|:---|
| 图模式调用 | [test_geir_bn_inference](./examples/arch35/test_geir_bn_inference.cpp) | 在<term>Ascend 950PR/Ascend 950DT</term>上，通过[算子IR](./op_graph/bn_inference_proto.h)构图并调用BNInference算子，可选择NCHW、NHWC、NCDHW、NDHWC或4D/5D ND storage format，并覆盖四种可选输入组合。ND storage下的通道轴遵循公开origin format。 |
