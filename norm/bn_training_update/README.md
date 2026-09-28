# BNTrainingUpdate

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

>注：本表按算子在各产品的注册/交付支持面判定——<term>Ascend 950PR&950DT系列产品</term>为本仓arch35实现（4维NCHW/NHWC、5维NCDHW，见参数说明与约束说明中的产品限定）；其余产品由CANN交付的TBE实现（NC1HWC0/NDC1HWC0/NCDHW及动态模板下的NCHW/NHWC）。

## 功能说明

- 算子功能：批归一化训练前向的update阶段（含moving average更新）。给定BNTrainingReduce产出的逐通道sum/square_sum，结合缩放因子scale与偏置offset，对输入x做批归一化仿射变换，输出归一化结果y；同时以factor为权重更新running mean/variance，并输出本batch的统计量batch_mean/batch_variance（有偏方差）。与[BNTrainingReduce](../bn_training_reduce/README.md)配套使用。

- 计算公式：

  设x的shape为[N, C, H, W]，num = N * H * W（样本数），batch_variance在有偏估计下计算，Bessel修正仅用于running variance更新：

  $$
  batch\_mean = {sum\over num}
  $$

  $$
  batch\_variance = {square\_sum\over num} - batch\_mean^2
  $$

  $$
  y = {scale\over\sqrt {batch\_variance + \epsilon}} * x + (offset - {scale * batch\_mean\over\sqrt {batch\_variance + \epsilon}})
  $$

  $$
  mean = factor * batch\_mean + (1 - factor) * mean
  $$

  $$
  variance = factor * {num\over num - 1} * batch\_variance + (1 - factor) * variance
  $$

  其中num=1时Bessel修正项取0。

## 参数说明

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
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
<td><ul><li>待归一化的输入张量，对应公式中的<code>x</code>，为4维（NCHW/NHWC）或5维（NCDHW）张量。</li><li>任一维度为0（含C=0）时的空tensor语义见约束说明。</li><li>fp16/bf16输入在算子内升fp32计算、写回时转回原类型。</li></ul></td>
<td>FLOAT16、FLOAT32、BFLOAT16</td>
<td>NCHW、NHWC、NCDHW、NC1HWC0、NDC1HWC0</td>
</tr>
<tr>
<td>sum</td>
<td>输入</td>
<td><ul><li>x在N、H、W维上的逐通道求和结果，即BNTrainingReduce的sum输出，对应公式中的<code>sum</code>。</li><li>shape约束见约束说明。</li></ul></td>
<td>FLOAT32</td>
<td>ND、NC1HWC0、NDC1HWC0、NCDHW、NCHW、NHWC</td>
</tr>
<tr>
<td>square_sum</td>
<td>输入</td>
<td><ul><li>x在N、H、W维上的逐通道平方求和结果，即BNTrainingReduce的square_sum输出，对应公式中的<code>square_sum</code>。</li><li>shape约束同<code>sum</code>。</li></ul></td>
<td>FLOAT32</td>
<td>ND、NC1HWC0、NDC1HWC0、NCDHW、NCHW、NHWC</td>
</tr>
<tr>
<td>scale</td>
<td>输入</td>
<td><ul><li>逐通道缩放因子，对应公式中的<code>scale</code>。</li><li>shape约束同<code>sum</code>。</li></ul></td>
<td>FLOAT32</td>
<td>ND、NC1HWC0、NDC1HWC0、NCDHW、NCHW、NHWC</td>
</tr>
<tr>
<td>offset</td>
<td>输入</td>
<td><ul><li>逐通道缩放偏置，对应公式中的<code>offset</code>。</li><li>shape约束同<code>sum</code>。</li></ul></td>
<td>FLOAT32</td>
<td>ND、NC1HWC0、NDC1HWC0、NCDHW、NCHW、NHWC</td>
</tr>
<tr>
<td>mean</td>
<td>输入</td>
<td><ul><li>更新前的running mean，对应公式中的<code>mean</code>。</li><li>shape约束同<code>sum</code>。</li></ul></td>
<td>FLOAT32</td>
<td>ND、NC1HWC0、NDC1HWC0、NCDHW、NCHW、NHWC</td>
</tr>
<tr>
<td>variance</td>
<td>输入</td>
<td><ul><li>更新前的running variance，对应公式中的<code>variance</code>。</li><li>shape约束同<code>sum</code>。</li></ul></td>
<td>FLOAT32</td>
<td>ND、NC1HWC0、NDC1HWC0、NCDHW、NCHW、NHWC</td>
</tr>
<tr>
<td>factor</td>
<td>必选属性</td>
<td><ul><li>新batch统计量在running statistics更新中的权重，对应公式中的<code>factor</code>。</li><li>取值约束见约束说明。</li></ul></td>
<td>FLOAT32</td>
<td>-</td>
</tr>
<tr>
<td>epsilon</td>
<td>必选属性</td>
<td><ul><li>添加到batch_variance上的小量，用于保证数值稳定，对应公式中的<code>ε</code>。</li><li>取值约束见约束说明。</li></ul></td>
<td>FLOAT32</td>
<td>-</td>
</tr>
<tr>
<td>y</td>
<td>输出</td>
<td><ul><li>归一化仿射结果，对应公式中的<code>y</code>。</li><li>shape与数据类型均与<code>x</code>一致。</li></ul></td>
<td>与x一致</td>
<td>与x相同</td>
</tr>
<tr>
<td>mean</td>
<td>输出</td>
<td><ul><li>更新后的running mean，对应公式中的<code>mean</code>。</li><li>shape约束见约束说明。</li></ul></td>
<td>FLOAT32</td>
<td>ND、NC1HWC0、NDC1HWC0、NCDHW、NCHW、NHWC</td>
</tr>
<tr>
<td>variance</td>
<td>输出</td>
<td><ul><li>更新后的running variance（Bessel修正后），对应公式中的<code>variance</code>。</li><li>shape约束同输出<code>mean</code>。</li></ul></td>
<td>FLOAT32</td>
<td>ND、NC1HWC0、NDC1HWC0、NCDHW、NCHW、NHWC</td>
</tr>
<tr>
<td>batch_mean</td>
<td>输出</td>
<td><ul><li>本batch的逐通道均值，对应公式中的<code>batch_mean</code>。</li><li>shape约束同输出<code>mean</code>。</li></ul></td>
<td>FLOAT32</td>
<td>ND、NC1HWC0、NDC1HWC0、NCDHW、NCHW、NHWC</td>
</tr>
<tr>
<td>batch_variance</td>
<td>输出</td>
<td><ul><li>本batch的逐通道有偏方差，对应公式中的<code>batch_variance</code>。</li><li>shape约束同输出<code>mean</code>。</li></ul></td>
<td>FLOAT32</td>
<td>ND、NC1HWC0、NDC1HWC0、NCDHW、NCHW、NHWC</td>
</tr>
</tbody></table>

## 约束说明

- 参数表中的数据格式为各产品所支持格式的并集，具体支持范围以本节的产品说明为准。

**<term>Ascend 950PR&950DT系列产品</term>：**

- x为4维（NCHW/NHWC）或5维（NCDHW）张量，支持NCHW、NHWC、NCDHW三种数据格式；NCHW/NHWC下通道维为dim 1/dim 3，NCDHW下通道维为dim 1，格式与维数不匹配时算子在host侧拒绝执行。
- x支持FLOAT16/FLOAT32/BFLOAT16；sum/square_sum/scale/offset/mean/variance恒为FLOAT32、ND格式，shape为[C]，元素数必须等于x的通道维C。
- factor取值范围为[0.0, 1.0]，epsilon必须大于0；越界时算子在host侧拒绝执行。
- 空tensor语义与TensorFlow FusedBatchNormV3对齐：任一非通道维为0且C>0时合法——y为空、更新后的mean/variance全为NaN、batch_mean/batch_variance全为0；C=0（空通道向量）不支持，host侧拒绝执行。
- 归一化分母为sqrt(|batch_variance + epsilon|)，batch_variance不做非负钳制；batch_variance输出为消减误差产生的微小负值时按原值输出。fp16输出的y饱和到±65504、NaN置0后写回。

**其余产品（<term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas训练系列产品</term>、<term>Atlas推理系列产品</term>、<term>Atlas 200I/500 A2推理产品</term>）：**

- 数据格式为NC1HWC0/NDC1HWC0/NCDHW（动态模板下支持NCHW/NHWC），x为4维张量。
- 统计量sum/square_sum/scale/offset/mean/variance与x同rank、同布局，恒为FLOAT32；NC1HWC0/NDC1HWC0下C按C0=16对齐，统计量元素数等于C1*C0。
- 各维必须为正数（proto声明不支持空tensor）。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE图模式 | [test_geir_bn_training_update](./examples/test_geir_bn_training_update.cpp) | 通过[算子IR](op_graph/bn_training_update_proto.h)构图方式调用BNTrainingUpdate算子（静态shape用例，校验全部5个输出的数值）。 |
| GE图模式 | [test_geir_bn_training_update_dynamic](./examples/test_geir_bn_training_update_dynamic.cpp) | 动态shape（-1）与动态rank（-2）构图用例，同一Session连续运行3组shape，校验全部5个输出的数值。 |
