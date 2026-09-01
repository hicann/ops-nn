# BN3DTrainingUpdate

## 产品支持情况

| 产品 | 是否支持 |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | √ |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：BN3DTrainingUpdate算子用于3D BatchNorm训练流程中的"更新"环节。算子接收前驱
  Bn3dTrainingReduce算子产出的逐通道统计量（sum、square_sum），结合scale、offset对输入x做
  批归一化得到输出y；同时计算当前batch的save统计量（batch_mean、batch_variance）供反向传播
  复用，并以factor（等价于PyTorch的momentum）为EMA权重更新running mean与running variance
  （输出mean、variance，inplace写回输入mean、variance）。

- 计算公式：

  记通道数 `C = sum.shape[0]`，reduce域元素个数 `num = x.size / C`，逐通道有：

  $$
  \mu = \frac{sum}{num}
  $$

  $$
  \sigma^2_{biased} = \frac{square\_sum}{num} - \mu^2
  $$

  $$
  y = \frac{x - \mu}{\sqrt{\sigma^2_{biased} + \epsilon}} \cdot scale + offset
  $$

  $$
  mean\_out = factor \cdot \mu + (1 - factor) \cdot mean
  $$

  $$
  variance\_out = factor \cdot \sigma^2_{unbiased} + (1 - factor) \cdot variance
  $$

  其中 $\sigma^2_{unbiased} = \frac{num}{num - 1} \cdot \sigma^2_{biased}$（Bessel修正）；
  当 `num == 1` 时Bessel修正分母为0，显式置无偏batch variance为0，此时running variance不并入
  本batch统计量，仅按EMA衰减：$variance\_out = (1 - factor) \cdot variance$，`batch_variance`
  （有偏）正常输出。

  $$
  batch\_mean = \mu, \quad batch\_variance = \sigma^2_{biased}
  $$

  当 `num == 0`（x任一非通道轴为0，即空batch）时，按零统计契约输出：`batch_mean`、
  `batch_variance`均为0，`mean_out = (1 - factor) \cdot mean`、
  `variance_out = (1 - factor) \cdot variance`，y无元素输出。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 352px">
<col style="width: 213px">
<col style="width: 100px">
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
  <td>表示3D批归一化的源数据张量，对应公式中的x。</td>
  <td>FLOAT16、FLOAT32、BFLOAT16</td>
  <td>NCHW、NCDHW、NHWC、NDHWC、NDC1HWC0</td>
</tr>
<tr>
  <td>sum</td>
  <td>输入</td>
  <td>前驱Bn3dTrainingReduce产出的逐通道求和，1D张量，长度为通道数C，对应公式中的sum。</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
<tr>
  <td>square_sum</td>
  <td>输入</td>
  <td>前驱Bn3dTrainingReduce产出的逐通道平方和，1D张量，长度为通道数C，对应公式中的square_sum。</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
<tr>
  <td>scale</td>
  <td>输入</td>
  <td>批归一化的缩放系数，1D张量，长度为通道数C，对应公式中的scale。</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
<tr>
  <td>offset</td>
  <td>输入</td>
  <td>批归一化的偏置项，1D张量，长度为通道数C，对应公式中的offset。</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
<tr>
  <td>mean</td>
  <td>输入</td>
  <td>running mean（历史均值），1D张量，长度为通道数C，EMA更新后被原地写回，对应公式中的mean。</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
<tr>
  <td>variance</td>
  <td>输入</td>
  <td>running variance（历史方差），1D张量，长度为通道数C，EMA更新后被原地写回，对应公式中的variance。</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
<tr>
  <td>factor</td>
  <td>属性</td>
  <td>EMA权重，等价于PyTorch的momentum，建议取值范围[0, 1]。</td>
  <td>FLOAT32</td>
  <td>-</td>
</tr>
<tr>
  <td>epsilon</td>
  <td>属性</td>
  <td>归一化分母的防除零小常数，参与sqrt(var + epsilon)，要求不得小于0。</td>
  <td>FLOAT32</td>
  <td>-</td>
</tr>
<tr>
  <td>y</td>
  <td>输出</td>
  <td>批归一化输出，对应公式中的y。</td>
  <td>FLOAT16、FLOAT32、BFLOAT16</td>
  <td>NCHW、NCDHW、NHWC、NDHWC、NDC1HWC0</td>
</tr>
<tr>
  <td>mean</td>
  <td>输出</td>
  <td>EMA更新后的running mean，inplace写回输入mean，对应公式中的mean_out。</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
<tr>
  <td>variance</td>
  <td>输出</td>
  <td>EMA更新后的running variance（合并的是无偏batch variance），inplace写回输入variance，对应公式中的variance_out。</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
<tr>
  <td>batch_mean</td>
  <td>输出</td>
  <td>当前batch的save mean（= sum / num），供反向传播复用，对应公式中的batch_mean。</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
<tr>
  <td>batch_variance</td>
  <td>输出</td>
  <td>当前batch的有偏save variance（= square_sum/num - μ²），供反向传播复用，对应公式中的batch_variance。</td>
  <td>FLOAT32</td>
  <td>ND</td>
</tr>
</tbody>
</table>

- <term>Ascend 950PR/Ascend 950DT</term>：数据格式不支持NDC1HWC0。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据格式不支持NHWC、NDHWC
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据格式不支持NHWC、NDHWC。
- <term>Atlas 200I/500 A2 推理产品</term>：数据类型不支持BFLOAT16，数据格式不支持NCHW、NCDHW、NHWC、NDHWC。
- <term>Atlas 推理系列产品</term>：数据类型不支持BFLOAT16，数据格式不支持NCHW、NHWC、NDHWC。
- <term>Atlas 训练系列产品</term>：数据类型不支持BFLOAT16，数据格式不支持NCDHW、NHWC、NDHWC。

## 约束说明

- 通道轴C由x的数据格式决定。
- Ascend 950PR/950DT：x的维度rank仅支持4（NCHW/NHWC）与5（NCDHW/NDHWC）。
- y的数据类型、数据格式与shape均与x保持一致。
- mean、variance、batch_mean、batch_variance的shape须与sum一致。
- mean、variance为inplace输入输出：调用完成后原tensor内容被EMA更新后的running统计量覆盖。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_bn3_d_training_update](examples/arch35/test_geir_bn3_d_training_update.cpp) | 通过[算子IR](op_graph/bn3_d_training_update_proto.h)构图方式调用BN3DTrainingUpdate算子。 |
