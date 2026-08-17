# BNTrainingUpdateV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     √    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：批归一化训练前向的update阶段（Batch Normalization Training Update V2）。给定BNTrainingReduce产出的逐通道sum/square_sum，结合缩放因子scale与偏置offset，对输入x做批归一化仿射变换，输出归一化结果y；同时输出本batch的统计量batch_mean/batch_variance（供反向传播使用）。适用于不含moving average更新的场景，与[BNTrainingReduce](../bn_training_reduce/README.md)配套使用。

- 计算公式：

  设x的shape为[N, C, R...]（dim0为N、dim1为C、后导维展平为R），num = N * R：

  $$
  batch\_mean = {sum\over num}
  $$

  $$
  batch\_variance = {square\_sum\over num} - batch\_mean^2
  $$

  $$
  y = {scale\over\sqrt {batch\_variance + ε}} * x + (offset - {scale * batch\_mean\over\sqrt {batch\_variance + ε}})
  $$

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
      <td><ul><li>表示待归一化的输入张量，对应公式中的<code>x</code>。</li><li>shape为[N, C, R...]，支持≥2维，dim0为N、dim1为C、后导维展平为归一化轴R。</li><li>不支持空tensor（各维必须为正数）。</li><li>fp16/bf16输入在算子内升fp32计算、单次舍入写回。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum</td>
      <td>输入</td>
      <td><ul><li>表示x在N与R维上的逐通道求和结果，即BNTrainingReduce的sum输出，对应公式中的<code>sum</code>。</li><li>shape为[C]，元素数必须等于x的dim1。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>square_sum</td>
      <td>输入</td>
      <td><ul><li>表示x在N与R维上的逐通道平方求和结果，即BNTrainingReduce的square_sum输出，对应公式中的<code>square_sum</code>。</li><li>shape为[C]，元素数必须等于x的dim1。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td><ul><li>表示逐通道缩放因子，对应公式中的<code>scale</code>。</li><li>shape为[C]，元素数必须等于x的dim1。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td><ul><li>表示逐通道缩放偏置，对应公式中的<code>offset</code>。</li><li>shape为[C]，元素数必须等于x的dim1。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>必选属性</td>
      <td><ul><li>表示添加到batch_variance上的小量，以确保数值稳定，对应公式中的<code>ε</code>。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td><ul><li>表示归一化仿射结果，对应公式中的<code>y</code>。</li><li>shape与数据类型均与<code>x</code>一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_mean</td>
      <td>输出</td>
      <td><ul><li>表示本batch的逐通道均值，对应公式中的<code>batch_mean</code>。</li><li>shape为[C]。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_variance</td>
      <td>输出</td>
      <td><ul><li>表示本batch的逐通道方差，对应公式中的<code>batch_variance</code>。</li><li>shape为[C]。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 仅支持ND格式（dim0=N、dim1=C、后导维为归一化轴R；图模式下NCHW标签会被框架归一化下发，布局相同）。
- sum/square_sum/scale/offset的元素数必须等于x的dim1（C）。
- 不支持空tensor：x任一维为0时算子拒绝执行（num=N*R作为分母无法定义）。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_bn_training_update_v2](./examples/arch35/test_geir_bn_training_update_v2.cpp) | 通过[算子IR](op_graph/bn_training_update_v2_proto.h)构图方式调用BNTrainingUpdateV2算子（含两组shape/epsilon用例）。 |
