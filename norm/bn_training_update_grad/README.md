# BNTrainingUpdateGrad

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

- 算子功能：批归一化训练反向的update-grad阶段（Batch Normalization Training Update Grad）。给定上游梯度grads、前向输入x、前向落盘的逐通道统计量batch_mean/batch_variance，在batch与空间维（N与后导维R）上做逐通道归约，输出仿射参数scale/offset的梯度diff_scale/diff_offset，供scale/offset参数更新使用（x的梯度由BNTrainingReduceGrad另算）。与批归一化训练前向的update阶段（[BNTrainingUpdateV3](../bn_training_update_v3/README.md)）配套使用。

- 计算公式：

  设grads/x的shape为[N, C, R...]（dim0为N、dim1为C、后导维展平为归一化轴R），归约轴为N维与全部R维：

  $$
  rstd = {1\over\sqrt {batch\_variance + ε}}
  $$

  $$
  diff\_scale = \sum_{n,r} grads * (x - batch\_mean) * rstd
  $$

  $$
  diff\_offset = \sum_{n,r} grads
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
      <td>grads</td>
      <td>输入</td>
      <td><ul><li>表示上游梯度（损失函数对批归一化输出y的梯度），对应公式中的<code>grads</code>。</li><li>shape为[N, C, R...]，支持≥2维，dim0为N、dim1为C、后导维展平为归一化轴R。</li><li>不支持空tensor（各维必须为正数）。</li><li>fp16/bf16输入在算子内升fp32计算。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td><ul><li>表示前向传播的输入张量，对应公式中的<code>x</code>。</li><li>shape与数据类型均与<code>grads</code>一致。</li><li>fp16/bf16输入在算子内升fp32计算。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_mean</td>
      <td>输入</td>
      <td><ul><li>表示x的逐通道均值（前向update阶段的batch_mean输出），对应公式中的<code>batch_mean</code>。</li><li>shape为[C]，元素数必须等于grads的dim1。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_variance</td>
      <td>输入</td>
      <td><ul><li>表示x的逐通道方差（前向update阶段的batch_variance输出），对应公式中的<code>batch_variance</code>。</li><li>shape为[C]，元素数必须等于grads的dim1。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>表示添加到batch_variance上的小量，以确保数值稳定，对应公式中的<code>ε</code>。</li><li>缺省值为0.0001。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>diff_scale</td>
      <td>输出</td>
      <td><ul><li>表示逐通道缩放因子scale的梯度，对应公式中的<code>diff_scale</code>。</li><li>shape为[C]。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>diff_offset</td>
      <td>输出</td>
      <td><ul><li>表示逐通道偏置offset的梯度，对应公式中的<code>diff_offset</code>。</li><li>shape为[C]。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 仅支持ND格式（dim0=N、dim1=C、后导维为归一化轴R；图模式下NCHW标签会被框架归一化下发，布局相同）。
- x的shape与数据类型必须与grads一致。
- batch_mean/batch_variance的元素数必须等于grads的dim1（C）。
- 不支持空tensor：grads任一维为0时算子拒绝执行（归约轴为空时和数无定义）。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_bn_training_update_grad](./examples/arch35/test_geir_bn_training_update_grad.cpp) | 通过[算子IR](op_graph/bn_training_update_grad_proto.h)构图方式调用BNTrainingUpdateGrad算子（含两组shape/epsilon用例）。 |
