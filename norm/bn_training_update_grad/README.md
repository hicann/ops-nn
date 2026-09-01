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

  设grads/x为ND布局[N, C, R...]（dim0为N、dim1为C、后导维展平为归一化轴R），归约轴为N维与全部R维（NHWC布局时C为最后一维，归约轴为全部前导维N·H·W）：

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
      <td><ul><li>表示上游梯度（损失函数对批归一化输出y的梯度），对应公式中的<code>grads</code>。</li><li>Ascend 950PR/Ascend 950DT：ND布局shape为[N, C, R...]（支持≥2维，dim0为N、dim1为C、后导维展平为归一化轴R；NCDHW为5D标签，内存布局与ND相同）；NHWC布局C为最后一维（shape任意rank≥2，前导维展平为归一化轴）。其余产品rank由输入格式固定（4D/5D/6D），通道轴随格式。</li><li>不支持空tensor（各维必须为正数）。</li><li>fp16/bf16输入在算子内升fp32计算。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND、NCDHW、NHWC</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td><ul><li>表示前向传播的输入张量，对应公式中的<code>x</code>。</li><li>shape、数据类型与布局格式均与<code>grads</code>一致。</li><li>fp16/bf16输入在算子内升fp32计算。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND、NCDHW、NHWC</td>
    </tr>
    <tr>
      <td>batch_mean</td>
      <td>输入</td>
      <td><ul><li>表示x的逐通道均值，对应公式中的<code>batch_mean</code>。前向落盘统计量（对应<a href="../bn_training_update_v3/README.md">BNTrainingUpdateV3</a>的reserve_1输出，即save_mean；TF FusedBatchNormGrad场景对应reserve_space_1），非V3的batch_mean输出。</li><li>shape为[C]，元素数必须等于grads的通道数C（ND为dim1，NHWC为最后一维）。</li><li>恒为FLOAT32。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_variance</td>
      <td>输入</td>
      <td><ul><li>表示x的逐通道<code>有偏方差save_variance</code>（前向落盘统计量，对应<a href="../bn_training_update_v3/README.md">BNTrainingUpdateV3</a>的reserve_2输出；TF FusedBatchNormGrad场景对应reserve_space_2），<strong>非</strong>V3的batch_variance输出（无偏方差，含num/(num-1)修正）——按无偏方差接线会使梯度系统性偏小sqrt((num-1)/num)。</li><li>shape为[C]，元素数必须等于grads的通道数C。</li><li>恒为FLOAT32。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>表示添加到batch_variance上的小量，以确保数值稳定，对应公式中的<code>ε</code>。</li><li>缺省值为0.0001。</li><li>建议非负：host不校验符号，batch_variance+ε为负时开方输出NaN。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>diff_scale</td>
      <td>输出</td>
      <td><ul><li>表示逐通道缩放因子scale的梯度，对应公式中的<code>diff_scale</code>。</li><li>shape与batch_mean一致（元素数为C；infershape原样复制batch_mean的shape，传[1,C,1,1]时输出同形）。</li><li>恒为FLOAT32。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>diff_offset</td>
      <td>输出</td>
      <td><ul><li>表示逐通道偏置offset的梯度，对应公式中的<code>diff_offset</code>。</li><li>shape与batch_mean一致（元素数为C）。</li><li>恒为FLOAT32。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- **Ascend 950PR/Ascend 950DT**：支持ND（dim0=N、dim1=C、后导维为归一化轴R；图模式下NCHW/NCDHW标签会被框架归一化下发或直接透传，NCDHW为5D标签、内存布局与ND相同，布局一致）与NHWC（C=最后一维，前导维N·H·W展平为归一化轴，任意rank≥2、C无上限）两类布局。
- **其余产品（Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品、Atlas 推理系列产品、Atlas 200I/500 A2 推理产品 等）：grads/x支持NCHW/NHWC/NC1HWC0/NCDHW（5D场景NDC1HWC0），rank由格式固定（4D/5D/6D），通道轴随格式；统计量与输出和grads同format。其中 Atlas A2 训练系列产品/Atlas A2 推理系列产品 及 Atlas A3 训练系列产品/Atlas A3 推理系列产品 支持 BFLOAT16；Atlas 200I/500 A2 推理产品 仅支持 NC1HWC0/NCHW（4D，不支持 NHWC/NCDHW/NDC1HWC0 与 BFLOAT16），Atlas 推理系列产品 不支持 BFLOAT16。
- x的shape、数据类型与布局格式必须与grads一致。
- batch_mean/batch_variance恒为FLOAT32，元素数必须等于grads的通道数C（ND为dim1，NHWC为最后一维）。
- 不支持空tensor：grads任一维为0时算子拒绝执行。归约轴（N与R维）为空时和数虽在数学上可定义为0，但通道轴C=0时输出元素数与统计量均无定义，且num（N·R）作为运算分母不可为空——与 Atlas A2 训练系列产品/Atlas A2 推理系列产品 同族算子（BNTrainingReduce/BNTrainingUpdate/BNTrainingUpdateV3）proto的"Empty tensors are not supported"统一口径（该产品侧 BNTrainingUpdateGrad的proto注释未附此句属其文档遗漏，非行为差异）。
- Ascend 950PR/Ascend 950DT 的ND布局下C==1且总元素数≥2^20、或R==1且C>2048时，内部按NHWC同构布局多核切分（语义不变）；NHWC小C多行场景的跨核部分和以浮点原子加合并，输出为各核部分和之浮点和（顺序不定，浮点原子加的舍入误差为机器精度量级O(eps)，部分和个数=核数≤64）。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_bn_training_update_grad](./examples/arch35/test_geir_bn_training_update_grad.cpp) | 通过[算子IR](op_graph/bn_training_update_grad_proto.h)构图方式调用BNTrainingUpdateGrad算子（含两组shape/epsilon用例）。 |
