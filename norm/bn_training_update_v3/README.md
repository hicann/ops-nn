# BNTrainingUpdateV3

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     √    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

> 注：本表按算子在各产品的注册/交付支持面判定——Ascend 950PR/Ascend 950DT 为本仓 arch35 实现（ND/NHWC、rank 2~8，见参数说明与约束说明中的产品限定）；其余产品由 CANN 交付的 TBE 实现（输入4维NHWC/NCHW，内部NC1HWC0/NDC1HWC0布局）。数据类型 BFLOAT16 仅 Ascend 950PR/Ascend 950DT、Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品支持；Atlas 训练系列产品、Atlas 推理系列产品、Atlas 200I/500 A2 推理产品仅 FLOAT16/FLOAT32。

## 功能说明

- 算子功能：批归一化训练前向的update阶段（Batch Normalization Training Update V3）。给定BNTrainingReduce产出的逐通道sum/square_sum，结合缩放因子scale与偏置offset，对输入x做批归一化仿射变换，输出归一化结果y；同时输出本batch的统计量batch_mean/batch_variance（batch_variance为无偏估计）以及反向传播用的中间量reserve_1（save_mean）/reserve_2（save_variance，有偏方差）。适用于不含moving average更新的场景，与[BNTrainingReduce](../bn_training_reduce/README.md)配套使用。

- 计算公式：

  设x的shape为[N, C, R...]（dim0为N、dim1为C、后导维展平为R），num = N * R：

  $$
  save\_mean = {sum\over num}
  $$

  $$
  save\_variance = {square\_sum\over num} - save\_mean^2
  $$

  $$
  batch\_mean = save\_mean
  $$

  $$
  batch\_variance = save\_variance * {num\over num - 1} \quad (num=1时为0)
  $$

  $$
  reserve\_1 = save\_mean
  $$

  $$
  reserve\_2 = save\_variance
  $$

  $$
  y = {scale\over\sqrt {save\_variance + ε}} * x + (offset - {scale * save\_mean\over\sqrt {save\_variance + ε}})
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
      <td><ul><li>表示待归一化的输入张量，对应公式中的<code>x</code>。</li><li>Ascend 950PR/Ascend 950DT：ND布局shape为[N, C, R...]，支持2~8维，dim0为N、dim1为C、后导维展平为归一化轴R；NHWC布局任意rank≥2，C=最后一维。</li><li>其余产品：4维（来源格式NHWC或NCHW，内部按5HD/6HD布局计算）。</li><li>不支持空tensor（各维必须为正数）。</li><li>fp16/bf16输入在算子内升fp32计算、单次舍入写回。</li></ul></td>
      <td>FLOAT32、FLOAT16；BFLOAT16仅Ascend 950PR/Ascend 950DT、Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品</td>
      <td>Ascend 950PR/Ascend 950DT：ND/NHWC；其余产品：NC1HWC0/NCHW/NDC1HWC0/NHWC</td>
    </tr>
    <tr>
      <td>sum</td>
      <td>输入</td>
      <td><ul><li>表示x在N与R维上的逐通道求和结果，即BNTrainingReduce的sum输出，对应公式中的<code>sum</code>。</li><li>Ascend 950PR/Ascend 950DT：shape为[C]（推荐；按元素数校验，[1,C]等元素数相同的形态同样放行），元素数必须等于通道数C（ND/NCHW布局下即x的dim1，NHWC布局下为最后一维）。</li><li>其余产品：shape与x同rank（如NC1HWC0下为[1,C1,1,1,C0]，元素数等于C1*C0）。</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR/Ascend 950DT：ND；其余产品：与x同布局</td>
    </tr>
    <tr>
      <td>square_sum</td>
      <td>输入</td>
      <td><ul><li>表示x在N与R维上的逐通道平方求和结果，即BNTrainingReduce的square_sum输出，对应公式中的<code>square_sum</code>。</li><li>Ascend 950PR/Ascend 950DT：shape为[C]（推荐；按元素数校验，[1,C]等元素数相同的形态同样放行），元素数必须等于通道数C（ND/NCHW布局下即x的dim1，NHWC布局下为最后一维）。</li><li>其余产品：shape与x同rank（如NC1HWC0下为[1,C1,1,1,C0]，元素数等于C1*C0）。</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR/Ascend 950DT：ND；其余产品：与x同布局</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td><ul><li>表示逐通道缩放因子，对应公式中的<code>scale</code>。</li><li>Ascend 950PR/Ascend 950DT：shape为[C]（推荐；按元素数校验，[1,C]等元素数相同的形态同样放行），元素数必须等于通道数C（ND/NCHW布局下即x的dim1，NHWC布局下为最后一维）。</li><li>其余产品：shape与x同rank（如NC1HWC0下为[1,C1,1,1,C0]，元素数等于C1*C0）。</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR/Ascend 950DT：ND；其余产品：与x同布局</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td><ul><li>表示逐通道缩放偏置，对应公式中的<code>offset</code>。</li><li>Ascend 950PR/Ascend 950DT：shape为[C]（推荐；按元素数校验，[1,C]等元素数相同的形态同样放行），元素数必须等于通道数C（ND/NCHW布局下即x的dim1，NHWC布局下为最后一维）。</li><li>其余产品：shape与x同rank（如NC1HWC0下为[1,C1,1,1,C0]，元素数等于C1*C0）。</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR/Ascend 950DT：ND；其余产品：与x同布局</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>必选属性</td>
      <td><ul><li>表示添加到save_variance上的小量，以确保数值稳定，对应公式中的<code>ε</code>。</li><li>取值应大于0；传入0或负值且save_variance为负时，host不校验、将静默输出NaN。</li><li>本产品为必选属性、无默认值（其余产品TBE侧缺省1e-7）。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>before_split_ori_shape / before_split_ori_format</td>
      <td>不涉及（Ascend 950PR/Ascend 950DT 与 Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品 差异点，历史背景说明）</td>
      <td><ul><li><strong>Ascend 950PR/Ascend 950DT：无此入参/属性。</strong>该两参数不属于算子原型输入，且与FFTS切分特性强相关——FFTS为已废弃特性，Ascend 950PR/Ascend 950DT 代码与原型均不包含、不支持。</li><li>历史背景：Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品等将其注册为可选属性（LIST_LIST_INT/LIST_INT，默认[[]]/[]），用于BN FFTS切分场景按切分前原始shape/format计算num；该特性已废弃，Ascend 950PR/Ascend 950DT 的num按实际输入shape计算，传入不生效（原型未注册，GE侧按未知属性忽略）。</li></ul></td>
      <td>不涉及</td>
      <td>不涉及</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td><ul><li>表示归一化仿射结果，对应公式中的<code>y</code>。</li><li>shape与数据类型均与<code>x</code>一致。</li></ul></td>
      <td>与x一致</td>
      <td>与x一致</td>
    </tr>
    <tr>
      <td>batch_mean</td>
      <td>输出</td>
      <td><ul><li>表示本batch的逐通道均值，对应公式中的<code>batch_mean</code>。</li><li>Ascend 950PR/Ascend 950DT：shape与scale一致（scale为[C]时即[C]）。</li><li>其余产品：与x同rank。</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR/Ascend 950DT：ND；其余产品：与x同布局</td>
    </tr>
    <tr>
      <td>batch_variance</td>
      <td>输出</td>
      <td><ul><li>表示本batch的逐通道方差（无偏估计，即save_variance * num/(num-1)），对应公式中的<code>batch_variance</code>。</li><li>Ascend 950PR/Ascend 950DT：shape与scale一致（scale为[C]时即[C]）。</li><li>其余产品：与x同rank。</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR/Ascend 950DT：ND；其余产品：与x同布局</td>
    </tr>
    <tr>
      <td>reserve_1</td>
      <td>输出</td>
      <td><ul><li>表示本batch的逐通道均值（save_mean，与batch_mean相同），供反向传播使用，对应公式中的<code>reserve_1</code>。</li><li>Ascend 950PR/Ascend 950DT：shape与scale一致（scale为[C]时即[C]）。</li><li>其余产品：与x同rank。</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR/Ascend 950DT：ND；其余产品：与x同布局</td>
    </tr>
    <tr>
      <td>reserve_2</td>
      <td>输出</td>
      <td><ul><li>表示本batch的逐通道方差（save_variance，有偏估计），供反向传播使用，对应公式中的<code>reserve_2</code>。</li><li>Ascend 950PR/Ascend 950DT：shape与scale一致（scale为[C]时即[C]）。</li><li>其余产品：与x同rank。</li></ul></td>
      <td>FLOAT32</td>
      <td>Ascend 950PR/Ascend 950DT：ND；其余产品：与x同布局</td>
    </tr>
  </tbody></table>

## 约束说明

**Ascend 950PR/Ascend 950DT：**

- 支持ND与NCHW格式（tiling对两者都放行；dim0=N、dim1=C、后导维为归一化轴R，rank 2~8；图模式下NCHW标签会被框架归一化下发，布局相同）。
- 支持NHWC格式（x任意rank≥2，C=最后一维，num=numel/C；统计量仍按元素数=C校验）。NHWC下通道数C无上限：C为64的倍数时按向量窗口流式处理，C非64倍数且整行预算内按整行tile处理，超大C按c窗口流式处理（系数按窗重建，UB占用与C无关）。NC1HWC0/NDC1HWC0来源格式在本产品被tiling拒收（GRAPH_FAILED）——Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品 上按这些格式下发的BN图迁移到 Ascend 950PR/Ascend 950DT 前需先转ND/NCHW/NHWC。
- x维度数须≥2；x为FLOAT16/FLOAT32/BFLOAT16。
- tiling阶段要求shape全为确定正值：任一维为0拒绝（空tensor），任一维为负/未知动态维报"dynamic shape dim is not supported in tiling"拒绝。
- 不支持BN FFTS切分场景（Ascend 950PR/Ascend 950DT 与 Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品 差异点）：before_split_ori_shape/before_split_ori_format不属于算子原型输入，与FFTS切分特性强相关且该特性已废弃——Ascend 950PR/Ascend 950DT 的代码与原型不包含这两个入参，num按实际输入shape计算。历史背景：Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品 侧将其注册为可选属性用于按切分前原始shape计算num，详见参数表该行说明。
- sum/square_sum/scale/offset恒为FLOAT32，元素数必须等于通道数C（ND/NCHW布局下即x的dim1，NHWC布局下为x的最后一维；shape推荐为[C]，按元素数校验）。
- batch_mean/batch_variance/reserve_1/reserve_2的shape与scale一致。
- 不支持空tensor：x任一维为0时算子拒绝执行（num=N*R作为分母无法定义）。

**其余产品（Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品、Atlas 训练系列产品、Atlas 推理系列产品）：**

- 数据格式为NC1HWC0/NCHW/NDC1HWC0/NHWC，x为5/6维。
- 统计量sum/square_sum/scale/offset与x同rank、恒为FLOAT32；NC1HWC0/NDC1HWC0下C按C0=16对齐，统计量元素数等于C1*C0。
- 不支持空tensor（各产品proto同声明）。
- 可选属性before_split_ori_shape（listListInt，默认[[]]）/before_split_ori_format（listInt，默认[]）：BN FFTS切分场景（已废弃特性）遗留注册，此时num按切分前原始shape计算。

**通用：** epsilon取值应大于0（host不校验，见参数说明）。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_bn_training_update_v3](./examples/arch35/test_geir_bn_training_update_v3.cpp) | 通过[算子IR](op_graph/bn_training_update_v3_proto.h)构图方式调用BNTrainingUpdateV3算子（含两组shape/epsilon用例）。 |
