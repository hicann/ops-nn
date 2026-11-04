# DequantBias

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |
|  <term>kirin X90处理器系列产品</term> | √ |
|  <term>kirin 9030处理器系列产品</term> | √ |

## 功能说明

- 算子功能：对输入x反量化操作，将输入的int32的数据转化为FLOAT16/BFLOAT16输出。
- 使用场景：量化感知训练（QAT）与量化推理的反向反量化。`weight_scale`承载per-channel权重量化尺度（沿M维广播），`activate_scale`承载per-token激活量化尺度（沿N维广播），`bias`为量化前回加的偏置；三者组合将int32量化值恢复到浮点域。
- 计算公式（按bias存在性与dtype选择，三者数学不等价）：

  - bias不存在时：

  $$
  y = A \times \text{weight\_scale} \times \text{activate\_scale}
  $$

  - bias存在且为INT32时（整型域先相加再统一缩放）：

  $$
    y = (A + \text{bias}) \times \text{weight\_scale} \times \text{activate\_scale}

  $$

  - bias存在且为浮点（FLOAT/FLOAT16/BFLOAT16）时（反量化后在浮点域加bias）：

  $$
    y = A \times \text{weight\_scale} \times \text{activate\_scale} + \text{bias}

  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>公式中的输入A。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight_scale</td>
      <td>输入</td>
      <td>公式中的输入weight_scale（aclnn接口层名为weightScale）。per-channel权重缩放尺度，shape为[N]，长度与x的N维一致。</td>
      <td>BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activate_scale</td>
      <td>输入（可选）</td>
      <td>公式中的输入activate_scale（aclnn接口层名为activateScaleOptional）。per-token激活缩放尺度，shape为[M]，长度与x的M维一致。可选输入：不参与计算时省略（aclnn层传空指针或空Tensor均视为不存在）。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>bias</td>
      <td>输入（可选）</td>
      <td>公式中的输入bias（aclnn接口层名为biasOptional）。shape为[N]，长度与x的N维一致。可选输入：不参与计算时省略（aclnn层传空指针或空Tensor均视为不存在）。存在且为INT32时走公式2（整型域先加），浮点类型走公式3。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_dtype</td>
      <td>属性</td>
      <td><ul><li>输出y的数据类型（aclnn接口层名为outputDtype）。</li><li>合法值仅1和27：1表示FLOAT16（weight_scale为FLOAT时使用），27表示BFLOAT16（weight_scale为BFLOAT16时使用），与weight_scale数据类型绑定。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出y（aclnn接口层名为out）。shape为[M, N]，与输入x一致。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/kirin 9030处理器系列产品: 不支持BFLOAT16。

## 约束说明

- 输入和输出参数中shape的N和M必须是正整数，且M的取值小于等于25000。
- x必须为2维tensor [M, N]，支持空tensor（M或N为0时返回空输出）。
- weight_scale/bias的长度须等于N，activate_scale的长度须等于M；可选输入（activate_scale/bias）传空tensor视为不存在。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn API | [test_aclnn_dequant_bias](./examples/test_aclnn_dequant_bias.cpp) | 通过[aclnnDequantBias](./docs/aclnnDequantBias.md)接口方式调用DequantBias算子。 |
| GE图模式 | - | 通过[算子IR](./op_graph/dequant_bias_proto.h)构图方式调用DequantBias算子。 |
| PyTorch API | - | 经torch_npu插件注册的`torch.ops.npu.npu_dequant_bias`接口调用（op-plugin已实现，torchair已注册ge.dequant_bias图转换）；样例参见aclnn样例的输入构造方式。 |
