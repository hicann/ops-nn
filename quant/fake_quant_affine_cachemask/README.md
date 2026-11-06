# FakeQuantAffineCachemask

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|<term>Ascend 950PR&950DT系列产品</term>|√|
|<term>Atlas A3系列产品</term>|√|
|<term>Atlas A2系列产品</term>|√|
|<term>Atlas 200I/500 A2推理产品</term>|×|
|<term>Atlas推理系列产品</term>|×|
|<term>Atlas训练系列产品</term>|×|
|<term>kirin X90处理器系列产品</term>|√|
|<term>kirin 9030处理器系列产品</term>|√|

## 功能说明

- 算子功能：对于输入数据x，使用scale和zero_point对输入x在指定轴axis上进行伪量化处理，并根据quant_min和quant_max对伪量化输出进行值域更新，最终返回结果y及对应位置掩码mask。
- 计算公式（与 ATen `fake_quantize_per_*_affine_cachemask` 实测语义一致，按 zero_point 数据类型分支；实现中 zero_point 统一提升到 fp32 域参与计算，x/scale 为真除法）：

  zero_point 为 INT32 时（两步式）：

  $$
  qval = Round(x / scale) + zeroPoint
  $$

  zero_point 为 FLOAT/FLOAT16 时（一步式）：

  $$
  qval = Round(x / scale + zeroPoint)
  $$

  $$
  y = (Min(quantMax, Max(quantMin, qval)) - zeroPoint) * scale
  $$

  $$
  mask = (qval >= quantMin)   \&  (qval <= quantMax)
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
  <col style="width: 250px">
  <col style="width: 200px">
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
      <td>公式中的`x`。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>公式中的`scale`，表示输入伪量化的缩放系数。shape支持0维（标量，仅per-tensor接口与图模式）或1维；per-channel接口size与`x`在axis轴上的长度一致、dtype与`x`相同，per-tensor接口为单元素。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_point</td>
      <td>输入</td>
      <td>公式中的`zero_point`，表示输入伪量化的零基准参数。shape支持0维（标量，仅per-tensor接口与图模式）或1维；per-channel接口size与`x`在axis轴上的长度一致。aclnn per-channel接口支持INT32、FLOAT16、FLOAT，per-tensor接口仅支持INT32；GEIR图模式另支持BFLOAT16（浮点zp及与`x`交叉组合的产品域见下方产品差异说明）。</td>
      <td>INT32（per-tensor aclnn）；INT32、FLOAT、FLOAT16（per-channel aclnn）；INT32、FLOAT、FLOAT16、BFLOAT16（GEIR图模式）</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>属性</td>
      <td>表示计算维度（仅per-channel接口含此参数）。合法范围为[-x.dim(), x.dim()-1]；x为0维（标量）时仅支持0或-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_min</td>
      <td>属性</td>
      <td>表示输入数据伪量化后的最小值。须满足quant_min不大于quant_max，且绝对值小于2^24。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_max</td>
      <td>属性</td>
      <td>表示输入数据伪量化后的最大值。须满足quant_min不大于quant_max，且绝对值小于2^24。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的`y`。维度和计算输入`x`的维度一致，dtype与`x`相同。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输出</td>
      <td>公式中的`mask`。维度和计算输入`x`一致。</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
  </tbody></table>

产品差异说明（按产品）：

- <term>Ascend 950PR&950DT系列产品</term>：GEIR 图模式支持 BFLOAT16 类型及 zero_point 浮点与 `x` 的交叉组合。
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：支持 FLOAT、FLOAT16 及同型浮点 zero_point（亦支持 INT32）。
- <term>kirin X90处理器系列产品</term>、<term>kirin 9030处理器系列产品</term>：zero_point 仅支持 INT32。
- aclnn 接口：x/scale/y 仅支持 FLOAT、FLOAT16；zero_point 在 per-channel 接口支持 INT32、FLOAT16、FLOAT（浮点形态实际可执行域同上），在 per-tensor 接口仅支持 INT32。

## 约束说明

- x 的维度范围为0-8维（ND格式）。支持动态shape（维度值-1）与动态rank（-2占位，推导阶段跳过静态校验，由运行时tiling按实际shape判定）。
- scale与x的数据类型一致。scale/zero_point的shape支持0维（标量，仅per-tensor接口与图模式）或1维；per-channel接口size与x在axis轴上的长度一致，per-tensor接口为单元素。
- quant_min不大于quant_max，且二者绝对值均小于2^24。
- zero_point数值的绝对值应小于2^24（fp32可精确表示的整数域）。超出时zero_point在计算中按fp32舍入，y/mask可能偏离公式一个量化级——该约束为精度建议值，接口不拒绝传入但行为按实现定义处理。
- x为0维（标量）时axis仅支持0或-1。
- 非有限值（NaN/±Inf）输入：与 PyTorch 语义对齐（详见 docs 特殊值语义章节）——mask 恒为 False；out 钳到量化边界，即 NaN/−Inf 时 out=(quant_min−zero_point)·scale，+Inf 时 out=(quant_max−zero_point)·scale。
- 空Tensor输入（任一输入/输出元素数为0）：aclnn接口直接返回成功且不写输出缓冲（workspaceSize=0），调用方不应读取输出。
- <term>Atlas A2系列产品</term>（非regbase路径）图模式直调仅支持axis=0；aclnn入口通过内部Transpose归一化不受此限制，如需axis≠0请使用aclnn接口或预先转置输入。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_fake_quant_per_channel_affine_cachemask.cpp](examples/test_aclnn_fake_quant_per_channel_affine_cachemask.cpp) | 通过[aclnnFakeQuantPerChannelAffineCachemask](docs/aclnnFakeQuantPerChannelAffineCachemask.md)接口方式调用FakeQuantAffineCachemask算子。 |
| aclnn接口  | [test_aclnn_fake_quant_per_tensor_affine_cachemask.cpp](examples/test_aclnn_fake_quant_per_tensor_affine_cachemask.cpp) | 通过[aclnnFakeQuantPerTensorAffineCachemask](docs/aclnnFakeQuantPerTensorAffineCachemask.md)接口方式调用FakeQuantAffineCachemask算子。 |
| 图模式 | - | 通过[算子IR](op_graph/fake_quant_affine_cachemask_proto.h)构图方式调用FakeQuantAffineCachemask算子。         |
