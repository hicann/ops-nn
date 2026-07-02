# SwigluGroupQuant

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：融合实现SwiGLU激活和分组低比特量化，支持FP8和FP4输出。输入`x`的最后一维被均分为`A`和`B`，先计算`silu(A) * B`，再执行量化。

- 计算公式：

  $$
  y_{tmp}=silu(A) \times B
  $$

  当传入`clamp_limit`时：

  $$
  A=min(A, clamp\_limit)
  $$

  $$
  B=min(max(B, -clamp\_limit), clamp\_limit)
  $$

  当传入`weight`时，量化前执行：

  $$
  y_{tmp}=y_{tmp} \times weight
  $$

  进行量化：

  $$
    scale=row\_max(abs(y_{tmp}))/dstTypeScale
  $$

  $$
    y = Cast(Mul(y_{tmp}, 1/scale))
  $$

  quant_mode为0时输出FP8类型的y和FLOAT32类型的y_scale；quant_mode为1时输出FP8/FP4类型的y和FLOAT8_E8M0类型的y_scale；quant_mode为2、3时输出HIFP8类型的y和FLOAT32类型的y_scale。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 100px">
  <col style="width: 420px">
  <col style="width: 240px">
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
      <td>输入张量，shape为[...,D]，最后一维D按左右两半做SwiGLU。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>可选输入</td>
      <td>量化前按token乘到SwiGLU输出上的权重。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>可选输入</td>
      <td>count模式的group token数。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>可选输入</td>
      <td>静态量化输入的scale张量。仅quant_mode=2时使用，quant_mode=3时不使用。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>属性</td>
      <td>目标量化类型：27=HIFLOAT8，35=FLOAT8_E5M2，36=FLOAT8_E4M3FN，40=FLOAT4_E2M1，41=FLOAT4_E1M2。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>属性</td>
      <td>量化模式。0表示Block FP8，1表示MX，2表示HIFP8静态量化，3表示HIFP8动态量化。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>属性</td>
      <td>量化块大小。0表示使用默认值；Block FP8支持128，MX支持32；quant_mode=2或3时不生效。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_scale</td>
      <td>属性</td>
      <td>是否将scale取整为2的幂。MX模式必须为true；quant_mode=2或3时不生效。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clamp_limit</td>
      <td>属性</td>
      <td>SwiGLU计算前的clamp阈值。默认不启用clamp。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type_max</td>
      <td>属性</td>
      <td>目标量化类型的最大有限值。quant_mode=2或3时用于计算scale = amax / dst_type_max，默认值为15.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_origin</td>
      <td>属性</td>
      <td>是否输出量化前的SwiGLU结果。MX FP4模式下该输出仅作占位。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>量化输出。FP8 shape为[...,D/2]；FP4每字节打包2个4-bit值，shape为[...,D/4]；HIFP8 shape为[...,D/2]。</td>
      <td>HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1、FLOAT4_E1M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_scale</td>
      <td>输出</td>
      <td>量化scale。Block FP8输出FLOAT32，shape为[...,ceil((D/2)/128)]；MX输出FLOAT8_E8M0，shape为[...,ceil(ceil((D/2)/32)/2),2]；HIFP8输出FLOAT32，无group_index时shape为[1]，有group_index时shape为[G]。</td>
      <td>FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_origin</td>
      <td>输出</td>
      <td>量化前的SwiGLU结果，shape为[...,D/2]。</td>
      <td>与x相同</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入`x`的rank必须大于0，最后一维`D`必须大于等于256且能被256整除。
- `dst_type`支持`HIFLOAT8`、`FLOAT8_E4M3FN`、`FLOAT8_E5M2`、`FLOAT4_E2M1`、`FLOAT4_E1M2`。
- `quant_mode=0`时仅支持FP8输出，`block_size`支持0或128。
- `quant_mode=1`时支持FP8/FP4输出，`block_size`支持0或32，`round_scale`必须为true。
- `quant_mode=2`时支持HIFP8静态量化输出，`dst_type`、`block_size`和`round_scale`不生效。
- `quant_mode=3`时支持HIFP8动态量化输出，`dst_type`、`block_size`和`round_scale`不生效。
- `dst_type`为`FLOAT4_E2M1`或`FLOAT4_E1M2`时，必须使用`quant_mode=1`。
- `dst_type`为`HIFLOAT8`时，必须使用`quant_mode=2`或`3`。
- `y_scale`的数据类型必须与`quant_mode`匹配：Block FP8为FLOAT32，MX为FLOAT8_E8M0，HIFP8为FLOAT32。
- `quant_mode=2`或`3`时，`group_index`可用于MoE场景的分组量化，`y_scale`的shape为[G]（G为group数量）。
- `clamp_limit`不启用时使用默认占位值-1.0；启用时必须大于0。

## 调用说明

|调用方式|调用样例|说明|
|:-------|:-------|:---|
|aclnn调用|[test_aclnn_swiglu_group_quant](./examples/test_aclnn_swiglu_group_quant.cpp)|通过[aclnnSwigluGroupQuant](./docs/aclnnSwigluGroupQuant.md)接口调用SwigluGroupQuant算子。|
|图模式调用|-|通过[算子IR](./op_graph/swiglu_group_quant_proto.h)构图方式调用SwigluGroupQuant算子。|
