# INInferV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     √    |
|  <term>Atlas 推理系列产品</term>    |     v    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：实例归一化推理（Instance Normalization Inference）。给定每个(N, C)的统计量mean/variance，对输入x做归一化，并可选地做gamma/beta仿射变换；同时将mean/variance透传到batch_mean/batch_variance输出。与[InstanceNorm](../instance_norm/README.md)相比，本算子不在算子内计算统计量，而是直接使用外部给定的mean/variance，常用于图融合场景。

- 计算公式：

  gamma/beta提供时：

  $$
  y = (x - mean) * {\gamma\over\sqrt {variance + ε}} + \beta
  $$

  gamma/beta不提供时：

  $$
  y = {{x - mean}\over\sqrt {variance + ε}}
  $$

  透传输出：

  $$
  batch\_mean = mean
  $$

  $$
  batch\_variance = variance
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
      <td><ul><li>表示归一化计算的输入张量，对应公式中的`x`。</li><li>shape为[N, C, R...]，支持≥2维，dim0为N、dim1为C、后导维展平为归一化轴R。</li><li>fp16输入在算子内升fp32计算、单次舍入写回。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>可选输入</td>
      <td><ul><li>表示仿射缩放参数，对应公式中的`γ`。</li><li>元素数必须等于N*C。</li><li>必须与beta同时提供或同时不提供。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>可选输入</td>
      <td><ul><li>表示仿射偏移参数，对应公式中的`β`。</li><li>元素数必须等于N*C。</li><li>必须与gamma同时提供或同时不提供。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>可选输入</td>
      <td><ul><li>表示每个(N, C)的均值，对应公式中的`mean`。</li><li>元素数必须等于N*C。</li><li>proto/def层声明为可选（与canndev逐字一致），实际必须提供，缺失时由tiling/infershape拦截报错。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>variance</td>
      <td>可选输入</td>
      <td><ul><li>表示每个(N, C)的方差，对应公式中的`variance`。</li><li>元素数必须等于N*C。</li><li>proto/def层声明为可选（与canndev逐字一致），实际必须提供，缺失时由tiling/infershape拦截报错。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>表示添加到variance上的小量，以确保数值稳定，对应公式中的`ε`。</li><li>默认值为1e-5。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td><ul><li>表示归一化结果，对应公式中的`y`。</li><li>shape与数据类型均与`x`一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_mean</td>
      <td>可选输出</td>
      <td><ul><li>表示mean的透传拷贝，对应公式中的`batch_mean`。</li><li>shape与`mean`一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_variance</td>
      <td>可选输出</td>
      <td><ul><li>表示variance的透传拷贝，对应公式中的`batch_variance`。</li><li>shape与`variance`一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 仅支持ND格式（dim0=N、dim1=C、后导维为归一化轴R；图模式下NCHW标签会被框架归一化下发，布局相同）。
- gamma/beta/mean/variance的元素数必须等于N*C；gamma与beta必须同时提供或同时不提供。
- mean/variance在proto/def层为可选声明，但实际必须提供。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_in_infer_v2](./examples/arch35/test_geir_in_infer_v2.cpp) | 通过[算子IR](op_graph/in_infer_v2_proto.h)构图方式调用INInferV2算子（含gamma/beta全输入与缺席两种图）。 |
