# AscendAntiQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：根据属性 `scale` 和 `offset` 对输入 `x` 进行反量化（per-tensor 标量反量化）。

- 计算公式：
  - `sqrt_mode` 为 false 时：

    $$
    y = cast\_to\_dst\_type((x + offset) * scale)
    $$

  - `sqrt_mode` 为 true 时：

    $$
    y = cast\_to\_dst\_type((x + offset) * scale * scale)
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
      <td><ul><li>表示算子输入的 Tensor，对应公式中的 x；</li><li>不支持空 Tensor；</li><li>shape 与输出 y 一致。</li></ul></td>
      <td>INT8、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>属性</td>
      <td><ul><li>表示反量化中的 scale 值，对应公式中的 scale；</li><li>per-tensor 标量。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>属性</td>
      <td><ul><li>表示反量化中的 offset 值，对应公式中的 offset；</li><li>per-tensor 标量。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>可选属性</td>
      <td><ul><li>表示输出的数据类型；</li><li>支持取值 1、0，分别表示 FLOAT16、FLOAT32（与 ge::DataType 枚举一致）；</li><li>缺省值为 FLOAT32（ge::DT_FLOAT，即 0）；</li><li>必须与输出 y 的 dtype 保持一致。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sqrt_mode</td>
      <td>可选属性</td>
      <td><ul><li>表示 scale 参与计算的逻辑，对应公式中的 sqrt_mode；</li><li>缺省值为 false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td><ul><li>表示反量化的计算输出，对应公式中的 y；</li><li>shape 和输入 x 一致。</li></ul></td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

- <term>Ascend 950PR/Ascend 950DT</term>：
  - 支持的 (输入 `x` dtype, 输出 `y` dtype) 组合：
    - (INT8, FLOAT16), (INT8, FLOAT32)
    - (HIFLOAT8, FLOAT16), (HIFLOAT8, FLOAT32)
    - (FLOAT8_E5M2, FLOAT16), (FLOAT8_E5M2, FLOAT32)
    - (FLOAT8_E4M3FN, FLOAT16), (FLOAT8_E4M3FN, FLOAT32)

## 约束说明

- `scale`、`offset` 为标量属性，仅支持 per-tensor 反量化。
- 输入 `x` 与输出 `y` 的 shape 必须完全一致，不支持空 Tensor，每个维度大小须大于 0。
- `dtype` 属性仅支持 FLOAT16、FLOAT32，且必须与输出 `y` 的实际 dtype 一致。
- 仅支持 <term>Ascend 950PR/Ascend 950DT</term>；其他产品形态不支持。
- 当前不提供 aclnn 接口，仅支持图模式调用。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_ascend_anti_quant](examples/test_geir_ascend_anti_quant.cpp) | 通过 GE IR 构图方式调用 AscendAntiQuant 算子（per-tensor scale/offset 标量属性）。 |
