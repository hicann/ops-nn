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

- 算子功能：根据属性`scale`和`offset`对输入`x`进行反量化（per-tensor标量反量化）。

- 计算公式：
  -`sqrt_mode`为false时：

    $$
    y = cast\_to\_dst\_type((x + offset) * scale)
    $$

  - `sqrt_mode`为true时：

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
      <td><ul><li>表示算子输入的Tensor，对应公式中的x；</li><li>不支持空Tensor；</li><li>shape与输出y一致。</li></ul></td>
      <td>INT8、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>属性</td>
      <td><ul><li>表示反量化中的scale值，对应公式中的scale；</li><li>per-tensor标量。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>属性</td>
      <td><ul><li>表示反量化中的offset值，对应公式中的offset；</li><li>per-tensor标量。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>可选属性</td>
      <td><ul><li>表示输出的数据类型；</li><li>支持取值1、0，分别表示FLOAT16、FLOAT32（与ge::DataType枚举一致）；</li><li>缺省值为FLOAT32（ge::DT_FLOAT，即0）；</li><li>必须与输出y的dtype保持一致。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sqrt_mode</td>
      <td>可选属性</td>
      <td><ul><li>表示scale参与计算的逻辑，对应公式中的sqrt_mode；</li><li>缺省值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td><ul><li>表示反量化的计算输出，对应公式中的y；</li><li>shape和输入x一致。</li></ul></td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

- <term>Ascend 950PR/Ascend 950DT</term>：
  - 支持的(输入`x`dtype，输出`y`dtype)组合：
    - (INT8, FLOAT16)、(INT8, FLOAT32)
    - (HIFLOAT8, FLOAT16)、(HIFLOAT8, FLOAT32)
    - (FLOAT8_E5M2, FLOAT16)、(FLOAT8_E5M2, FLOAT32)
    - (FLOAT8_E4M3FN, FLOAT16)、(FLOAT8_E4M3FN, FLOAT32)

## 约束说明

- `scale`、`offset`为标量属性，仅支持per-tensor反量化。
- 输入`x`与输出`y`的shape必须完全一致，不支持空Tensor，每个维度大小须大于0。
- `dtype`属性仅支持FLOAT16、FLOAT32，且必须与输出`y`的实际dtype一致。
- 仅支持 <term>Ascend 950PR/Ascend 950DT</term>；其他产品形态不支持。
- 当前不提供aclnn接口，仅支持图模式调用。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_ascend_anti_quant](examples/test_geir_ascend_anti_quant.cpp) | 通过GE IR构图方式调用AscendAntiQuant算子（per-tensor scale/offset标量属性）。 |
