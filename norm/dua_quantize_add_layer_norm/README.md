# DuaQuantizeAddLayerNorm

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：DuaQuantizeAddLayerNorm是一个复杂的计算，结合了量化、加法和层归一化（Layer Normalization）的功能。

- 计算公式：

  $$
  x = x1 + x2 + bias
  $$

  $$
  rstd = {{1}\over\sqrt {Var(x)+epsilon}}
  $$

  $$
  norm\_out = (x-\bar{x}) * rstd * gamma + beta
  $$

  - 当axis=-65535时：

    $$
    out1 = round((norm\_out * scales1)+zero\_points1)
    $$

    $$
    out2 = round((norm\_out * scales2)+zero\_points2)
    $$

  - 当axis≠-65535时：

    $$
    out1 = round((norm\_out/scales1)+zero\_points1)
    $$

    $$
    out2 = round((norm\_out/scales2)+zero\_points2)
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
      <td>x1</td>
      <td>输入</td>
      <td>为正向融合算子的输入x1，对应公式中的`x1`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>为正向融合算子的输入x2，对应公式中的`x2`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示进行归一化计算的权重，对应公式中的`gamma`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>表示进行归一化中的偏移量，对应公式中的`beta`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>表示DuaQuantizeAddLayerNorm中加法计算的输入，对应公式中的`bias`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales1</td>
      <td>输入</td>
      <td>用于量化第一个输入张量的缩放因子，对应公式中的`scales1`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales2</td>
      <td>输入</td>
      <td>用于量化第二个输入张量的缩放因子，对应公式中的`scales2`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points1</td>
      <td>可选输入</td>
      <td>用于量化第一个输入张量的偏移量，对应公式中的`zero_points1`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points2</td>
      <td>可选输入</td>
      <td>用于量化第二个输入张量的偏移量，对应公式中的`zero_points2`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>属性</td>
      <td><ul><li>指定量化后的输出张量y的数据类型。</li><li>没有默认值。</li><li>当前交付仅支持INT8（y1、y2固定为INT8）。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>可选属性</td>
      <td><ul><li>表示静态量化处理scale的方法是乘法或除法。当axis为-65535时使用乘法，axis取其他值时使用除法。</li><li>默认值为-1。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>添加到分母中的值，以确保数值稳定，对应公式中的`epsilon`。</li><li>默认值为1e-5。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additional_output</td>
      <td>可选属性</td>
      <td><ul><li>表示是否开启x的输出（预留开关）。</li><li>默认值为false。</li><li>当前实现中x为必选输出且始终写出，该属性不影响x的输出。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td>量化后的第一个输出张量，对应公式中的`out1`。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>量化后的第二个输出张量，对应公式中的`out2`。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输出</td>
      <td>表示x1和x2的和，对应公式中的`x`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 数据类型：9 个输入的 dtype 必须一致，取值 FLOAT16、BFLOAT16、FLOAT32。
- 维度：x1（及 x2）的 rank 取值范围为 [1, 8]；rank 为 0 的标量按 shape [1] 处理。
- 归一化维：x1 的最后一维（W = x1.shape[-1]）必须大于 0。
- 输入形状：x2.shape 必须等于 x1.shape。
- 参数形状：gamma、beta、bias、scales1、scales2、zero_points1、zero_points2 必须为 1 维，且长度等于 W。
- 输出：y1、y2、x 的 shape 必须等于 x1.shape；y1、y2 的 dtype 固定为 INT8，x 的 dtype 与 x1 一致。
- 量化参数：axis 不为 -65535（除法模式）时，scales1、scales2 不得为 0；实现对除零不作守卫，结果按 Inf/NaN 语义传播。
- 数值范围：epsilon 无值域校验；当 Var(x)+epsilon 的平方项超出 FLOAT32 可表示范围时按 IEEE 语义传播（例如方差平方上溢为 Inf 时 rstd 为 0，norm_out 退化为 beta）。
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：输入参数和输出参数`x`的数据类型仅支持BFLOAT16。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_dua_quantize_add_layer_norm](examples/test_geir_dua_quantize_add_layer_norm.cpp)  | 通过[算子IR](op_graph/dua_quantize_add_layer_norm_proto.h)构图方式调用DuaQuantizeAddLayerNorm算子。         |
