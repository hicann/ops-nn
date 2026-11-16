# QuantizeAddLayerNorm

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |
|  <term>Kirin X90处理器系列产品</term> | √ |
|  <term>Kirin 9030处理器系列产品</term> | √ |

## 功能说明

- 算子功能：是一种结合了量化、加法和层归一化的操作。依次完成以下操作：
  - 加法：将两个输入张量进行加法操作。
  - 层归一化：对加法后的张量进行层归一化操作。
  - 量化：将输入张量从浮点数量化为低精度整数或浮点数。

- 计算公式：

  $$
  x = x1 + x2 + bias
  $$

  $$
  rstd = {{1}\over\sqrt {Var(x)+eps}}
  $$

  $$
  norm_out = (x-\bar{x}) * rstd * gamma + beta
  $$

  $$
  out = round((norm_out/scales)+zero_points)
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
      <td>表示加法计算的第一个输入，对应公式中的`x1`。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示加法计算的第二个输入，对应公式中的`x2`。shape与`x1`的shape保持一致。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>用于缩放归一化后的输出，对应公式中的`gamma`。必须为一维张量，长度与`x1`的最后一维保持一致。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>用于偏移归一化后的输出，对应公式中的`beta`。shape与`gamma`的shape保持一致。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>表示用于加法计算的偏移量，对应公式中的`bias`。在Ascend 950上，shape可以与`gamma`或`x1`保持一致。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>输入</td>
      <td>表示量化的缩放因子，对应公式中的`scales`。公开axis=-1模式支持`gamma` shape或在其前增加值为1的维度。</td>
      <td>FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points</td>
      <td>可选输入</td>
      <td>表示量化过程中得到y的offset张量，对应公式中的`zero_points`。shape规则与`scales`一致。</td>
      <td>FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>属性</td>
      <td>
      <ul><li>必选属性，指定量化后的输出张量y的数据类型，当前仅支持`DT_INT8`（枚举值2）。</li><li>没有默认值。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>可选属性</td>
      <td><ul><li>表示量化模式。公开接口仅支持-1（per-channel除法）；实现保留-65535（乘法）和65535（per-tensor标量乘法）内部兼容模式，其他值按-1处理。内部兼容模式不构成公开功能承诺。</li><li>默认值为-1。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>添加到分母中的值，以确保数值稳定，对应公式中的`eps`，必须为有限正数。</li><li>默认值为1e-5。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additional_output</td>
      <td>可选属性</td>
      <td><ul><li>表示是否开启x的输出。为false时，输出槽位x的内容未写入，不可读取。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示QuantizeAddLayerNorm的最终输出结果，对应公式中的`out`，shape与`x1`的shape保持一致。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输出</td>
      <td>表示加法运算的输出结果，对应公式中的`x`。shape与`x1`的shape保持一致。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90处理器系列产品</term>、<term>Kirin 9030处理器系列产品</term>：所有输入、输出的数据类型不支持BFLOAT16。

## 约束说明

- 在<term>Ascend 950PR&950DT系列产品</term>上：
  - `x1`至少为一维，`gamma`必须为一维，且长度等于`x1`的最后一维。
  - `x1`和`gamma`不支持零长度维或空Tensor。
  - `x1`、`x2`、`gamma`、`beta`和`bias`的数据类型必须相同。
  - 当`x1`为FLOAT16或FLOAT32时，`scales`和`zero_points`必须为FLOAT32。
  - 当`x1`为BFLOAT16时，`scales`和`zero_points`支持BFLOAT16或FLOAT32。
- `zero_points`存在时，其数据类型必须与`scales`相同。
- 内部`axis=65535`兼容模式接受单元素`scales`/`zero_points`，并兼容`gamma` shape但只读取首元素。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_quantize_add_layer_norm](examples/test_geir_quantize_add_layer_norm.cpp)  | 通过[算子IR](op_graph/quantize_add_layer_norm_proto.h)构图方式调用QuantizeAddLayerNorm算子。         |
