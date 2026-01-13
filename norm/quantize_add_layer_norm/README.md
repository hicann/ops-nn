# QuantizeAddLayerNorm

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

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
      <td>用于缩放归一化后的输出，对应公式中的`gamma`。shape与`x1`的shape最后一维保持一致。</td>
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
      <td>表示用于加法计算的偏移量，对应公式中的`bias`。shape与`gamma`的shape保持一致。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>输入</td>
      <td>表示量化的缩放因子，对应公式中的`scales`。shape与`gamma`的shape保持一致。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points1</td>
      <td>可选输入</td>
      <td>表示量化过程中得到y的offset张量，对应公式中的`zero_points1`。shape与`gamma`的shape保持一致。</td>
      <td>FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>必选属性</td>
      <td>
      <ul><li>指定量化后的输出张量y的数据类型。</li><li>没有默认值。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>可选属性</td>
      <td><ul><li>表示需要进行量化的element-wise轴，其他的轴做broadcast，指定的轴不能超过输入x的维度数。当前仅支持-1，传其他值均不生效。</li><li>默认值为-1。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>添加到分母中的值，以确保数值稳定，对应公式中的`eps`。</li><li>默认值为1e-5f。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additional_output</td>
      <td>可选属性</td>
      <td><ul><li>表示是否开启x的输出。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示QuantizeAddLayerNorm的最终输出结果，对应公式中的`out`。</td>
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

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_quantize_add_layer_norm](examples/test_geir_quantize_add_layer_norm.cpp)  | 通过[算子IR](op_graph/quantize_add_layer_norm_proto.h)构图方式调用QuantizeAddLayerNorm算子。         |