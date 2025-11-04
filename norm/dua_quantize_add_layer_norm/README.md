# DuaQuantizeAddLayerNorm

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

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
      <td>可选输入</td>
      <td>表示DuaQuantizeAddLayerNorm中加法计算的输入，对应公式中的`bias`。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales1</td>
      <td>可选输入</td>
      <td>用于量化第一个输入张量的缩放因子，对应公式中的`scales1`。</td>
      <td>BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales2</td>
      <td>可选输入</td>
      <td>用于量化第二个输入张量的缩放因子，对应公式中的`scales2`。</td>
      <td>BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points1</td>
      <td>可选输入</td>
      <td>用于量化第一个输入张量的偏移量，对应公式中的`zero_points1`。</td>
      <td>INT8、UINT8、BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points2</td>
      <td>可选输入</td>
      <td>用于量化第二个输入张量的偏移量，对应公式中的`zero_points2`。</td>
      <td>INT8、UINT8、BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>必选属性</td>
      <ul><li>指定量化后的输出张量y的数据类型。</li><li>没有默认值。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>可选属性</td>
      <td><ul><li>表示需要进行量化的elewise轴，其他的轴做broadcast，指定的轴不能超过输入x的维度数。当前仅支持-1，传其他值均不生效。</li><li>默认值为-1。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>添加到分母中的值，以确保数值稳定，对应公式中的`epsilon`。</li><li>默认值为1e-5f。</li></ul></td>
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
      <td>y1</td>
      <td>输出</td>
      <td>量化后的第一个输出张量，对应公式中的`out1`。</td>
      <td>INT8、UINT8、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>量化后的第二个输出张量，对应公式中的`out2`。</td>
      <td>INT8、UINT8、INT32</td>
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

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_dua_quantize_add_layer_norm](examples/test_geir_dua_quantize_add_layer_norm.cpp)  | 通过[算子IR](op_graph/dua_quantize_add_layer_norm_proto.h)构图方式调用DuaQuantizeAddLayerNorm算子。         |

<!--[test_geir_dua_quantize_add_layer_norm](examples/test_geir_dua_quantize_add_layer_norm.cpp)-->