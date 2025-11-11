# AddLayerNormQuant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：LayerNorm算子是大模型常用的归一化操作。AddLayerNormQuant算子将LayerNorm前的Add算子和LayerNorm归一化输出给1个或2个下游的量化算子融合起来，减少搬入搬出操作。LayerNorm下游的量化算子可以是Quantize、AscendQuantV2或是DynamicQuant算子，具体的量化算子类型由attr入参divMode和quantMode决定。当下游有2个量化算子时，2个量化算子的算子类型、输入输出dtype组合和可选输入的组合需要完全一致。
- 计算公式*：

  $$
  x = x1 + x2 + bias
  $$
  
  $$
  y = {{x-E(x)}\over\sqrt {Var(x)+epsilon}} * gamma + beta
  $$
  
  - 当quantMode输入为"static"时，输出outScales1和outScales2无实际意义。取决于divMode的输入，融合的量化算子可能是Quantize或是AscendQuantV2：
    - 当divMode输入为true时，融合的量化算子为Quantize，计算公式如下所示：
  
      $$
      y1 = round(y / scales1 + zeroPoints1)
      $$
  
      $$
      y2 = round(y / scales2 + zeroPoints2), \quad \text{当且仅当scales2存在}
      $$
  
    - 当divMode输入为false时，融合的量化算子为AscendQuantV2，计算公式如下所示：
  
      $$
      y1 = round(y * scales1 + zeroPoints1)
      $$
  
      $$
      y2 = round(y * scales2 + zeroPoints2), \quad \text{当且仅当scales2存在}
      $$
  
  - 当quantMode输入为"dynamic"时，输入zeroPoints1和zeroPoints2无实际意义。融合的量化算子是DynamicQuant，此时divMode无效：
    - 若scales1和scales2均无输入，则y2和scale2输出无实际意义，可忽略。计算公式如下所示：
  
      $$
      outScales1 = row\_max(abs(y))/127
      $$
  
      $$
      y1 = round(y / outScales1)
      $$
  
    - 若仅输入scales1，则y2和scale2输出无实际意义，可忽略。计算公式如下所示：
  
      $$
      tmp1 = y * scales1
      $$
  
      $$
      outScales1 = row\_max(abs(tmp1))/127
      $$
  
      $$
      y1 = round(y / outScales1)
      $$
  
    - 若scales1和scales2均存在，则y2和scale2输出有效。计算公式如下所示：
  
      $$
      tmp1 = y * scales1, \quad tmp2 = y * scales2
      $$
  
      $$
      outScales1 = row\_max(abs(tmp1))/127, \quad outScales2 = row\_max(abs(tmp2))/127
      $$
  
      $$
      y1 = round(y / outScales1),\quad y2 = round(y / outScales2)
      $$
  
    其中row\_max代表对每行求最大值。

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
      <td>表示AddLayerNorm中加法计算的输入，对应公式中的`x1`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示AddLayerNorm中加法计算的输入，对应公式中的`x2`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示层归一化中的gamma参数，对应公式中的`gamma`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>表示层归一化中的beta参数，对应公式中的`beta`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>可选输入</td>
      <td>表示AddLayerNormQuant中加法计算的输入，对应公式中的`bias`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales1</td>
      <td>可选输入</td>
      <td>表示第一个被融合的量化计算子中的scale/smooth输入，对应公式中的`scales1`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales2</td>
      <td>可选输入</td>
      <td>表示第二个被融合的量化计算子中的scale/smooth输入，对应公式中的`scales2`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points1</td>
      <td>可选输入</td>
      <td>表示第一个被融合的量化计算子中的zeroPoints输入，对应公式中的`zeroPoints1`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points2</td>
      <td>可选输入</td>
      <td>表示第二个被融合的量化计算子中的zeroPoints输入，对应公式中的`zeroPoints2`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>可选属性</td>
      <td><ul><li>用于确定融合算子融合的是静态还是动态量化算子，对应公式中的`quantMode`。取值可以是 "static"或是 "dynamic"。</li><li>默认值为"dynamic"。</li></ul></td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>添加到分母中的值，以确保数值稳定，用于防止除0错误，对应公式中的`epsilon`。</li><li>默认值为1e-5f。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additional_output</td>
      <td>可选属性</td>
      <td><ul><li>表示是否开启x=x1+x2+bias的输出。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>div_mode</td>
      <td>可选属性</td>
      <td><ul><li>表示静态量化处理scale的方法是乘法或是除法，对应公式中的`divMode`。</li><li>默认值为true。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的`y1`。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>表示量化输出Tensor，对应公式中的`y2`。</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输出</td>
      <td>表示x1和x2的和，对应公式中的`x`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out_scales1</td>
      <td>输出</td>
      <td>表示通过scales1计算的动态量化缩放结果，对应公式中的`outScales1`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out_scales2</td>
      <td>输出</td>
      <td>表示通过scales2计算的动态量化缩放结果，对应公式中的`outScales2`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式 | [test_geir_add_layer_norm_quant](examples/test_geir_add_layer_norm_quant.cpp)  | 通过[算子IR](op_graph/add_layer_norm_quant_proto.h)构图方式调用AddLayerNormQuant算子。         |


<!--
| aclnn接口  | [test_aclnn_add_layer_norm_quant](examples/test_aclnn_add_layer_norm_quant.cpp) | 通过[aclnnAddLayerNormQuant](docs/aclnnAddLayerNormQuant.md)接口方式调用AddLayerNormQuant算子。 |
-->