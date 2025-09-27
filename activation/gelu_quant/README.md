# GeluQuant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>昇腾910_95 AI处理器</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品 </term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas 200/300/500 推理产品</term>       |     ×    |



## 功能说明

- 算子功能：将GeluV2与DynamicQuant/AscendQuantV2进行融合，对输入的数据self进行GELU激活后，对激活的结果进行量化，输出量化后的结果。

- 计算公式：
1. 先计算GELU计算得到geluOut
  - approximate = tanh
  $$
  geluOut=Gelu(self)=self × Φ(self)=0.5 * self * (1 + Tanh( \sqrt{2 / \pi} * (self + 0.044715 * self^{3})))
  $$
  - approximate = none
  $$
   geluOut=Gelu(self)=self × Φ(self)=0.5 * self *[1 + erf(self/\sqrt{2})]
  $$
2. 再对geluOut进行量化操作
  - quant_mode = static
  $$
  y = round\_to\_dst\_type(geluOut * inputScaleOptional + inputOffsetOptional, round\_mode)
  $$
  - quant_mode = dynamic
    $$
    geluOut = geluOut * inputScaleOptional
    $$
    $$
    Max = max(abs(geluOut))
    $$
    $$
    outScaleOptional = Max/maxValue
    $$
    $$
    y = round\_to\_dst\_type(geluOut / outScaleOptional, round\_mode)
    $$

  - maxValue: 对应数据类型的最大值。

    |   DataType    | maxValue |
    | :-----------: | :------: |
    |     INT8      |  127    |
    | FLOAT8_E4M3FN |  448   |
    |  FLOAT8_E5M2  |  57344  |
    |   HIFLOAT8    |  32768   |

## 参数说明

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 181px">
  <col style="width: 144px">
  <col style="width: 273px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>self</td>
      <td>输入</td>
      <td>公式中的输入self。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputScaleOptional</td>
      <td>输入</td>
      <td>公式中的输入inputScaleOptional。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputOffsetOptional</td>
      <td>输入</td>
      <td>公式中的输入inputOffsetOptional。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>approximate</td>
      <td>属性</td>
      <td><ul><li>GELU激活函数的模式。</li><li>approximate仅支持{"none", "tanh"}。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
      <td>quantMode</td>
      <td>属性</td>
      <td><ul><li>量化的模式。</li><li>quantMode仅支持{"static", "dynamic"}。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
      <td>roundMode</td>
      <td>属性</td>
      <td><ul><li>数据转换的模式。</li><li>支持{"rint", "round", "hybrid"}模式。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
      <td>dstType</td>
      <td>属性</td>
      <td><ul><li>数据转换后y的类型。</li><li>输入范围为{2, 34, 35, 36}。</li></ul></td>
      <td>INT64_T</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出y。</td>
      <td>FLOAT8_E4M3FN、FLOAT8_E5M2、HiFLOAT8、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>outScaleOptional</td>
      <td>输出</td>
      <td>公式中的outScaleOptional。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

inputScaleOptional的数据类型与self的类型一致，或者在类型不一致时采用精度更高的类型。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_gelu_quant](./examples/test_aclnn_gelu_quant.cpp) | 通过[aclnnGeluQuant](./docs/aclnnGeluQuant.md)接口方式调用GeluQuant算子。    |
| 图模式调用 | -   | 通过[算子IR](./op_graph/gelu_quant_proto.h)构图方式调用GeluQuant算子。 |

