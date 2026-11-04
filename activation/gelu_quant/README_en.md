# GeluQuant

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>kirin X90 processor series products</term>    |     √    |
|  <term>Kirin 9030 processor series products</term>    |     √    |

## Function

- Description: fuses GeluV2 with DynamicQuant/AscendQuantV2. After GELU activation is performed on the input data self, the activation result is quantized, and the quantized result is output.

- Formula:

1. First compute GELU to obtain geluOut

    - approximate = tanh

    $$
    geluOut=Gelu(self)=self × Φ(self)=0.5 * self * (1 + Tanh( \sqrt{2 / \pi} * (self + 0.044715 * self^{3})))
    $$

    - approximate = none

    $$
    geluOut=Gelu(self)=self × Φ(self)=0.5 * self *[1 + erf(self/\sqrt{2})]
    $$
2. Then perform the quantization operation on geluOut

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

  - maxValue: the maximum value of the corresponding data type.

    |   DataType    | maxValue |
    | :-----------: | :------: |
    |     INT8      |  127    |
    | FLOAT8_E4M3FN |  448   |
    |  FLOAT8_E5M2  |  57344  |
    |   HIFLOAT8    |  32768   |

## Parameters

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 181px">
  <col style="width: 144px">
  <col style="width: 273px">
  <col style="width: 256px">
  <col style="width: 116px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>The input self in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputScaleOptional</td>
      <td>Input</td>
      <td>The input inputScaleOptional in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputOffsetOptional</td>
      <td>Input</td>
      <td>The input inputOffsetOptional in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>approximate</td>
      <td>Attribute</td>
      <td><ul><li>The mode of the GELU activation function.</li><li>approximate supports only {"none", "tanh"}.</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode</td>
      <td>Attribute</td>
      <td><ul><li>The quantization mode.</li><li>quantMode supports only {"static", "dynamic"}.</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
     <tr>
      <td>roundMode</td>
      <td>Attribute</td>
      <td><ul><li>The data conversion mode.</li><li>The {"rint", "round", "hybrid"} modes are supported.</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
     <tr>
      <td>dstType</td>
      <td>Attribute</td>
      <td><ul><li>The type of y after data conversion.</li><li>The input range is {2, 34, 35, 36}.</li></ul></td>
      <td>INT64_T</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output y in the formula.</td>
      <td>FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>outScaleOptional</td>
      <td>Output</td>
      <td>The outScaleOptional in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- For <term>kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>, the data types of self, inputScaleOptional, and inputOffsetOptional do not support BFLOAT16. y does not support FLOAT8_E4M3FN, FLOAT8_E5M2, or HIFLOAT8.

## Constraints

The data type of inputScaleOptional is consistent with the type of self, or a type with higher precision is used when the types are inconsistent.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_gelu_quant](./examples/arch35/test_aclnn_gelu_quant.cpp) | Invokes the GeluQuant operator through the [aclnnGeluQuant](./docs/aclnnGeluQuant_en.md) API.    |
| Graph-mode invocation | [test_geir_gelu_quant](./examples/test_geir_gelu_quant.cpp)   | Invokes the GeluQuant operator through graph construction using the [operator IR](./op_graph/gelu_quant_proto.h). |
