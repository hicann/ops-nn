# DequantSwigluQuant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |

## 功能说明
- 算子功能：在Swish门控线性单元激活函数前后添加dequant和quant操作，实现x的DequantSwigluQuant计算。
- 计算公式：

  $$
  dequantOut = Dequant(x, weightScaleOptional, activationScaleOptional, biasOptional)
  $$

  $$
  swigluOut = Swiglu(dequantOut)=Swish(A)*B
  $$

  $$
  out = Quant(swigluOut, quantScaleOptional, quantOffsetOptional)
  $$

  其中，A表示dequantOut的前半部分，B表示dequantOut的后半部分。


## 参数说明

<table style="undefined;table-layout: fixed; width: 851px"><colgroup>
  <col style="width: 121px">
  <col style="width: 144px">
  <col style="width: 213px">
  <col style="width: 257px">
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
      <td>x</td>
      <td>输入</td>
      <td>输入待处理的数据，公式中的x。</td>
      <td>FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>weightScaleOptional</td>
      <td>输入</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
      <td>activationScaleOptional</td>
      <td>输入</td>
      <td>激活函数的反量化scale。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>biasOptional</td>
      <td>输入</td>
      <td>Matmul的bias，公式中的biasOptional。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quantScaleOptional</td>
      <td>输入</td>
      <td>量化的scale，公式中的quantScaleOptional。</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>quantOffsetOptional</td>
      <td>输入</td>
      <td>量化的offset。</td>
      <td>FLOAT</td>
      <td>ND</td>
     </tr>
      <tr>
      <td>groupIndexOptional</td>
      <td>输入</td>
      <td>MoE分组需要的group_index。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>activateLeft</td>
      <td>属性</td>
      <td>表示是否对输入的左半部分做swiglu激活。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantModeOptional</td>
      <td>属性</td>
      <td>表示使用动态量化。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
     <tr>
      <td>dstType</td>
      <td>属性</td>
      <td>表示指定输出y的数据类型。</td>
      <td>INT64</td>
      <td>-</td>
     </tr>
     <tr>
      <td>roundModeOptional</td>
      <td>输属性</td>
      <td>表示对输出y结果的舍入模式。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
      <tr>
      <td>activateDim</td>
      <td>属性</td>
      <td>表示进行swish计算时，选择的指定切分轴。</td>
      <td>INT64</td>
      <td>-</td>
     </tr>
       <tr>
      <td>yOut</td>
      <td>输出</td>
      <td>-</td>
      <td>INT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleOut</td>
      <td>输出</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - DequantSwigluQuant不包含dstType、roundModeOptional和activateDim参数。

## 约束说明

- 输入x对应activateDim的维度需要是2的倍数，且x的维数必须大于1维。
- 当输入x的数据类型为INT32时，weightScaleOptional不能为空；当输入x的数据类型不为INT32时，weightScaleOptional不允许输入，传入空指针。
- 当输入x的数据类型不为INT32时，activationScaleOptional不允许输入，传入空指针。
- 当输入x的数据类型不为INT32时，biasOptional不允许输入，传入空指针。
- 当输出yOut的数据类型为FLOAT4_E2M1、FLOAT4_E1M2时，yOut的最后一维需要是2的倍数。
- 输出yOut的尾轴不超过5120.

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_dequant_swiglu_quant](./examples/test_aclnn_dequant_swiglu_quant.cpp) | 通过[aclnnDequantSwigluQuant](./docs/aclnnDequantSwigluQuant.md)接口方式调用DequantSwigluQuant算子。    |
| aclnn调用 | - | 通过[aclnnDequantSwigluQuantV2](./docs/aclnnDequantSwigluQuantV2.md)接口方式调用DequantSwigluQuant算子。    |
| 图模式调用 | - | 通过[算子IR](./op_graph/dequant_swiglu_quant_proto.h)构图方式调用DequantSwigluQuant算子。 |

<!-- [test_aclnn_dequant_swiglu_quant_v2](./examples/test_aclnn_dequant_swiglu_quant_v2.cpp) -->