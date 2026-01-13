# DequantBias

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：对输入x反量化操作，将输入的int32的数据转化为FLOAT16/BFLOAT16输出。
- 计算公式：

  $$
  y = A \times \text{weight\_scale} \times \text{activate\_scale}
  $$


  $$
    y = (A + \text{bias}) \times \text{weight\_scale} \times \text{activate\_scale}

  $$


  $$
    y = A \times \text{weight\_scale} \times \text{activate\_scale} + \text{bias}

  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
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
      <td>公式中的输入A。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weightScale</td>
      <td>输入</td>
      <td>公式中的输入weight_scale。</td>
      <td>BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activateScaleOptional</td>
      <td>输入</td>
      <td>公式中的输入activate_scale。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
     <tr>
      <td>biasOptional</td>
      <td>输入</td>
      <td>公式中的输入bias。</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputDtype</td>
      <td>属性</td>
      <td><ul><li>输出out的数据类型。</li><li>值为[1,27]。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的输出y。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

输入和输出参数中shape的N和M必须是正整数，且M的取值小于等于25000。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_dequant_bias](./examples/test_aclnn_dequant_bias.cpp) | 通过[aclnnDequantBias](./docs/aclnnDequantBias.md)接口方式调用DequantBias算子。    |
| 图模式调用 | -   | 通过[算子IR](./op_graph/dequant_bias_proto.h)构图方式调用DequantBias算子。 |

