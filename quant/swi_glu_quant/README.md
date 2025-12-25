# SwiGluQuant

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √    |

## 功能说明

- 算子功能：在SwiGlu激活函数后添加quant操作，实现输入x的SwiGluQuant计算，支持int8或int4量化输出。
- 算子功能差异点说明：相比于aclnnSwiGluQuant接口，aclnnSwiGluQuantV2新增支持groupIndexOptional传入cumsum模式和count模式，通过groupListType控制不同的模式；新增支持非MoE（groupIndexOptional传空）的场景；新增支持int8或int4量化输出yOut，通过dstType控制不同的量化输出数据类型。
- 算子支持范围：当前SwiGluQuant支持MoE场景（传入groupIndexOptional）和非MoE场景（groupIndexOptional传空），SwiGluQuant的输入x和group_index来自于GroupedMatMul算子和MoeInitRouting的输出，通过group_index入参实现MoE分组动态量化、静态per_tensor量化、静态per_channel量化功能。
- MoE场景动态量化计算公式：  

  $$
    Act = SwiGLU(x) = Swish(A)*B \\
    Y_{tmp}^0 = Act[0\colon g[0],\colon] * smooth\_scales[0\colon g[0],\colon], i=0 \\
    Y_{tmp}^i = Act[g[i]\colon g[i+1], \colon] *  smooth\_scales[g[i]\colon g[i+1], \colon], i \in (0, G) \cap \mathbb{Z}\\
    scale=row\_max(abs(Y_{tmp}))/dstTypeScale
  $$

  $$
    Y = Cast(Mul(Y_{tmp}, Scale))
  $$

     其中，A表示输入x的前半部分，B表示输入x的后半部分，g表示group_index，G为group_index的分组数量。int8量化时，$dstTypeScale = 127$（127是int8的最大值）；int4量化时，$dstTypeScale = 7$（7是int4的最大值）。
  
- MoE场景静态量化计算公式：  

  $$
    Act = SwiGLU(x) = Swish(A)*B \\
    Y_{tmp}^0 = Act(0\colon g[0],\colon) * smooth\_scales[0\colon g[0],\colon] + offsets[0\colon g[0],\colon], i=0 \\
    Y_{tmp}^i = Act[g[i]\colon g[i+1], \colon] *  smooth\_scales[g[i]\colon g[i+1], \colon] + offsets[g[i]\colon g[i+1], \colon], i \in (0, G) \cap \mathbb{Z}\\
  $$


  $$
    Y = Cast(Y_{tmp})
  $$

  其中，A表示输入x的前半部分，B表示输入x的后半部分，g表示group_index，G为group_index的分组数量。

- 非MoE场景（groupIndexOptional传空）动态量化计算公式：  

  $$
    Act = SwiGLU(x) = Swish(A)*B \\
    Y_{tmp} = Act* smooth\_scales(0,\colon)\\
    scale=row\_max(abs(Y_{tmp}))/dstTypeScale
  $$

  $$
    Y = Cast(Mul(Y_{tmp}, Scale))
  $$

     其中，A表示输入x的前半部分，B表示输入x的后半部分。int8量化时，$dstTypeScale = 127$（127是int8的最大值）；int4量化时，$dstTypeScale = 7$（7是int4的最大值）。
  
- 非MoE场景（groupIndexOptional传空）静态量化计算公式：  

  $$
    Act = SwiGLU(x) = Swish(A)*B \\
    Y_{tmp} = Act * smooth\_scales(0,\colon) + offsets(0,\colon) \\
  $$


  $$
    Y = Cast(Y_{tmp})
  $$

  其中，A表示输入x的前半部分，B表示输入x的后半部分。

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
      <td>x</td>
      <td>输入</td>
      <td>公式中的输入x。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smoothScalesOptional</td>
      <td>输入</td>
      <td>公式中的smooth_scales变量。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offsetsOptional</td>
      <td>输入</td>
      <td>公式中的输入offsets。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>groupIndexOptional</td>
      <td>输入</td>
      <td>公式中的group_index变量。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>activateLeft</td>
      <td>属性</td>
      <td><ul><li>表示左矩阵是否参与运算。</li><li>用户必须传参。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantModeOptional</td>
      <td>属性</td>
      <td><ul><li>用户必须传参。</li><li>"static"表示静态量化、"dynamic"表示动态量化、"dynamic_msd"表示动态MSD量化。当前仅支持"dynamic"动态量化，"static"静态量化。静态量化仅支持per_tensor量化和per_channel量化。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
      <td>groupListType</td>
      <td>属性</td>
      <td><ul><li>用户必须传参。</li><li>0表示cumsum模式、1表示count模式。当前仅支持0 cumsum模式，1 count模式。</li></ul></td>
      <td>INT64_T</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType</td>
      <td>属性</td>
      <td><ul><li>用户必须传参。</li><li>2表示yOut为int8量化输出、29表示yOut为int4量化输出。当前仅支持输入2和29，默认值是2。</li></ul></td>
      <td>INT64_T</td>
      <td>-</td>
    </tr>
    <tr>
    <tr>
      <td>yOut</td>
      <td>输出</td>
      <td>公式中的y。</td>
      <td>INT8、INT4</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleOut</td>
      <td>输出</td>
      <td>公式中的scale。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    
  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_swi_glu_quant](./examples/test_aclnn_swi_glu_quant.cpp) | 通过[aclnnSwiGluQuant](./docs/aclnnSwiGluQuant.md)接口方式调用SwiGluQuant算子。    |
| aclnn调用 | [test_aclnn_swi_glu_quant_v2](./examples/test_aclnn_swi_glu_quant_v2.cpp) | 通过[aclnnSwiGluQuantV2](./docs/aclnnSwiGluQuantV2.md)接口方式调用SwiGluQuant算子。    |
| 图模式调用 | -   | 通过[算子IR](./op_graph/swi_glu_quant_proto.h)构图方式调用SwiGluQuant算子。 |
