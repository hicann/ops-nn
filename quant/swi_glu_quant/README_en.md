# SwiGluQuant

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         |  Support   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √    |
| <term>Atlas A2 products</term> |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                             |    ×    |
| <term>Atlas training products</term>                              |    ×    |
| <term>kirin X90 processor series products</term> | √ |
| <term>kirin 9030 processor series products</term> | √ |

## Function

- Description: Adds a quant operation after the SwiGlu activation function to implement the SwiGluQuant computation of the input x. int8 or int4 quantized output is supported.
- Operator support scope: currently, SwiGluQuant supports the MoE scenario (with groupIndexOptional passed in) and the non-MoE scenario (with groupIndexOptional left empty). The input x and group_index of SwiGluQuant come from the outputs of the GroupedMatMul operator and MoeInitRouting. Through the group_index input parameter, it implements MoE grouped dynamic quantization, static per_tensor quantization, and static per_channel quantization.
- Formula for dynamic quantization in the MoE scenario:
  $$
    Act = SwiGlu(x) = Swish(A)*B \\
    Y_{tmp}^0 = Act[0\colon g[0],\colon] * smooth\_scales[0\colon g[0],\colon], i=0 \\
    Y_{tmp}^i = Act[g[i]\colon g[i+1], \colon] *  smooth\_scales[g[i]\colon g[i+1], \colon], i \in (0, G) \cap \mathbb{Z}\\
    scale=dstTypeScale/row\_max(abs(Y_{tmp}))
  $$

  $$
    Y = Cast(Mul(Y_{tmp}, Scale))
  $$
     Here, A denotes the first half of the input x, B denotes the second half of the input x, g denotes group_index, and G is the number of groups of group_index. For int8 quantization, $dstTypeScale = 127$ (127 is the maximum value of int8); for int4 quantization, $dstTypeScale = 7$ (7 is the maximum value of int4).

- Formula for static quantization in the MoE scenario:
  $$
    Act = SwiGLU(x) = Swish(A)*B \\
    Y_{tmp}^0 = Act(0\colon g[0],\colon) * smooth\_scales[0\colon g[0],\colon] + offsets[0\colon g[0],\colon], i=0 \\
    Y_{tmp}^i = Act[g[i]\colon g[i+1], \colon] *  smooth\_scales[g[i]\colon g[i+1], \colon] + offsets[g[i]\colon g[i+1], \colon], i \in (0, G) \cap \mathbb{Z}\\
  $$
  $$
    Y = Cast(Y_{tmp})
  $$
  Here, A denotes the first half of the input x, B denotes the second half of the input x, g denotes group_index, and G is the number of groups of group_index.

- Formula for dynamic quantization in the non-MoE scenario (groupIndexOptional left empty):
  $$
    Act = SwiGLU(x) = Swish(A)*B \\
    Y_{tmp} = Act* smooth\_scales(0,\colon)\\
    scale=dstTypeScale/row\_max(abs(Y_{tmp}))
  $$

  $$
    Y = Cast(Mul(Y_{tmp}, Scale))
  $$
     Here, A denotes the first half of the input x, B denotes the second half of the input x. For int8 quantization, $dstTypeScale = 127$ (127 is the maximum value of int8); for int4 quantization, $dstTypeScale = 7$ (7 is the maximum value of int4).

- Formula for static quantization in the non-MoE scenario (groupIndexOptional left empty):
  $$
    Act = SwiGLU(x) = Swish(A)*B \\
    Y_{tmp} = Act * smooth\_scales(0,\colon) + offsets(0,\colon) \\
  $$
  $$
    Y = Cast(Y_{tmp})
  $$
  Here, A denotes the first half of the input x, B denotes the second half of the input x.

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
      <td>x</td>
      <td>Input</td>
      <td>Input x in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smoothScalesOptional</td>
      <td>Input</td>
      <td>The smooth_scales variable in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offsetsOptional</td>
      <td>Input</td>
      <td>Input offsets in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>groupIndexOptional</td>
      <td>Input</td>
      <td>The group_index variable in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>activateLeft</td>
      <td>Attribute</td>
      <td><ul><li>Denotes whether the left matrix participates in the computation.</li><li>Users must pass this parameter.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantModeOptional</td>
      <td>Attribute</td>
      <td><ul><li>Users must pass this parameter.</li><li>"static" indicates static quantization, "dynamic" indicates dynamic quantization, and "dynamic_msd" indicates dynamic MSD quantization. Currently, only "dynamic" dynamic quantization and "static" static quantization are supported. Static quantization supports only per_tensor quantization and per_channel quantization.</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType</td>
      <td>Attribute</td>
      <td><ul><li>Users must pass this parameter.</li><li>0 indicates the cumsum mode, and 1 indicates the count mode. Currently, only 0 (cumsum mode) and 1 (count mode) are supported.</li></ul></td>
      <td>INT64_T</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType</td>
      <td>Attribute</td>
      <td><ul><li>Users must pass this parameter.</li><li>2 indicates that yOut is the int8 quantized output, and 29 indicates that yOut is the int4 quantized output. Currently, only 2 and 29 are supported as inputs, and the default value is 2.</li></ul></td>
      <td>INT64_T</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>Output</td>
      <td>y in the formula.</td>
      <td>INT8, INT4</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleOut</td>
      <td>Output</td>
      <td>scale in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/kirin 9030 processor series products: the `x` data type does not support BFLOAT16; the `y` data type does not support INT4.

## Constraints

None.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_swi_glu_quant](./examples/test_aclnn_swi_glu_quant.cpp) | Invokes the SwiGluQuant operator through the [aclnnSwiGluQuant](./docs/aclnnSwiGluQuant_en.md) interface.    |
| aclnn invocation | [test_aclnn_swi_glu_quant_v2](./examples/test_aclnn_swi_glu_quant_v2.cpp) | Invokes the SwiGluQuant operator through the [aclnnSwiGluQuantV2](./docs/aclnnSwiGluQuantV2_en.md) interface.    |
| Graph-mode invocation | -   | Invokes the SwiGluQuant operator by building a graph through the [operator IR](./op_graph/swi_glu_quant_proto.h). |
