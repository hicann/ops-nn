# GroupedDynamicBlockQuant

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: based on the passed-in start values of group indices (group_list), it quantizes each group at the basic block granularity into FP8/HiFP8, and outputs the quantization parameter scale (FP32).

- Formula:

  $$
    input\_max = block\_reduce\_max(abs(input))
  $$

  $$
    scale = min(input\_max/(FP8\_MAX/HiF8\_MAX/DST\_TYPE\_MAX), 1/min\_scale)
  $$

  $$
    y = cast\_to\_[HiF8/FP8](input/scale)
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1401px"><colgroup>
  <col style="width: 149px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 200px">
  <col style="width: 100px">
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
      <td>Denotes the input tensor of the operator, shaped [M, N] or [B, M, N], corresponding to input in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_list</td>
      <td>Input</td>
      <td>Denotes the offset of each group on the M axis (cumsum mode).</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min_scale</td>
      <td>Input</td>
      <td>Denotes the minimum scale value that takes part in the scale computation, corresponding to min_scale in the formula.</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Optional attribute</td>
      <td>Denotes the approximation mode for casting the high-bit data to the target data type at the end.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Input</td>
      <td>Denotes the data type of y after the data conversion.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>row_block_size</td>
      <td>Input</td>
      <td>Denotes the quantization granularity specified on the M axis.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>col_block_size</td>
      <td>Input</td>
      <td>Denotes the quantization granularity specified on the N axis.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group_list_type</td>
      <td>Input</td>
      <td>Denotes the function type of group_list.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type_max</td>
      <td>Optional attribute</td>
      <td><ul><li>The maximum value of the target data type.</li><li>Currently only the target data type HIFLOAT8 is supported, and the selectable value range is [0.0, 32768.0]</li><li>0.0 indicates that the original maximum value of the data type is used.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Denotes the quantized output tensor, corresponding to y in the formula.</td>
      <td>HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Output</td>
      <td>Denotes the quantization scale corresponding to each group, corresponding to scale in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| aclnn API | [test_aclnn_grouped_dynamic_block_quant](examples/arch35/test_aclnn_grouped_dynamic_block_quant.cpp) | Calls the GroupedDynamicBlockQuant operator through the [aclnnGroupedDynamicBlockQuant](docs/aclnnGroupedDynamicBlockQuant_en.md) API. |
| aclnn API | [test_aclnn_grouped_dynamic_block_quant_v2](examples/arch35/test_aclnn_grouped_dynamic_block_quant_v2.cpp) | Calls the GroupedDynamicBlockQuant operator through the [aclnnGroupedDynamicBlockQuantV2](docs/aclnnGroupedDynamicBlockQuantV2_en.md) API. |
| Graph mode | - | Calls the GroupedDynamicBlockQuant operator by building the graph through the [operator IR](op_graph/grouped_dynamic_block_quant_proto.h). |
