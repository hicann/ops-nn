# DynamicBlockQuant

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: Divides the input tensor into multiple data blocks by using the given row_block_size and col_block_size, and performs quantization with the data block as the basic granularity. In each block, it first computes the quantization parameter scale of the current block, and then quantizes the input based on scale. It outputs the final quantization result and the quantization parameter scale of each block.

- Formula:

  $$
  input\_max = block\_reduce\_max(abs(x))
  $$

  $$
  scale = min((FP8\_MAX/HiF8\_MAX /DST\_TYPE\_MAX / INT8\_MAX) / input\_max, 1/min_scale)
  $$

  $$
  y = cast\_to\_[FP8/HiF8/INT8](x / scale)
  $$

  block\_reduce\_max denotes computing the maximum value in each block.

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
      <td>Tensor input of the operator, corresponding to `x` in the formula. The shape supports 2 to 3 dimensions.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min_scale</td>
      <td>Optional attribute</td>
      <td><ul><li>Minimum scale value involved in the scale computation, corresponding to `min_scale` in the formula. It must be a positive floating-point number.</li><li>Default value: 0.0.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>Rounding mode of quantization. Values rint and round are supported.</li><li>Default value: rint.</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Optional attribute</td>
      <td><ul><li>Data type of output y. Values 2, 34, 35, and 36 are supported, representing INT8, HIFLOAT8, FLOAT8_E5M2, and FLOAT8_E4M3FN, respectively.</li><li>Default value: 35, corresponding to the data type FLOAT8_E5M2.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>row_block_size</td>
      <td>Optional attribute</td>
      <td><ul><li>Number of elements contained in each block along the row dimension. Values 1, 128, 256, and 512 are supported.</li><li>Default value: 1.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>col_block_size</td>
      <td>Optional attribute</td>
      <td><ul><li>Number of elements contained in each block along the column dimension. Values 64, 128, 192, and 256 are supported.</li><li>Default value: 128.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type_max</td>
      <td>Optional attribute</td>
      <td><ul><li>Maximum value of the target data type.</li><li>Currently only the target data type HIFLOAT8 is supported, and the optional values are 0.0, 15.0, 56.0, 224.0, and 32768.0</li><li>0.0 indicates that the original maximum value of the data type is used.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Quantized tensor, corresponding to `y` in the formula. The shape is the same as the shape of input `x`, and the data type is determined by the value of parameter `dst_type`.</td>
      <td>INT8, HIFLOAT8, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Output</td>
      <td><ul><li>Scale tensor used for quantization, corresponding to `scale` in the formula.</li><li>If the shape of input `x` is [M, N], the shape dimensions of output `scaleOut` are [ceil(M/rowBlockSize), ceil(N/colBlockSize)]; if the shape of input `x` is [B, M, N], the shape dimensions of output `scaleOut` are [B, ceil(M/rowBlockSize), ceil(N/colBlockSize)].</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - The shape of parameters `x`, `y`, and `scale` supports only 2 dimensions
    - Parameter `round_mode` supports only rint.
    - Parameter `dst_type` supports only the value 2, representing INT8.
    - Parameter `row_block_size` supports only the values 1 or 128.
    - Parameter `col_block_size` supports only the value 1.
    - Parameter `dst_type_max` supports only the value 0.
    - The data type of parameter `y` supports only INT8.

  - <term>Ascend 950PR&950DT products</term>:
    - The shape of parameters `x`, `y`, and `scale` supports 2 or 3 dimensions.
    - The value of parameter `round_mode` has a mapping with the data type of parameter `y`:
      - When the data type of output `y` is HIFLOAT8, parameter `round_mode` supports round.
      - When the data type of output `y` is INT8, FLOAT8_E4M3FN, or FLOAT8_E5M2, parameter `round_mode` supports rint.
    - Parameter `dst_type` supports the values 2, 34, 35, and 36, representing INT8, HIFLOAT8, FLOAT8_E5M2, and FLOAT8_E4M3FN, respectively.
    - The data type of parameter `y` supports INT8, HIFLOAT8, FLOAT8_E4M3FN, and FLOAT8_E5M2.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_dynamic_block_quant](examples/test_aclnn_dynamic_block_quant.cpp) | Calls the DynamicBlockQuant operator through the [aclnnDynamicBlockQuant](docs/aclnnDynamicBlockQuant_en.md) interface. |
| Graph-mode invocation | -  | Calls the DynamicBlockQuant operator by building a graph with the [operator IR](op_graph/dynamic_block_quant_proto.h).         |

<!--Graph mode supports only the Ascend 950PR&950DT products-->
