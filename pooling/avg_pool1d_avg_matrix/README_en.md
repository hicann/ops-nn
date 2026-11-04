# AvgPool1DAvgMatrix

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>| √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: generates the average matrix weights for the 1D AvgPool backward propagation scenario.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| ---- | ---- | ---- | ---- | ---- |
| `x` | Input | Input tensor. Only the shape and format information is used. | FLOAT, FLOAT16, INT8, INT16, UINT8, INT32, INT64, DOUBLE | NCHW, NHWC, NC1HWC0 |
| `ksize` | Attribute | Pooling window size. | INT | - |
| `strides` | Attribute | Sliding stride. | INT | - |
| `pads` | Attribute | Left and right padding. | ListInt | - |
| `ceil_mode` | Attribute | Whether to use ceil rounding. The default value is `false`. | BOOL | - |
| `count_include_pad` | Attribute | Whether the count includes padding. The default value is `false`. | BOOL | - |
| `y` | Output | Average matrix weight output. | Same as `x`, supporting FLOAT, FLOAT16, INT8, INT16, UINT8, INT32, INT64, DOUBLE | NC1HWC0 |

## Constraints

- `pads` contains at least two elements.
- `strides` cannot be 0.
- The input format supports only `NCHW`, `NHWC`, and `NC1HWC0`.

## Invocation Description

| Invocation method | Invocation sample | Description |
| ---- | ---- | ---- |
| Graph-mode invocation | [test_geir_avg_pool1d_avg_matrix](./examples/test_geir_avg_pool1d_avg_matrix.cpp) | Calls the AvgPool1DAvgMatrix operator by constructing a graph through the [operator IR](./op_graph/avg_pool1d_avg_matrix_proto.h). |
