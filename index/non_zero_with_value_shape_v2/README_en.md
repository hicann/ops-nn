# NonZeroWithValueShapeV2

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

- Operator function: Updates the output shapes of `value` and `index` according to the `count` input of `NonZeroWithValue`.

## Parameters

| Parameter name | Input/Output | Description | Data type | Data format |
| ---- | ---- | ---- | ---- | ---- |
| `value` | Input | The value output of `NonZeroWithValue`. | DOUBLE, FLOAT, FLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BOOL | ND |
| `index` | Input | The index output of `NonZeroWithValue`. | INT32 | ND |
| `count` | Input | Number of non-zero elements. | INT32 | ND |
| `value` | Output | The value output after the shape update, with shape `[count[0]]`. | DOUBLE, FLOAT, FLOAT16, INT8, UINT8, INT16, UINT16, INT32, UINT32, INT64, UINT64, BOOL | ND |
| `index` | Output | The index output after the shape update, with shape `[2, count[0]]`. | INT32 | ND |

## Constraints

- The data types of `index` and `count` must be INT32.
- The `count` input contains at least 1 INT32 element.

## Invocation Description

| Invocation method | Invocation sample | Description |
| ---- | ---- | ---- |
| Graph-mode invocation | [test_geir_non_zero_with_value_shape_v2](./examples/test_geir_non_zero_with_value_shape_v2.cpp) | Calls the NonZeroWithValueShapeV2 operator through graph construction with the [operator IR](./op_graph/non_zero_with_value_shape_v2_proto.h). |
