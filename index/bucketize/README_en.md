# Bucketize

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

- Operator function: computes the interval index of each element in the input tensor based on the attribute `boundaries`.

## Parameters

| Parameter Name | Input/Output/Attribute | Description | Data type | Data format |
| ---- | ---- | ---- | ---- | ---- |
| `x` | Input | The input tensor to be bucketized, with any shape. | INT32, INT64, FLOAT, DOUBLE | ND |
| `boundaries` | Attribute | The bucket boundaries sorted in ascending order. | ListFloat | - |
| `dtype` | Attribute | The output tensor type. INT32 and INT64 are supported. The default value is INT32. | TYPE | - |
| `right` | Attribute | `true` indicates right insertion, and `false` indicates left insertion. The default value is `true`. | BOOL | - |
| `y` | Output | The bucketization result with the same shape as `x`. | INT32, INT64 | ND |

## Constraints

- `boundaries` must be in ascending order.
- The output shape is consistent with the input `x`.

## Invocation Description

| Invocation method | Invocation sample | Description |
| ---- | ---- | ---- |
| Graph-mode invocation | [test_geir_bucketize](./examples/test_geir_bucketize.cpp) | Invokes the Bucketize operator through graph construction using the [operator IR](./op_graph/bucketize_proto.h). |
