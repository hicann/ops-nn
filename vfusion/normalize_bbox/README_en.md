# NormalizeBBox

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>| × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: Converts the absolute coordinates of the object detection boxes into the relative scale according to the height and width of each image.
  The operator does not truncate the results. Therefore, when the input boxes exceed the image range, the output may also exceed `[0, 1]`.
- Formula:

  For batch `b`, let $h=shape\_hw[b,0]$ and $w=shape\_hw[b,1]$. Then each box is computed as follows:

  $$
  y_{b,i} = boxes_{b,i} / [h, w, h, w]
  $$

  `reversed_box` only changes the position of the coordinate axis and does not change the computation semantics:

  - `reversed_box=false`: The coordinate axis is the last dimension, and the shape of `boxes` is `[batch, ..., 4]`.
  - `reversed_box=true`: The coordinate axis is the first dimension, and the shape of `boxes` is `[batch, 4, ...]`.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| boxes | Input | Absolute coordinates of the object detection boxes, corresponding to `boxes` in the formula. | FLOAT, FLOAT16 | ND |
| shape_hw | Input | Image size of each batch, with the shape `[batch, 3]`. The first two elements are the height and width respectively, and the third element is a reserved position. | INT32 | ND |
| y | Output | Converted relative coordinates. The shape and data type are consistent with `boxes`. | FLOAT, FLOAT16 | ND |
| reversed_box | Optional attribute | Specifies the position of the coordinate axis of `boxes`. The default value is false. | BOOL | - |

### Product Difference Description

| Product | Static shape capability | Dynamic shape capability | Shape/rank and empty tensor restrictions |
| --- | --- | --- | --- |
| <term>Ascend 950PR&950DT products</term> | Both inputs and outputs are ND | Dynamic shapes and dynamic ranks are supported, and both inputs and outputs are ND | The rank of `boxes` is 2 to 8; empty tensors with a batch or a non-coordinate dimension of 0 are supported, and the coordinate dimension must be 4 |

## Constraints

- The batch of `shape_hw` must be consistent with the batch of `boxes`.
- When `reversed_box=false`, the last dimension of `boxes` must be 4. When `reversed_box=true`,
  the first dimension of `boxes` must be 4.
- The height and width in `shape_hw` are directly used as divisors, and the operator does not check their value ranges.
- <term>Ascend 950PR&950DT products</term>:
  - Data format: The inputs and outputs of static shapes and dynamic shapes are all ND.
  - Dynamic shape capability: Dynamic shapes and dynamic ranks are supported.
  - Input shape constraint: The rank of `boxes` is 2 to 8; empty tensors with a batch or a non-coordinate dimension of 0 are supported, and the coordinate dimension must be 4.
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, and <term>Atlas training products</term>:
  - Data format: The inputs and outputs of static shapes are all ND.
  - Dynamic shape capability: Dynamic shapes are not supported.
  - Input shape constraint: Only 3-dimensional `boxes` are supported, and empty tensors are not supported.

## Invocation Description

This operator does not provide an aclnn single-operator interface and is invoked in graph mode of the Graph Engine (GE).

| Invocation method | Invocation sample | Description |
| --- | --- | --- |
| GE graph mode | [test_geir_normalize_bbox.cpp](examples/arch35/test_geir_normalize_bbox.cpp) | Builds a graph through the [NormalizeBBox operator IR](op_graph/normalize_bbox_proto.h) and invokes the operator. |
