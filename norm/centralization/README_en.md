# Centralization

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: Computes the mean of the input tensor on the specified axes and elementwise subtracts it from the corresponding elements of the input.
- Formula:

  $$
  y = x - mean(x, axes)
  $$

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Tensor to be centralized. | FLOAT, FLOAT16 | ND |
| axes | Attribute | Mean reduction axes, LIST_INT. Negative axes are supported, and the axes must be unique and within `[-rank, rank)`; whether the parameter is optional and its default value are described in the constraints. | LIST_INT | - |
| y | Output | Has the same shape and dtype as x. | FLOAT, FLOAT16 | ND |

## Constraints

- <term>Ascend 950PR&950DT products</term>:
  - The input and output support the FLOAT and FLOAT16 data types and the ND format, and the shape and dtype must be the same.
  - The input rank is 1 to 8. Dynamic shape/rank graph building is supported, and each dimension must already be determined at execution time.
  - Each execution-time dimension and the total number of input elements must not exceed 9223372036854775807 (the int64 upper limit; actually constrained by the device memory).
  - `axes` is optional, with the default value `{-1}`; when it is not set or is set to empty, the last axis is used.
  - Empty tensors are supported, and the output keeps the same shape and dtype as the input.

## Invocation Description

| Invocation method | Sample code | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_centralization](examples/test_geir_centralization.cpp) | Calls the Centralization operator by building a graph through the operator IR. The IR definition is in [centralization_proto.h](op_graph/centralization_proto.h). |
| TensorFlow parsing path | - | Executed through GE parsing with the TensorFlow custom operator `Centralization` (OriginOpType=Centralization). The parameters are the same as those of the operator IR (x input, axes optional attribute, and y output). |

Currently, the GE graph mode and the TensorFlow parsing path are supported, and no dedicated aclnn, PyTorch, or ONNX interface is provided. This operator supports only the ND format and does not support FRACTAL_NZ, FRACTAL_Z, or other formats.
