# Assert

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                         |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function

- Operator function: asserts that the given condition is true. If the input tensor input_condition evaluates to false, the operator prints the tensor list in input_data.

## Parameters

| Parameter name             | Input/Output/Attribute | Description                                                         | Data type                                                                             | Data format |
|-----------------|----------|------------------------------------------------------------|----------------------------------------------------------------------------------|------|
| input_condition | Input       | Input tensor, the condition to check.                                           | BOOL                                                                             | ND   |
| input_data      | Input       | Dynamic input tensors (variable count), representing the set of tensors to print when the condition is false.             | FLOAT, FLOAT16, INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, BOOL, DOUBLE, STRING | ND   |
| summarize       | Optional attribute     | • Optional integer that specifies printing the first summarize elements of each tensor.<br>• The default value is 3. | INT                                                                              | -    |

## Constraints

- `input_condition` must be a scalar (rank 0, with shape `[]`); a one-dimensional single-element tensor with shape `[1]` does not meet the requirement.

## Invocation Description

| Invocation method | Invocation sample                                                         | Description                                                 |
|--------------|--------------------------------------------------------------|----------------------------------------------------|
| Graph-mode invocation | [test_geir_assert](./examples/test_geir_assert.cpp) | Invokes the Assert operator by building a graph through the [operator IR](./op_graph/assert_proto.h). |
