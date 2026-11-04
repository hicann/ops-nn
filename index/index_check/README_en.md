# IndexCheck

English | [简体中文](./README.md)

## Applicable Products

| Product                                           | Support |
| :------------------------------------------- | :--: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: checks whether an index is out of bounds. When an index exceeds the boundary value range, the operator triggers a hardware exception and terminates the program.
- Application scenario: it is used by index-type aclnn l2 APIs to perform an index validity check before the index operation is executed.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| :--- | :--- | :--- | :--- | :--- |
| bounds | Input | Boundary values of the dimensions (that is, the size of each dimension). The length must be consistent with the number of tensors in the indices list. | INT64 | ND |
| indices_list | Input | Index tensor list (variable-length input). Each tensor in the list corresponds to a boundary value in bounds in order. | INT32, INT64 | ND |

## Constraints

- bounds must be a 1-dimension tensor, and the length range is 1～8.
- The length of bounds must equal the number of tensors in the indices list.
- The indices list cannot be empty, and the number of tensors in the list cannot exceed 8.
- The data types of all tensors in the indices list must be consistent (all INT32 or all INT64).
- Each tensor in the indices list supports 0～8 dimensions and empty tensors; tensors with 9 or more dimensions are not supported.
- An empty indices tensor does not trigger the index check; an empty indices list is an invalid input.
- Index values in indices support negative numbers.
- If an index is out of bounds, the operator triggers a hardware exception and terminates the program.

## Invocation Description

IndexCheck is called internally by index-type aclnn l2 APIs (such as aclnnIndex and aclnnIndexPutV2) through the l0op API (`l0op::IndexCheck` in `op_api/index_check.h`) during the `GetWorkspaceSize` stage, to perform an index validity check before the index operation is executed. This operator has no independent external aclnn single-operator API.
