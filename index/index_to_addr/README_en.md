# IndexToAddr

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

- Operator function: Generates the address table corresponding to each row within a block based on the block index, the original matrix shape, the block shape, and the base address information.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| ---- | ---- | ---- | ---- | ---- |
| `base_addr` | Input | Base address tensor, with the shape `[2]`. | INT64, UINT64 | ND |
| `x` | Input | Block index tensor, with the shape `[2]`. The first element is the block row index, and the second element is the block column index. | INT64, UINT64 | ND |
| `ori_shape` | Attribute | Original matrix shape, with length 2. | ListInt | - |
| `block_size` | Attribute | Block shape, with length 2. | ListInt | - |
| `ori_storage_mode` | Attribute | Storage mode of the original matrix. Default: `Matrix`. Currently only `Matrix` is supported. | STRING | - |
| `block_storage_mode` | Attribute | Block storage mode. Default: `Matrix`. Currently only `Matrix` is supported. | STRING | - |
| `rank_id` | Attribute | rank id. Default: 0. | INT | - |
| `dtype` | Attribute | Base data type. Default: DT_FLOAT. | TYPE | - |
| `addrs_table` | Output | Address table tensor, with the shape `[block_size[0], 4]`. | INT64, UINT64 | ND |

## Constraints

- The data types of `base_addr` and `x` must be identical.
- `base_addr` and `x` must be 1D tensors with 2 elements.
- `ori_shape` and `block_size` must be ListInt values of length 2.
- `ori_storage_mode` and `block_storage_mode` currently support only `Matrix`.
- `x[0] * block_size[0]` must be less than `ori_shape[0]`, and `x[1] * block_size[1]` must be less than `ori_shape[1]`.
- The number of elements of the output tensor must not be less than `block_size[0] * 4`.

## Invocation Description

| Invocation method | Invocation sample | Description |
| ---- | ---- | ---- |
| Graph-mode invocation | [test_geir_index_to_addr](./examples/test_geir_index_to_addr.cpp) | Invokes the IndexToAddr operator through graph construction with the [operator IR](./op_graph/index_to_addr_proto.h). |
