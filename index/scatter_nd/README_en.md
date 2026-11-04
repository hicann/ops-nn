# ScatterNd

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    √  |
| <term>Atlas A3 products</term>    |    √    |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×   |

## Function

Operator function: copies the data of data to out, and at the same time updates the data in out at the specified indices according to updates.

## Parameters

  - data(aclTensor*, computation input): aclTensor on the Device side, with the data type consistent with updates and out, and the shape satisfying 1<=rank(data)<=8. Supports [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md), and the [data format](../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the data types support FLOAT16, FLOAT, BOOL, and BFLOAT16

  - indices(aclTensor*, computation input): aclTensor on the Device side, with the data types supporting INT32 and INT64. indices.shape[-1] <= rank(data), and 1<=rank(indices)<=8. Supports [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md), and the [data format](../../docs/en/context/data_format.md) supports ND. Only non-negative indices are supported. Out-of-bounds index data in indices is not supported.

  - updates(aclTensor*, computation input): aclTensor on the Device side, with the data type consistent with data and out. The shape requires rank(updates)=rank(data)+rank(indices)-indices.shape[-1] -1, and satisfies 1<=rank(updates)<=8. Supports [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md), and the [data format](../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the data types support FLOAT16, FLOAT, BOOL, and BFLOAT16

  - out(aclTensor*, computation output): aclTensor on the Device side, with the data type consistent with data and out, and the shape consistent with data. Supports [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md), and the [data format](../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the data types support FLOAT16, FLOAT, BOOL, and BFLOAT16

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface | [test_aclnn_scatter_nd](examples/arch35/test_aclnn_scatter_nd.cpp) | Calls the ScatterNdUpdate operator through the [aclnnScatterNd](docs/aclnnScatterNd_en.md) interface. |
