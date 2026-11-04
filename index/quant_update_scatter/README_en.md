# QuantUpdateScatter

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

The operator first quantizes updates along the quantAxis dimension: quantScales scales updates, and quantZeroPoints applies the offset. It then updates the values in selfRef at the corresponding positions one by one with the values in the quantized updates along the specified dimension axis, according to the index tensor indices.

## Parameters

  - selfRef (aclTensor*, calculation input|calculation output): a tensor on the Device side, the source data tensor. The data type supports INT8, 3 to 8 dimensions are supported, the [data format](../../docs/en/context/data_format.md) supports ND, and [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported.
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the size of the last dimension must be 32B aligned.
  - indices (aclTensor*, calculation input): a tensor on the Device side, the index tensor. The data types support INT32 and INT64, only 1 or 2 dimensions are supported, the [data format](../../docs/en/context/data_format.md) supports ND, and [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported.\
  When the shape of indices is 1D, the value range of indices is [0, selfRef.shape(axis) - updates.shape(axis));\
  When the shape of indices is 2D, the 0th value of each item of indices has the value range [0, selfRef.shape(0)), and the 1st value of each item of indices has the value range [0, selfRef.shape(axis) - updates.shape(axis)).
  - updates (aclTensor*, calculation input): a tensor on the Device side, the update data tensor. The number of dimensions of updates must match the number of dimensions of selfRef. The size of its 1st dimension equals the size of the 1st dimension of indices and is not greater than the size of the 1st dimension of selfRef. The size of its axis dimension is not greater than the size of the axis dimension of selfRef. The sizes of the other dimensions must equal the sizes of the corresponding dimensions of selfRef. The [data format](../../docs/en/context/data_format.md) supports ND, and [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported.
    - <term>Atlas inference products</term>: the data type supports FLOAT16. The size of the last dimension must be 32B aligned.
    - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: the data types support BFLOAT16 and FLOAT16. The size of the last dimension must be 32B aligned.
    - <term>Ascend 950PR&950DT products</term>: the data types support BFLOAT16 and FLOAT16.
  - quantScales (aclTensor*, calculation input): a tensor on the Device side, the quantization scale tensor. 1 to 8 dimensions are supported. The number of elements of quantScales must equal the size of updates along the quantAxis dimension. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, and the [data format](../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas inference products</term>: the data type supports FLOAT32.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: the data types support BFLOAT16 and FLOAT32.
  - quantZeroPoints (aclTensor*, calculation input): a tensor on the Device side, the quantization offset tensor. 1 to 8 dimensions are supported. The number of elements of quantZeroPoints must equal the size of updates along the quantAxis dimension. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, the [data format](../../docs/en/context/data_format.md) supports ND, and it is an optional parameter.
    - <term>Atlas inference products</term>: the data type supports INT32.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: the data types support BFLOAT16 and INT32.
  - axis (int64_t, calculation input): the dimension on updates used for updating. The data type is INT64, and the value range is [-len(updates.shape) + 1, -1) or [1, len(updates.shape) - 1).
  - quantAxis (int64_t, calculation input): the dimension on updates used for quantization. The data type is INT64, and the supported values are -1 or len(updates.shape) - 1.
  - reduction (int64_t, calculation input): specifies the data operation mode. The data type is INT64, and the supported value is 1 (update).

## Constraints

- The number of dimensions of indices can only be 1 or 2. If it is 2D, the size of its 2nd dimension must be 2. Out-of-bounds indexing is not supported, and out-of-bounds indices are not validated. The selfRef data segments mapped by indices cannot overlap; if they overlap, multiple executions may produce different results due to multi-core concurrency.
- The input data type combinations of selfRef, indices, updates, quantScales, and quantZeroPoints include:
  - <term>Atlas inference products</term>:

    |selfRef|indices|updates|quantScales|quantZeroPoints|
    |---|---|---|---|---|
    |INT8|INT32|FLOAT16|FLOAT32|INT32|
    |INT8|INT64|FLOAT16|FLOAT32|INT32|

  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>:

    |selfRef|indices|updates|quantScales|quantZeroPoints|
    |---|---|---|---|---|
    |INT8|INT32|BFLOAT16|BFLOAT16|BFLOAT16|
    |INT8|INT64|BFLOAT16|BFLOAT16|BFLOAT16|
    |INT8|INT32|FLOAT16|FLOAT32|INT32|
    |INT8|INT64|FLOAT16|FLOAT32|INT32|

## Invocation Example

| Invocation method | Sample code | Description |
| --- | --- | --- |
| aclnn API | [test_aclnn_inplace_quant_scatter](examples/test_aclnn_inplace_quant_scatter.cpp) | Calls the QuantUpdateScatter operator through the [aclnnInplaceQuantScatter](docs/aclnnInplaceQuantScatter_en.md) API. |
