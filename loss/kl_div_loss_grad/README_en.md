# KlDivLossGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

Operator function: Performs the backward computation of the result of the [aclnnKlDiv](https://gitcode.com/cann/ops-math/blob/master/math/kl_div_v2/docs/aclnnKlDiv.md) API.

## Parameters

- gradOutput(aclTensor*, computation input): An aclTensor on the Device side. The [data format](../../docs/en/context/data_format.md) supports ND. The shape needs to satisfy the [broadcast relationship](../../docs/en/context/broadcast_relationship.md) with self. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: The data types support FLOAT, FLOAT16, and BFLOAT16.
- self(aclTensor*, computation input): An aclTensor on the Device side. The [data format](../../docs/en/context/data_format.md) supports ND. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: The data types support FLOAT, FLOAT16, and BFLOAT16.
- target(aclTensor*, computation input): An aclTensor on the Device side. The [data format](../../docs/en/context/data_format.md) supports ND. The shape needs to satisfy the [broadcast relationship](../../docs/en/context/broadcast_relationship.md) with self. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: The data types support FLOAT, FLOAT16, and BFLOAT16.
- reduction(int64_t, computation input): An int64_t on the Host side. The [data format](../../docs/en/context/data_format.md) supports ND. It specifies the reduction to be applied to the output. 0 ("none") | 1 ("mean") | 2 ("sum") | 3 ("batchmean") are supported. "none" means no reduction is applied, "mean" means the total of the output is divided by the number of elements in the output, "sum" means the output is summed, and "batchmean" means the total of the output is divided by the number of batches.
- logTarget(bool, computation input): A BOOL type on the Host side, indicating whether to convert target to log space.
  - out(aclTensor*, computation output): An aclTensor on the Device side, with the same shape as self. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: The data types support FLOAT, FLOAT16, and BFLOAT16.

## Constraints

- The shape of self must be the same as the shape of out.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_kl_div_loss_grad.cpp](examples/test_aclnn_kl_div_loss_grad.cpp) | Calls the KlDivLossGrad operator through the [aclnnKlDivBackward](docs/aclnnKlDivBackward_en.md) API. |
