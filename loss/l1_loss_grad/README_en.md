# L1LossGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Computes the backpropagation of [aclnnL1Loss](../lp_loss/docs/aclnnL1Loss_en.md). reduction specifies the computation method of the loss function. 'none', 'mean', and 'sum' are supported. 'none' means that no reduction is applied, 'mean' means that the sum of the output is divided by the number of elements in the output, and 'sum' means that the output is summed.

## Parameters

- gradOutput (aclTensor*, computation input): Backward input gradient, the input `gradOutput` in the formula, an aclTensor on the Device side. The data type of gradOutput must satisfy the data type deduction rules with self and target (see [mutual deduction relationship](../../docs/en/context/deduction_relationship.md)). The shape needs to satisfy the [broadcast relationship](../../docs/en/context/broadcast_relationship.md) with self and target and be less than 9 dimensions. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, and the [data format](../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: The supported data types are BFLOAT16, FLOAT16, and FLOAT32.
- self (aclTensor*, computation input): The input `self` in the formula, an aclTensor on the Device side. The data type of self must satisfy the data type deduction rules with gradOutput and target (see [mutual deduction relationship](../../docs/en/context/deduction_relationship.md)). The shape needs to satisfy the [broadcast relationship](../../docs/en/context/broadcast_relationship.md) with gradOutput and target and be less than 9 dimensions. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, and the [data format](../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: The supported data types are BFLOAT16, FLOAT16, and FLOAT32.
- target (aclTensor*, computation input): The input `target` in the formula, an aclTensor on the Device side. The data type of target must satisfy the data type deduction rules with self and gradOutput (see [mutual deduction relationship](../../docs/en/context/deduction_relationship.md)). The shape needs to satisfy the [broadcast relationship](../../docs/en/context/broadcast_relationship.md) with gradOutput and self and be less than 9 dimensions. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, and the [data format](../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: The supported data types are BFLOAT16, FLOAT16, and FLOAT32.
- reduction (int64_t, computation input): Specifies the computation method of the loss function, an integer on the Host side. The supported values are 0('none') | 1('mean') | 2('sum'). 'none' means that no reduction is applied, 'mean' means that the sum of the output is divided by the number of elements in the output, and 'sum' means that the output is summed.
- gradInput (aclTensor*, computation output): The output `gradInput` in the formula, an aclTensor on the Device side. The data type of gradInput must satisfy the data type deduction rules with self (see [mutual deduction relationship](../../docs/en/context/deduction_relationship.md)). The shape needs to satisfy the broadcast relationship [broadcast relationship](../../docs/en/context/broadcast_relationship.md) with target, self, and gradOutput and be less than 9 dimensions. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, and the [data format](../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: The supported data types are BFLOAT16, FLOAT16, and FLOAT32.

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                                                      |
| ---------------- | --------------------------- |-------------------------------------------------------------------------|
| aclnn invocation  | [test_aclnn_l1_loss_grad.cpp](examples/test_aclnn_l1_loss_grad.cpp) | Calls the L1LossGrad operator through the [aclnnL1LossBackward](docs/aclnnL1LossBackward_en.md) interface. |
