# SmoothL1LossBackward

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

- Operator function: computes the backpropagation of the [aclnnSmoothL1Loss](../smooth_l1_loss_v2/docs/aclnnSmoothL1Loss_en.md) API.
- Formula:

  The backpropagation of SmoothL1Loss can be computed by differentiation. For the first case of SmoothL1Loss, that is, |x-y|<1, the derivative is:
  $$
  \frac{\partial SmoothL1Loss(x,y)}{\partial x} = x - y
  $$

  For the second case of SmoothL1Loss, that is, |x-y|≥1, the derivative is:
  $$
  \frac{\partial SmoothL1Loss(x,y)}{\partial x} = sign(x - y)
  $$

  In this formula, sign(x) denotes the sign function of x, that is:
  $$
  sign(x) =\begin{cases}
  1,& if　x>0 \\
  0,& if　x=0 \\
  -1,& if　x<0
  \end{cases}
  $$

## Parameters

  - gradOut (aclTensor*, computation input): gradient backpropagation input, `SmoothL1Loss` in the formula, an aclTensor on the Device side. The shape must satisfy the [broadcast relationship](../../docs/en/context/broadcast_relationship.md) with self and target, the data type must satisfy the data type deduction rules with the data types of self and target (see [mutual deduction relationship](../../docs/en/context/deduction_relationship.md)), [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, and the [data format](../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: the data types support BFLOAT16, FLOAT16, and FLOAT32.

  - self (aclTensor*, computation input): `x` in the formula, an aclTensor on the Device side. The shape must satisfy the [broadcast relationship](../../docs/en/context/broadcast_relationship.md) with gradOut and target, the data type must satisfy the data type deduction rules with the data types of gradOut and target (see [mutual deduction relationship](../../docs/en/context/deduction_relationship.md)), [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, and the [data format](../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: the data types support BFLOAT16, FLOAT16, and FLOAT32.

  - target (aclTensor*, computation input): `y` in the formula, an aclTensor on the Device side. The shape must satisfy the [broadcast relationship](../../docs/en/context/broadcast_relationship.md) with gradOut and self, the data type must satisfy the data type deduction rules with the data types of gradOut and target (see [mutual deduction relationship](../../docs/en/context/deduction_relationship.md)), [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, and the [data format](../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: the data types support BFLOAT16, FLOAT16, and FLOAT32.

  - reduction (int64_t, computation input): computation attribute that specifies the reduction to apply to the output, an integer on the Host side. The supported values are 0('none')|1('mean')|2('sum'). 'none' means no reduction is applied, 'mean' means the sum of the output is divided by the number of elements in the output, and 'sum' means the output is summed.

  - beta (float, computation input): computation attribute that specifies the value at which the loss changes between L1 and L2. The data type is float, and the value must be non-negative.

  - gradInput (aclTensor*, computation output): the shape is the broadcast result of gradOut, self, and target; [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, and the [data format](../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>: the data types support BFLOAT16, FLOAT16, and FLOAT32.

## Constraints

- Deterministic Computation:

    - aclnnSmoothL1LossBackward is implemented as deterministic by default.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_smooth_l1_loss_backward.cpp](examples/arch35/test_aclnn_smooth_l1_loss_backward.cpp) | Invokes the SmoothL1LossBackward operator through the [aclnnSmoothL1LossBackward](docs/aclnnSmoothL1LossBackward_en.md) API. |
