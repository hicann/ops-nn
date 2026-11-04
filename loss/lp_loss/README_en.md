# LpLoss

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

- Operator function: Computes the mean absolute error (MAE) between each element of the input self and the target target. reduction specifies the reduction to apply to the output. The supported values are 'none', 'mean', and 'sum'. 'none' means no reduction is applied, 'mean' means the sum of the output is divided by the number of elements in the output, and 'sum' means the output is summed.

- Formula:

  When `reduction` is `none`

  $$
  \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
  l_n = \left| x_n - y_n \right|,
  $$

  Where $x$ is self, $y$ is target, and $N$ is the batch size. If `reduction` is not `'none'`, then

  $$
  \ell(x, y) =
  \begin{cases}
  \operatorname{mean}(L), &  \text{if reduction} = \text{'mean';}\\
  \operatorname{sum}(L),  &  \text{if reduction} = \text{'sum'.}
  \end{cases}
  $$

## Parameters

- self (aclTensor*, computation input): Input `self` in the formula, an aclTensor on the Device. Its data type and the data type of target satisfy the data type inference rules (see [mutual inference relationship](../../docs/en/context/deduction_relationship.md)). The shape supports 0-8 dimensions and must satisfy the [broadcast rules](../../docs/en/context/broadcast_relationship.md) with target. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, and the [data format](../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas A2 products</term>, <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>: the supported data types are BFLOAT16, FLOAT16, FLOAT32, and INT64.
- target (aclTensor*, computation input): Input `target` in the formula, an aclTensor on the Device. Its data type and the data type of self satisfy the data type inference rules (see [mutual inference relationship](../../docs/en/context/deduction_relationship.md)). The shape supports 0-8 dimensions and must satisfy the [broadcast rules](../../docs/en/context/broadcast_relationship.md) with self. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, and the [data format](../../docs/en/context/data_format.md) supports ND.
    - <term>Atlas A2 products</term>, <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>: the supported data types are BFLOAT16, FLOAT16, FLOAT32, and INT64.
- reduction (int64_t, computation input, as an attribute): Specifies the input in the formula that determines the reduction to apply to the output, that is, `reduction` in the formula, an integer on the Host. The supported values are 0('none') | 1('mean') | 2('sum'). 'none' means no reduction is applied, 'mean' means the sum of the output is divided by the number of elements in the output, and 'sum' means the output is summed.
- out (aclTensor*, computation output): `out` in the formula, an aclTensor on the Device. Its data type must be convertible from the data type inferred between self and target (see [mutual conversion relationship](../../docs/en/context/conversion_relationship.md)). [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, and the [data format](../../docs/en/context/data_format.md) supports ND. When the value of reduction is 0, the shape of out is the same as the shape of self and target after broadcasting; when the value of reduction is not 0, the shape of out supports 0 dimensions.
    - <term>Atlas A2 products</term>, <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>: the supported data types are BFLOAT16, FLOAT16, FLOAT32, INT64, COMPLEX64, and COMPLEX128.

## Constraints

- Deterministic Computation:
    - aclnnL1Loss is deterministic by default.
- In LpLoss, p is the parameter for computing the loss, and only p=1 is supported. The aclnn interface name is aclnnL1Loss.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_l1_loss.cpp](examples/test_aclnn_l1_loss.cpp) | Invokes the LpLoss operator through the [aclnnL1Loss](docs/aclnnL1Loss_en.md) interface. |
