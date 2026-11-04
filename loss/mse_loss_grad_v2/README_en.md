# MseLossGradV2

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: backward propagation of the mean square error loss operator ([MseLossV2](../mse_loss_v2/README.md)), which outputs the gradient of the loss with respect to the predicted value predict.

- Formula:

  When `reduction` is `mean`:

  $$
  MselossBackward(grad, x, y) = grad * (x - y) * 2 / x.numel()
  $$

  where `x.numel()` denotes the number of elements in `x`. If `reduction` is not `mean`:

  $$
  MselossBackward(grad, x, y) = grad * (x - y) * 2
  $$

- where:
  - grad: the gradient (dout) passed back from the upper layer.
  - x: the forward predicted value (predict).
  - y: the forward label value (label).

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>predict</td>
      <td>Input</td>
      <td>Input x in the formula, the forward predicted value. It must satisfy the broadcast relationship with label and dout.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>label</td>
      <td>Input</td>
      <td>Input y in the formula, the forward label value. It must satisfy the broadcast relationship with predict and dout.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dout</td>
      <td>Input</td>
      <td>Input grad in the formula, the gradient passed back from the upper layer. It must satisfy the broadcast relationship with predict and label.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Attribute</td>
      <td>Specifies the computation method of the loss function. Supports none | mean | sum, with the default value mean. 'none' means that no reduction is applied, 'mean' means that the output sum is divided by the number of elements of predict, and 'sum' means that the output is summed.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
     <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output MselossBackward in the formula. Its shape is the result after predict, label, and dout are broadcast, and its dtype is the same as predict.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas inference products</term>: the BFLOAT16 data type is not supported.

## Constraints

- The data types of inputs predict, label, and dout must be consistent, and they must satisfy the broadcast relationship. The shape of output y is the result after the three are broadcast.
- Empty tensor: when any of predict, label, or dout is an empty tensor, the output is an empty tensor (empty in, empty out).
- Deterministic computation:
  - aclnnMseLossBackward is a default deterministic implementation.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_mse_loss_grad_v2](./examples/test_aclnn_mse_loss_grad_v2.cpp) | Invokes the MseLossGradV2 operator through the [aclnnMseLossBackward](./docs/aclnnMseLossBackward_en.md) interface. |

## Reference Resources

- [aclnnMseLossBackward interface document](./docs/aclnnMseLossBackward_en.md)
- [aclnnMseLoss forward interface document](../mse_loss/docs/aclnnMseLoss_en.md)
- [aclnn invocation sample](./examples/test_aclnn_mse_loss_grad_v2.cpp)
