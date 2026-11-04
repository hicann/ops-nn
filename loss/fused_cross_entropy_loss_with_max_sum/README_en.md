# FusedCrossEntropyLossWithMaxSum

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

- Operator function: this operator is part of the cross entropy computation module in the vocabulary parallel scenario. It solves the memory and computation efficiency problems under an ultra-large vocabulary. The current part computes the loss and the softMax result.
- Formula:

    $$
    lossOut = log(sum\_exp\_logits) - predicted\_logits
    $$

    $$
    softMaxOutOptional = exp(vocab\_parallel\_logits -logits\_max.unsqueeze(dim = -1)) \ sum\_exp\_logits.unsqueeze(dim = -1)
    $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 502px">
  <col style="width: 213px">
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
      <td>logitsMax</td>
      <td>Input</td>
      <td>Maximum value of each row after the matmul computation, corresponding to logitsMax in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sumExpLogits</td>
      <td>Input</td>
      <td>Result of exp after the matmul computation result subtracts the maximum value of each row. Corresponding to sumExpLogits in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>predictedLogits</td>
      <td>Input</td>
      <td>Indicates the result of predictedLogits after the matmul computation result subtracts the maximum value of each row and is filtered by maskedTargetOut. Corresponding to predictedLogits in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputOptional</td>
      <td>Input</td>
      <td>Left matrix of the matmul input.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weightOptional</td>
      <td>Input</td>
      <td>Right matrix of the matmul input.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vocabParallelLogitsOptional</td>
      <td>Input</td>
      <td>Matmul computation result.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>labelSmoothing</td>
      <td>Attribute</td>
      <td>Label smoothing coefficient, used to mitigate overfitting.</td>
      <td>float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>lossOut</td>
      <td>Output</td>
      <td>Intermediate variable. Corresponding to loss in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softMaxOutOptional</td>
      <td>Output</td>
      <td>Intermediate variable. Corresponding to vocabParallelLogits in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface | [test_aclnn_fused_cross_entropy_loss_with_max_sum](examples/test_aclnn_fused_cross_entropy_loss_with_max_sum.cpp) | Invokes the FusedCrossEntropyLossWithMaxSum operator through the [aclnnFusedCrossEntropyLossWithMaxSum](docs/aclnnFusedCrossEntropyLossWithMaxSum_en.md) interface. |
