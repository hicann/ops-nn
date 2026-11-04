# CrossEntropyLossGrad

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

- **Operator function**: Backpropagation of aclnnCrossEntropyLoss.
- **Formula**:
  $$
  ignoreMask_{target(t)}=\begin{cases}
  1, &target(t) ≠ ignoreIndex \\
  0, &target(t) = ignoreIndex
  \end{cases}
  $$

  $$
  smoothLossGrad=\begin{cases}
  grad / sum(weight_{target}* ignoreMask) * labelSmoothing / C, &reduction = mean \\
  grad * labelSmoothing / C, &reduction = sum \\
  grad * labelSmoothing / C, &reduction = none
  \end{cases}
  $$

  $$
  lossOutGrad=\begin{cases}
  grad * (1-labelSmoothing) / sum(weight_{target}* ignoreMask) * ignoreMask, &reduction = mean \\
  grad * (1-labelSmoothing) * ignoreMask, &reduction = sum \\
  grad * (1-labelSmoothing) * ignoreMask, &reduction = none
  \end{cases}
  $$

  $$
  nllLossGrad = lossOutGrad * weight_{target}
  $$

  $$
  logSoftmaxGradLossOutSubPart = exp(logProb) * nllLossGrad
  $$

  $$
  predictionsGradLossOut_{ij}=\begin{cases}
  nllLossGrad_i, & j=target(i)  \\
  0, & j ≠ target(i)
  \end{cases}
  $$

  $$
  predictionsGradLossOut = logSoftmaxGradLossOutSubPart - predictionsGradLossOut
  $$

  $$
  smoothLossGrad = smoothLossGrad * ignoreMask
  $$

  $$
  logSoftmaxGradSmoothLoss = smoothLossGrad * weight
  $$

  $$
  predictionsGradSmoothLoss = exp(logProb) * sum(logSoftmaxGradSmoothLoss) - logSoftmaxGradSmoothLoss
  $$

  Output in the scenario without zloss:

  $$
  xGrad_{out} = predictionsGradLossOut + predictionsGradSmoothLoss
  $$

  zloss scenario:

  $$
  gradZ=\begin{cases}
  grad + gradZloss, & gradZloss is passed  \\
  grad, & gradZloss is not passed
  \end{cases}
  $$

  $$
  zlossGrad=\begin{cases}
  gradZ / sum(ignoreMask), & &reduction = mean  \\
  gradZ, & &reduction = sum \\
  gradZ, & &reduction = none
  \end{cases}
  $$

  $$
  lseGrad = 2 * lseSquareScaleForZloss * lseForZloss * ignoreMask * zlossGrad
  $$

  $$
  zlossOutputGrad = exp(logProb) * lseGrad
  $$

  Output in the zloss scenario:

  $$
  xGrad_{out} = xGrad_{out} + zlossOutputGrad
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 312px">
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
      <td>grad</td>
      <td>Input</td>
      <td><ul>Gradient of the forward output loss, grad in the formula.</ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>logProb</td>
      <td>Input</td>
      <td><ul>logSoftmax computation result of the forward input, logProb in the formula.</ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>Input</td>
      <td><ul>Class indexes, target in the formula.</ul></td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Optional input</td>
      <td><ul><li>Scale weight specified for each class, weight in the formula.</li><li>Default: all 1.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gradZloss</td>
      <td>Optional input</td>
      <td><ul><li>Gradient of the forward output zloss, gradZloss in the formula.</li><li>Currently not supported yet.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lseForZloss</td>
      <td>Optional input</td>
      <td><ul><li>zloss-related input. If lse_square_scale_for_zloss is non-zero, the forward additionally outputs the lse_for_zloss intermediate result for the backward to compute lse, lseForZloss in the formula.</li><li>Currently not supported yet.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the reduction mode to apply to the output.</li><li>Default value: "mean".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ignoreIndex</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the ignored label value.</li><li>Default value: -100.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>labelSmoothing</td>
      <td>Optional attribute</td>
      <td><ul><li>Smoothing amount when computing the loss.</li><li>Currently only the input 0.0 is supported.</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>lseSquareScaleForZloss</td>
      <td>Optional attribute</td>
      <td><ul><li>Scale required for the zloss computation.</li><li>Currently not supported yet.</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td><ul>Gradient computation result, corresponding to out in the formula.</ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

  - target supports only class label indexes, and probability inputs are not supported.
  - The data types of gradLoss, logProb, gradZlossOptional, lseForZlossOptional, and xGradOut must be the same.
  - The zloss function is currently not supported yet. Passing the related inputs, that is, gradZlossOptional, lseForZlossOptional, and lseSquareScaleForZloss, does not take effect.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_cross_entropy_loss_grad](examples/test_aclnn_cross_entropy_loss_grad.cpp) | Calls the CrossEntropyLossGrad operator through the [aclnnCrossEntropyLossGrad](docs/aclnnCrossEntropyLossGrad_en.md) interface. |
