# CrossEntropyLoss

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: computes the cross-entropy loss of the input.
- Computation expression:

  When reductionOptional is mean, the formula for computing the cross-entropy loss loss is:
  $$
  l_n = -weight_{y_n}*log\frac{exp(x_{n,y_n})}{\sum_{c=1}^Cexp(x_{n,c})}*1\{y_n\ !=\ ignoreIndex \}
  $$
  $$
  loss=\begin{cases}\sum_{n=1}^N\frac{1}{\sum_{n=1}^Nweight_{y_n}*1\{y_n\ !=\ ignoreIndex \}}l_n,&\text{if reductionOptional = ‘mean’} \\\sum_{n=1}^Nl_n,&\text {if reductionOptional = ‘sum’}\\\{l_0,l_1,...,l_n\},&\text{if reductionOptional = ‘None’}\end{cases}
  $$
  The formula for computing log_prob is:
  $$
  lse_n = log*\sum_{c=1}^{C}exp(x_{n,c})
  $$
  $$
  logProb_{n,c} = x_{n,c} - lse_n
  $$
  The formula for computing zloss is:
  $$
  zloss_n = lseSquareScaleForZloss *（lse_n）^2
  $$
  Here N is the number of batches, and C is the number of labels.

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
      <td>x</td>
      <td>Input</td>
      <td><ul>x in the formula.</ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Input</td>
      <td><ul>Represents the label, y in the formula.</ul></td>
      <td>INT64, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Optional input</td>
      <td><ul><li>Represents the scaling weight specified for each class, weight in the formula.</li><li>The default is all 1s.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reductionOptional</td>
      <td>Optional attribute</td>
      <td><ul><li>Represents the reduction mode of loss.</li><li>The default value is "mean".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ignoreIndex</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the label value to be ignored.</li><li>The default value is -100.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>labelSmoothing</td>
      <td>Optional attribute</td>
      <td><ul><li>Represents the smoothing amount when computing loss.</li><li>The default value is 0.</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>lseSquareScaleForZloss</td>
      <td>Optional attribute</td>
      <td><ul><li>Represents the scale required for the zloss computation.</li><li>Currently not supported.</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>returnZloss</td>
      <td>Optional attribute</td>
      <td><ul><li>Controls whether to return the zloss output. It is a Boolean value on the Host side. Pass True when zLoss needs to be output; otherwise, pass False.</li><li>Currently not supported.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>lossOut</td>
      <td>Output</td>
      <td><ul>Represents the output loss, corresponding to loss in the formula.</ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>logProbOut</td>
      <td>Output</td>
      <td><ul>The output provided to the backward computation, corresponding to logProb in the formula.</ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zlossOut</td>
      <td>Output</td>
      <td><ul><li>Represents the auxiliary loss, corresponding to zlossOut in the formula.</li><li>Currently not supported.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lseForZlossOut</td>
      <td>Output</td>
      <td><ul><li>Represents the Tensor output to the backward in the zloss scenario. When lseSquareScaleForZloss is 0, the output is None, corresponding to lse in the formula.</li><li>Currently not supported.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

  - target supports only class label indices, and probability inputs are not supported.
  - Currently, zloss-related functions are not supported. Passing the related inputs, namely lseSquareScaleForZloss and returnZloss, has no effect.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_cross_entropy_loss](examples/test_aclnn_cross_entropy_loss.cpp) | Invoke the CrossEntropyLoss operator through the [aclnnCrossEntropyLoss](docs/aclnnCrossEntropyLoss_en.md) interface. |
