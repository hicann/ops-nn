# SoftmaxFocalLoss

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    √     |
| <term>Atlas inference products</term>                              |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function

- Operator function: Computes the Softmax Focal Loss in multi-classification scenarios. Focal Loss introduces the modulation factor $(1-p)^\gamma$ on the basis of cross entropy, reduces the loss weight of easy samples, and makes training focus more on hard samples. The input pred is the probability after softmax, and this operator does not build in softmax.

- Formula:

  Let pred be $p$ (with the shape $[N, C]$), target be $t$ (one-hot labels with the same shape as $p$), and weight be $w$ (with the same shape as $p$, and treated as all 1s when not provided). For the $n$-th sample:

  $$
  \text{CE}_n = \sum_{c=0}^{C-1} -\log(p_{n,c}) \cdot t_{n,c} \cdot w_{n,c}
  $$

  $$
  \text{FW}_n = \sum_{c=0}^{C-1} \alpha \cdot \exp\big(\gamma \cdot \log(1 - p_{n,c})\big) \cdot t_{n,c}
  $$

  $$
  y_{n,c} = \text{CE}_n \cdot \text{FW}_n
  $$

  where:

  - $N$ is the batch count, $C$ is the number of classes, and the reduction axis is the last dimension.
  - $\alpha$ and $\gamma$ are the attributes alpha and gamma respectively.
  - $y$ has the same shape as $p$, and the elements in the same row have the same value.

## Parameters

<table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 40px">
  <col style="width: 60px">
  <col style="width: 266px">
  <col style="width: 60px">
  <col style="width: 40px">
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
      <td>pred</td>
      <td>Input</td>
      <td>Probability output by the preceding softmax, corresponding to p in the formula. The values fall in the open interval (0, 1), the length of each dimension must be greater than 0 (empty tensors are not supported), the last dimension is the class dimension, and the other dimensions are sample dimensions. <term>Ascend 950PR&950DT products</term>: rank≥1 is supported; other products: only the 2D (batch_size, num_classes) is supported.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>Input</td>
      <td>One-hot labels, corresponding to t in the formula. The shape is the same as pred, and the values must be 0 or 1.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Optional input</td>
      <td>Element-wise weight, corresponding to w in the formula. The shape is the same as pred; when it is not passed in, it is treated as all 1s. The data type can be different from pred.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Attribute</td>
      <td>Exponent of the modulation factor, corresponding to gamma in the formula, with the default value 2.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Attribute</td>
      <td>Weight coefficient of the modulation factor, corresponding to alpha in the formula, with the default value 0.25.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Attribute</td>
      <td>Reduction method. Supports 'none', 'mean', and 'sum', with the default value 'mean'. The shape of the output y depends on this attribute: it has the same shape as pred when set to 'none', and it is a scalar when set to 'mean' or 'sum'. Currently only 'none' is supported, and it must be passed in explicitly.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Loss value, with the shape the same as pred, and the elements in the same row have the same value. The data type is the same as pred.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method | Sample code | Description |
| -------- | -------- | ---- |
| Graph-mode invocation | [test_geir_softmax_focal_loss.cpp](examples/test_geir_softmax_focal_loss.cpp) | Calls the SoftmaxFocalLoss operator by constructing a graph through the [operator IR](./op_graph/softmax_focal_loss_proto.h). |
