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

- Operator function: Computes the mean absolute error (MAE) between each element of the input predict and the target label. reduction specifies the reduction to apply to the output. The supported values are "none", "mean", and "sum". "none" means no reduction is applied, "mean" means the sum of the output is divided by the number of elements in the output, and "sum" means the output is summed.

- Formula:

  When `reduction` is `none`

  $$
  \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
  l_n = \left| x_n - y_n \right|,
  $$

  Where $x$ is predict, $y$ is label, and $N$ is the batch size. If `reduction` is not `"none"`, then

  $$
  \ell(x, y) =
  \begin{cases}
  \operatorname{mean}(L), &  \text{if reduction} = \text{'mean';}\\
  \operatorname{sum}(L),  &  \text{if reduction} = \text{'sum'.}
  \end{cases}
  $$

## Parameters

This chapter describes the parameters of the operator (LpLoss) in its definition file `op_host/lp_loss_def.cpp`, that is, the IR layer semantics in GE graph mode. The parameter semantics of the aclnn API (aclnnL1Loss) differ from this table. For details, see [aclnnL1Loss](docs/aclnnL1Loss_en.md) and the "Layer semantics mapping table" in the "Constraints" section below.

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>predict</td>
      <td>Input</td>
      <td>Input $x$ in the formula, i.e., the predicted value. The shape supports 0-8 dimensions, dynamic shape (-1), and dynamic rank (-2).</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>label</td>
      <td>Input</td>
      <td>Input $y$ in the formula, i.e., the ground-truth label. The data type and shape must be exactly the same as predict. The IR layer does not perform data type inference or broadcast.</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>p</td>
      <td>Attribute</td>
      <td><ul><li>The order for computing the loss. Currently only p=1 (i.e., mean absolute error) is supported.</li><li>Required attribute with no default value.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the reduction to apply to the output, i.e., `reduction` in the formula. The supported values are "none", "mean", and "sum".</li><li>Default value: "mean".</li></ul></td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output in the formula. The data type is the same as predict. When reduction is "none", the shape is the same as predict; when reduction is "mean" or "sum", it is a 0-dimensional tensor (scalar).</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- In LpLoss, p is the order for computing the loss. Only p=1 is supported, and the corresponding aclnn API name is aclnnL1Loss.

- Layer semantics mapping table: This operator has two paths: GE graph mode (IR/Kernel layer) and direct aclnn invocation (aclnnL1Loss). The two layers use different parameter names and data type domains. Do not mix the two layers when using the operator.

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 160px">
  <col style="width: 200px">
  <col style="width: 300px">
  <col style="width: 320px">
  </colgroup>
  <thead>
    <tr>
      <th>Comparison Item</th>
      <th>IR/Kernel Layer (LpLoss)</th>
      <th>aclnn Layer (aclnnL1Loss)</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>Parameter naming</td>
      <td>predict, label, y</td>
      <td>self, target, out</td>
      <td>Two sets of names for the same parameters, one-to-one correspondence.</td>
    </tr>
    <tr>
      <td>Data type domain</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>After inference: FLOAT, FLOAT16, INT64, and BFLOAT16 (by product)</td>
      <td>INT64 is implemented by small-op composition in the aclnn layer, not through the LpLoss kernel. The IR/Kernel layer does not support INT64. Neither layer supports COMPLEX64/COMPLEX128.</td>
    </tr>
    <tr>
      <td>Relationship between inputs</td>
      <td>The dtype and shape of label must be exactly the same as predict</td>
      <td>self and target satisfy the data type inference rules and broadcast rules</td>
      <td>Inference and broadcast are performed by the aclnn layer before the kernel is invoked.</td>
    </tr>
    <tr>
      <td>reduction type</td>
      <td>String: "none"/"mean"/"sum", default "mean"</td>
      <td>int64_t enum: 0('none')/1('mean')/2('sum')</td>
      <td>When the attribute is omitted, the IR layer takes effect with the default "mean".</td>
    </tr>
    <tr>
      <td>Non-contiguous tensor</td>
      <td>Not supported; inputs must be contiguous ND tensors</td>
      <td>Supported</td>
      <td>The kernel applies the access view derived from predict to label as well, so the IR/Kernel layer requires a consistent layout of the two inputs; the aclnn layer performs contiguous transformation before the kernel is invoked.</td>
    </tr>
  </tbody></table>

- Data type differences across products:
    - <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: The IR/Kernel layer supports FLOAT16, BFLOAT16, and FLOAT; the aclnn layer supports FLOAT, FLOAT16, BFLOAT16, and INT64 after inference.
    - <term>Atlas inference products</term>, <term>Atlas training products</term>: The aclnn layer supports FLOAT, FLOAT16, and INT64 after inference, and **does not support BFLOAT16** (passing BFLOAT16 returns 161002).

- Obligation to reject out-of-range data types and shapes: The data type domains, data format domains, and dimension ranges declared in the table above are the complete support range of the operator. Inputs beyond the declared range (unlisted data types, non-ND format, dimensions greater than 8) must be rejected and return an error; callers must not rely on the behavior of undeclared combinations.

- Dimension range: The shapes of predict and label support 0-8 dimensions (including 0-dimensional scalars).

- Empty tensor behavior (when the input contains 0 elements):
    - When reduction is "none" (0 in the aclnn layer): outputs an empty tensor.
    - When reduction is "mean" (1 in the aclnn layer): outputs NaN. Note that this result differs from the intuition of "mean", due to the divide-by-zero semantics when the number of elements is 0.
    - When reduction is "sum" (2 in the aclnn layer): outputs 0.

- Input/output overlap: Overlap (aliasing) between input and output memory is not supported. predict, label, and y must be independent memory. The operator does not provide in-place computation semantics and does not detect overlap; the behavior with overlapping memory is undefined.

- Dynamic shape: The operator registration declares support for dynamic shape and dynamic rank. In a shape, -1 indicates that the dimension size is unknown and -2 indicates that the rank is unknown, which are handled by the dynamic branch at compile time.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_l1_loss.cpp](examples/test_aclnn_l1_loss.cpp) | Invokes the LpLoss operator through the [aclnnL1Loss](docs/aclnnL1Loss_en.md) interface. |
| GE graph mode  | - | Invokes the LpLoss operator through [operator IR](op_graph/lp_loss_proto.h) graph construction. |
