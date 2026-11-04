# BatchNormV3

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: performs batch normalization on a batch of data. The statistics of the generated data after normalization are 0 mean and 1 standard deviation.

- Formula:

  $$
  y = \frac{(x - E(x))}{\sqrt{Var(x) + ε}} * γ + β
  $$
  E(x) is the mean and Var(x) is the variance, both computed inside the operator. ε is an extremely small floating-point number that prevents the denominator from being 0.

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
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
      <td>
      <ul><li>The input tensor for batch normalization, corresponding to `x` in the formula.</li><li>The shape supports 4D and 5D.</li></ul>
      </td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW/NHWC/NCDHW/NDHWC</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>
      <ul><li>The weight for batch normalization, corresponding to `γ` in the formula.</li><li>A 1D tensor whose shape is the same as dimension C of the input x. The data type and the data type of x support the following combinations: [x: float16, weight: float16/float32], [x: bfloat16, weight: bfloat16/float32], [x: float32, weight: float32].</li></ul>
      </td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td><ul><li>The bias value for batch normalization, corresponding to `β` in the formula.</li><li>A 1D tensor whose data type and shape are consistent with those of the input parameter weight.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>running_mean</td>
      <td>Input/Output</td>
      <td><ul><li>Training scenario: the mean before and after the momentum update during training. Inference scenario: the mean used during inference, corresponding to `E(x)` in the formula.</li><li>A 1D tensor whose shape is the same as dimension C of the input x. The data type and the data type of x support the following combinations: [x: float16, running_mean: float16/float32], [x: bfloat16, running_mean: bfloat16/float32], [x: float32, running_mean: float32].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>running_var</td>
      <td>Input/Output</td>
      <td>Training scenario: the variance before and after the momentum update during training. Inference scenario: the variance used during inference, corresponding to `Var(x)` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>A small value added to the variance to avoid division by zero, corresponding to `ε` in the formula.</li><li>The default value is 1e-5.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>momentum</td>
      <td>Optional attribute</td>
      <td><ul><li>The momentum parameter, used to update the mean and variance during training.</li><li>The default value is 0.1.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_training</td>
      <td>Optional attribute</td>
      <td><ul><li>Marks whether the scenario is training. true indicates the training scenario, and false indicates the inference scenario.</li><li>The default value is true.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td><ul><li>Represents the output result after batch normalization, corresponding to `y` in the formula.</li><li>The data type, data format, and shape are consistent with those of the input x.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW/NHWC/NCDHW/NDHWC</td>
    </tr>
    <tr>
      <td>running_mean</td>
      <td>Output</td>
      <td><ul><li>Output only in the training scenario, the mean after the momentum update during training.</li><li>The data type and shape are consistent with those of the input running_mean.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>running_var</td>
      <td>Output</td>
      <td><ul><li>Output only in the training scenario, the variance after the momentum update during training.</li><li>The data type and shape are consistent with those of the input running_var.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>save_mean</td>
      <td>Output</td>
      <td><ul><li>Output only in the training scenario, the saved mean of x, corresponding to `E(x)` in the formula.</li><li>A 1D tensor.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>save_rstd</td>
      <td>Output</td>
      <td><ul><li>Output only in the training scenario, the saved variance of x or the reciprocal of the standard deviation of x, corresponding to `Var(x)` and the reciprocal of the square root of (Var(x) + ε) in the formula, respectively.</li><li>A 1D tensor.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:
    - The data formats NHWC and NDHWC are not supported for the input x and the output y.
    - The output parameter `save_rstd` stores the variance of x.

  - <term>Atlas training products</term>:
    - Data type: BFLOAT16 is not supported for all inputs and outputs.
    - Data format: NHWC and NDHWC are not supported for the input x and the output y.
    - The output parameter `save_rstd` stores the variance of x.

  - <term>Atlas inference products</term>:
    - Data type: BFLOAT16 is not supported for all inputs and outputs.
    - Data format: NHWC and NDHWC are not supported for the input x and the output y.
    - The output parameter `save_rstd` stores the variance of x.

  - <term>Ascend 950PR&950DT products</term>:

    The output parameter `save_rstd` stores the reciprocal of the standard deviation of x.

## Constraints

For <term>Atlas A2 products</term> and <term>Atlas A3 products</term>, only the training scenario is supported.

## Invocation Description

| Invocation method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_batch_norm_v3](examples/test_aclnn_batch_norm_v3.cpp) | Invokes the BatchNormV3 operator through the [aclnnBatchNorm](docs/aclnnBatchNorm_en.md) API. |
| Graph mode | -  | Invokes the BatchNormV3 operator through graph construction using the [operator IR](op_graph/batch_norm_v3_proto.h).         |
