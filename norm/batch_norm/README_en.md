# BatchNorm

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
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
      <ul><li>The input tensor for batch normalization, corresponding to `x` in the formula.</li><li>The shape has at least 2 dimensions.</li></ul>
      </td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW/NHWC/NCDHW/NDHWC/ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>
      <ul><li>The weight for batch normalization, corresponding to `γ` in the formula.</li><li>A 1D tensor whose shape is the same as dimension C of the input x. If the input x is in ND format, the shape is the same as the second dimension of the input x.</li></ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Input</td>
      <td><ul><li>The bias value for batch normalization, corresponding to `β` in the formula.</li><li>A 1D tensor whose shape is consistent with that of the input parameter scale.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Optional input</td>
      <td><ul><li>Training scenario: the mean before the momentum update during training, as an optional input. Inference scenario: the mean used during inference, as a required input, corresponding to `E(x)` in the formula.</li><li>A 1D tensor whose shape is consistent with that of the input parameter scale.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>variance</td>
      <td>Optional input</td>
      <td><ul><li>Training scenario: the variance before the momentum update during training, as an optional input. Inference scenario: the variance used during inference, as a required input, corresponding to `Var(x)` in the formula.</li><li>A 1D tensor whose shape is consistent with that of the input parameter scale.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>A small value added to the variance to avoid division by zero, corresponding to `ε` in the formula.</li><li>The default value is 1e-4f.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the data format of the input x. Supported values: "NHWC", "NCHW", "NDHWC", and "NCDHW".</li><li>The default value is "NHWC".</li></ul></td>
      <td>STRING</td>
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
      <td>exponential_avg_factor</td>
      <td>Optional attribute</td>
      <td><ul><li>The factor used to update the momentum average during training.</li><li>The default value is 1.0f.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td><ul><li>Represents the output result after batch normalization, corresponding to `y` in the formula.</li><li>The data type, data format, and shape are consistent with those of the input x.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW/NHWC/NCDHW/NDHWC/ND</td>
    </tr>
    <tr>
      <td>batch_mean</td>
      <td>Output</td>
      <td><ul><li>Training mode: the mean of the current batch. Inference mode: equal to the input mean.</li><li>A 1D tensor whose shape is consistent with that of the input parameter scale.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_variance</td>
      <td>Output</td>
      <td><ul><li>Training mode: the variance of the current batch. Inference mode: equal to the input variance.</li><li>A 1D tensor whose shape is consistent with that of the input parameter scale.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reserve_space_1</td>
      <td>Output</td>
      <td><ul><li>Reserved for gradient computation. Training mode: equal to the output batch_mean. Inference mode: equal to the input mean.</li><li>A 1D tensor whose shape is consistent with that of the input parameter scale.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reserve_space_2</td>
      <td>Output</td>
      <td><ul><li>Reserved for gradient computation. Training mode: stores inv_var: (1/sqrt(epsilon + variance)), used for reuse in the backward gradient computation. Inference mode: equal to the input variance.</li><li>A 1D tensor whose shape is consistent with that of the input parameter scale.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reserve_space_3</td>
      <td>Output</td>
      <td><ul><li>A 1D tensor containing only one element. It exists only for TensorFlow compatibility, and the data is meaningless.</li><li>The shape is [1].</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_batch_norm](./examples/test_geir_batch_norm.cpp)   | Invokes the BatchNorm operator through graph construction using the [operator IR](./op_graph/batch_norm_proto.h). |
