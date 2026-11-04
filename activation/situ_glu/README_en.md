# SituGlu

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------- | ------|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×    |

## Function

- Operator function: SiTU Gated Linear Unit activation function. It splits the input tensor x along the specified dimension into the gate and up halves, and computes the output by the SiTU formula.

- Calculation formula or calculation process:

  For a given input tensor x whose dimensions are [a, b, c, d, e, f, g, ...], the operator SituGlu performs the following computations:

  1. Folds x based on the input parameter dim into [pre, cut], where cut must be an even number. Let h = cut // 2.

  2. Splits x into the front and back parts based on the input parameter activate_left:

     - When activate_left is true (default):

     $$
     gate = x[:, :h], \quad up = x[:, h:]
     $$

     - When activate_left is false:

     $$
     gate = x[:, h:], \quad up = x[:, :h]
     $$

  3. Performs the SiTU computation based on the input parameters beta and linear_beta:

     $$
     situ\_a = \beta \cdot \tanh\left(\frac{gate}{\beta}\right) \cdot \text{sigmoid}(gate)
     $$

     When linear_beta > 0:

     $$
     up = linear\_beta \cdot \tanh\left(\frac{up}{linear\_beta}\right)
     $$

  4. Output:

     $$
     y = situ\_a \cdot up
     $$

  5. Reshapes the output tensor y so that the number of dimensions is the same as that of x before folding, the size on the dim axis is half of x, and the other dimensions are the same as x.

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 170px">
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
      <td>Input x in the formula. The dim dimension is evenly split into the gate and up parts.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Attribute</td>
      <td><ul><li>Dimension to split.</li><li>The value range is [-x.dim(), x.dim()-1].</li><li>The default value is -1.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Attribute</td>
      <td><ul><li>Scaling coefficient of the SiTU gate part, controlling the strength of the tanh nonlinearity.</li><li>The default value is 1.0.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linear_beta</td>
      <td>Attribute</td>
      <td><ul><li>Scaling coefficient of the linear tanh on the up path.</li><li>When it is greater than 0, the bounding transformation is applied to up; when it is less than or equal to 0, up is passed through directly.</li><li>The default value is 0.0.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>Attribute</td>
      <td><ul><li>Whether gate is the front half when x is split.</li><li>true means that gate is the front half and up is the back half; false means that gate is the back half and up is the front half.</li><li>The default value is true.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>y in the formula, the SiTU activation result. The size of the dim dimension is half of x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Non-contiguous tensors are supported.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_situ_glu](./examples/test_aclnn_situ_glu.cpp) | Calls the SituGlu operator through the [aclnnSituGlu](./docs/aclnnSituGlu_en.md) interface. |
| GE graph mode | -  | Calls the SituGlu operator by building a graph through [operator IR](./op_graph/situ_glu_proto.h).         |
| PyTorch API |  -  | Calls the SituGlu operator through the [situ_glu](./docs/torchapi_situ_glu_en.md) interface.         |
