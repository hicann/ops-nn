# InstanceNorm

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: Performs Instance Normalization. Compared with [BatchNorm](../batch_norm_v3/README.md), InstanceNorm normalizes over the instances of each sample instead of the whole batch, which makes this function more suitable for processing data such as images.
- Formula:

  $$
  y = {{x-E(x)}\over\sqrt {Var(x) + ε}} * γ + β
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 372px">
  <col style="width: 213px">
  <col style="width: 105px">
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
      <td><ul><li>Empty tensors are supported. Only the reduce axis supports the input value 0; the N and C axes do not support the input value 0.</li><li>Input data for the InstanceNorm computation, corresponding to `x` in the formula.</li><li>The actual data format is determined by the parameter data_format.</li><li>NCHW/NHWC: the shape supports 4D.</li><li>NCDHW/NDHWC: the shape supports 5D.</li><li>ND: the shape supports 2 to 8 dimensions, where the 2nd dimension is fixed to the C axis.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW/NHWC/NCDHW/NDHWC/ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td><ul><li>Empty tensors are not supported.</li><li>A 1D tensor, representing the scale factor (weight) for the InstanceNorm computation, corresponding to `γ` in the formula.</li><li>The data type is the same as x by default; if not, it is explicitly set to FLOAT32.</li><li>The shape is the same as the C axis of input x.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td><ul><li>Empty tensors are not supported.</li><li>A 1D tensor, representing the bias for the InstanceNorm computation, corresponding to `β` in the formula.</li><li>The data type and shape are the same as input gamma.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the data format of input x.</li><li>Default value: NDHWC.</li><li>The parameter data_format supports only "NCHW", "NHWC", "NCDHW", "NDHWC", and "ND".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Value added to the variance to avoid division by zero. It corresponds to `ε` in the formula.</li><li>Default value: 1e-6.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td><ul><li>Empty tensors are supported.</li><li>Output result of InstanceNorm, corresponding to `y` in the formula.</li><li>The shape, data type, and data format are the same as those of input x.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW/NHWC/NCDHW/NDHWC/ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Output</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Mean of InstanceNorm, corresponding to `E(x)` in the formula.</li><li>The data type is the same as gamma.</li><li>The shape and input x satisfy a <a href="../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> (the number of shape dimensions is the same as input x, the sizes of non-reduction axes are the same as input x, and the sizes of reduction axes are 1).</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>variance</td>
      <td>Output</td>
      <td><ul><li>Empty tensors are not supported.</li><li>Variance of InstanceNorm, corresponding to `Var(x)` in the formula.</li><li>The data type and shape are the same as input mean.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_instance_norm](./examples/test_geir_instance_norm.cpp) | Calls the InstanceNorm operator by building a graph with the [operator IR](op_graph/instance_norm_proto.h). |
