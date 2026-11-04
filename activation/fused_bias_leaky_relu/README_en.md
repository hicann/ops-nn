# FusedBiasLeakyRelu

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Description: adds the bias to the input tensor x and then applies the LeakyReLU activation function with a scale factor.
- Formula:

  $$
  FusedBiasLeakyRelu(x, bias) =
  \begin{cases}
  (x + bias) \ast scale, \quad x + bias \geq 0\\
  (x + bias) \ast negative\_slope \ast scale, \quad x + bias < 0
  \end{cases}
  $$

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1300px"><colgroup>
  <col style="width: 101px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 200px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage instructions</th>
      <th>Data type</th>
      <th>Data format</th>
      <th>Dimensions (shape)</th>
      <th>Non-contiguous tensor</th>
    </tr></thead>
  <tbody>
      <tr>
      <td>x</td>
      <td>Input</td>
      <td>Represents the input data tensor of FusedBiasLeakyRelu.</td>
      <td>Supports empty tensors.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>bias</td>
      <td>Input</td>
      <td>Represents the bias tensor, whose shape is exactly the same as x.</td>
      <td>The data type and shape must be consistent with x.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>negative_slope</td>
      <td>Attribute</td>
      <td>The slope coefficient of the negative half-axis of LeakyReLU.</td>
      <td>A float optional attribute. The default value is 0.2.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>scale</td>
      <td>Attribute</td>
      <td>The scale factor.</td>
      <td>A float optional attribute. The default value is 1.414213562373 (that is, √2).</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output tensor of FusedBiasLeakyRelu.</td>
      <td>The shape and dtype are the same as x.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
  </tbody>
  </table>

## Constraints

- The data types (dtypes) of x, bias, and y must be consistent, and only FLOAT16 and FLOAT are supported (the Tiling side checks the dtypes of x and bias against a whitelist and for consistency).
- The shapes of x and bias must be exactly the same dimension by dimension, and the shape of y is the same as x. Broadcasting is not supported (the Tiling side checks shape consistency dimension by dimension).
- negative_slope and scale are float optional attributes with default values (negative_slope=0.2, scale=1.414213562373). When they are not set, the Tiling side dispatches with the default values.
- The supported data format is ND (the Tiling side checks that the formats of x and bias are ND).
- Tensors support up to 8 dimensions, and rank 0 (scalar) inputs are supported (the Tiling side checks the number of dimensions, and scalars are processed as single elements).
- Empty tensors are supported (with 0 elements; the Tiling side returns directly and does not start the computation kernel).
- Dynamic shapes (the graph describes a shape containing -1, an unknown dimension) and dynamic ranks (the graph describes a shape of [-2], an unknown number of dimensions) are supported. The same graph can run continuously with multiple groups of concrete shapes.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode (GEIR) invocation | [test_geir_fused_bias_leaky_relu](./examples/arch35/test_geir_fused_bias_leaky_relu.cpp) | Invoke the FusedBiasLeakyRelu operator in GE IR graph mode (REG_OP registration). |
| Graph-mode (GEIR) dynamic shape invocation | [test_geir_fused_bias_leaky_relu_dynamic](./examples/arch35/test_geir_fused_bias_leaky_relu_dynamic.cpp) | A GE IR invocation sample covering the two dynamic scenarios: -1 unknown dimension and -2 unknown rank. |
