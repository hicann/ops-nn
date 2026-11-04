# FusedMulApplyMomentum

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:-----|:-------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>| √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: Implements the parameter update of the SGD with Momentum optimizer and fuses the gradient multiplication (`grad = x1 * x2`). It is used in the parameter update phase after backward propagation in deep learning training, and it is applicable to gradient scaling (Loss Scaling) in mixed-precision training scenarios.
- Formula:

  **Standard mode** (use_nesterov = false):

  $$
  \text{accum}_{t} = \text{accum}_{t-1} \cdot m + x_1 \cdot x_2
  $$

  $$
  \text{var}_{t} = \text{var}_{t-1} - \text{accum}_{t} \cdot \eta
  $$

  **Nesterov mode** (use_nesterov = true):

  $$
  \text{accum}_{t} = \text{accum}_{t-1} \cdot m + x_1 \cdot x_2
  $$

  $$
  \text{var}_{t} = \text{var}_{t-1} - x_1 \cdot x_2 \cdot \eta - \text{accum}_{t} \cdot m \cdot \eta
  $$

  where $var$ denotes the model parameters, $accum$ denotes the momentum accumulator, $x_1$ denotes the gradient tensor, $x_2$ denotes the gradient scaling factor, $m$ denotes the momentum coefficient, and $\eta$ denotes the learning rate.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
|:-------|:--------------|:-----|:--------|:--------|
| var | Input | Model parameters to be updated, corresponding to var<sub>t-1</sub> in the formula. | FLOAT, FLOAT16 | ND |
| accum | Input | Momentum accumulator, corresponding to accum<sub>t-1</sub> in the formula. The shape is the same as var. | FLOAT, FLOAT16 | ND |
| lr | Input | Learning rate, corresponding to η in the formula, a scalar. | FLOAT, FLOAT16 | ND |
| x1 | Input | Gradient tensor, corresponding to x<sub>1</sub> in the formula. The shape is the same as var. | FLOAT, FLOAT16 | ND |
| momentum | Input | Momentum coefficient, corresponding to m in the formula, a scalar, with the typical value 0.9. | FLOAT, FLOAT16 | ND |
| x2 | Input | Gradient scaling factor, corresponding to x<sub>2</sub> in the formula, a scalar. | FLOAT, FLOAT16 | ND |
| use_nesterov | Attribute | <ul><li>Switch of the Nesterov mode.</li><li>The default value is false.</li></ul> | BOOL | - |
| use_locking | Attribute | <ul><li>Lock control (kept for TensorFlow compatibility and not used).</li><li>The default value is false.</li></ul> | BOOL | - |
| outVar | Output | Updated model parameters, corresponding to var<sub>t</sub> in the formula. The shape and dtype are the same as var. | FLOAT, FLOAT16 | ND |
| outAccum | Output | Updated momentum accumulator, corresponding to accum<sub>t</sub> in the formula. The shape and dtype are the same as var. | FLOAT, FLOAT16 | ND |

## Constraints

- The shapes and data types of the three tensors var, accum, and x1 must be exactly the same.
- lr, momentum, and x2 are scalars (shape [1]), and their data types must be the same as the data type of the tensors.
- For FP16 inputs, the internal computation is promoted to FP32, and the output is converted back to FP16, to ensure the precision of the momentum accumulator in long-term training.
- The use_locking attribute is kept for TensorFlow compatibility and is not used in the Ascend C implementation.
- The operator is deterministic by default, and the same input produces the same output.
- Empty tensors (0 elements) are not supported.
- The tensor rank range is 0 to 8.
- Dynamic shape support: Kernel mode supports -1/-2 (unknown dimensions); GEIR mode does not support -1/-2, because the infershape built into canndev cannot correctly propagate unknown dimensions to the inplace outputs.

## Invocation Description

| Invocation method | Invocation sample | Description |
|:--------|:--------|:-----|
| Graph-mode invocation | [test_geir_fused_mul_apply_momentum](examples/test_geir_fused_mul_apply_momentum.cpp) | Calls the FusedMulApplyMomentum operator by constructing a graph through the [operator IR](op_graph/fused_mul_apply_momentum_proto.h). |
