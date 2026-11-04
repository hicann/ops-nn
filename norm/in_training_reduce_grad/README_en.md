# INTrainingReduceGrad

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:-----|:-------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: Implements the Instance Normalization training backward gradient computation. In deep learning training, the forward stage computes the variance and mean of normalization, and the backward stage computes the gradient of the original input (pd_x) based on the gradient of the loss with respect to the input (dy). This operator receives the variance, mean, residual scaling, and bias parameters cached in the forward stage, and computes the complete gradient of the inverse normalization transformation.
- Formula:

  $$
  \text{data\_sqrt} = \sqrt{\text{variance} + \epsilon}
  $$

  $$
  \text{multiplier} = -\frac{\text{res\_gamma}}{N \cdot \text{data\_sqrt}}
  $$

  $$
  \text{addend} = \frac{\text{mean}}{\text{data\_sqrt}} \cdot \frac{\text{res\_gamma}}{N} - \frac{\text{res\_beta}}{N}
  $$

  $$
  \text{coef} = \text{dy} + \text{multiplier} \cdot x + \text{addend}
  $$

  $$
  \text{mul\_scale} = \frac{\text{gamma}}{\text{data\_sqrt}}
  $$

  $$
  \text{pd\_x} = \text{coef} \cdot \text{mul\_scale}
  $$

  where $N$ is the number of elements in the spatial dimensions ($H \times W$), $\epsilon$ is a small constant for preventing division by zero (1e-6), $\text{variance}$ and $\text{mean}$ are the statistics cached in the forward stage, $\text{res\_gamma}$ and $\text{res\_beta}$ are the residual scaling and bias, and $\text{gamma}$ is the learnable scaling parameter.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
|:-------|:--------------|:-----|:--------|:--------|
| dy | Input | Gradient of the loss with respect to the normalized output. | FLOAT16, FLOAT | NCHW, NHWC |
| x | Input | Original input of the forward stage. The shape is the same as dy. | FLOAT16, FLOAT | NCHW, NHWC |
| variance | Input | Variance cached in the forward stage, a channel-dimension tensor. | FLOAT | NCHW, NHWC |
| mean | Input | Mean cached in the forward stage, a channel-dimension tensor. | FLOAT | NCHW, NHWC |
| res_gamma | Input | Residual scaling parameter, a channel-dimension tensor. | FLOAT | NCHW, NHWC |
| res_beta | Input | Residual bias parameter, a channel-dimension tensor. | FLOAT | NCHW, NHWC |
| gamma | Input | Learnable scaling parameter, a channel-dimension tensor. | FLOAT | NCHW, NHWC |
| pd_x | Output | Gradient of the original input x. The shape and dtype are the same as dy. | FLOAT16, FLOAT | NCHW, NHWC |

## Constraints

- The shape and data type of dy and x must be exactly the same.
- The shapes of variance, mean, res_gamma, res_beta, and gamma must be the same, and they are channel-dimension tensors: in the NCHW format, the shape is [N, C, 1, 1]; in the NHWC format, the shape is [N, 1, 1, C].
- The input rank must be 4 (only 4D NCHW/NHWC is supported).
- For FP16 inputs, the internal computation is promoted to FP32, and the output is converted back to FP16 (CAST_RINT), to ensure the precision of gradient computation.
- Empty tensors are not supported (H×W=0, that is, num=0). In this case, the Tiling stage reports an error and returns GRAPH_FAILED.
- The operator is deterministic by default, and the same input produces the same output.

## Invocation Description

| Invocation method | Sample code | Description |
|:--------|:--------|:-----|
| GE graph mode | [test_geir_in_training_reduce_grad](examples/test_geir_in_training_reduce_grad.cpp) | Invoke the INTrainingReduceGrad operator by building a graph through the [operator IR](op_graph/in_training_reduce_grad_proto.h). |
