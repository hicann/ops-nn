# BatchNormGrad

English | [简体中文](./README.md)

## Product Support Status

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: computes the gradients of BatchNorm for backpropagation, producing the gradient of input x and the gradients of scale and offset.
- Inputs y_backprop and x, and output x_backprop, support FLOAT, FLOAT16 and BF16; the remaining inputs and outputs are FLOAT.

## Parameter Description

| Name | Input/Output/Attribute | Description | Data Type | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| y_backprop | Input | Backpropagated gradient of the output. | FLOAT, FLOAT16, BF16 | NCDHW, NCHW, NHWC, NDHWC |
| x | Input | Input of the forward computation. | FLOAT, FLOAT16, BF16 | NCDHW, NCHW, NHWC, NDHWC |
| scale | Input | Scale parameter of the forward computation. | FLOAT | ND |
| reserve_space_1 | Input | Intermediate value saved by the BatchNorm forward pass: the saved mean of x in training mode, and the running mean of x in inference mode. | FLOAT | ND |
| reserve_space_2 | Input | Intermediate value saved by the BatchNorm forward pass: the saved inverse standard deviation of x in training mode, and the running variance of x in inference mode. | FLOAT | ND |
| reserve_space_3 | Optional input | Intermediate result reserved by the forward computation; not used and not involved in calculations. | FLOAT | ND |
| x_backprop | Output | Gradient of input x. | FLOAT, FLOAT16, BF16 | NCDHW, NCHW, NHWC, NDHWC |
| scale_backprop | Output | Gradient of scale. | FLOAT | ND |
| offset_backprop | Output | Gradient of offset. | FLOAT | ND |
| reserve_space_4 | Output | Intermediate result reserved by the backward computation. | FLOAT | ND |
| reserve_space_5 | Output | Intermediate result reserved by the backward computation. | FLOAT | ND |
| epsilon | Attribute | Numerical stability parameter, default 1e-4. | FLOAT | - |
| data_format | Attribute | Data layout, default NHWC. | STRING | - |
| is_training | Attribute | Whether training mode is used, default true. | BOOL | - |
| output_mask | Attribute | Output mask, default {true, false, false}. Valid only in inference mode, it determines whether x_backprop, scale_backprop and offset_backprop contain actual results. | LIST_BOOL | - |

## Constraint Description

- The preceding layer of this operator must be the BatchNorm operator.
- scale, reserve_space_1 and reserve_space_2 are 1D tensors whose shape must be the C channel.
- x has the same shape as y_backprop; x_backprop has the same shape as x.
- Inputs y_backprop and x, and output x_backprop, support NCDHW, NCHW, NHWC and NDHWC; the remaining inputs and outputs use ND.
- reserve_space_3 is an optional input.
- In the Data Types column of the product lists, only the AI Core of Ascend 950 lists bfloat16; the AI Core and AI CPU of Atlas A2/A3 list only float16 and float32.

## Invocation Description

| Invocation Method | Sample Code | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | - | Invokes the BatchNormGrad operator by building a graph through the [operator IR](./op_graph/batch_norm_grad_proto.h) |
