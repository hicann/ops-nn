# BatchNorm3D

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| --- | --- |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

BatchNorm3D performs batch normalization channel by channel on a 5D input tensor, and supports training and inference modes.

Let each channel contain $M=N\times D\times H\times W$ elements. Training mode computes the channel mean $\mu$ and the biased variance $v$:

$$
\mu=\frac{1}{M}\sum_{i=1}^{M}x_i,\qquad
v=\frac{1}{M}\sum_{i=1}^{M}(x_i-\mu)^2
$$

Both training and inference modes compute the output with the following formula. In inference mode, $\mu$ and $v$ come from the inputs `mean` and `variance` respectively.

$$
y=(x-\mu)\frac{1}{\sqrt{v+\text{epsilon}}}\times\text{scale}+\text{offset}
$$

In training mode, when $M>1$, `batch_variance` outputs the unbiased variance $vM/(M-1)$. For the result when $M=1$ and the meaning of `reserve_space_2`, see [Constraints](#constraints).

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| x | Input | The input tensor with a logical 5D shape and C channels; see below for the storage format constraints. | FLOAT16, FLOAT32 | NCDHW, NDHWC, NDC1HWC0 |
| scale | Input | The per-channel scaling coefficient. | FLOAT32 | ND, NDC1HWC0 |
| offset | Input | The per-channel bias; its shape matches scale. | FLOAT32 | ND, NDC1HWC0 |
| mean | Optional input | The per-channel inference mean; its shape matches scale. It is required for inference; see below for the passing requirements in training. | FLOAT32 | ND, NDC1HWC0 |
| variance | Optional input | The per-channel inference variance; its shape matches scale. It is required for inference; see below for the passing requirements in training. | FLOAT32 | ND, NDC1HWC0 |
| y | Output | The normalization result; its shape, data type, and data format match x. | FLOAT16, FLOAT32 | NCDHW, NDHWC, NDC1HWC0 |
| batch_mean | Output | Its shape matches scale. In training it is the batch mean; in inference it is the input mean. | FLOAT32 | ND, NDC1HWC0 |
| batch_variance | Output | Its shape matches scale. In training it is the batch variance statistic; in inference it is the input variance. | FLOAT32 | ND, NDC1HWC0 |
| reserve_space_1 | Output | Its shape matches scale. In training it stores the batch mean for backward computation; in inference it is the input mean. | FLOAT32 | ND, NDC1HWC0 |
| reserve_space_2 | Output | Its shape matches scale. In training it stores an auxiliary statistic; see below for details. In inference it is the input variance. | FLOAT32 | ND, NDC1HWC0 |
| epsilon | Optional attribute | The value added to the variance before the square root operation. Default value: 0.0001. A finite positive number is recommended. | FLOAT32 | - |
| data_format | Optional attribute | The logical data format of x. Supports NCDHW and NDHWC. Default value: NCDHW. | STRING | - |
| is_training | Optional attribute | true indicates training, and false indicates inference. Default value: true. | BOOL | - |

## Constraints

- At runtime, each dimension of the input must be greater than 0; empty tensors are not supported. The channel axis of NCDHW is 1, and the channel axis of NDHWC is 4.
- The input is stored contiguously in the declared format; the operator does not provide a custom stride interface.
- epsilon does not reject values through a numeric domain check. Non-positive or non-finite values may cause division by zero, infinity, or NaN. At inference, a non-negative variance is recommended.
- <term>Ascend 950PR&950DT products</term>:
  - x and y use the NCDHW or NDHWC format with rank 5; data_format must match the format of x. The other tensors use the ND format, are all one-dimensional, and have a length equal to C.
  - In training mode, mean and variance can be omitted. When passed in, they must meet the specifications above, and their values do not take part in the computation. All five outputs are required outputs.
  - In training mode, `reserve_space_2` is the reciprocal standard deviation $1/\sqrt{v+\text{epsilon}}$. When $M=1$, `batch_variance` is NaN.
  - In the dynamic graph, `-1` denotes a dimension to be determined, and `[-2]` denotes a rank to be determined. At execution, you must provide a concrete shape that satisfies the constraints above. The shape and type inference callbacks do not serve as input validity determination interfaces.
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, <term>Atlas training products</term>:
  - It uses the CANN built-in implementation, and all tensors use the NDC1HWC0 storage format at execution. The storage shape of x and y is `[N,D,C1,H,W,C0]`, and the storage shape of channel tensors is `[1,1,C1,1,1,C0]`, where C1 and C0 match the corresponding dimensions of x. The logical formats support NCDHW and NDHWC.
  - In training mode, mean and variance must be omitted.
  - In training mode, `reserve_space_2` is the biased variance $v$. With finite inputs and $M=1$, `batch_variance` is 0.

## Invocation Description

| Invocation method | Invocation sample | Description |
| --- | --- | --- |
| GE graph mode | [test_geir_batch_norm3d](examples/test_geir_batch_norm3d.cpp) | Builds the graph and calls it through the [operator IR](op_graph/batch_norm3d_proto.h). |
