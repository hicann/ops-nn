# INTrainingReduceV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: serves as the reduction stage of the Instance Normalization training forward pass.
  It computes the element sum `sum` and the square sum `square_sum` over the spatial dimensions for each instance channel respectively, and keeps the reduction dimensions.
  This operator is used together with INTrainingUpdateV2.
- Formula:

  $$
  sum_{n,c} = \sum_{r \in R} x_{n,c,r}
  $$

  $$
  square\_sum_{n,c} = \sum_{r \in R} x_{n,c,r}^{2}
  $$

  Here $R$ is the set of spatial dimensions. The outputs are the raw sums, and no mean scaling is performed.
  FLOAT16 inputs are converted to FLOAT before multiplication and accumulation, and the data types of both outputs are FLOAT.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| x | Input | The input tensor to be reduced, corresponding to `x` in the formula. | FLOAT, FLOAT16 | NC1HWC0, NDC1HWC0, ND, NCHW, NHWC, NCDHW, NDHWC |
| sum | Output | The result of summing `x` along the spatial dimensions. The instance dimension and the channel dimension are kept, and the size of each reduced dimension is 1. | FLOAT | NC1HWC0, NDC1HWC0, ND |
| square_sum | Output | The result of the square sum of `x` along the spatial dimensions. The shape is the same as `sum`. | FLOAT | NC1HWC0, NDC1HWC0, ND |

## Constraints

- The logical shape of NCHW/NHWC inputs is 4D, and the spatial dimensions are H and W. NCDHW/NDHWC inputs are 5D, and the spatial dimensions are D, H, and W.
- <term>Ascend 950PR&950DT products</term>:
  - For static shapes, `x` is NCHW, NCDHW, ND, NHWC, or NDHWC, and the two outputs are ND. Dynamic shapes of these formats and ND dynamic rank are supported. NHWC/NDHWC inputs keep the corresponding formats, NCHW/NCDHW/ND inputs use ND storage, and both outputs are ND.
  - The NCHW input is `[N,C,H,W]` and the output is `[N,C,1,1]`; the NCDHW input is `[N,C,D,H,W]` and the output is `[N,C,1,1,1]`.
  - The NHWC input is `[N,H,W,C]` and the output is `[N,1,1,C]`; the NDHWC input is `[N,D,H,W,C]` and the output is `[N,1,1,1,C]`.
  - The ND input rank is 2~8. dim0 and dim1 are the instance dimension and the channel dimension respectively, and dim2 and later dimensions are spatial dimensions reduced to 1.
  - Each dimension of `x` must be greater than 0. Empty tensors are not supported.
  - The total number of input elements (the product of all dimension sizes) must not exceed `2^63-1`.
  - The channel row byte count `C*sizeof(x.dtype)` of NHWC/NDHWC inputs must not exceed `2^40-1`, that is, `C` of FLOAT16 must not exceed `2^39-1`, and `C` of FLOAT must not exceed `2^38-1`.
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, and <term>Atlas training products</term>:
  - The 4D NCHW/NHWC original formats are mapped to NC1HWC0, and the 5D NCDHW/NDHWC original formats are mapped to NDC1HWC0. The storage formats of the input and the output are consistent.
  - The registered dynamic shape and dynamic rank paths of the corresponding models use NC1HWC0 or NDC1HWC0 storage, and the format mapping is consistent with the static shape.
  - The original input is 4D or 5D. The logical output keeps N and C, and H, W or D, H, W is 1. The storage format is mapped by the graph engine.

## Invocation Description

This operator does not provide an aclnn single-operator interface. It is invoked in the graph mode of the graph engine (Graph Engine, GE).

| Invocation method | Invocation sample | Description |
| --- | --- | --- |
| GE graph mode | [test_geir_in_training_reduce_v2](examples/arch35/test_geir_in_training_reduce_v2.cpp) | Construct a graph and invoke the operator through the [INTrainingReduceV2 operator IR](op_graph/in_training_reduce_v2_proto.h). |
