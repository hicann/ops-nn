# INTrainingUpdateGradGammaBeta

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:---|:---:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

This operator performs parameter gradient reduction for InstanceNorm training backpropagation. It sums `res_gamma` and `res_beta` along dimension 0 respectively and keeps the reduced dimension, producing `pd_gamma` and `pd_beta`:

$$
pd\_gamma = \sum_{n=0}^{N-1} res\_gamma[n, \ldots], \qquad
pd\_beta = \sum_{n=0}^{N-1} res\_beta[n, \ldots].
$$

The output shape is the same as the input shape, but dimension 0 is fixed to 1.

## Parameters

<table>
<col style="width: 170px"><col style="width: 170px"><col style="width: 200px"><col style="width: 200px"><col style="width: 170px">
<tr><td>Parameter name</td><td>Input/Output/Attribute</td><td>Description</td><td>Data type</td><td>Data format</td></tr>
<tr><td>res_gamma</td><td>Input</td><td>Partial sum of the `gamma` gradient, with dimension 0 as the reduction axis.</td><td>FLOAT</td><td>NCHW, NHWC, NCDHW, NDHWC, ND, NDC1HWC0</td></tr>
<tr><td>res_beta</td><td>Input</td><td>Partial sum of the `beta` gradient. Its shape, data type, and data format must be identical to those of `res_gamma`.</td><td>FLOAT</td><td>Same as `res_gamma`</td></tr>
<tr><td>pd_gamma</td><td>Output</td><td>Result of summing `res_gamma` along dimension 0 and keeping the reduced dimension.</td><td>FLOAT</td><td>Same as input</td></tr>
<tr><td>pd_beta</td><td>Output</td><td>Result of summing `res_beta` along dimension 0 and keeping the reduced dimension. Its shape is the same as that of `pd_gamma`.</td><td>FLOAT</td><td>Same as input</td></tr>
</table>

### Product Differences

| Product | Parameter or scenario | Static shape capability | Dynamic shape capability | Shape/rank and combination restrictions |
|:---|:---|:---|:---|:---|
| <term>Ascend 950PR&950DT products</term> | Inputs and outputs are both FLOAT | NCHW, NHWC, NCDHW, NDHWC, ND | NCHW, NHWC, NCDHW, NDHWC, ND, with dynamic rank support | NCHW and NHWC correspond to 4D; NCDHW and NDHWC correspond to 5D; ND supports 4D or 5D; the empty tensor semantics described below are supported; NDC1HWC0 is not supported. |
| <term>Atlas A3 products</term><br><term>Atlas A2 products</term><br><term>Atlas 200I/500 A2 inference products</term><br><term>Atlas inference products</term><br><term>Atlas training products</term> | The public GE IR logical interface is FLOAT and 4D NCHW/NHWC; CANN built-in internal execution uses NDC1HWC0 | The internal execution/storage format is NDC1HWC0 | For existing products that support dynamic shapes, the internal execution/storage format for unknown shapes is NDC1HWC0 | NDC1HWC0 is a built-in internal format and is not the user input format for <term>Ascend 950PR&950DT products</term>; the new 5D, ND, and empty tensor extensions added in this directory do not change the behavior of existing products. |

## Constraints

- This operator has no attributes. The reduction axis is fixed to dimension 0, and the reduced dimension is kept.
- The two inputs must have the same shape, data type, and data format. Broadcasting is not supported.
- <term>Ascend 950PR&950DT products</term>
  - The input rank is always 4 or 5, and this applies to all data formats (ND is no exception): NCHW and NHWC correspond to 4D inputs, and NCDHW and NDHWC correspond to 5D inputs. ND serves as a dense tensor execution view (element-wise identical to the dense layouts of the four public formats), and its rank must also be 4 or 5. Inputs with rank 0, 1, 2, 3, or 6 or greater are rejected.
  - There is no additional upper limit on dimension values: each dimension is a non-negative int64 (the actual limit is constrained by device memory and the total number of elements across the full shape), and no per-dimension numeric cap is set. The boundary behavior of dimension 0 is consistent with that of other dimensions (a 0 dimension triggers the empty-set/empty-tensor semantics, see below).
  - Numerical semantics: the operator accumulates in fp32 and sums in dimension 0 order. Internally it can select different compensation strategies based on the input, and externally it is accepted uniformly against fp32sum precision tolerances. A column that contains NaN outputs NaN; a column that contains both +Inf and -Inf outputs NaN; when a finite true value exceeds the FLOAT range, the output is an Inf with the corresponding sign. This operator does not guarantee bit-level consistency with implementations on other products or with different tiling configurations.
  - Dynamic shape and dynamic rank are supported: during shape/dtype inference, unknown-rank inputs are passed through as is. For a shape that contains -1 unknown dimensions, non-reduction dimensions are copied as is, and dimension 0 is still set to 1 by the reduction semantics (for example, input [-1,64,3,5] infers output [1,64,3,5]). During execution, the concrete shape after materialization must still satisfy the rank constraints above, and every dimension must be non-negative (any negative dimension is rejected).
  - Extended empty tensor semantics: when dimension 0 is 0, the operator performs an empty-set sum and the output element is 0; when a non-reduction dimension is 0, the output is an empty tensor.
  - Only the dense data formats listed in the parameter table are supported. `NDC1HWC0` and `NC1HWC0` are not supported.
- Both inputs and outputs must be contiguous dense tensors (caller memory obligation): non-contiguous views such as transposes, slices, and stride changes are not supported. When the caller holds such views, it must first convert them to contiguous memory before building the graph. A contiguous view with a storage offset is equivalent to a dense tensor based at the data start address. The output must be a contiguous dense tensor, and non-contiguous outputs are not supported.
- Inputs are read-only and outputs are written independently (caller memory obligation): memory overlap (alias) between inputs and outputs and in-place modification are not supported, and input content remains unchanged after the invocation.
- The operator rejects the following constraints during GE graph compilation (shape/dtype inference or tiling validation) and returns a failure, so graph execution does not start: data types and data formats (including consistency between the two inputs), rank and shape consistency, pairing of the origin data format with the rank (unknown-rank inputs are deferred during inference and rechecked in tiling after materialization), and negative values of dynamic dimensions. Each rejection branch produces operator-side diagnostic logs. You can obtain the failure reason through GE runtime logs (plog) or `GEGetErrorMsgV2`. This operator does not promise public numeric error codes, and the error semantics are subject to the log messages.
- The output shape and dtype are determined by inference: in GE graph mode, the framework allocates the output tensor according to the inference result (the shape is the input shape with dimension 0 set to 1, and the dtype is float32). If the execution path retains the concrete output shape or dtype declared by the caller, the operator validates it against the inference result.
- Contiguity and memory non-overlap are not validated during graph compilation: shape/dtype inference and tiling validation have no input channel for memory addresses or strides. Invocations that violate the contiguity or memory overlap obligations are not rejected at compilation time, and the behavior is undefined.
- This directory does not provide an aclnn single-operator interface or a PyTorch/TensorFlow extension path. The only invocation method is GE graph mode.

## Invocation Description

| Invocation method | Sample code | Description |
|:---|:---|:---|
| GE graph mode | [Graph construction sample](examples/test_geir_in_training_update_grad_gamma_beta.cpp) | Creates an `INTrainingUpdateGradGammaBeta` node through the [operator IR](op_graph/in_training_update_grad_gamma_beta_proto.h). |
