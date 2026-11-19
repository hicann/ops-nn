# MVNV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----: |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |
| Atlas 200I/500 A2 inference products | × |
| Atlas inference products | × |
| Atlas training products | × |

## Function

- Operator function: computes the mean and standard deviation of a 4-D NCHW input tensor along the specified axes and performs normalization. In the ideal mathematical sense, the output has a mean of 0 along the reduction axes; when the standard deviation is significantly greater than eps, the variance approaches 1.

- Formula:

$$
y = \frac{x - mean}{\sqrt{var} + eps}
$$

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Denotes the 4-D NCHW input tensor, which is `x` in the formula. | FLOAT16, FLOAT | NCHW |
| y | Output | Denotes the normalized output tensor, which is `y` in the formula. The shape and dtype are the same as those of x. | FLOAT16, FLOAT | NCHW |
| eps | Optional attribute | Denotes the floating-point value added to std (std + eps). The default value is 1e-9. Negative and non-finite values propagate according to IEEE floating-point semantics. | FLOAT | - |
| axes | Optional attribute | Denotes the list of reduction axes. Each element must be in the range [-4, 4); negative values are normalized as axis + 4. If the parameter is not specified or is an empty list, [0, 2, 3] is used by default. Duplicate axes are deduplicated, and the order does not affect the result. | INT64 | - |

## Constraints

- x must be a 4-D tensor. The graph description may use the dynamic dimension marker -1, and InferShape propagates it to the output. Before entering runtime tiling, however, the storage shape must have been materialized as 4-D. The tiling stage rejects any non-4-D input and any negative dimension.
- After negative values are normalized, each element of axes must be in [0, 4). Out-of-range values are rejected by the tiling stage. If axes is not specified or is an empty list, [0, 2, 3] is used by default. A non-empty axes list is treated as a set: duplicate axes are idempotent, and the order does not affect the result.
- eps is constrained only to the FLOAT data type. Negative values, NaN, and ±Inf are compatible inputs and propagate according to the literal formula and IEEE floating-point semantics. A genuine FP32 subnormal eps follows the unscaled compatibility path, and the execution of subnormal values still depends on the platform FTZ mode.
- The denominator is strictly computed as `sqrt(var) + eps` according to the formula, with no additional clamping to the minimum normal number. When the variance is 0 and eps is 0, the result propagates according to floating-point division-by-zero semantics.
- FLOAT16 inputs are promoted to FLOAT for mean and variance statistics, and FLOAT inputs use FLOAT intermediate computation. Precision reduction is disabled in OpDef, so the FLOAT graph path cannot be automatically converted to FLOAT16. The long reduction path uses compensated accumulation.
- When a reduction group is a finite constant and eps is greater than 0, the output is 0. NaN and Inf are not treated as constant groups and propagate according to floating-point operations. Subnormal inputs and intermediate results follow the floating-point execution mode of the current Ascend platform. Accuracy acceptance follows the cross_check L0 standard at precision 2.1.
- Empty tensor inputs are valid: an empty tensor output of the same shape is returned directly, and no kernel computation is executed.
- Only the FLOAT16 and FLOAT data types are supported, and the output shape and data type must be consistent with those of the input. A conflicting description is rejected if it reaches tiling.
- The logical format of inputs and outputs is NCHW. OpDef declares AutoContiguous for inputs and outputs, and the framework performs contiguity conversion when needed. Tiling accepts NCHW or GE-derived dense ND storage, while the kernel itself still requires contiguous storage.
- The product of the 4-D shape must be representable by INT64, and the arch35 DMA row stride must be less than 2^40 bytes; otherwise, tiling rejects the input. The actual executable scale is also subject to the tensor description capability of the graph framework, available device memory, and runtime resource limits. Internally, the implementation uses INT64 to represent shapes, strides, and element offsets.
- Invalid inputs such as rank, dimensions, shape product, input/output description, DMA stride, UB budget, and axes are rejected at the earliest stage reachable among GE graph validation, compilation, and operator tiling. When the rejection occurs in tiling, GRAPH_FAILED is returned, and the runtime typically reports EZ9999. When the framework rejects the input at an earlier stage, the error code is defined by the corresponding GE stage. The operator's log messages use the operator name as the prefix and include the invalid parameter and its actual value.

## Invocation Description

| Invocation method | Invocation sample | Description |
| :--- | :--- | :--- |
| GE graph mode | [test_geir_mvn_v2](./examples/test_geir_mvn_v2.cpp) | Invokes the operator in the Graph Engine graph mode through the operator IR definition [op_graph/mvn_v2_proto.h](./op_graph/mvn_v2_proto.h). |
