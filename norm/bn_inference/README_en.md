# BNInference

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: Performs inference-mode batch normalization on the input feature map. The output `y` has the same shape, data type, and data format as `x`.
- In the following formulas, `i` denotes the element index, `c` denotes the channel that contains the element, and `momentum_0` denotes the first element of `momentum`.
- Under full batch normalization semantics, when both `scale` and `offset` are missing, the calculation formula is:

  $$
  factor = momentum_0 == 0 ? 0 : 1 / momentum_0
  $$

  $$
  alpha_c = -factor \times mean_c
  $$

  $$
  beta_c = \frac{1}{\sqrt{factor \times variance_c+epsilon}}
  $$

  $$
  y_i = (x_i+alpha_c) \times beta_c
  $$

- Under full batch normalization semantics, when both `scale` and `offset` are present, the value of `momentum` does not participate in the calculation. The calculation formula is:

  $$
  s_c = \sqrt{variance_c+epsilon}
  $$

  $$
  inv\_s_c = \frac{1}{s_c}
  $$

  $$
  beta_c = scale_c \times inv\_s_c
  $$

  $$
  alpha_c = \frac{offset_c}{scale_c} \times s_c-mean_c
  $$

  $$
  y_i = (x_i+alpha_c) \times beta_c
  $$

  For the valid combinations of optional inputs and the default values, refer to [Constraints](#constraints).

Under full batch normalization semantics on <term>Ascend 950PR&950DT products</term>, when only `scale` or only `offset` is present,
the value of `momentum` does not participate in the calculation:

$$
rstd_c = \frac{1}{\sqrt{variance_c+epsilon}}
$$

$$
\begin{aligned}
\text{Only scale:}\quad &y_i = ((x_i-mean_c) \times rstd_c) \times scale_c \\
\text{Only offset:}\quad &y_i = (x_i-mean_c) \times rstd_c+offset_c
\end{aligned}
$$

On <term>Ascend 950PR&950DT products</term>, when `mode=0`, `mean` and `variance` represent the pre-folded coefficients `alpha` and `beta` respectively:

$$
base_i = (x_i+mean_c) \times variance_c
$$

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Feature map to be normalized. | FLOAT, FLOAT16, BFLOAT16 | NC1HWC0, NCHW, NHWC, NCDHW, NDHWC, ND |
| mean | Input | First per-channel parameter. Its specific mathematical meaning is determined by the calculation mode. See the formula and [Constraints](#constraints). | FLOAT, FLOAT16, BFLOAT16 | NC1HWC0, ND |
| variance | Input | Second per-channel parameter. Its specific mathematical meaning is determined by the calculation mode. See the formula and [Constraints](#constraints). | FLOAT, FLOAT16, BFLOAT16 | NC1HWC0, ND |
| momentum | Input | Scaling parameter for the mean and variance. For the conditions where it participates in the calculation, see the formula and [Constraints](#constraints). | FLOAT, FLOAT16, BFLOAT16 | NC1HWC0, ND |
| scale | Optional input | Per-channel scaling parameter, corresponding to `scale` in the formula. For the default behavior, refer to [Constraints](#constraints). | FLOAT, FLOAT16, BFLOAT16 | NC1HWC0, ND |
| offset | Optional input | Per-channel offset parameter, corresponding to `offset` in the formula. For the default behavior, refer to [Constraints](#constraints). | FLOAT, FLOAT16, BFLOAT16 | NC1HWC0, ND |
| epsilon | Optional attribute | Value added to `variance` under full batch normalization semantics. The default value is `1e-5`. | FLOAT | - |
| use_global_stats | Optional attribute | Whether to use global statistics. The default value is `true`. For valid values, refer to [Constraints](#constraints). | BOOL | - |
| mode | Optional attribute | Calculation mode. The default value is `1`. For the value semantics on each product, refer to [Constraints](#constraints). | INT | - |
| y | Output | Batch normalization result. Its shape, data type, and data format are the same as those of `x`. | FLOAT, FLOAT16, BFLOAT16 | NC1HWC0, NCHW, NHWC, NCDHW, NDHWC, ND |

## Constraints

- The data types and data formats in the parameter table are the union of the capabilities of all supported products. When you call the operator, you must use the valid combinations for the corresponding product listed below. You cannot combine data types and data formats arbitrarily.
- <term>Ascend 950PR&950DT products</term>:
  - Data type combinations: The following items represent the data types of `x`, `mean` and `variance`, `momentum`, `scale` and `offset`, and `y` in order, for 11 combinations in total:
    - `(FLOAT, FLOAT, FLOAT, FLOAT, FLOAT)`.
    - `(FLOAT16, FLOAT, FLOAT, FLOAT16, FLOAT16)`.
    - `(BFLOAT16, FLOAT, FLOAT, BFLOAT16, BFLOAT16)`.
    - `(FLOAT16, FLOAT16, FLOAT16, FLOAT16, FLOAT16)`.
    - `(FLOAT16, FLOAT16, FLOAT, FLOAT16, FLOAT16)`.
    - `(BFLOAT16, BFLOAT16, BFLOAT16, BFLOAT16, BFLOAT16)`.
    - `(BFLOAT16, BFLOAT16, FLOAT, BFLOAT16, BFLOAT16)`.
    - `(FLOAT16, FLOAT, FLOAT, FLOAT, FLOAT16)`.
    - `(BFLOAT16, FLOAT, FLOAT, FLOAT, BFLOAT16)`.
    - `(BFLOAT16, FLOAT16, FLOAT16, FLOAT16, BFLOAT16)`.
    - `(FLOAT, FLOAT16, FLOAT16, FLOAT16, FLOAT)`.
  - When `scale` or `offset` is missing, the data type of the missing tensor is not required. When both are present, you must use the data types of the same combination listed above.
  - Static shape and dynamic shape formats of `x` and `y`: NCHW→NCHW, NHWC→NHWC, NCDHW→NCDHW, NDHWC→NDHWC, ND→ND.
  - Input shape constraints: NCHW and NHWC are 4D, NCDHW and NDHWC are 5D, and ND is 4D or 5D. For NCHW and NCDHW, the channel axis is axis 1. For NHWC and NDHWC, the channel axis is the last axis. For ND, the channel axis is determined by the original format (origin format). When the original format is ND, the channel axis is axis 1.
  - Per-channel parameter constraints: The static and dynamic shape formats of `mean`, `variance`, `momentum`, `scale`, and `offset` are all ND. The shapes of `mean`, `variance`, and the present `scale` and `offset` are [C]. The shape of `momentum` is [], [1], or [C].
  - Calculation mode: `mode=0` uses the pre-folded semantics, and `momentum` and `epsilon` do not participate in the calculation. Any non-zero integer uses the full batch normalization semantics.
  - Optional input combinations: Full batch normalization supports both missing, only `scale`, only `offset`, and both present. A missing `scale` is treated as 1 and a missing `offset` is treated as 0. The pre-folded semantics supports both missing, only `scale`, and both present, and does not support only `offset`. The pre-folded results are `base`, `base*scale`, and `base*scale+offset` in order.
  - `use_global_stats`: Both `true` and `false` can be used. The value does not change the calculation result.
  - Empty tensor: Any dimension of `x` can be 0, and the output is an empty tensor with the same shape. All inputs must still satisfy the parameter constraints. When C=0, the shape of each per-channel parameter must be [0].
  - Scale limits: Shapes and element counts are processed as 64-bit values. The actually allocatable size is limited by the available memory of the runtime environment.
  - During cross-product migration, if a full batch normalization call explicitly sets `mode=0`, you must change `mode` to a non-zero value on this product.
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
  - Data types: All inputs and outputs use the same data type. FLOAT, FLOAT16, and BFLOAT16 are supported.
  - Static shape and dynamic shape formats of `x` and `y`: NC1HWC0→NC1HWC0, ND→ND. `x` is a logical 4D or 5D feature map, and `y` has the same shape and format as `x`. NC1HWC0 is a private data format of this product group.
  - Per-channel parameter constraints: The logical shapes of `mean`, `variance`, `momentum`, `scale`, and `offset` are [C]. Both static and dynamic shapes use NC1HWC0 or ND. The parameter formats are the same as that of `x`: all are NC1HWC0 when `x` is NC1HWC0, and all are ND when `x` is ND.
  - Calculation mode: Full batch normalization semantics is used when `mode` is any integer.
  - Optional input combinations: Only both `scale` and `offset` missing or both present are supported.
  - `use_global_stats`: Both `true` and `false` can be used. The value does not change the calculation result.
- <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, and <term>Atlas training products</term>:
  - Data types: All inputs and outputs use the same data type. FLOAT and FLOAT16 are supported.
  - Static shape and dynamic shape formats of `x` and `y`: NC1HWC0→NC1HWC0, ND→ND. `x` is a logical 4D or 5D feature map, and `y` has the same shape and format as `x`. NC1HWC0 is a private data format of this product group.
  - Per-channel parameter constraints: The logical shapes of `mean`, `variance`, `momentum`, `scale`, and `offset` are [C]. Both static and dynamic shapes use NC1HWC0 or ND. The parameter formats are the same as that of `x`: all are NC1HWC0 when `x` is NC1HWC0, and all are ND when `x` is ND.
  - Calculation mode: Full batch normalization semantics is used when `mode` is any integer.
  - Optional input combinations: Only both `scale` and `offset` missing or both present are supported.
  - `use_global_stats`: Both `true` and `false` can be used. The value does not change the calculation result.

## Invocation Description

This operator does not provide an aclnn single-operator interface and only supports GE graph-mode invocation.

The sample shows the data formats and optional input combinations supported by <term>Ascend 950PR&950DT products</term>.

| Invocation method | Invocation sample | Description |
| :--- | :--- | :--- |
| GE graph mode | [test_geir_bn_inference](./examples/arch35/test_geir_bn_inference.cpp) | Builds a graph through [operator IR](./op_graph/bn_inference_proto.h) and calls the BNInference operator. |
