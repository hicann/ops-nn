# LpNormReduce

English | [简体中文](./README.md)

## Applicable Products

| Product                                                     | Support |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                   |    √     |
| <term>Atlas A3 products</term> |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                  |    √     |
| <term>Atlas inference products</term>                          |    √     |
| <term>Atlas training products</term>                          |    √     |

## Function

- Operator function: Computes the reduction segment of the Lp norm. After taking the absolute value of the input `x` element-wise, the operator performs reduction along the axes specified by the attribute `axes`, and the reduction operator is determined by the attribute `p`. This operator only performs the reduction and does not perform the final $\frac{1}{p}$-th root. The complete Lp norm is obtained by chaining LpNormReduce and the matching LpNormUpdate operators. Currently, the LpNormUpdate of the legacy chain supports only FLOAT16 and FLOAT, so BFLOAT16 is only a single-node capability of this operator.
- Formula: Let $a_i = \lvert x_i \rvert$, and let $S$ be the normalized set of reduction axes (an empty `axes` means all axes; negative axes are converted by `axis + rank`; duplicate axes are deduplicated).

  When `p` is 0, count the number of non-zero elements:

  $$
  y = \sum_{i \in S}{[a_i \neq 0]}
  $$

  When `p` is 1:

  $$
  y = \sum_{i \in S}{a_i}
  $$

  When `p` is another non-negative finite integer:

  $$
  y = \sum_{i \in S}{a_i^p}
  $$

  When `p` is 2147483647 (representing $+\infty$):

  $$
  y = \max\limits_{i \in S}{a_i}
  $$

  When `p` is -2147483648 (representing $-\infty$):

  $$
  y = \min\limits_{i \in S}{a_i}
  $$

## Parameters

| Parameter name  | Input/Output/Attribute | Description | Data type | Data format |
| ------- | -------------- | ---- | -------- | -------- |
| x       | Required input       | `x` in the formula, the tensor to be reduced. The rank range is 0 to 8, and dynamic shape and dynamic rank are supported. | FLOAT, FLOAT16, BFLOAT16 | ND |
| y       | Required output       | Output `y` in the formula. The data type is the same as `x`, and the shape is determined by `axes` and `keepdim`. | FLOAT, FLOAT16, BFLOAT16 | ND |
| p       | Optional attribute       | `p` in the formula, the integer order of the norm. The default value is 2. 2147483647 represents positive infinity, and -2147483648 represents negative infinity. Otherwise, `p` must be a non-negative integer. | INT | - |
| axes    | Optional attribute       | Set of reduction axes $S$ in the formula. The default value is {}. An empty list means reduction over all axes. The value range is [-rank(`x`), rank(`x`)), and negative axes and duplicate axes are supported. | LISTINT | - |
| keepdim | Optional attribute       | Determines whether the output tensor keeps the reduction axes. When it is true, each reduction axis is kept with length 1; when it is false, the reduction axes are removed, and full-axis reduction outputs a 0D scalar. The default value is false. | BOOL | - |
| epsilon | Optional attribute       | Attribute kept for compatibility with the legacy ABI. It does not participate in the computation of this operator and is consumed by the subsequent LpNormUpdate. The default value is 1e-12. | FLOAT | - |

### Product Differences

| Product | Implementation and data types | shape/rank capabilities and restrictions |
| ---- | -------------- | -------------------- |
| <term>Ascend 950PR&950DT products</term> | Uses the Ascend C implementation of this repository; supports FLOAT16, FLOAT, BFLOAT16, and the ND format. | Supports static shape, dynamic shape, and dynamic rank, with a rank range of 0 to 8. Except for the two infinity sentinels, the complete non-negative INT64 `p` range is supported, and finite-order power operations keep the original integer values. BFLOAT16 applies only to the single-node LpNormReduce; the LpNormUpdate of the current legacy chain does not support BFLOAT16. |
| <term>Atlas A3 products</term><br><term>Atlas A2 products</term><br><term>Atlas 200I/500 A2 inference products</term><br><term>Atlas inference products</term><br><term>Atlas training products</term> | Uses the TBE implementation built into CANN; supports FLOAT16, FLOAT, and the ND format, and does not support BFLOAT16. | Supports static shape, dynamic shape, and dynamic rank, with a rank range of 0 to 8. |

## Constraints

- This operator is a compatibility operator for legacy GEIR solidified graphs. For new scenarios, use the LpNormReduceV2 operator.
- This operator outputs only the reduction result of the Lp norm and does not perform the $\frac{1}{p}$-th root. For FLOAT16/FLOAT, the complete norm can be obtained by chaining the matching LpNormUpdate operator. The LpNormUpdate of the current legacy chain does not support BFLOAT16.
- The following input boundary descriptions describe the implementation on <term>Ascend 950PR&950DT products</term>.
- The input rank range is 0 to 8. A rank-0 input allows only an empty `axes`; with a known rank, an out-of-bounds axis reports an error, and duplicate axes are deduplicated by their first occurrence.
- During graph inference, unknown rank and dynamic dimensions with the value -1 are allowed, and other negative dimensions are illegal. During execution, both the logical shape and the storage shape of `x` must be concrete; for a non-scalar input, the rank and every dimension of the logical shape must be exactly the same as those of the storage shape, and a rank-0 scalar allows the storage shape to be `[]` or `[1]`. During execution, the logical shape and the storage shape of `y` must be concrete and dimension-wise consistent with the result derived from `axes` and `keepdim`; for a scalar output, the logical/storage shape allows both materialization forms `[]` and `[1]`. To ensure that byte offset and alignment calculations are representable, the product of all non-zero dimensions of `x` must not exceed `floor((INT64_MAX-4096)/4)` (including empty tensors). The actually runnable shape is also limited by the device and host memory capacities.
- When the reduction domain of a finite `p` is empty, the output is 0; if a zero-length dimension is located only on a non-reduction axis, the output is an empty tensor. When the positive or negative infinity sentinel encounters an empty reduction domain, an error is reported because `max`/`min` is undefined.
- Except for the -2147483648 sentinel, a finite negative `p` reports an error. Non-negative values of 2147483648 and greater are still processed as finite integer orders.
- For FLOAT16 and BFLOAT16 inputs, the intermediate processes of absolute value, power, and reduction accumulation are completed in FLOAT, and the result is converted back to the input data type at the end. The output data type is always the same as the input data type.
- The total size of aligned output rows at each layer of the reduction tree must not exceed the 16KiB cache. If it exceeds this size, the Host rejects the shape to avoid a device-side cache overflow.
- When `axes` specifies an axis of `x` with length 1, the computation result may have precision differences.
- Only the GEIR/Kernel compatibility path is provided, and no new aclnn API is added.

## Invocation Description

| Invocation method | Sample code | Description |
| -------- | -------- | ---- |
| Graph mode   | -        | Constructs a graph through the [LpNormReduce operator IR entry](op_graph/lp_norm_reduce_proto.h). This header file keeps the REG_OP declaration with redefinition-guard macros, and no aclnn API is provided. |
