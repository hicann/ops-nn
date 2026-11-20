# AscendDequant

English | [简体中文](./README.md)

## Applicable Products

|Product|Supported|
|:----------|:----------:|
|<term>Ascend 950PR&950DT series</term>|√|
|<term>Atlas A3 series</term>|×|
|<term>Atlas A2 series</term>|×|
|<term>Atlas 200I/500 A2 inference products</term>|×|
|<term>Atlas inference series</term>|×|
|<term>Atlas training series</term>|×|

## Function

- Function: dequantizes the INT32 input `x` element-wise with the dequantization scale `deq_scale` and writes the FLOAT16 or FLOAT32 result `y`; `deq_scale` may be right-aligned broadcast to `x`.

- Formulas:

    $$
    s = reinterpret\_f32(low\_32\_bits(deq\_scale))
    $$

    $$
    acc = cast\_to\_fp32(x) \times s
    $$

    $$
    acc = relu\_flag\ ?\ \max(acc, 0) : acc
    $$

    $$
    acc = sqrt\_mode\ ?\ acc \times s : acc
    $$

    $$
    y = cast\_to\_dst\_type(acc)
    $$

  Notes:

  - For a UINT64 `deq_scale`, the low 32 bits carry the IEEE-754 bit pattern of the FLOAT32 scale; the high 32 bits are not used.
  - `sqrt_mode=true` multiplies the result by the same scale `s` once more, that is, multiplies by the square of `s`; it is not a square root.
  - The ReLU of `relu_flag` is applied between the two multiplications.
  - A FLOAT16 output is rounded only once (RINT); a FLOAT32 output skips the narrowing cast.

## Parameters

| Parameter | Input/Output/Attr | Description | Data Type | Format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | The tensor to dequantize, `x` in the formulas. It supports 1 to 8 dimensions, its shape equals the output `y`, and empty tensors (any dimension is 0) are supported. | INT32 | ND |
| deq_scale | Input | The dequantization scale, `s` in the formulas. Its shape must be right-aligned broadcastable to `x`, that is, its rank does not exceed that of `x`, and every right-aligned axis equals the corresponding axis of `x` or is 1. | UINT64 | ND |
| sqrt_mode | Optional attr | When true, multiplies the result by the same scale `s` once more. Default false. | BOOL | - |
| relu_flag | Optional attr | When true, applies `max(acc, 0)` between the two multiplications. Default false. | BOOL | - |
| dtype | Optional attr | Output data type: 0 for FLOAT32, 1 for FLOAT16. Default 1, and it must match the data type of the output `y`. | INT | - |
| y | Output | The dequantized result, `y` in the formulas. Its shape equals the input `x` exactly. | FLOAT16, FLOAT32 | ND |

## Data Type Combinations

| x | deq_scale | y | dtype |
| :--- | :--- | :--- | :--- |
| INT32 | UINT64 | FLOAT16 | 1 |
| INT32 | UINT64 | FLOAT32 | 0 |

## Constraints

- `deq_scale` follows right-aligned broadcast semantics (the same as numpy broadcasting): its rank does not exceed that of `x`, and after right alignment every axis either equals the corresponding axis of `x` or is 1; otherwise a shape-related error is reported.
- The shape of the output `y` must equal the input `x` exactly, and its data type must match the `dtype` attribute.
- Empty tensors are supported: when any dimension of `x` is 0, no computation is launched (`usedCoreNum=0`, blockDim=1, and the kernel returns immediately).
- The operator is purely element-wise and uses ND positional semantics, with no format-dependent layout derivation. The NHWC, NCHW, NCDHW and NDHWC formats registered in the operator definition are interface registration only, and are computed exactly like ND.
- `sqrt_mode` is not mutually exclusive with the data type of `deq_scale`; `sqrt_mode=true` is supported with UINT64 as well.

## Invocation Description

| Invocation method | Sample code | Description |
| :--- | :--- | :--- |
| GE graph mode | - | Calls the AscendDequant operator by building a graph through [operator IR](./op_graph/ascend_dequant_proto.h). |

No aclnn interface is provided (`ACLNNTYPE aclnn_exclude`), so only GE graph mode is supported.
