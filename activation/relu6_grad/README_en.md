# Relu6Grad

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: The backward of the activation function Relu6 computes the backpropagation gradient of the corresponding Relu6 operation. For positions where the forward input falls in the open interval `(0, 6)`, the upstream gradient is passed through; other positions (including the endpoints `features = 0` and `features = 6`) output 0.

- Formula:

  $$
  backprops_{i} =
  \begin{cases}
  gradients_{i}, & 0 < features_{i} < 6 \\
  0, & otherwise
  \end{cases}
  $$

  Here:
  - gradients: The backpropagation gradient of the corresponding Relu6 operation.
  - features: The features passed as input to the corresponding Relu6 operation, or its output. Using either one produces the same result.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| gradients | Input | The backpropagation gradient passed to the corresponding Relu6 operation, gradients in the formula. | FLOAT16, FLOAT, BFLOAT16 | ND |
| features | Input | The features passed as input to the corresponding Relu6 operation, or its output. Using either one produces the same result, features in the formula. For the shape relationship with gradients, see Constraints. | FLOAT16, FLOAT, BFLOAT16 | ND |
| backprops | Output | The output tensor in the formula. The shape is determined from the input shapes according to the broadcast constraints of each product, see Constraints. | FLOAT16, FLOAT, BFLOAT16 | ND |

## Constraints

- The data types of gradients, features, and backprops must be consistent. Mixed data types are not supported.
- Dynamic shapes and dynamic ranks are supported.
- <term>Ascend 950PR&950DT products</term>:
  - Input shape constraints: gradients and features support two-way broadcasting, and the rank of each does not exceed 8; GE graph mode supports empty tensors that satisfy the broadcast relationship.
  - Output shape constraints: The output is the broadcast shape of the two inputs; the number of output elements of an empty tensor is 0.
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:
  - Input shape constraints: The shape of features must be consistent with gradients or be [1].
  - Output shape constraints: Consistent with gradients.

## Implementation

| Layer | File | Description |
| --- | --- | --- |
| Graph prototype | `op_graph/relu6_grad_proto.h` | `REG_OP(Relu6Grad)`, two inputs and one output |
| Operator definition | `op_host/relu6_grad_def.cpp` | `OpDef::AddConfig("ascend950", ...)` |
| InferShape | `op_host/relu6_grad_infershape.cpp` | Reuses `Ops::Base::InferShape4Broadcast` |
| Graph-mode InferDataType | `op_graph/relu6_grad_graph_infer.cpp` | Checks that the dtypes of the two inputs are consistent and infers the output dtype |
| Tiling | `op_host/arch35/relu6_grad_tiling_arch35.{h,cpp}` | Calls `Ops::Base::BroadcastBaseTiling<OpDag>` in branches by dtype |
| DAG | `op_kernel/arch35/relu6_grad_dag.h` | Native computation on the fp32 path; the fp16/bf16 path promotes intermediate precision to fp32 |
| Struct | `op_kernel/arch35/relu6_grad_struct.h` | `BRC_TEMP_SCH_MODE_KEY_DECL/SEL` |
| Kernel entry | `op_kernel/relu6_grad_apt.cpp` | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` + `BroadcastSch<schMode, OpDag>` |

### fp32 path

```text
In0(dy) ─CopyInBrc─+
                   +──────────────────────────────+
                                                  |
In1(x)  ─CopyInBrc─+                              |
                   +─Compare(GT, x, 0) ─ mask_gt0 |
                   +─Compare(LT, x, 6) ─ mask_lt6 |
                          mask = And(mask_gt0, mask_lt6)
                                                  |
   Const(0,fp32) ─Duplicate─ Zero ─────────────── ┴ Select(mask, dy, Zero) ─ CopyOut ─ Out0
```

`Vec::Compare<u8, fp32, GT/LT>` outputs a bitmask, and `Vec::And<u8>` combines the `(0 < x < 6)`
interval mask; `Vec::Select<u8, fp32, TENSOR_TENSOR>` selects element by element according to
`mask ? dy : 0`, avoiding branches, implemented with the native hardware vsel.

### fp16 / bf16 path

```text
In0/In1 ─CopyInBrc─ Cast(->fp32) ─+
                                  +─ Compare ── And ─ Select ── Cast(->T,RINT) ─ CopyOut ─ Out0
               Const(0,fp32) ── Zero ─+
               Const(6,fp32) ── Six ──+
```

Promote fp16/bf16 entirely to fp32 for cmp/and/sel, and at the end use `CAST_MODE_RINT`
(round-to-nearest-even) to fall back to the original dtype. Reasons:

1. The comparison and selection operations uniformly use fp32 intermediate precision;
2. The bounds 0.0 and 6.0 can both be represented exactly in fp16/bf16, but running cmp on the fp32 path is more robust,
   and when dy is NaN/Inf, the fp32 intermediate is not truncated by the half-precision range.

## Invocation Description

| Invocation method   | Sample code           | Description                                            |
| ---------------- | --------------------------- |-----------------------------------------------|
| GE graph mode | [test_geir_relu6_grad.cpp](examples/arch35/test_geir_relu6_grad.cpp)  | Calls the Relu6Grad operator by building a graph through the [operator IR](op_graph/relu6_grad_proto.h). |
