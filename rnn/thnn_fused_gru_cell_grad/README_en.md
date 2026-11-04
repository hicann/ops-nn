# ThnnFusedGruCellGrad

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :--- |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: completes the fused backward gradient computation for a single time step of GRU (gated recurrent unit). Given the upstream gradient `grad_hy (B, H)` and the intermediate result `storage (B, 5H)` saved by the forward `_thnn_fused_gru_cell` (stored in the row in the order of the five H-wide planes `[r, z, n, hx, hn]`), a single kernel computes the 5 gradient outputs in a fused manner — the two gate pre-activation gradients, the hidden-state direct gradient, and the two bias gradients. It is aligned with the PyTorch `aten::_thnn_fused_gru_cell_backward`.
- Formula:

  Let $go = \mathrm{grad\_hy}$. Split the five planes $r, z, n, hx, hn$ from each row of `storage` (each $H$ wide). The five gradient chains are:

  $$\mathrm{gin} = go \cdot (1 - z) \cdot (1 - n^2) \tag{tanh\_backward}$$

  $$\mathrm{gig} = go \cdot (hx - n) \cdot (1 - z) \cdot z \tag{sigmoid\_backward}$$

  $$\mathrm{grg} = \mathrm{gin} \cdot hn \cdot (1 - r) \cdot r \tag{sigmoid\_backward}$$

  $$\mathrm{ghn} = \mathrm{gin} \cdot r$$

  $$\mathrm{ghx} = go \cdot z$$

  The outputs are concatenated as `grad_input_gates = [grg, gig, gin]` and `grad_hidden_gates = [grg, gig, ghn]` (both $(B, 3H)$), and `grad_hx = ghx` ($(B, H)$). The two biases are the reductions of the corresponding gate gradients along the batch dimension (each $(3H,)$). When `has_bias = false`, the two bias outputs are empty tensors $(0,)$.

## Parameters

|Parameter name|Input/Output/Attribute|Description|Data type|Data format|
|-----|-----------|----|---------|------|
|grad_hy|Input|The upstream gradient ∂L/∂hy (the gradient of the forward output hy), shape (B, H), corresponding to go in the formula.|FLOAT, FLOAT16, BFLOAT16|ND|
|storage|Input|The intermediate result saved by the forward _thnn_fused_gru_cell, stored in the row in the order of the five planes [r, z, n, hx, hn], shape (B, 5H). The data type is the same as grad_hy.|FLOAT, FLOAT16, BFLOAT16|ND|
|has_bias|Attribute|Indicates whether the forward input_bias is defined (bool, default false). When it is true, the two bias gradients (3H,) are output. When it is false, the empty tensor (0,) is output.|BOOL|-|
|grad_input_gates|Output|The gate pre-activation gradient on the input side, concatenated in the row as three segments [grg, gig, gin], shape (B, 3H).|FLOAT, FLOAT16, BFLOAT16|ND|
|grad_hidden_gates|Output|The gate pre-activation gradient on the hidden side, concatenated in the row as three segments [grg, gig, ghn], shape (B, 3H).|FLOAT, FLOAT16, BFLOAT16|ND|
|grad_hx|Output|The hidden-state direct gradient, shape (B, H).|FLOAT, FLOAT16, BFLOAT16|ND|
|grad_input_bias|Output|The result of reducing grad_input_gates along the batch dimension, shape (3H,) (has_bias=true) or (0,) (has_bias=false).|FLOAT, FLOAT16, BFLOAT16|ND|
|grad_hidden_bias|Output|The result of independently reducing grad_hidden_gates along the batch dimension, shape (3H,) (has_bias=true) or (0,) (has_bias=false).|FLOAT, FLOAT16, BFLOAT16|ND|

## Constraints

- Data types: the data types of grad_hy and storage must be consistent, and only FLOAT, FLOAT16, and BFLOAT16 are supported. The data types of all outputs are the same as grad_hy.
- Data formats: only ND is supported for the inputs and outputs.
- Dimension constraints: grad_hy and storage must be 2D, and the shape of storage must satisfy (B, 5H) (the second dimension is 5 times H, and the B dimension is consistent with grad_hy).
- Non-contiguous tensors: both inputs and outputs support non-contiguous tensors — inputs are normalized to contiguous through Contiguous before computation, and outputs are scattered back according to the user view through ViewCopy (empty outputs are skipped).
- Empty tensors: empty tensor inputs are supported. B=0 or H=0 is a valid degenerate case — the gate gradients and the grad_hx output are empty. When has_bias=true and B=0, the two bias outputs are all-zero (3H,). When has_bias=false, the two bias outputs are empty tensors (0,).
- Determinism: the default implementation is deterministic, and multiple runs with the same input produce bitwise identical results.

## Invocation Description

- <term>Ascend 950PR&950DT products</term>:

    | Invocation method | Invocation sample | Description |
    |---------|---------|------|
    | aclnn API | [test_aclnn_thnn_fused_gru_cell_grad.cpp](examples/arch35/test_aclnn_thnn_fused_gru_cell_grad.cpp) | Invoke the ThnnFusedGruCellGrad operator through the [aclnnThnnFusedGruCellBackward](docs/aclnnThnnFusedGruCellBackward.md) interface |
    | GE graph mode | - | Invoke the ThnnFusedGruCellGrad operator by constructing a graph through the [operator IR](op_graph/thnn_fused_gru_cell_grad_proto.h) |
