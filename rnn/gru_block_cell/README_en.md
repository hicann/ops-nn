# GruBlockCell

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:-----|:-------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: Performs the forward computation of a single-step GRU (Gated Recurrent Unit) cell, aligned with `tf.raw_ops.GRUBlockCell` (reset_before semantics). Given the current-step input x, the previous-step hidden state hPrev, and the fused weights and biases, one call fusion-completes the gating GEMM, candidate GEMM, sigmoid/tanh gate activations, and hidden-state update. Among them, r/u/c are exposed outputs as intermediate quantities required for backpropagation, and h is the new hidden state. B=batch, I=inputSize, H=hiddenSize.
- Formula:

  $$
  ru\_bar = [x,\ h_{prev}] \cdot w_{ru} + b_{ru} \in \mathbb{R}^{B \times 2H}
  $$

  $$
  r = \sigma(ru\_bar[:,\ 0{:}H]), \qquad u = \sigma(ru\_bar[:,\ H{:}2H])
  $$

  $$
  c = \tanh\left([x,\ h_{prev} \odot r] \cdot w_c + b_c\right)
  $$

  $$
  h = u \odot (h_{prev} - c) + c
  $$

  In the formulas, $\sigma$ is sigmoid, $\odot$ is element-wise multiplication, and $[x,\ h_{prev}]$ denotes concatenation along axis 1.
- Semantic key points:
  - reset_before variant: The reset gate r **first** multiplies hPrev element-wise and **then** enters the candidate GEMM. This differs from the reset_after semantics of PyTorch GRU (`torch.nn.functional.gru_cell` is not equivalent to this operator, and no mapping is performed).
  - Gate order r|u: The first H columns of wRu correspond to the r gate, and the last H columns correspond to the u gate.
  - Intra-step serial dependency: The candidate GEMM depends on the output r of the gating GEMM. The two Cube computation segments run serially within a single kernel.
  - Special value semantics: NaN propagates element-wise. ±Inf saturates through sigmoid/tanh ($\sigma(\pm\infty) \in \{0,1\}$ and $\tanh(\pm\infty) = \pm 1$), and no new NaN is generated. For all-zero inputs, the outputs are deterministic values (r=u=0.5, c=0, h=0).

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
|:-------|:--------------|:-----|:--------|:--------|
| x | Input | Current-step input, corresponding to x in the formula, with shape [B, I]. | FLOAT | ND |
| hPrev | Input | Previous-step hidden state, corresponding to $h_{prev}$ in the formula, with shape [B, H]. dim0 is the same as that of x. | FLOAT | ND |
| wRu | Input | Fused r/u gate weight, corresponding to $w_{ru}$ in the formula, with shape [I+H, 2H] (rows: x block + h block; columns: r block \| u block). | FLOAT | ND |
| wC | Input | Fused candidate weight, corresponding to $w_c$ in the formula, with shape [I+H, H]. | FLOAT | ND |
| bRu | Input | <ul><li>Fused r/u gate bias, corresponding to $b_{ru}$ in the formula, with shape [2H].</li><li>Must be rank-1. The [1, 2H] size-1 broadcast view is not supported yet. See Constraints.</li></ul> | FLOAT | ND |
| bC | Input | <ul><li>Candidate bias, corresponding to $b_c$ in the formula, with shape [H].</li><li>Must be rank-1. The [1, H] size-1 broadcast view is not supported yet. See Constraints.</li></ul> | FLOAT | ND |
| r | Output | Reset gate activation (training intermediate quantity), corresponding to r in the formula, with shape [B, H]. | FLOAT | ND |
| u | Output | Update gate activation (training intermediate quantity), corresponding to u in the formula, with shape [B, H]. | FLOAT | ND |
| c | Output | Candidate state (training intermediate quantity), corresponding to c in the formula, with shape [B, H]. | FLOAT | ND |
| h | Output | New hidden state, corresponding to h in the formula, with shape [B, H]. | FLOAT | ND |

The parameter order is consistent with the registration order in [op_host/gru_block_cell_def.cpp](op_host/gru_block_cell_def.cpp) (this file is the source of truth for the operator definition, and the GE IR prototype in [op_graph/gru_block_cell_proto.h](op_graph/gru_block_cell_proto.h) corresponds to it item by item). This operator has no attributes. The attribute surface aligned with TF REGISTER_OP only has the dtype constraint T: {float}.

## Constraints

- Data types: Only float32. The dtypes of the 6 inputs and 4 outputs must be the same. No fp16/bf16 path is available.
- Data format: Only ND.
- Shape ranges: B∈[1, 2³¹−1], I∈[1, 65535], H∈[1, 65535]. x/hPrev/wRu/wC are rank-2, and bRu/bC are rank-1.
- Empty tensor support: Not supported. B, I, and H must all be ≥1.
- Bias broadcast view limitation: The [1, 2H]/[1, H] size-1 broadcast views are currently rejected at the tiling stage (GRAPH_FAILED). Rationale (recorded from actual tests): the GEIR online graph path has a framework-level defect in delivering rank-2 size-1 view inputs. After the view is allowed through, the pass1 (c/h) outputs are numerically corrupted while pass0 (r/u) is normal, and this is independent of the bias values (zero bias still fails, wC=0 recovers). Reproduction and disappearance depend only on changing the declared input shape. The original kernel reproduces the issue as well (it is not introduced by this implementation). Allowing the view would turn a clean error into silent numerical corruption, so the rank-1 requirement is maintained and will be relaxed after the framework is fixed (at that point the kernel reads a flat ND contiguous table and needs no change).
- On-chip capacity constraint: The actually supported range of the hidden-state dimension is H≤8152 (Hp≤8152 after alignment to 8). An excessive H is rejected at the host tiling stage (GRAPH_FAILED) and is not dispatched to the device for execution. The boundary is determined by the L1 full-width feedback slot aH of h⊙r (the h-segment Mmad of pass1 requires the entire A matrix and cannot be split along H). The capacity figure uses the platform query value. Near the upper bound, the column tile width nL0c is forced to narrow, and throughput drops significantly (H=8152 is about 1/8 of H=6144). This is graceful degradation, not a functional defect.
- Weight layout validation moved earlier: Constraints such as x/hPrev rank-2, x.shape[0]==hPrev.shape[0], wRu [I+H, 2H], wC [I+H, H], and element counts 2H/H of bRu/bC are validated at the InferShape stage (better than the kernel runtime validation of TF). Violations report `shape_mismatch`.
- Accuracy: Compared against an independent fp64 golden model, with tolerance rtol=1e-5/atol=1e-6. When K=I+H exceeds the noise calibration range (K>2048), an accuracy floor caused by fp32 L0C sequential accumulation exists. Actual measurement shows a max_abs of about 1.85e-6 at K=8160. This floor is suppressed by split-K grouping (group width 16) plus Neumaier compensation merge on the AIV side. The group width is an accuracy contract, not a tuning knob.
- Implementation form: fp32 is the only path, and tilingKey is always 0 with a single tier (unified kernel for static shapes). GM data movement has no 32B alignment requirement, and any valid shape can execute.

## Invocation Description

Ascend 950PR&950DT products:

| Invocation method | Invocation sample | Description |
|:--------|:--------|:-----|
| GE graph mode | [test_geir_gru_block_cell](examples/arch35/test_geir_gru_block_cell.cpp), [test_geir_gru_block_cell_dynamic](examples/arch35/test_geir_gru_block_cell_dynamic.cpp) | Calls the GruBlockCell operator by building a graph with `op::GruBlockCell` through [operator IR](op_graph/gru_block_cell_proto.h), and compiles and executes the graph through `ge::Session`. The former is the static shape path (13 suites, including capacity boundaries and special values), and the latter covers the dynamic shape-setting path with the two unknown dimension declarations `-1`/`-2`. Compile and run: `bash build.sh --run_example gru_block_cell graph --soc=ascend950`. |
| TensorFlow native interface | [gru_block_cell_tf_plugin](framework/gru_block_cell_tf_plugin.cpp) | `tf.raw_ops.GRUBlockCell` is automatically mapped 1:1 to the GruBlockCell operator through the TF plugin. The input/output order, shapes, and fused weight layout are exactly the same on both sides, so no weight rearrangement or subgraph expansion is required. Online execution depends on the TF-NPU adapter `npu_device`, and the operator must be called within `tf.function`. |
| aclnn API | - | This operator is delivered GEIR-only and does not provide an aclnn interface. |
| PyTorch API | - | No torch_extension interface is provided. In addition, PyTorch GRU uses reset_after semantics, is not equivalent to this operator, and is not mapped. |
