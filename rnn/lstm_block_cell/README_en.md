# LstmBlockCell

English | [简体中文](./README.md)

The LSTM (long short-term memory network) **single-step forward fully-fused operator** is aligned with the TensorFlow `LstmBlockCell` (the single-step semantics of `tf.raw_ops.LSTMBlockCell`, TensorFlow path). Given the current-step input `x`, the previous cell state `cs_prev` and hidden state `h_prev`, the merged projection weight `w`, and the bias `b`, one call completes the projection GEMM, the ICFO four-gate activation, and the state update, and outputs the seven `[B, H]` tensors `i`/`cs`/`f`/`o`/`ci`/`co`/`h`.

This operator is **delivered as GEIR only**: it is invoked in GE graph mode by constructing a graph with `op::LstmBlockCell`, or the TF-side operator `LstmBlockCell` with the same name is automatically mapped to this operator through the TensorFlow plugin.

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- **Operator function**: LSTM single-step cell forward computation (B=batch, I=inputSize, H=hiddenSize). The fusion completes the projection GEMM and the ICFO gate activation. `i`/`f`/`o`/`ci` are gate and candidate intermediate values; `cs`/`co`/`h` are the new cell state, the new output state, and the new hidden state.
- **Formula** (gate order `i|ci|f|o`, σ is sigmoid, ⊙ is element-wise multiplication, `[x, h_prev]` denotes concatenation along axis 1):

$$
gates = [x,\ h_{prev}] \cdot w + b \in \mathbb{R}^{B \times 4H}
$$

$$
i = \sigma\left(gates[:,\ 0{:}H] + p_i\right), \qquad
ci = \tanh\left(gates[:,\ H{:}2H]\right)
$$

$$
f = \sigma\left(gates[:,\ 2H{:}3H] + f_b + p_f\right)
$$

$$
cs = \mathrm{clip}\left(f \odot cs_{prev} + i \odot ci,\ \pm c_{clip}\right)
$$

$$
co = \tanh(cs), \qquad
o = \sigma\left(gates[:,\ 3H{:}4H] + p_o\right), \qquad
h = o \odot co
$$

  Here `forget_bias` ($f_b$, default 1.0) is added before the f gate sigmoid. When `cell_clip` ($c_{clip}$, default 3.0) is greater than 0, cs is clipped to $[-c_{clip}, c_{clip}]$; when it equals 0, no clipping is performed. $p_i/p_f/p_o$ are peephole terms. When `use_peephole` (default false) is true, they are $cs_{prev} \odot wci$, $cs_{prev} \odot wcf$, and $co \odot wco$ respectively; otherwise they are 0.

- **Semantic key points**:

  - **Single-step cell**: aligned with `tf.raw_ops.LSTMBlockCell` (8 inputs, no seq_len_max), not the multi-step loop form of `BlockLSTMV2`.
  - **Gate order i|ci|f|o**: in the column order of `w`, the first H columns are the i gate, the next H columns are ci, the next H columns are the f gate, and the last H columns are the o gate (ICFO, not IFCO).
  - **Peephole weights are always passed in**: when `use_peephole` is false, `wci/wcf/wco` do not participate in the computation, but the 8 input slots are fixed (the TF side uses zero padding instead of absence by convention).
  - **Special value semantics**: ±Inf saturates through sigmoid/tanh (σ(±Inf) ∈ {0,1}, tanh(±Inf) = ±1). Gate activation and cell_clip clamping are implemented with Compares + Select (a comparison is always false for NaN, the original value takes the else branch, NaN propagates element-wise, and the operand order is the same as std::min/std::max of TF).
  - **Implementation form**: MIX 1 AIC : 2 AIV. The operator performs full-axis splitting on M (batch)/N (hidden)/K (I+H). Any valid shape with B/I/H ≥ 1 can execute, and oversized batches are automatically consumed in multiple rounds (no capacity rejection).

## Parameters

|Parameter name|Input/Output/Attribute|Description|Data type|Data format|
|-----|-----------|----|---------|------|
|x|Input|Current-step input, shape [B, I]|FLOAT/FLOAT16|ND|
|cs_prev|Input|Cell state of the previous step, shape [B, H]; dim0 is the same as x|FLOAT/FLOAT16|ND|
|h_prev|Input|Hidden state of the previous step, shape [B, H]; dim0 is the same as x|FLOAT/FLOAT16|ND|
|w|Input|Merged projection weight, shape [I+H, 4H] (rows: x block + h block; column order i\|ci\|f\|o)|FLOAT/FLOAT16|ND|
|wci|Input|Peephole weight of the i gate, shape [H]|FLOAT/FLOAT16|ND|
|wcf|Input|Peephole weight of the f gate, shape [H]|FLOAT/FLOAT16|ND|
|wco|Input|Peephole weight of the o gate, shape [H]|FLOAT/FLOAT16|ND|
|b|Input|Bias, shape [4H] (column order i\|ci\|f\|o)|FLOAT/FLOAT16|ND|
|i|Output|Input gate activation, shape [B, H]|FLOAT/FLOAT16|ND|
|cs|Output|New cell state, shape [B, H]|FLOAT/FLOAT16|ND|
|f|Output|Forget gate activation, shape [B, H]|FLOAT/FLOAT16|ND|
|o|Output|Output gate activation, shape [B, H]|FLOAT/FLOAT16|ND|
|ci|Output|Candidate state, shape [B, H]|FLOAT/FLOAT16|ND|
|co|Output|New output state (tanh(cs)), shape [B, H]|FLOAT/FLOAT16|ND|
|h|Output|New hidden state, shape [B, H]|FLOAT/FLOAT16|ND|
|forget_bias|Attribute|Forget gate bias bonus, added before the f gate sigmoid; default 1.0|floating point|-|
|cell_clip|Attribute|Cell state clipping magnitude; when it is greater than 0, cs is clipped to [-cell_clip, cell_clip]; default 3.0|floating point|-|
|use_peephole|Attribute|Whether to enable peephole connections; default false|boolean|-|

## Constraints

- **Data types**: data types other than float32/float16 are not supported. The dtype consistency of the 8 inputs and 7 outputs is checked at two levels, graph-level deduction and tiling; violations return `GRAPH_FAILED`.
- **Data formats**: ND only.
- **Shape constraints**: the dimensions of x/cs_prev/h_prev/w must be 2, and the dimensions of wci/wcf/wco/b must be 1 (checked in the tiling stage; violations return `GRAPH_FAILED` and output an error log). B/I/H take values in [1, 2^24]. The total number of elements of a single tensor must be < 2^32 (the kernel addresses elements with uint32 offsets). The shape contract (h_prev/cs_prev are [B, H], w is [I+H, 4H], b is [4H], and wci/wcf/wco are [H]) is also checked in the tiling stage.
- **Empty inputs**: B/I/H are all ≥ 1; zero-value dimensions are rejected in the tiling stage.
- **Attributes**: all values of the 3 attributes are supported, with no rejection path (`cell_clip=0` means no clipping).
- **Dynamic shapes**: -1 unknown dimensions and -2 unknown rank declarations are supported (checked and shaped using the actually supplied shape at execution time).
- **Implementation form**: the fp32/fp16 dual tilingKeys (0/1) use a unified kernel for static shapes. GM transfer has no 32B alignment requirement, and any valid shape can execute.

## Invocation Description

Ascend 950PR&950DT products:

| Invocation method | Invocation sample | Description |
|---------|---------|------|
| GE graph mode | [test_geir_lstm_block_cell](examples/arch35/test_geir_lstm_block_cell.cpp) | Construct a graph with `op::LstmBlockCell` through the [operator IR](op_graph/lstm_block_cell_proto.h), then compile and execute it through `ge::Session` |
| GE graph mode (dynamic shape) | [test_geir_lstm_block_cell_dynamic](examples/arch35/test_geir_lstm_block_cell_dynamic.cpp) | Declare -1 unknown dimensions/-2 unknown rank, and call `RunGraph` multiple times on the same graph with different shapes |
| Native TensorFlow interface | [lstm_block_cell_tf_plugin](framework/lstm_block_cell_tf_plugin.cpp) | The TF-side operator `LstmBlockCell` with the same name is automatically mapped to this operator through the TF plugin |

This operator provides only the two GE graph mode paths above and does not provide a single-operator C API for direct invocation.

### Graph-mode invocation

Prerequisite: after you build and install the operator package, GE resolves the prototype and tiling library of this operator through `$ASCEND_OPP_PATH/vendors/<vendor_name>_nn`.

Call chain: `op::LstmBlockCell` graph construction (8 `op::Data` inputs + 7 [B, H] output descriptions) → `ge::Session::AddGraph` (compilation: prototype → InferShape → tiling → kernel binary → task generation) → `RunGraph` (dispatch for execution) → output collection (i/cs/f/o/ci/co/h, each [B, H]).

```cpp
#include "graph.h"
#include "ge_api.h"    // ge::GEInitialize / ge::Session
#include "lstm_block_cell_proto.h"  // operator prototype (op::LstmBlockCell)

// 1) Graph construction: 8 inputs → LstmBlockCell → 7 outputs
auto lstmCellOp = op::LstmBlockCell("lstm_block_cell");
lstmCellOp.set_input_x(dataX);           // x      [B, I]
lstmCellOp.set_input_cs_prev(dataCsP);   // cs_prev [B, H]
lstmCellOp.set_input_h_prev(dataHP);     // h_prev  [B, H]
lstmCellOp.set_input_w(dataW);           // w       [I+H, 4H]
lstmCellOp.set_input_wci(dataWci);       // wci     [H]
lstmCellOp.set_input_wcf(dataWcf);       // wcf     [H]
lstmCellOp.set_input_wco(dataWco);       // wco     [H]
lstmCellOp.set_input_b(dataB);           // b       [4H]

ge::TensorDesc outDesc(ge::Shape({B, H}), ge::FORMAT_ND, ge::DT_FLOAT);
lstmCellOp.update_output_desc_i(outDesc);  // cs/f/o/ci/co/h have the same shape [B, H]

ge::Graph graph("lstm_block_cell");
graph.SetInputs({dataX, dataCsP, dataHP, dataW, dataWci, dataWcf, dataWco, dataB})
     .SetOutputs({lstmCellOp});

// 2) Session compilation + execution + output collection (feeds are the 8 host input tensors; GE handles H2D/D2H)
std::map<ge::AscendString, ge::AscendString> options = {
    {"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
ge::GEInitialize(options);
ge::Session session(std::map<ge::AscendString, ge::AscendString>{});
session.AddGraph(graphId, graph, options);  // compilation (including InferShape/tiling/kernel binary)
session.RunGraph(graphId, feeds, outputs);  // execution; outputs are i/cs/f/o/ci/co/h in order
```

For a complete runnable sample (including deterministic input construction and an independent double-precision golden numerical comparison with tolerances rtol=1e-5/atol=1e-6), see [test_geir_lstm_block_cell.cpp](examples/arch35/test_geir_lstm_block_cell.cpp).

### TensorFlow integration

This operator has a built-in TF plugin (build artifact `libcust_tf_parsers.so`). The TF side must register the operator `LstmBlockCell` with the same name (8 inputs, 7 outputs, 3 attributes; the slots are exactly consistent with the OpDef of this operator, loaded through `tf.load_op_library`). The plugin automatically maps inputs by position and matches attributes by name:

```cpp
// framework/lstm_block_cell_tf_plugin.cpp
REGISTER_CUSTOM_OP("LstmBlockCell")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LstmBlockCell")
    .ParseParamsByOperatorFn(ParseLstmBlockCellByOpFn);
```

On the basis of `AutoMappingByOpFn`, `ParseLstmBlockCellByOpFn` forward-sets the input dtype of the source node (TF attribute `T`) on all 8 inputs and 7 outputs, preventing ATC from selecting an fp16 kernel for an all-fp32 graph and bridging the whole graph with trans_Cast. Online execution depends on the TF-NPU adapter `npu_device`, and the operator must be called within `tf.function`:

```python
import npu_device
import tensorflow as tf

lstm_block_cell_module = tf.load_op_library("libcustom_ops.so")  # TF-side LstmBlockCell registration library

with npu_device.open().as_default():
    @tf.function(autograph=False)
    def call_lstm(x, cs_prev, h_prev, w, wci, wcf, wco, b):
        return lstm_block_cell_module.lstm_block_cell(
            x=x, cs_prev=cs_prev, h_prev=h_prev, w=w,
            wci=wci, wcf=wcf, wco=wco, b=b,
            forget_bias=1.0, cell_clip=3.0, use_peephole=False)

    i, cs, f, o, ci, co, h = call_lstm(x, cs_prev, h_prev, w, wci, wcf, wco, b)
```
