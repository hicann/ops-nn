# ApplyCamePart4

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>   |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function:

  This is the fourth stage of the CAME optimizer (the parameter update stage). Given the parameter param_in to be updated, the first momentum m with the shape (N,M), and the confidence factors r_in(N) and c_in(M), this operator writes back param_out/r_out/c_out according to the CAME update rule. sum_r (the reduction sum of r_in over all N rows) and
  global_shape (the global N,M) are optional inputs: in the distributed scenario, the preceding operator/the global shape passes them in; in the single-device scenario, they are omitted by default, and the kernel
  completes the reduction internally and takes the local n/m.

- Formula (n = len(r_in),m = len(c_in); N/M take global_shape (when given), otherwise n/m):

  $$
  sum\_r=\sum_{i} r\_in_i \quad (\text{reduced inside the kernel when sum\_r is omitted})
  $$

  $$
  r\_out=\beta_3 \cdot r\_in+\frac{(1-\beta_3)}{M} \cdot sum\_u\_r
  $$

  $$
  c\_out=\beta_3 \cdot c\_in+\frac{(1-\beta_3)}{N} \cdot sum\_u\_c
  $$

  $$
  denom=\beta_3 \cdot \frac{sum\_r}{N}+(1-\beta_3)\cdot\frac{sum\_u\_rc}{M \cdot N}
  $$

  $$
  param\_out=(1-lr\cdot weight\_decay)\cdot param\_in-\frac{lr \cdot m}{\sqrt{r\_out \otimes c\_out / denom}}
  $$

  In this formula, $r\_out \otimes c\_out$ is the outer product of (N,1)×(1,M). fp16/bf16 path: inputs are cast to fp32 for computation,
  and outputs are rounded back to low precision according to RNE (CAST_RINT); the param update takes the rounded r_out/c_out as inputs.

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 500px">
  <col style="width: 212px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>param_in</td>
      <td>Input</td>
      <td>Parameters to be updated, with the shape (N,M).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>m</td>
      <td>Input</td>
      <td>First momentum, with the shape (N,M), consistent with param_in.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>r_in</td>
      <td>Input</td>
      <td>Row confidence factor, with the shape (N,).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>c_in</td>
      <td>Input</td>
      <td>Column confidence factor, with the shape (M,).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight_decay</td>
      <td>Input</td>
      <td>Weight decay coefficient, scalar (1,).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>Input</td>
      <td>Learning rate, scalar (1,).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta3</td>
      <td>Input</td>
      <td>Confidence factor decay coefficient, scalar (1,).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_u_r</td>
      <td>Input</td>
      <td>Reduction of the update amount in the r direction, with the shape (N,).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_u_c</td>
      <td>Input</td>
      <td>Reduction of the update amount in the c direction, with the shape (M,).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_u_rc</td>
      <td>Input</td>
      <td>Reduction of the global update amount, scalar (1,).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum_r</td>
      <td>Input (optional)</td>
      <td>Reduction sum of r_in over all N rows, scalar (1,); reduced inside the kernel when omitted.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>global_shape</td>
      <td>Input (optional)</td>
      <td>Global (N,M), with the shape (2,); the local n/m is taken when omitted.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>param_out</td>
      <td>Output</td>
      <td>Updated parameters, with the shape (N,M).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>r_out</td>
      <td>Output</td>
      <td>Updated row confidence factor, with the shape (N,).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>c_out</td>
      <td>Output</td>
      <td>Updated column confidence factor, with the shape (M,).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- param_in/m must be 2D, and r_in/c_in must be 1D; the length of r_in = the 0th dimension of param_in, and the length of c_in = the 1st dimension of param_in (tiling validation; GRAPH_FAILED is returned when they are inconsistent).
- No attributes (attr).
- sqrt domain: s = r_out⊙c_out/denom must be ≥ 0 element-wise, and denom ≠ 0; otherwise inf/nan propagation occurs according to IEEE semantics (consistent with the torch formula semantics, not an operator error).
- Empty tensors (N=0 or M=0) are not guarded: the operator does not crash, but the outputs on non-empty dimensions are not written (undefined); the behavior is aligned with A2.
- Support relationship: for the Atlas A2 series (ascend910b/ascend910_93), the built-in ascendc implementation in the canndev repository provides support (ascendc_config.json compute_units=[ascend910b,ascend910_93], and the kernel binary build configuration binary_json_cfg.ini has included it; the legacy TBE aic-ops-info.ini has not included it, and it is provided in the form of dynamic compilation); Ascend 950PR/950DT is supported by the vendor package of this repository (arch35). The two implementations do not affect each other. The vendor package can only be installed in the Ascend 950 environment — its proto/tiling registration has no soc dimension, and if it mistakenly enters the A2 environment, it will shadow the built-in implementation of canndev and make the operator unavailable due to the mismatch of the tiling data structure.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| GE graph mode invocation | [test_geir_apply_came_part4](./examples/test_geir_apply_came_part4.cpp) | Invokes the ApplyCamePart4 operator by constructing the graph through the [operator IR](op_graph/apply_came_part4_proto.h).    |
