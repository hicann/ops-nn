# Shrink

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- **Operator function**: performs a non-linear shrink transformation on the input tensor, and maps elements to three regions (positive region, negative region, and dead zone) according to the relationship between the input value and the threshold `lambd`. It is often used for weight regularization in sparsified model inference and training.

- **Formula**:

  $$
  out_i=\begin{cases}
  self_i-bias, & self_i > lambd \\
  self_i+bias, & self_i < -lambd \\
  0, & -lambd \leq self_i \leq lambd
  \end{cases}
  $$

  Here, `lambd` is the threshold parameter (default 0.5), and `bias` is the offset parameter (default 0.0). When `lambd < 0`, the behavior is path-dependent: in Kernel/GEIR (graph mode), it is internally clamped to 0 at the tiling stage before computation; for the aclnn invocation (aclnnShrink), the parameter validation rejects it and returns 161002 (see [aclnnShrink](./docs/aclnnShrink.md)).

  **dtype semantics**: for FLOAT16 inputs, `lambd`/`bias` are first converted to FLOAT16 (`static_cast<T>`) before participating in the threshold comparison and addition/subtraction — near the threshold, the result may differ from a reference implementation constructed with the real-number formula (for example, with `x=fp16(0.5009765625)` and `lambd=0.5008`, the real-number formula outputs about 0.50098 in the positive region, while the device outputs 0 for the dead zone); FLOAT inputs do not have this difference.

## Parameters

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 300px">
<col style="width: 200px">
<col style="width: 170px">
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
    <td>input_x</td>
    <td>Input</td>
    <td>Input tensor, corresponding to self in the formula. The region of each element is determined by comparing the element value with lambd.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lambd</td>
    <td>Attribute</td>
    <td>Threshold parameter, corresponding to lambd in the formula. When self_i>lambd, the element enters the positive region; when self_i&lt;-lambd, the element enters the negative region. The default value is 0.5. When `lambd&lt;0`: in graph mode it is internally clamped to 0, and the aclnn invocation returns 161002.</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>Attribute</td>
    <td>Offset parameter, corresponding to bias in the formula. bias is subtracted in the positive region and added in the negative region. The default value is 0.0.</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
</tbody></table>

## Constraints

- ND input; empty tensors (0 elements), dynamic shapes (`-1` unknown dimension), and dynamic ranks (`-2`) are passed through according to the input shape, and the output is the same as the input.
- `lambd < 0`: in graph mode (Kernel/GEIR), it is internally clamped to 0 at the tiling stage; the aclnn invocation rejects it as an invalid parameter (161002).
- Non-finite value semantics (consistent with the kernel implementation): for NaN inputs, the threshold comparisons on both sides are both false, and 0 is output as the dead zone (NaN is not propagated per IEEE); +inf/-inf inputs fall into the positive/negative region respectively, and the output is the IEEE arithmetic result of `x-bias`/`x+bias` (when bias=0, the output is ±inf). When `lambd = +inf`, all inputs fall into the dead zone and 0 is output (including ±inf inputs, since the strict comparison inf>inf is false). When `bias = ±inf`, the active region is computed with IEEE arithmetic: finite inputs output the corresponding ±inf, and ±inf inputs forming inf∓inf output NaN (the NaN produced by the arithmetic propagates to the output, different from the dead-zone 0 for NaN inputs). When `lambd`/`bias` are NaN scalars, they are rejected as out-of-domain: the aclnn invocation returns 161002 (NaN is not a number and is not within the declared domain of "non-negative/any real number").

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_shrink](./examples/test_aclnn_shrink.cpp) | Invokes the Shrink operator through the [aclnnShrink](./docs/aclnnShrink_en.md) interface. |
| Graph-mode invocation | - | Invokes the Shrink operator through graph construction with the [operator IR](./op_graph/shrink_proto.h). |
