[中文](README.md)

# QuantMaxV2

## Product Support

| Product | Supported |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Operator function: the input `x` can be grouped by `group_list` (when `group_list` is absent the whole tensor is treated as a single group, i.e. Per-Tensor). For each group the maximum absolute value `amax` of `|x|` is computed, and the quantized result `y` is produced. Passing `scale` selects static quantization; omitting it selects dynamic quantization, in which case the quantization coefficient `y_scale` is additionally output. `amax` is a required output.

  Compared with [GroupedQuantMax](../grouped_quant_max/README.md), V2 provides three extra capabilities: "optional scale coefficient, optional group_list grouping, and y_scale output for dynamic quantization"; `amax` is upgraded to a required output, and the new attribute `dst_type_max` is added.

  | `scale` | `group_list` | Mode | Output |
| :----------------: | :----------------: |  ---------------- | ------------------------------------------------- |
| Absent (`nullptr`) | Absent (`nullptr`) | Dynamic Per-Tensor | `y` + `amax[1]` (required) + `y_scale[1]` (required) |
| Passed | Absent (`nullptr`) | Static Per-Tensor | `y` + `amax[1]` (required) + `y_scale` (not output, pass `nullptr`) |
| Absent (`nullptr`) | Passed | Dynamic Per-Group | `y` + `amax[G]` (required) + `y_scale[G]` (required) |
| Passed | Passed | Static Per-Group | `y` + `amax[G]` (required) + `y_scale` (not output, pass `nullptr`) |

- Formulas:

  `group_list` splits `x` into `numGroups` (denoted `G`) groups along dim-0, where `group_list[g]` is the cumulative end position on dim-0 of the first `g+1` groups (cumsum semantics). For each group (`g = 0, 1, …, G-1`):

  $$
  x_g = x\left[\,groupList[g-1] : groupList[g],\ \ldots\,\right] \qquad(\ \text{the lower bound is } 0 \text{ when } g = 0\ )
  $$

  $$
  a_g = \max\left(|x_g|\right)
  $$

  **Static quantization** (`scale` is passed, `scale[g]` is the reciprocal `1/s` of the regular coefficient `s`):

  $$
  q_g = scale[g], \qquad y_g = \mathrm{Cast}\left(x_g \times q_g,\ dstType,\ roundMode\right)
  $$

  **Dynamic quantization** (`scale` is absent, `dstTypeMax` is the maximum finite value of the destination type):

  $$
  q_g = \begin{cases}
  \dfrac{dstTypeMax}{a_g}, & a_g > 0 \\[4pt]
  0, & a_g = 0
  \end{cases}, \qquad
  y_g = \mathrm{Cast}\left(x_g \times q_g,\ dstType,\ roundMode\right)
  $$

  **Outputs**:

  $$
  amax[g] = a_g, \qquad
  y = \mathrm{Concat}\left(y_0,\ y_1,\ \ldots,\ y_{G-1}\right), \qquad
  yScale[g] = \dfrac{1}{q_g} \quad(\text{only output for dynamic quantization})
  $$

  where `y_scale` satisfies `x ≈ y × y_scale` (`y_scale[g] = amax[g] / dstTypeMax`; when `a_g = 0`, `q_g = 0` and the corresponding `y_scale[g] = 0`).

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input tensor to be quantized, corresponding to x in the formulas. Supports 2-8 dimensions and empty tensors.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Optional input</td>
      <td>Reciprocal <code>1/s</code> of the quantization scale coefficient s. <code>nullptr</code> means dynamic quantization (the coefficient is computed on the fly from the amax of the group); a non-null value means static quantization. Shape is [numGroups] ([1] for Per-Tensor).</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_list</td>
      <td>Optional input</td>
      <td>Cumulative end position of each group of x along axis 0 (cumsum semantics). <code>nullptr</code> means Per-Tensor; a non-null value means Per-Group, with shape [numGroups]; num_groups ranges from 1 to 384.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Quantization result, corresponding to y in the formulas. Shape is exactly the same as x; dtype is determined by dst_type.</td>
      <td>HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>amax</td>
      <td>Output</td>
      <td>Maximum absolute value of x per group before quantization (required output). Shape is [numGroups] ([1] for Per-Tensor); dtype is the same as x.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_scale</td>
      <td>Optional output</td>
      <td>Quantization coefficient of y (<code>y_scale = 1/q</code>, equal to <code>amax/dstTypeMax</code> for dynamic quantization). Output only for dynamic quantization, with shape [numGroups] ([1] for Per-Tensor); pass <code>nullptr</code> for static quantization.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Attribute</td>
      <td>Rounding mode used when converting the quantization result to the destination type (required). dst_type=34 supports "round" and "hybrid"; dst_type=35/36 only supports "rint".</td>
      <td>string</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Attribute</td>
      <td>Destination data type of y (required). 34=HIFLOAT8, 35=FLOAT8_E5M2, 36=FLOAT8_E4M3FN.</td>
      <td>int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type_max</td>
      <td>Attribute</td>
      <td>Maximum finite value of the destination type, default 15.0, valid range 0~32768. Only participates in coefficient computation for dynamic quantization.</td>
      <td>float</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- Matching between `round_mode` and `dst_type`: when `dst_type` is `35` or `36`, `round_mode` must be `"rint"`; when `dst_type` is `34`, it must be `"round"` or `"hybrid"`.
- Requirements on `group_list` values: every value must be within `[0, x.shape[0]]`, non-decreasing (equal values are allowed, meaning an empty group), and the last value must equal the size of axis 0 of x. The PyTorch interface (`cann_ops_nn.quant_max`) validates this constraint and raises an exception; the aclnn C interface requires the caller to guarantee it.
- `dst_type_max` only takes effect for `dst_type=34` (HIFLOAT8); when `dst_type` is `35`/`36` the dynamic coefficient is derived from the maximum value of the type (57344/448) and the passed value does not participate in the computation (range validation still applies).
- The value of `dst_type_max` must be a finite value within 0~32768, otherwise the first-stage interface reports an error.
- Deterministic computation: aclnnQuantMaxV2 is deterministic by default on <term>Ascend 950PR&950DT products</term>.

## Usage

| Invocation | Example | Description |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_quant_max_v2](./examples/arch35/test_aclnn_quant_max_v2.cpp) | Invoke the QuantMaxV2 operator through the [aclnnQuantMaxV2](./docs/aclnnQuantMaxV2.md) interface. |
| Graph mode invocation | - | Invoke the QuantMaxV2 operator by building a graph with the [operator IR](./op_graph/quant_max_v2_proto.h). |
