# DynamicMxQuantWithDualAxis

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: Performs MX quantization with the FLOAT4-class or FLOAT8-class destination data type on the -1 axis and the -2 axis at the same time. On the given -1 axis and -2 axis, for every 32 values, the operator computes the quantization scales mxscale1 and mxscale2 corresponding to the two groups of values, divides all elements of each group by the corresponding mxscale1 or mxscale2, converts them to the corresponding dst_type according to round_mode, and obtains the quantization results y1 and y2. When dst_type is FLOAT8_E4M3FN or FLOAT8_E5M2, the value of scale_alg specifies different algorithms for computing mxscale.

- Axis merging description: During implementation, the operator merges all axes before the -2 axis (exclusive). For a tensor whose input shape is $(d_0, d_1, ..., d_{n-3}, d_{n-2}, d_{n-1})$, the dimensions before the -2 axis, $(d_0, d_1, ..., d_{n-3})$, are merged into one dimension; this is equivalent to reshaping the input to $(d_0 \times d_1 \times ... \times d_{n-3}, d_{n-2}, d_{n-1})$ and then performing the quantization computation.

- Formula:
  - Scenario 1: when scale_alg is 0, the implementation follows the OCP Microscaling Formats (Mx) Specification:
  - Groups the input x into groups of 32 values along the -1 axis; a group of 32 values $\{\{V_i\}_{i=1}^{32}\}$ is quantized to $\{mxscale1, \{P_i\}_{i=1}^{32}\}$

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale1 = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale1, round\_mode), \space i\space from\space 1\space to\space 32
    $$

  - At the same time, groups the input x into groups of 32 values along the -2 axis; a group of 32 values $\{\{V_j\}_{j=1}^{32}\}$ is quantized to $\{mxscale2, \{P_j\}_{j=1}^{32}\}$

    $$
    shared\_exp = floor(log_2(max_j(|V_j|))) - emax
    $$

    $$
    mxscale2 = 2^{shared\_exp}
    $$

    $$
    P_j = cast\_to\_dst\_type(V_j/mxscale2, round\_mode), \space j\space from\space 1\space to\space 32
    $$

  - The $P_{i}$ values quantized along the -1 axis form the output y1 at the positions of the corresponding $V_{i}$, and mxscale1 forms the output mxscale1 according to the groups on the -1 axis dimension. The $P_{j}$ values quantized along the -2 axis form the output y2 at the positions of the corresponding $V_{j}$, and mxscale2 forms the output mxscale2 according to the groups on the -2 axis dimension.

  - emax: the exponent of the largest normal number of the corresponding data type.

    |   DataType    | emax |
    | :-----------: | :--: |
    |  FLOAT4_E2M1  |  2   |
    |  FLOAT4_E1M2  |  0   |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |

  - Scenario 2: when scale_alg is 1, only FP8 types are involved (round-up algorithm):
    - Groups the input x into groups of 32 values along the -1 axis (or the -2 axis), computes the block scaling factor $S_{fp32}^b$ for each group separately, and then maps all elements in the group to the target low-precision type FP8.
    - Finds the maximum absolute value of the values in the group: $Amax(D_{fp32}^b) = max(\{|d_i|\}_{i=1}^{32})$
    - Maps FP32 into the range representable by the target data type FP8: $S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}$
    - Extracts the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from the block scaling factor and performs conditional rounding up.
    - Computes the block scaling factor: $S_{ue8m0}^b=2^{E_{int}^b}$; computes the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
    - Applies quantization to each element in the group: $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$

  - Scenario 3: when scale_alg is 2, only the FP4_E2M1 type is involved (DynamicDtypeRange algorithm, supported only by the V2 interface):
    - When dstTypeMax = 0.0/6.0/7.0, the operator computes scale using the exponent-field addValueBit carry method.
    - When dstTypeMax is another custom value, the operator computes scale using the FP32-precision invDstTypeMax multiplication method.

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 330px">
  <col style="width: 280px">
  <col style="width: 120px">
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
      <td>x</td>
      <td>Input</td>
      <td>Data to be quantized, corresponding to V<sub>i</sub> and d<sub>i</sub> in the formulas.<br>When the destination type is FLOAT4_E2M1 or FLOAT4_E1M2, the last dimension of x must be an even number.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Optional Attribute</td>
      <td>Data conversion mode.<br>When dst_type is 40/41(FLOAT4_E2M1/FLOAT4_E1M2), {"rint", "floor", "round"} are supported;<br>when dst_type is 35/36(FLOAT8_E5M2/FLOAT8_E4M3FN), only {"rint"} is supported;<br>when a null pointer is passed in, the "rint" mode is used.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Optional Attribute</td>
      <td>Specifies the types of y1 and y2 after data conversion.<br>The input range is {35, 36, 40, 41}, corresponding to {35:FLOAT8_E5M2, 36:FLOAT8_E4M3FN, 40:FLOAT4_E2M1, 41:FLOAT4_E1M2} respectively.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>Optional Attribute</td>
      <td>Computation methods of mxscale1 and mxscale2.<br>Supported values are 0, 1, and 2; 0 represents the OCP implementation (scenario 1), 1 represents the round-up implementation (scenario 2), and 2 represents the DynamicDtypeRange implementation (scenario 3).<br>When dst_type is FLOAT4_E1M2, only 0 is supported.<br>When dst_type is FLOAT4_E2M1, only 0 and 2 are supported.<br>When dst_type is FLOAT8, only 0 and 1 are supported.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type_max</td>
      <td>Optional Attribute</td>
      <td>Value of maxType, corresponding to Amax(DType) in the formulas.<br>Supported values are 0.0 and 6.0-12.0; 0.0 means that Amax(DType) is the maximum value of the quantization result data type, and 6.0-12.0 means that Amax(DType) is the passed-in value.<br>This value can be set only when FP4_E2M1 is used and scale_alg is 2.</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td>Result corresponding to the input x after quantization along the -1 axis, corresponding to P<sub>i</sub> and d<sup>i</sup> in the formulas.<br>The shape is the same as that of the input x.</td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxscale1</td>
      <td>Output</td>
      <td>Quantization scale corresponding to each group on the -1 axis, corresponding to mxscale1 and S<sup>b</sup> in the formulas.<br>The shape is the -1 axis value of x divided by 32 and rounded up, with even padding and a pad fill value of 0.</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td>Result corresponding to the input x after quantization along the -2 axis, corresponding to P<sub>j</sub> and d<sup>j</sup> in the formulas.<br>The shape is the same as that of the input x.</td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxscale2</td>
      <td>Output</td>
      <td>Quantization scale corresponding to each group on the -2 axis, corresponding to mxscale2 and S<sup>b</sup> in the formulas.<br>The shape is the -2 axis value of x divided by 32 and rounded up, with even padding and a pad fill value of 0.<br>The mxscale2 output requires interleaving processing for every two rows of data.</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

 - The shape constraints on x, mxscale1, and mxscale2 are as follows:
    - The number of dimensions of x should be greater than or equal to 2.
    - rank(mxscale1) = rank(x) + 1.
    - rank(mxscale2) = rank(x) + 1.
    - mxscale1.shape[-2] = (ceil(x.shape[-1] / 32) + 2 - 1) / 2.
    - mxscale2.shape[-3] = (ceil(x.shape[-2] / 32) + 2 - 1) / 2.
    - mxscale1.shape[-1] = 2.
    - mxscale2.shape[-1] = 2.
    - Other dimensions are the same as those of the input x.
    - Example: when the shape of the input x is [B, M, N] and the destination data type is FP8-class, the corresponding y1 and y2 have the shape [B, M, N], mxscale1 has the shape [B, M, (ceil(N/32)+2-1)/2, 2], and mxscale2 has the shape [B, (ceil(M/32)+2-1)/2, N, 2].

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation (V1)  | [test_aclnn_dynamic_mx_quant_with_dual_axis](examples/arch35/test_aclnn_dynamic_mx_quant_with_dual_axis.cpp) | Invokes the operator through the [aclnnDynamicMxQuantWithDualAxis](docs/aclnnDynamicMxQuantWithDualAxis_en.md) interface; supports scale_alg=0/1. |
| aclnn invocation (V2)  | [test_aclnn_dynamic_mx_quant_with_dual_axis_v2](examples/arch35/test_aclnn_dynamic_mx_quant_with_dual_axis_v2.cpp) | Invokes the operator through the [aclnnDynamicMxQuantWithDualAxisV2](docs/aclnnDynamicMxQuantWithDualAxisV2_en.md) interface; supports scale_alg=0/1/2. |
| Graph-mode invocation | -  | Builds the graph and invokes the DynamicMxQuantWithDualAxis operator through the [operator IR](op_graph/dynamic_mx_quant_with_dual_axis_proto.h).         |
