# GroupedDynamicMxQuant

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: Based on the start values of the passed-in group indices, this operator performs dynamic MX quantization with the target data type float4/float8 on each group at the basic block granularity.

- Formula:
  - Scenario 1, when `scaleAlg` is 0:
    - First, group the input `x` along dimension 0 by `groupIndex`, and then group the elements within each group by k = blocksize. A group of k numbers {{x<sub>i</sub>}<sub>i=1</sub><sup>k</sup>} is used to compute the quantization scale `mxscale_pre` of this group of numbers, {mxscale_pre, {P<sub>i</sub>}<sub>i=1</sub><sup>k</sup>}. The formulas are formulas (1) and (2) below.

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax  \tag{1}
    $$

    $$
    mxscale\_pre = 2^{shared\_exp}  \tag{2}
    $$

    - Each number in the group is divided by `mxscale_pre` and converted to the corresponding `dst_type` according to `round_mode` to obtain the quantization result `y`. The formula is formula (3) below.

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize \tag{3}
    $$

    - The quantized $P_{i}$ values are placed at the positions of the corresponding $V_{i}$ values to form the output `y`, and `mxscale_pre` is grouped by the corresponding `groupIndex`. The first dimension within a group is padded to an even number to form the output `mxscale`.

    - `emax`: the exponent of the largest normal number of the corresponding data type.

        |   DataType    | emax |
        | :-----------: | :--: |
        | FLOAT4_E2M1   |  2   |
        | FLOAT4_E1M2   |  0   |
        | FLOAT8_E4M3FN |  8   |
        | FLOAT8_E5M2   |  15  |

  - Scenario 2, when `scaleAlg` is 1, only FP8 types are involved:
    - Split the long vector into blocks, each of length k, and compute a block scaling factor $S_{fp32}^b$ for each block separately. Then map all elements in the block to the target low-precision type FP8 by using the same $S_{fp32}^b$. If the last block contains fewer than k elements, treat the missing values as 0 and process the block as a complete block.
    - Find the maximum absolute value of the numbers in the block:

      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$

    - Map FP32 to the range representable by the target data type FP8, where $Amax(DType)$ is the maximum value representable by the target precision.

      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
      $$

    - Convert the block scaling factor $S_{fp32}^b$ to a scaling value $S_{ue8m0}^b$ representable in the FP8 format.
    - Extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from the floating-point block scaling factor $S_{fp32}^b$.
    - To prevent overflow during quantization, round the exponent up, and keep it within the range representable by FP8:

      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a normal number, and } E_{int}^b < 254 \text{ and } M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a subnormal number, and } M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{otherwise} \end{cases}
      $$

    - Compute the block scaling factor: $S_{ue8m0}^b=2^{E_{int}^b}$
    - Compute the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
    - Apply the final step of quantization. For each element in the block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$. The final quantization output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ denotes the block scaling factor, namely $S_{ue8m0}^b$, and $[d^i]_{i=1}^k$ denotes the quantized data in the block.

  - Scenario 3, when `scaleAlg` is 2, only FP4 types are involved:
    - When `dstTypeMax = 0.0/6.0/7.0(FP4_E2M1)` or `dstTypeMax = 1.875(FP4_E1M2)`:
      - Group the input `x` along dimension 0 by k = blocksize numbers. A group of k numbers $\{\{V_i\}_{i=1}^{k}\}$ is dynamically quantized to $\{mxscale, \{P_i\}_{i=1}^{k}\}$, k = blocksize:
      $$
      shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{if the one/two/three bits preceding the high bits of the mantissa are 1, and the mantissa is not all 0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{otherwise} \end{cases} \\
      $$
      $$
      P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
      $$
      - The quantized $P_{i}$ values are placed at the positions of the corresponding $V_{i}$ values to form the output `y`, and the output `mxscale` is formed by the groups along the corresponding dimension.
    - When `dstTypeMax != 0.0/6.0/7.0(FP4_E2M1)` or `dstTypeMax != 1.875(FP4_E1M2)`:
      - Split the long vector into blocks, each of length k, and compute a block scaling factor $S_{fp32}^b$ for each block separately. Then map all elements in the block to the target low-precision type by using the same $S_{fp32}^b$. If the last block contains fewer than k elements, treat the missing values as 0 and process the block as a complete block.
      - Find the maximum absolute value of the numbers in the block:
      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$
      - Map FP32 to the range representable by the target data type. When `dst_type_max` is 0, $Amax(DType)$ is the maximum value representable by the target precision; when `dst_type_max` is not 0, $Amax(DType)$ is the value passed in by `dst_type_max`.
      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
      $$
      - Convert the block scaling factor $S_{fp32}^b$ to a scaling value $S_{ue8m0}^b$ representable in the FP8 format.
      - Extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from the floating-point block scaling factor $S_{fp32}^b$.
      - To prevent overflow during quantization, round the exponent up, and keep it within the range representable by FP8:
        $$
        E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a normal number, and } E_{int}^b < 254 \text{ and } M_{fixp}^b > 0 \\ E_{int}^b, & \text{otherwise} \end{cases}
        $$
      - Compute the block scaling factor: $S_{ue8m0}^b=2^{E_{int}^b}$
      - Compute the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
      - Apply the final step of quantization. For each element in the block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$. The final quantization output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ denotes the block scaling factor, namely $S_{ue8m0}^b$, and $[d^i]_{i=1}^k$ denotes the quantized data in the block.
      - The quantized $P_{i}$ values are placed at the positions of the corresponding $V_{i}$ values to form the output `y`, and the output `mxscale` is formed by the groups along the corresponding dimension.

## Parameters

<table style="undefined;table-layout: fixed; width: 1010px"><colgroup>
  <col style="width: 130px">
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
        <td>aclTensor on the Device. It is the input x in the formula. Only 2-dimensional shapes are supported. Non-contiguous tensors are supported, and empty tensors are supported.</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>groupIndex</td>
        <td>Input</td>
        <td>aclTensor on the Device. It is the start index of quantization groups. Only 1-dimensional shapes are supported. Non-contiguous tensors are supported, but empty tensors are not supported.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>roundMode</td>
        <td>Attribute</td>
        <td>string on the host. It is round_mode in the formula, which specifies the mode of data conversion. "rint"/"round"/"floor" are supported. For FLOAT8, only "rint" is supported; for FLOAT4, all three options "rint"/"round"/"floor" are supported.</td>
        <td>STRING</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dstType</td>
        <td>Attribute</td>
        <td>int64_t on the host. It is dst_type in the formula, which specifies the type of y after data conversion. The input range is {35, 36, 40, 41}, which correspond to the data types of output y {35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN, 40: FLOAT4_E2M1, 41: FLOAT4_E1M2}.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>blocksize</td>
        <td>Attribute</td>
        <td>int64_t on the host. It is blocksize in the formula, which specifies the number of elements quantized each time. Only 32 is supported.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleAlg</td>
        <td>Attribute</td>
        <td>int64_t on the host, which specifies the algorithm used to compute mxscale. 0, 1, and 2 are supported.</td>
        <td>INT64</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dstTypeMax</td>
        <td>Attribute</td>
        <td>float32 on the host. It takes effect when scale_alg=2. The default value 0.0 means that max_type is the maximum value of the target data type. If another value is passed in, mxscale needs to be computed according to the passed-in value. Currently, the supported values are 0.0 and [1.75, 3.5) (FLOAT4_E1M2), and 0.0 and [6.0, 12.0] (FLOAT4_E2M1).</td>
        <td>FLOAT</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>Output</td>
        <td>aclTensor on the Device. It is the output y in the formula, that is, the quantization result corresponding to the input x. It must correspond to dstType. Only 2-dimensional shapes are supported, empty tensors are supported, and the shape is the same as that of the input x.</td>
        <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, FLOAT4_E1M2</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>mxscale</td>
        <td>Output</td>
        <td>aclTensor on the Device. It is the output mxscale composed of mxscale_pre in the formula, that is, the quantization scale corresponding to each group. Only 3-dimensional shapes are supported, and empty tensors are supported. Assume that the shape of x is [m,n] and the shape of groupIndex is [g]; then the shape of mxscale is [(m/(blocksize * 2)+g), n, 2].</td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
      </tr>
    </tbody></table>

## Constraints

- The constraints on x, groupIndex, y, and mxscale are as follows:
  - The values in groupIndex must be non-decreasing and must not be less than 0, and the last element must be the length of the first dimension of x.
  - $rank(mxscale) = rank(x) + 1$.
  - Assume that the shape of x is $[m,n]$ and the shape of groupIndex is $[g]$; then the shape of mxscale is $[(m/(blocksize * 2)+g), n, 2]$.
  - $mxscale.shape[-1] = 2$.
  - The shape of the output y is the same as that of the input x.
  - When the data type of the output y is FLOAT4_E2M1 or FLOAT4_E1M2, the length of the trailing axis (dimension 1) of x must be an even number.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_grouped_dynamic_mx_quant](./examples/arch35/test_aclnn_grouped_dynamic_mx_quant.cpp) | Invoke the GroupedDynamicMxQuant operator through the [aclnnGroupedDynamicMxQuant](./docs/aclnnGroupedDynamicMxQuant_en.md) API. |
| aclnn invocation | [test_aclnn_grouped_dynamic_mx_quant_v2](./examples/arch35/test_aclnn_grouped_dynamic_mx_quant_v2.cpp) | Invoke the GroupedDynamicMxQuant operator through the [aclnnGroupedDynamicMxQuantV2](./docs/aclnnGroupedDynamicMxQuantV2_en.md) API. |
