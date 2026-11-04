# RmsNormDynamicMxQuant

English | [简体中文](./README.md)


## Applicable Products

| Product | Support |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | × |
|  <term>Kirin 9030 processor series products</term> | × |

## Function

- Description: The RmsNorm operator is a normalization operation commonly used in large models. Compared with the LayerNorm operator, it removes the mean-subtraction part. The DynamicMxQuant operator performs dynamic MX quantization on the tail axis in groups by blocksize=32. The RmsNormDynamicMxQuant operator fuses the DynamicMxQuant operator with the normalization output of RmsNorm, reducing move-in and move-out operations. On the input tail axis, for each group of blocksize=32 values, the operator computes the quantization scale mxscale of that group, then divides each value in the group by mxscale, converts each value to the corresponding dst_type according to round_mode, and obtains the quantized result y. When dst_type is FLOAT8_E4M3FN or FLOAT8_E5M2, the value of scale_alg specifies the algorithm used to compute mxscale.
- Formula:

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  When scale_alg is 0:

   - Groups the RmsNorm output y along the tail axis dimension by k = 32 values, and dynamically quantizes a group of k values $\{\{V_i\}_{i=1}^{k}\}$ into $\{mxscale,\{P_i\}_{i=1}^{k}\}$
    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
    $$
    - emax: the exponent bits of the largest normal number of the corresponding data type.

        |   DataType    | emax |
        | :-----------: | :--: |
        |  FLOAT4_E2M1  |  2   |
        |  FLOAT4_E1M2  |  0   |
        | FLOAT8_E4M3FN |  8   |
        |  FLOAT8_E5M2  |  15  |

  When scale_alg is 1, only FP8 types are involved:
    - Divides the long vector into blocks, each block of length k, and computes a block scaling factor $S_{fp32}^b$ for each block separately; then maps all elements in the block to the target low-precision type FP8 using the same $S_{fp32}^b$.
    - Finds the maximum absolute value of the values in the block:

      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$

    - Maps FP32 into the representable range of the target data type FP8:

      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
      $$

    - Converts the value to the scaling value $S_{ue8m0}^b$ representable in the FP8 format
    - Extracts the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from the floating-point block scaling factor $S_{fp32}^b$
    - To prevent overflow during quantization, rounds the exponent up:

      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a normal number, } E_{int}^b < 254 \text{, and } M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a subnormal number, and } M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{otherwise} \end{cases}
      $$

    - Computes the block scaling factor: $S_{ue8m0}^b=2^{E_{int}^b}$
    - Computes the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
    - Applies the final step of quantization: $d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$

## Parameters

<table style="undefined;table-layout: fixed; width: 910px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
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
      <td>x</td>
      <td>Input</td>
      <td>Denotes the source data tensor in the standardization process, corresponding to x in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Denotes the weight tensor in the standardization process, corresponding to gamma in the formula. The shape must be consistent with the last dimension of x.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Optional input</td>
      <td>Denotes the bias term in the standardization process, corresponding to beta in the formula. The shape must be consistent with gamma.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Denotes the value added to the denominator to ensure numerical stability, corresponding to epsilon in the formula.</li><li>The default value is 1e-6.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>Optional attribute</td>
      <td><ul><li>Denotes the method for computing mxscale, corresponding to scale_alg in the formula.</li><li>Supports the values 0 and 1. Value 0 indicates the Open Compute Project (OCP) implementation, and value 1 indicates the cuBLAS implementation. When dst_type is FLOAT4_E2M1/FLOAT4_E1M2, only the value 0 is supported.</li><li>The default value is 0.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>Denotes the data conversion mode, corresponding to round_mode in the formula.</li><li>When dst_type is 40/41, {"rint", "floor", "round"} is supported.</li><li>When dst_type is 36/35, only {"rint"} is supported.</li><li>The default value is "rint".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Optional attribute</td>
      <td><ul><li>Denotes the type of y after data conversion, corresponding to DType in the formula.</li><li>The input range is {35, 36, 40, 41}, corresponding to {FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2}.</li><li>The default value is 40.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_rstd</td>
      <td>Optional attribute</td>
      <td><ul><li>Denotes whether to output a valid rstd.</li><li>Supports True and False.</li><li>The default value is False.</li><li>When output_rstd is False, the operator does not write out rstd, and rstd is an invalid placeholder output.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td><ul><li>Denotes the normalized and quantized result, corresponding to Pi and di in the formula; the shape is the same as x.</li></ul></td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxscale</td>
      <td>Output</td>
      <td><ul><li>Denotes the quantization scale corresponding to each group, corresponding to mxscale and Sb in the formula; see Constraints for the shape.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Output</td>
      <td><ul><li>Denotes the reciprocal of the normalization denominator, corresponding to the reciprocal of Rms(x) in the formula.</li><li>When output_rstd is True, the rank is the same as x, and the shape is x.shape[:-1] + (1,). For example, when x is (2,3,4,8), rstd is (2,3,4,1). When output_rstd is False, rstd is an invalid placeholder output.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The ranks of x and y are 1–7; gamma and the optional beta are one-dimensional, with a length equal to the tail axis length of x, and the shape and data type of beta are the same as those of gamma.
- A valid rstd has the same rank as x, with only the last dimension being 1; the rank of mxscale is rank(x)+1.
- Empty tensors are supported. When an empty dimension exists outside the tail axis of x, no valid data is produced; when the tail axis is empty, y and mxscale are empty, and the enabled rstd is NaN.
- Graph inference supports unknown dimensions (-1) and unknown ranks (-2); before execution, you must concretize the tensor to satisfy the preceding rank, axis length, and data type constraints.

- <term>Ascend 950PR&950DT products</term>:

  The shape constraints of mxscale are as follows:
  - rank(mxscale) = rank(x) + 1.
  - mxscale.shape[-2] = ceil(ceil(x.shape[-1] / 32) / 2).
  - mxscale.shape[-1] = 2.
  - The other dimensions are consistent with the input x.

- When the data type of the output y is FLOAT4_E2M1 or FLOAT4_E1M2, the value of the tail axis of x must be an even number.

- The data types of the input gamma and the optional input beta can only be the same as the data type of x or be FLOAT32.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_rms_norm_dynamic_mx_quant](examples/test_aclnn_rms_norm_dynamic_mx_quant.cpp) | Calls the RmsNormDynamicMxQuant operator through the [aclnnRmsNormDynamicMxQuant](docs/aclnnRmsNormDynamicMxQuant_en.md) interface.         |
| GE graph mode | -  | Calls the RmsNormDynamicMxQuant operator by building a graph through [operator IR](op_graph/rms_norm_dynamic_mx_quant_proto.h).         |
