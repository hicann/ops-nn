# AddRmsNormDynamicMxQuant

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
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

- Operator function: fuses the Add operator, RmsNorm normalization (which removes the mean-subtraction part compared with the LayerNorm operator), and DynamicMxQuant dynamic MX quantization into a single computation, reducing data transfer in and out. The operator first performs addition on the inputs, then performs RmsNorm normalization, and finally performs dynamic MX quantization on the trailing axis of the input in groups of blocksize=32. It outputs the quantization result y, the addition result x, the quantization scale mxscale, and the reciprocal root mean square rstd. The mxscale computation algorithm is specified by scale_alg (when dst_type is FP8, both the OCP and cuBLAS implementations are supported; when dst_type is FP4, only OCP is supported). For the detailed computation process, see the formula below.
- V2 interface (CANN >= 9.2.0): adds the optional input x3. When x3 is not empty, the formula becomes x = (x3 + x1) + x2, supporting three-way residual addition. For details, see [aclnnAddRmsNormDynamicMxQuantV2](docs/aclnnAddRmsNormDynamicMxQuantV2_en.md).
- Formula:

  When x3 is empty (the V1 interface, or the V2 interface with x3 passed as empty):
  $$
  x=x_{1}+x_{2}
  $$

  When x3 is not empty (V2 interface only):
  $$
  x=(x_{3}+x_{1})+x_{2}
  $$

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  When scale_alg is 0:

  - Group the RmsNorm output y along the trailing axis dimension in groups of k = 32 values. A group of k values $\{V_i\}_{i=1}^{k}$ is dynamically quantized to $\{mxscale,\{P_i\}_{i=1}^{k}\}$

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \quad i = 1, 2, \ldots, k
    $$

  - emax: the exponent of the maximum normal number of the corresponding data type.

    |   DataType    | emax |
    | :-----------: | :--: |
    |  FLOAT4_E2M1  |  2   |
    |  FLOAT4_E1M2  |  0   |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |

  When scale_alg is 1, only FP8 types are involved:
  - Divide the long vector into blocks, each with length k. Compute a separate block scale factor $S_{fp32}^b$ for each block, and then map all elements in the block to the target low-precision type FP8 using the same $S_{fp32}^b$.
  - Find the maximum absolute value in the block:

    $$
    Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
    $$

  - Map FP32 into the range representable by the target data type FP8:

    $$
    S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
    $$

  - Convert the factor to the scaling value $S_{ue8m0}^b$ representable in FP8 format
  - Extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from the floating-point block scale factor $S_{fp32}^b$
  - To prevent overflow during quantization, round the exponent up:

    $$
    E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{否则} \end{cases}
    $$

  - Compute the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
  - Compute the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
  - Apply the factor in the final quantization step: $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
      <td>x1</td>
      <td>Input</td>
      <td>Represents the source data tensor in the normalization process, corresponding to x1 in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Represents the source data tensor in the normalization process, corresponding to x2 in the formula. The shape and data type must be the same as x1.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Represents the weight tensor in the normalization process, corresponding to gamma in the formula. The shape must be the same as the last dimension of x1.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Optional input</td>
      <td>Represents the bias term in the normalization process, corresponding to beta in the formula. The shape and data type must be the same as gamma.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x3</td>
      <td>Optional input</td>
      <td>Represents the optional input in the addition computation, that is, the residual path, corresponding to x3 in the formula. The shape and data type must be the same as x1. When it is not passed in, the operator computes x = x1 + x2.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Represents the value added to the denominator to ensure numerical stability. It corresponds to epsilon in the formula.</li><li>The default value is 1e-6.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>Optional attribute</td>
      <td><ul><li>Represents the mxscale computation method, corresponding to scale_alg in the formula.</li><li>The supported values are 0 and 1. Value 0 indicates the Open Compute Project (OCP) implementation, and value 1 indicates the cuBLAS implementation. When dst_type is FLOAT4_E2M1/FLOAT4_E1M2, only value 0 is supported.</li><li>The default value is 0.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>Represents the data conversion mode, corresponding to round_mode in the formula.</li><li>When dst_type is 35/36, only {"rint"} is supported.</li><li>When dst_type is 40/41, {"rint", "floor", "round"} are supported.</li><li>The default value is "rint".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Optional attribute</td>
      <td><ul><li>Represents the type of y after data conversion, corresponding to DType in the formula.</li><li>The input range is {35, 36, 40, 41}, corresponding to {FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2}.</li><li>The default value is 40.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_rstd</td>
      <td>Optional attribute</td>
      <td><ul><li>Represents whether to output a valid rstd_out.</li><li>Supports True and False.</li><li>The default value is False.</li><li>When output_rstd is False, rstd is an invalid placeholder output.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td><ul><li>Represents the normalized and quantized result, corresponding to Pi and di in the formula. The shape is the same as x1.</li></ul></td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Output</td>
      <td><ul><li>Represents the addition result of the inputs, corresponding to x in the formula. The shape and data type are the same as x1.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxscale</td>
      <td>Output</td>
      <td><ul><li>Represents the quantization scale corresponding to each group, corresponding to mxscale and Sb in the formula. For the shape, see Constraints.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Output</td>
      <td><ul><li>Represents the reciprocal root mean square (including epsilon) of the input addition result, corresponding to the reciprocal of Rms(x) in the formula.</li><li>When output_rstd is True, the number of dimensions is the same as x1. The dimensions that do not need normalization (dimensions of x1 minus dimensions of gamma) must be the same as the corresponding dimensions of x1, and the dimensions that need normalization must be 1. For example, when the shape of x1 is (2,3,4,8) and the shape of gamma is (8,), the shape of rstd is (2,3,4,1).</li><li>When output_rstd is False, rstd is an invalid placeholder output.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- <term>Ascend 950PR&950DT products</term>:

  The shape constraints of mxscale are as follows:
  - rank(mxscale) = rank(x1) + 1.
  - mxscale.shape[-2] = ceil(ceil(x1.shape[-1] / 32) / 2), that is, the number of blocks after grouping the trailing axis of x1 by blocksize=32 is rounded up to an even number and then divided by 2.
  - mxscale.shape[-1] = 2.
  - The other dimensions are the same as the input x1.

  The output behavior for empty tensors is as follows:
  - When the length of any dimension of x1 other than the last dimension is 0, y, x, mxscale, and the enabled rstd are all empty tensors.
  - When only the last dimension of x1 has a length of 0, y, x, and mxscale are empty tensors. If rstd is enabled, the last dimension of rstd has a length of 1, the other dimensions are the same as x1, and the elements are NaN.

- When the data type of the output y is FLOAT4_E2M1 or FLOAT4_E1M2, the value of the trailing axis of x1 must be even.

- The data type of the input gamma can only be the same as the data type of x1 or FLOAT32. The shape and data type of the optional input beta must be exactly the same as gamma (that is, when gamma is FLOAT32, beta must also be FLOAT32 and cannot use the data type of x1).

- **Boundary value scenario description**
  - When the input is Inf: 1. the output y is 0; 2. the output x is Inf; 3. the output mxscale is 255, and the even pad fill value is 0; 4. if rstd is enabled, the output rstd is 0.
  - When the input is NaN: 1. the output y is 0; 2. the output x is NaN; 3. the output mxscale is 255, and the even pad fill value is 0; 4. if rstd is enabled, the output rstd is NaN.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_add_rms_norm_dynamic_mx_quant](examples/test_aclnn_add_rms_norm_dynamic_mx_quant.cpp) | Invoke the AddRmsNormDynamicMxQuant operator through the [aclnnAddRmsNormDynamicMxQuant](docs/aclnnAddRmsNormDynamicMxQuant_en.md) interface. |
| aclnnV2 invocation  | [test_aclnn_add_rms_norm_dynamic_mx_quant_v2](examples/test_aclnn_add_rms_norm_dynamic_mx_quant_v2.cpp) | Invoke the operator through the [aclnnAddRmsNormDynamicMxQuantV2](docs/aclnnAddRmsNormDynamicMxQuantV2_en.md) interface, supporting the optional input x3. |
| Graph-mode invocation | -  | Invoke the AddRmsNormDynamicMxQuant operator by constructing a graph through the [operator IR](op_graph/add_rms_norm_dynamic_mx_quant_proto.h).         |
| PyTorch API | [test_torch_extension](tests/ut/torch_extension/test_torch_extension.py) | Invoke the AddRmsNormDynamicMxQuant operator through the [cann_ops_nn.add_rms_norm_dynamic_quant](docs/torchapi_add_rms_norm_dynamic_quant_en.md) interface. You need to install the `cann_ops_nn` whl package. Single-operator mode and graph mode (torch.compile + torchair) are supported. For the interface usage, refer to the linked document. |
