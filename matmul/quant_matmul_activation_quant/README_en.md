# QuantMatmulActivationQuant

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|<term>Ascend 950PR&950DT products</term>|√|
|<term>Atlas A3 products</term>|×|
|<term>Atlas A2 products</term>|×|
|<term>Atlas 200I/500 A2 inference products</term>|×|
|<term>Atlas inference products</term>|×|
|<term>Atlas training products</term>|×|

## Function

- Operator function: fuses quantized matrix multiplication, activation, and dynamic quantization. Currently, the activations GELU, SwiGLU, and SiLU and the MX [quantization mode](../../docs/en/context/quant_mode_introduction.md) are supported. FP8 (FLOAT8_E4M3FN/FLOAT8_E5M2) and FP4 (FLOAT4_E2M1) data types are supported for input and output. x2 supports both the ND and NZ data formats. When x2 is in NZ format, call the [aclnnQuantMatmulActivationQuantWeightNz](docs/aclnnQuantMatmulActivationQuantWeightNz_en.md) API. When x2 is in ND format, call the [aclnnQuantMatmulActivationQuant](docs/aclnnQuantMatmulActivationQuant_en.md) API.

For the data types, ND/WeightNZ formats, transposition, and output shape limits of each activation, see [Activation Capability Comparison Table](#activation-capability-comparison-table).

- Formula:

  - <term>Ascend 950PR&950DT products</term>:

    - QuantMatmul MX quantization mode:

      $$
      matmulOut[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice * x2Slice))* (x1Scale[m/gsM, j] * x2Scale[j, n/gsN]))+bias[n]
      $$

      Here, gsM, gsN, and gsK represent groupSizeM, groupSizeN, and groupSizeK, respectively. x1Slice is the vector of length groupSizeK in row m of x1, and x2Slice is the vector of length groupSizeK in column n of x2. Both K-axis slices start at j*groupSizeK, where j ranges over [0, kLoops) and kLoops = ceil(K / groupSizeK). K is the length of the K axis. The last slice may be shorter than groupSizeK.

    - Activation formulas:

      - GELU (tanh, high-performance approximation):

        $$
        activationOut=GELU(matmulOut)=matmulOut × Φ(matmulOut)=0.5 * matmulOut * (1 + tanh( \sqrt{2 / \pi} * (matmulOut + 0.044715 * matmulOut^{3})))
        $$

      - GELU (erf):

        $$
        activationOut=GELU(matmulOut)=0.5 * matmulOut * (1 + erf(matmulOut / \sqrt{2}))
        $$

      - SwiGLU: the original N must be a positive number and a multiple of 64. First add bias to the complete FP32 matrix multiplication result, then evenly split it along the trailing axis into two contiguous parts as `matmulOut=[gate | linear]`:

        $$
        activationOut=SiLU(gate) * linear
        $$

        MX quantization is performed after the SwiGLU result is rounded to BF16 in RNE mode. The output width is half of the original N.

      - SiLU: add bias to the complete FP32 matrix multiplication result, compute
        `SiLU(matmulOut)` element-wise (`SiLU(x) = x * sigmoid(x)`), then perform MX quantization after rounding to BF16 in RNE mode. The output keeps the original N columns.

    - Dynamic quantization formula:

      - Scenario 1: when scaleAlg is 0:
        - Group the input activationOut along the trailing axis into groups of $k = 32$ values. Each group of k values $\{\{V_i\}_{i=1}^{k}\}$ is dynamically quantized to $\{mxscale1, \{P_i\}_{i=1}^{k}\},\space k = 32$.

        $$
        shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
        mxscale = 2^{shared\_exp}\\
        P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space k\\
        $$

        - The quantized $P_{i}$ values form the output `y` at the positions of the corresponding $V_{i}$ values, and the mxscale values form the output `y_scale` according to the groups on the trailing axis.

        - emax: the exponent bits of the largest normal number of the corresponding data type.

            |   DataType    | emax |
            | :-----------: | :--: |
            |  FLOAT4_E2M1  |  2   |
            |  FLOAT4_E1M2  |  0   |
            | FLOAT8_E4M3FN |  8   |
            |  FLOAT8_E5M2  |  15  |

      - Scenario 2: when scaleAlg is 1, only FP8 types are involved:
        - Split the input activationOut along the trailing axis into blocks of $k = 32$ values. Compute a block scaling factor $S_{fp32}^b$ for each block separately, then map all elements in the block to the target low-precision type FP8 using the same $S_{fp32}^b$. If the last block has fewer than $k = 32$ elements, treat the missing values as 0 and process the block as a full block.
        - Find the maximum absolute value of the values in the block:

          $$
          Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
          $$

        - Map FP32 to the range representable by the target data type FP8, where $Amax(DType)$ is the maximum value representable by the target precision:

          $$
          S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
          $$

        - Convert the block scaling factor $S_{fp32}^b$ to the scaling value $S_{ue8m0}^b$ representable in FP8 format.
        - Extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from the floating-point block scaling factor $S_{fp32}^b$.
        - To prevent overflow during quantization, round the exponent up and keep it within the range representable by FP8:

          $$
          E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a normal number, } E_{int}^b < 254 \text{, and } M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a subnormal number and } M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{otherwise} \end{cases}
          $$

        - Compute the block scaling factor: $S_{ue8m0}^b=2^{E_{int}^b}$
        - Compute the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
        - Apply the final step of quantization. For each element in the block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$. The quantization result of the final output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ is the scaling factor of the block, that is, $S_{ue8m0}^b$, and $[d^i]_{i=1}^k$ is the quantized data in the block.

      - Scenario 3: when scaleAlg is 2, only the FP4_E2M1 type is involved:
        - When dstTypeMax = 0.0/6.0/7.0:
          - Group the input activationOut along the trailing axis into groups of $k = blocksize$ values. Each group of k values $\{\{V_i\}_{i=1}^{k}\}$ is dynamically quantized to $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = blocksize:

          $$
          shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{if one or two of the high bits preceding the mantissa bits are 1, and the mantissa is not all 0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{otherwise} \end{cases}
          $$

          $$
          P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize
          $$

          - The quantized $P_{i}$ values form the output `y` at the positions of the corresponding $V_{i}$ values, and the mxscale values form the output `y_scale` according to the groups on the trailing axis.

        - When dstTypeMax != 0.0/6.0/7.0:
          - Split the input activationOut along the trailing axis into blocks of $k = blocksize$ values. Compute a block scaling factor $S_{fp32}^b$ for each block separately, then map all elements in the block to the target low-precision type using the same $S_{fp32}^b$. If the last block has fewer than $k$ elements, treat the missing values as 0 and process the block as a full block.
          - Find the maximum absolute value of the values in the block:

            $$
            Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
            $$

          - Map FP32 to the range representable by the target data type. When dstTypeMax=0, $Amax(DType)$ is the maximum value representable by the target precision. When dstTypeMax!=0, $Amax(DType)$ is the passed-in value of dstTypeMax.

            $$
            S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
            $$

          - Convert the block scaling factor $S_{fp32}^b$ to the scaling value $S_{ue8m0}^b$ representable in FP8 format.
          - Extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from the floating-point block scaling factor $S_{fp32}^b$.
          - To prevent overflow during quantization, round the exponent up and keep it within the range representable by FP8:

            $$
            E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a normal number, } E_{int}^b < 254 \text{, and } M_{fixp}^b > 0 \\ E_{int}^b, & \text{otherwise} \end{cases}
            $$

          - Compute the block scaling factor: $S_{ue8m0}^b=2^{E_{int}^b}$
          - Compute the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
          - Apply the final step of quantization. For each element in the block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$. The quantization result of the final output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ is the scaling factor of the block, that is, $S_{ue8m0}^b$, and $[d^i]_{i=1}^k$ is the quantized data in the block.
          - The quantized $P_{i}$ values form the output `y` at the positions of the corresponding $V_{i}$ values, and the mxscale values form the output `y_scale` according to the groups on the trailing axis.

### FP8 Scaling Algorithm Selection

- `scale_alg=0` uses the OCP shared exponent algorithm, which is consistent with the OCP semantics of [DynamicMxQuant](../../quant/dynamic_mx_quant/README.md) and [SwigluMxQuant](../../quant/swiglu_mx_quant/README.md). This algorithm directly generates a power-of-two scale based on the maximum exponent and does not additionally increase the scale based on the maximum finite value of the target FP8. When the normalized result falls outside the FP8 representation boundary, the output is determined by the underlying FP8 type conversion semantics.
- `scale_alg=1` uses the BLAS algorithm, which computes the scale based on the maximum finite value of the target FP8 and rounds the E8M0 exponent up, reducing the risk of quantization overflow at boundary values. Use this algorithm when the input may reach the FP8 representation boundary and the service requires finite outputs.
- With the same output dtype, 32-element grouping, and `scale_alg`, the fused operator only adds matrix multiplication and activation before quantization. The scale generation method and FP8 conversion rules are consistent with those of separately invoked small operators.

## Activation Capability Comparison Table

All rows below use the MX quantization mode. `N` is the original output width of the matrix multiplication, and `H` is the output width after activation. GELU includes `gelu_tanh` and `gelu_erf`; SwiGLU and SiLU correspond to `swiglu` and `silu`, respectively.

| Activation | Input/Output Type | x2 Format | transposeX1 | transposeX2 | Original N Constraint | Output Width H | scaleAlg | roundMode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GELU | FP8 | ND | false/true | false/true | N > 0, no additional alignment requirement | N | 0/1 | rint |
| GELU | FP8 | WeightNZ | false/true | false/true | N > 1, no additional alignment requirement | N | 0/1 | rint |
| GELU | FP4 | ND | false/true | false/true | positive even number | N | 0/2 | rint/floor/round |
| GELU | FP4 | WeightNZ | false | false/true | positive even number | N | 0/2 | rint/floor/round |
| SwiGLU | FP8 | ND | false/true | false/true | positive number and a multiple of 64 | N/2 | 0/1 | rint |
| SwiGLU | FP8 | WeightNZ | false | false/true | positive number and a multiple of 64 | N/2 | 0/1 | rint |
| SiLU | FP8 | ND | false/true | false/true | N > 0, no additional alignment requirement, 65 allowed | N | 0/1 | rint |
| SiLU | FP8 | WeightNZ | false | false/true | N > 1, no additional alignment requirement, 65 allowed | N | 0/1 | rint |

- **SiLU is computed element-wise without N/2 split, and the output keeps the complete N columns. The multiple-of-64 N constraint of SwiGLU does not apply to SiLU. Neither SwiGLU nor SiLU supports FP4.**
- In FP8 scenarios, `x1` and `y` are both FLOAT8_E4M3FN or both FLOAT8_E5M2. ND `x2` can be either type, while WeightNZ `x2` supports only FLOAT8_E4M3FN. In FP4 scenarios, `x1`, `x2`, and `y` are all FLOAT4_E2M1. In all scenarios, the input and output scales are E8M0.
- The output logical shapes are `y=[..., M, H]` and `y_scale=[..., M, ceil(H/64), 2]`. Therefore, the `y_scale` of SiLU is `[..., M, ceil(N/64), 2]`, which is `[..., M, 2, 2]` when N=65. On the PyTorch side, the FP4 `y` is packed as two elements per uint8, and the physical trailing axis is H/2.
- `transposeX1/X2` in the table are ACLNN parameters, corresponding to the operator attributes `transpose_x1/x2`. The PyTorch API expresses them through transposed views of the tensors and the corresponding scales. WeightNZ also requires K > 1. FP4 also requires K > 2 and even inner axes of x1 and x2.

ND and WeightNZ are invoked through [aclnnQuantMatmulActivationQuant](docs/aclnnQuantMatmulActivationQuant_en.md) and [aclnnQuantMatmulActivationQuantWeightNz](docs/aclnnQuantMatmulActivationQuantWeightNz_en.md), respectively. The [PyTorch API](docs/torchapi_quant_matmul_activation_quant_en.md) selects the interface based on the x2 format.

## Parameters

<table class="tg" style="undefined;table-layout: fixed; width: 1166px"><colgroup>
<col style="width: 81px">
<col style="width: 121px">
<col style="width: 430px">
<col style="width: 390px">
<col style="width: 144px">
</colgroup>
<thead>
  <tr>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Parameter name</span></th>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Input/Output/Attribute</span></th>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Description</span></th>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Data type</span></th>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Data format</span></th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">x1</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Input</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">The left matrix in the matrix multiplication.</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">x2</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Input</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">The right matrix in the matrix multiplication.</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FRACTAL_NZ, ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">x1_scale_optional</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Optional input</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">The scaling factor of the quantization parameters in the matrix multiplication computation, corresponding to x1Scale in the formula.</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT8_E8M0</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">x2_scale</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Input</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">The scaling factor of the quantization parameters in the matrix multiplication computation, corresponding to x2Scale in the formula.</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FLOAT8_E8M0</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">bias_optional</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Optional input</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">The bias accumulated after the matrix multiplication, corresponding to bias in the formula.</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT32</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">y</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Output</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">The result of the matrix multiplication and activation after dynamic quantization.</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">y_scale</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Output</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">The quantization scale of each group after dynamic quantization.</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT8_E8M0</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
</tbody></table>

### Attribute Description

| Attribute Name | Description | Default Value |
| --- | --- | --- |
| transpose_x1 | Whether the input shape of x1 is transposed. | false |
| transpose_x2 | Whether the input shape of x2 is transposed. | false |
| group_size | The quantization group sizes along the m, n, and k directions of the input, composed of the three concatenated values groupSizeM, groupSizeN, and groupSizeK. Currently, only [1, 1, 32] is supported in the MX scenario. | 0 |
| activation_type | Activation function type. Supported values: "gelu_tanh", "gelu_erf", "swiglu", and "silu". | "gelu_tanh" |
| y_dtype | Data type of the output y. | DT_FLOAT8_E4M3FN |
| quant_mode | Quantization mode. Currently, "mx" is supported. | "mx" |
| round_mode | Rounding mode. When y is FLOAT4_E2M1, "rint", "floor", and "round" are supported. When y is FLOAT8, only "rint" is supported. | "rint" |
| scale_alg | Scaling algorithm. When y is FLOAT4_E2M1, 0 and 2 are supported. When y is FLOAT8, 0 and 1 are supported. | 0 |
| dst_type_max | Maximum value of the target data type, used to control the quantization range. When scale_alg is 0 or 1, this attribute does not take effect. When scale_alg is 2, supported values are 0.0 and 6.0-12.0. | 0.0 |

## Constraints

- Empty tensors are not supported.
- Contiguous tensors are supported. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) support only the scenario where the last two axes are transposed.
- The input and output support the following data type combinations:

    | x1            | x2            | x1_scale    | x2_scale    | bias         | y                         | y_scale      |
    |---------------|---------------|-------------|-------------|--------------|---------------------------|-------------|
    | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | null/FLOAT32 | FLOAT8_E4M3FN             | FLOAT8_E8M0 |
    | FLOAT8_E5M2   | FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | null/FLOAT32 | FLOAT8_E5M2               | FLOAT8_E8M0 |
    | FLOAT8_E5M2   | FLOAT8_E5M2   | FLOAT8_E8M0 | FLOAT8_E8M0 | null/FLOAT32 | FLOAT8_E5M2               | FLOAT8_E8M0 |
    | FLOAT8_E4M3FN | FLOAT8_E5M2   | FLOAT8_E8M0 | FLOAT8_E8M0 | null/FLOAT32 | FLOAT8_E4M3FN             | FLOAT8_E8M0 |
    | FLOAT4_E2M1   | FLOAT4_E2M1   | FLOAT8_E8M0 | FLOAT8_E8M0 | null/FLOAT32 | FLOAT4_E2M1               | FLOAT8_E8M0 |

  - When x2 is FLOAT8_E5M2, only the ND format is supported and the NZ format is not supported.
  - In the GELU, SwiGLU, and SiLU scenarios, the data type of y is determined by the data type of x1 and must stay consistent. SwiGLU and SiLU support only MXFP8 and do not support MXFP4.
  - MXFP4 scenario constraints (the data types of x1, x2, and y are all FLOAT4_E2M1):
    - scale_alg supports only the values 0 and 2.
    - When scale_alg is 2, dst_type_max supports the values 0.0 and 6.0-12.0.
    - round_mode supports "rint", "floor", and "round".
    - The inner axes of x1 and x2 must be even, and k must be greater than 2.
    - The N dimension of y must be even.
    - When x2 is in NZ format, transposition of x1 is not supported.

## Invocation Description

  | Invocation method   | Sample Code           | Description                                         |
  | ---------------- | --------------------------- | --------------------------------------------------- |
  | ACLNN: GELU + MXQuant, ND weight | [test_aclnn_quant_matmul_gelu_mx_quant](examples/arch35/test_aclnn_quant_matmul_gelu_mx_quant.cpp) | Uses [aclnnQuantMatmulActivationQuant](docs/aclnnQuantMatmulActivationQuant_en.md), with `gelu_tanh` as the activation and N as the output width. |
  | ACLNN: GELU + MXQuant, WeightNZ weight | [test_aclnn_quant_matmul_gelu_mx_quant_weight_nz](examples/arch35/test_aclnn_quant_matmul_gelu_mx_quant_weight_nz.cpp) | Converts the weight first, then calls [aclnnQuantMatmulActivationQuantWeightNz](docs/aclnnQuantMatmulActivationQuantWeightNz_en.md), with `gelu_tanh` as the activation. |
  | ACLNN: SwiGLU + MXQuant, ND weight | [test_aclnn_quant_matmul_swiglu_mx_quant](examples/arch35/test_aclnn_quant_matmul_swiglu_mx_quant.cpp) | Uses [aclnnQuantMatmulActivationQuant](docs/aclnnQuantMatmulActivationQuant_en.md), with `swiglu` as the activation, an `[N]` FP32 bias, and N/2 as the output width. |
  | PyTorch API | - | Use<br>[quant_matmul_activation_quant](docs/torchapi_quant_matmul_activation_quant_en.md)<br>to invoke the QuantMatmulActivationQuant operator. |

## SwiGLU Support Scope

When `activation_type="swiglu"`, after bias is added to the complete FP32 matrix multiplication result, the result is evenly split along the last dimension into two contiguous parts as `C=[gate | linear]`, `SiLU(gate) * linear` is computed, and MX quantization is performed after the SwiGLU result is rounded to BF16 in RNE mode.

- The original `N` must be a positive number and a multiple of 64. The output width `N/2` can have a trailing block of 32 columns.
- `x1` and `x2` are E4M3FN or E5M2, and the data type of `y` must be consistent with that of `x1`. The input and output scales are E8M0.
- `x2` supports ND or WeightNZ. For specific data type and transposition combinations, see [Activation Capability Comparison Table](#activation-capability-comparison-table).
- `bias` can be empty. When not empty, it is FP32 `[N]`; the three-dimensional output scenario also supports `[B,1,N]`. The bias applies to the complete original N width before the split, and `[N/2]` is not accepted.
- The input rank is 2 to 6, and the batch axes are right-aligned for broadcasting. The batch dimension of the scale must be consistent with the corresponding input.
- `quant_mode="mx"`, `round_mode="rint"`, and `scale_alg=0/1`. The group size is the default value or `[1,1,32]`.

The SwiGLU output shapes are `y=[...,M,N/2]` and `y_scale=[...,M,ceil((N/2)/64),2]`. The output width of GELU and SiLU is still N. Two 32-element quantization groups form one 64-element scale storage group, and trailing elements that do not fill a group are not written to y.

For the ND and WeightNZ interface constraints, see [aclnnQuantMatmulActivationQuant](docs/aclnnQuantMatmulActivationQuant_en.md) and [aclnnQuantMatmulActivationQuantWeightNz](docs/aclnnQuantMatmulActivationQuantWeightNz_en.md), respectively.
