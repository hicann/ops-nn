# RotateQuant

English | [简体中文](./README.md)

## Applicable Products

| Product                                                     | Support |
| :------------------------------------------------------- | :------: |
| Ascend 950PR&950DT products                   |    √     |
| Atlas A3 products |    √     |
| Atlas A2 products |    √     |
| Atlas 200I/500 A2 inference products                  |    ×     |
| Atlas inference products                          |    ×     |
| Atlas training products                          |    ×     |

## Function

- Operator function: Applies a rotation transform to tensor x, then performs an optional clamp operation, and finally performs symmetric dynamic quantization (with the target data type int8 or quint4x2) or MX quantization (with the target data type of the FLOAT4 class or FLOAT8 class).
- Formula:
  1. Rotation transform

    $$
    Y = (x.\text{reshape}(*,k) @ \text{rotation}).\text{reshape}(m, n)
    $$

    Where: $\mathbf{x} \in \mathbb{R}^{m \times n}$, $\mathbf{Y} \in \mathbb{R}^{m \times n}$, $\mathbf{rotation} \in \mathbb{R}^{k \times k}$.

  2. When alpha is within the valid value range (0.0, 1.0), the clamp computation is performed

    $$
    groupMaxVal = GroupMax(|Y|) \\
    limit = alpha * groupMaxVal \\
    Y = Y.clamp(min=-limit, max=limit)
    $$
    GroupMax means that every 32 values form a group, and the maximum value within each group is computed.

  3. Quantization
    - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: symmetric dynamic quantization (pertoken row-wise quantization)
      - Scale factor computation (row by row)

        $$
        s_i = \frac{\max_{j \in [0,\ n-1]} |Y_{i,j}|}{C_{\text{MAX}}}
        $$

        Where: $s_i$ is the scale factor of row $i$; $C_{\text{MAX}}$ is the maximum value of the quantization range, which is 127 for int8 and 7 for quint4x2.
      - Quantization computation

        $$
        y_{i,j} = \frac{Y_{i,j}}{s_i}
        $$

    - <term>Ascend 950PR&950DT products</term>: MX quantization

      - Scenario 1, when scaleAlg is 0:
        - Groups the input x along the axis dimension into groups of k = 32 values. A group of k values $\{\{V_i\}_{i=1}^{k}\}$ is dynamically quantized to $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, where k = 32.

          $$
          shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
          mxscale = 2^{shared\_exp}\\
          P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space 32\\
          $$

        - The quantized $P_{i}$ values form output yOut at the positions corresponding to $V_{i}$, and mxscale forms output mxscaleOut according to the groups along the corresponding axis dimension.

        - emax: the exponent of the largest regular number for the corresponding data type.

            |   DataType    | emax |
            | :-----------: | :--: |
            |  FLOAT4_E2M1  |  2   |
            | FLOAT8_E4M3FN |  8   |
            |  FLOAT8_E5M2  |  15  |

      - Scenario 2, when scaleAlg is 1, only FP8 types are involved:
        - Divides the long vector into blocks, each of length k, and computes a block scale factor $S_{fp32}^b$ for each block separately. Then all elements within the block are mapped to the target low-precision FP8 type using the same $S_{fp32}^b$. If the last block contains fewer than k elements, the missing values are treated as 0 and the block is processed as a complete block.
        - Finds the maximum absolute value among the values in the block:

          $$
          Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
          $$

        - Maps FP32 to the range representable by the target data type FP8, where $Amax(DType)$ is the maximum value representable by the target precision.

          $$
          S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
          $$

        - Converts the block scale factor $S_{fp32}^b$ to a scale value $S_{ue8m0}^b$ representable in FP8 format.
        - Extracts the unbiased exponent $E_{int}^b$ and mantissa $M_{fixp}^b$ from the floating-point block scale factor $S_{fp32}^b$.
        - To prevent overflow during quantization, the exponent is rounded up and must stay within the range representable by FP8:

          $$
          E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{if} S_{fp32}^b \text{is a normal number, and} E_{int}^b < 254 \text{and} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{if} S_{fp32}^b \text{is a subnormal number, and} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{otherwise} \end{cases}
          $$

        - Computes the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
        - Computes the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
        - Applies the final step of quantization. For each element in the block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$. The final quantization output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ is the block scale factor, here referring to $S_{ue8m0}^b$, and $[d^i]_{i=1}^k$ is the quantized data within the block.

      - Scenario 3, when scaleAlg is 2, only the FP4_E2M1 type is involved:
        - When dstTypeMax = 6.0/7.0:
          - Groups the input x along the axis dimension into groups of k = 32 values. A group of k values $\{\{V_i\}_{i=1}^{k}\}$ is dynamically quantized to $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, where k = 32:

            $$
            shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{if the one or two higher bits before the mantissa bits are 1, and the mantissa is not all zeros} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{otherwise} \end{cases} \\
            $$

            $$
            P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space 32\\
            $$

          - The quantized $P_{i}$ values form output yOut at the positions corresponding to $V_{i}$, and mxscale forms output mxscaleOut according to the groups along the corresponding axis dimension.
        - When dstTypeMax != 6.0/7.0:
          - Divides the long vector into blocks, each of length k, and computes a block scale factor $S_{fp32}^b$ for each block separately. Then all elements within the block are mapped to the target low-precision type using the same $S_{fp32}^b$. If the last block contains fewer than k elements, the missing values are treated as 0 and the block is processed as a complete block.
          - Finds the maximum absolute value among the values in the block:

            $$
            Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
            $$

          - Maps FP32 to the range representable by the target data type. When dst_max_value=0, $Amax(DType)$ is the maximum value representable by the target precision; when dst_max_value!=0, $Amax(DType)$ is the value passed in dst_max_value.

            $$
            S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
            $$

          - Converts the block scale factor $S_{fp32}^b$ to a scale value $S_{ue8m0}^b$ representable in FP4_E2M1 format.
          - Extracts the unbiased exponent $E_{int}^b$ and mantissa $M_{fixp}^b$ from the floating-point block scale factor $S_{fp32}^b$.
          - To prevent overflow during quantization, the exponent is rounded up and must stay within the range representable by FP4_E2M1:

            $$
            E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{if} S_{fp32}^b \text{is a normal number, and} E_{int}^b < 254 \text{and} M_{fixp}^b > 0 \\ E_{int}^b, & \text{otherwise} \end{cases}
            $$

          - Computes the block scale factor: $S_{ue8m0}^b=2^{E_{int}^b}$
          - Computes the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
          - Applies the final step of quantization. For each element in the block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$. The final quantization output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ is the block scale factor, here referring to $S_{ue8m0}^b$, and $[d^i]_{i=1}^k$ is the quantized data within the block.
          - The quantized $P_{i}$ values form output yOut at the positions corresponding to $V_{i}$, and mxscale forms output mxscaleOut according to the groups along the corresponding axis dimension.

## Parameters

  <table style="undefined;table-layout: fixed; width: 962px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input tensor to be rotated and quantized.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rotation</td>
      <td>Input</td>
      <td>Rotation matrix.</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
    </tr>
  <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>Scale coefficient for the range to be limited by clamp.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_list</td>
      <td>Optional input</td>
      <td>Represents the quantization group size distribution of input `x`.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>Output</td>
      <td>Output tensor after quantization.</td>
      <td>INT8, INT4, FLOAT4_E2M1, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleOut</td>
      <td>Output</td>
      <td>Scale coefficient computed by dynamic quantization.</td>
      <td>FLOAT32, FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_dtype</td>
      <td>Attribute</td>
      <td>Specifies the output data type of yOut.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Attribute</td>
      <td>Axis that participates in quantization.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Attribute</td>
      <td>Specifies the rounding mode for quantization. The supported values are rint, floor, and round.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>Attribute</td>
      <td>Indicates the computation method of scale.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type_max</td>
      <td>Attribute</td>
      <td>Indicates the maximum value of the quantization target type.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>trans</td>
      <td>Attribute</td>
      <td>Specifies whether to transpose the input.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group_list_type</td>
      <td>Optional attribute</td>
      <td><ul><li>Represents the grouping mode of the groupList input. When group_list is not nullptr, the supported values are 0 and 1.</li><li>The default value is 0.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

## Constraints

- <term>Ascend 950PR&950DT products</term>:
  - The shape of x is (*, N) with dimensions in the range [1, 7]; the shape of rotation is (K, K) or (N/K, K, K) with dimensions in the range [2, 3]. In the current version, K supports only 32, 64, 128, and 256.
  - The length (N) of the last dimension of x must be an integer multiple of K.
  - The output type of yOut is FLOAT4_E2M1, FLOAT8_E4M3FN, or FLOAT8_E5M2, and its shape is the same as x.
  - x and rotation must have the same data type.
  - The shape of scaleOut must be (*, CeilDiv(N,64), 2), and its data type is FLOAT8_E8M0.
  - alpha is an optional input. When it is not a null pointer, its shape is (1,), its data type is BFLOAT16, and its valid value range is (0.0, 1.0). No clamp processing is performed if a null pointer is passed in or the value is outside the valid range.
  - Currently, axis supports only -1 or D-1, where D is the number of dimensions of the shape of x.
  - roundMode supports "rint", "round", and "floor". When a null pointer is passed in, the "rint" mode is used. When the data type of yOut is FLOAT8_E4M3FN or FLOAT8_E5M2, roundMode supports only "rint".
  - scaleAlg supports the values 0, 1, and 2. When the data type of yOut is FLOAT8_E4M3FN or FLOAT8_E5M2, only 0 and 1 are supported; when the data type of yOut is FLOAT4_E2M1, only 0 and 2 are supported.
  - dstTypeMax: when scaleAlg=2, dstTypeMax must be in the range [6.0, 12.0]; in other scenarios, only 0.0 is supported.
  - Currently, trans supports only false.
  - group_list supports nullptr.
  - When group_list is not nullptr:
    - group_list_type must take a value from [0, 1].
    - The shape of group_list is (E,), where E is the number of groups. E must be greater than or equal to 1 and less than or equal to 1024.
    - The values of group_list must satisfy the following conditions; otherwise the correctness of the output cannot be guaranteed:
      - When group_list_type is 0, group_list must be a non-negative, monotonically non-decreasing sequence, representing the cumsum result (cumulative sum) of the size of each group after grouping. The last value must be less than or equal to the first dimension of the tensor in x.
      - When group_list_type is 1, group_list must be a non-negative sequence, representing the size of each group after grouping. The sum of the values must be less than or equal to the first dimension of the tensor in x.

- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
  - group_list supports only nullptr.
  - The shape of x is (M, N), and the shape of rotation is (K, K).
  - The shape of rotation must be a square matrix (K, K).
  - The length (N) of the second dimension of x must be an integer multiple of K, and N must be divisible by 8.
  - When the output type of yOut is int8, the shape of y must be the same as x, that is (M, N).
  - When the output type of yOut is int32, the shape of y must be (M, N//8).
  - x and rotation must have the same data type.
  - The shape of scaleOut must be (M), and its data type is FLOAT.
  - alpha is an optional input. In the current version, only a null pointer is supported.
  - Currently, axis supports only -1 or 1.
  - roundMode supports "rint", "round", and "floor". When a null pointer is passed in, the "rint" mode is used.
  - Currently, scaleAlg supports only 0.
  - Currently, dstTypeMax supports only 0.0.
  - Currently, trans supports only false.
  - The range of N is [128, 16000].
  - The range of K is [16, 1024].

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface (V1) | [test_aclnn_rotate_quant](examples/test_aclnn_rotate_quant.cpp) | Invokes the RotateQuant operator through the [aclnnRotateQuant](docs/aclnnRotateQuant_en.md) interface. |
| aclnn interface (V2) | [test_aclnn_rotate_quant_v2](examples/test_aclnn_rotate_quant_v2.cpp) | Invokes the RotateQuant operator through the [aclnnRotateQuantV2](docs/aclnnRotateQuantV2.md) interface. The V2 interface supports the grouped quantization feature. |
| PyTorch interface | - | Invokes the RotateQuant operator through the [rotate_quant](docs/torchapi_rotate_quant.md) interface.|
