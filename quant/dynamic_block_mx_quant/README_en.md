# DynamicBlockMxQuant

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

- Operator function: converts the input variable to the destination data type through MX quantization, using data blocks (32\*32) as the basic blocks. In each basic block, based on the value of scale_alg, a different scale algorithm computes the quantization parameter scale (1\*1) of the current block, which is broadcast to scale1 (32\*1) and scale2 (1\*32) as outputs. Meanwhile, each value in the basic block is divided by scale and converted to the corresponding dst_type according to round_mode, producing the quantized result y.

- Formula:
    - Scenario 1: when scaleAlg is 0:
      - Group the input x into basic blocks of data blocks (32\*32). The values in one data block $\{\{V_i\}_{i=1}^{32*32}\}$ are quantized to $\{scale, \{P_i\}_{i=1}^{32*32}\}$.

        $$
        shared\_exp = floor(log_2(max_i(|V_i|))) - emax
        $$

        $$
        scale = 2^{shared\_exp}
        $$

        $$
        P_i = cast\_to\_dst\_type(V_i/scale, round\_mode), \space i\space from\space 1\space to\space 32*32
        $$

      - Meanwhile, scale (1\*1) is broadcast to scale1 (32\*1) and scale2 (1\*32) as the outputs scale1 and scale2. The quantized $P_{i}$ values form the output y at the positions of the corresponding $V_{i}$ values.

      - emax: the exponent bits of the largest normal number of the corresponding data type.

        |   DataType    | emax |
        | :-----------: | :--: |
        |  FLOAT4_E2M1  |  2   |
        |  FLOAT4_E1M2  |  0   |
        | FLOAT8_E4M3FN |  8   |
        |  FLOAT8_E5M2  |  15  |

    - Scenario 2: when scaleAlg is 2, only the FLOAT4_E2M1 type is involved:
      - Split the long vector into blocks of length k. Compute a block scaling factor $S_{fp32}^b$ for each block separately, then map all elements in the block to the target low-precision type FP8 using the same $S_{fp32}^b$. If the last block has fewer than k elements, treat the missing values as 0 and process the block as a full block.
      - Find the maximum absolute value of the values in the block:
      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$
      - Introduce the new attribute dst_type_max. dst_type_max is of type float and defaults to 0. In the default output, it indicates that max_type is the maximum value of the target data type. If another value is passed in, the scale must be computed based on the passed-in value. Currently, the valid values are 0.0, 6.0, and 7.0, and this value can be set only in the FLOAT4_E2M1 scenario.
      - Map FP32 to the range representable by the target data type FLOAT4_E2M1.
      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{dst\_type\_max}
      $$
      - Convert the block scaling factor $S_{fp32}^b$ to the scaling value $S_{ue8m0}^b$ representable in FP8 format.
      - Extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from the floating-point block scaling factor $S_{fp32}^b$.
      - To prevent overflow during quantization, round the exponent up and keep it within the range representable by FP8:
      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a normal number, } E_{int}^b < 254 \text{, and } M_{fixp}^b > 0 \\ E_{int}^b, & \text{otherwise} \end{cases}
      $$
      - Compute the block scaling factor: $S_{ue8m0}^b=2^{E_{int}^b}$
      - Compute the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
      - Apply the final step of quantization. For each element in the block, $d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$. The quantization result of the final output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $S^b$ is the scaling factor of the block, that is, $S_{ue8m0}^b$, and $[d^i]_{i=1}^k$ is the quantized data in the block.

## Parameters

  | Parameter Name | Input/Output/Attribute | Description | Data type | Data format |
  | ----- | ----- |----- |----- |----- |
  | x | Input | Represents the input x, corresponding to $V_i$ in the formula.<br>When dst_type is FLOAT4_E2M1 or FLOAT4_E1M2, the last dimension of x must be even. | FLOAT16, BFLOAT16 | ND |
  | round_mode  | Optional attribute | Represents the data conversion mode, corresponding to round_mode in the formula.<br>When dst_type is 40/41, corresponding to the output y data types FLOAT4_E2M1/FLOAT4_E1M2, {"rint", "floor", "round"} is supported;<br>When dst_type is 35/36, corresponding to the output y data types FLOAT8_E5M2/FLOAT8_E4M3FN, only {"rint"} is supported;<br>When a null pointer is passed in, the "rint" mode is used. | STRING | - |
  | dst_type | Input | Represents the type of y after the specified data conversion.<br>The input range is {35, 36, 40, 41}, corresponding to the output y1 and y2 data types {35:FLOAT8_E5M2, 36:FLOAT8_E4M3FN, 40:FLOAT4_E2M1, 41:FLOAT4_E1M2}, respectively. | INT64 | - |
  | scale_alg | Input | Represents the scale computation method.<br>Currently, only the values 0 and 2 are supported, representing the OCP Microscaling Formats (Mx) Specification and Dynamic Dtype Range implementations, respectively. | INT64 | - |
  | dst_type_max | Input | Represents that max_type is the maximum value of the target data type.<br>This parameter takes effect when scale_alg=2 and dst_type is FLOAT4_E2M1, and the scale must be computed based on the passed-in value.<br>Currently, the valid values are 0.0/6.0/7.0, and this value can be set only in the FLOAT4_E2M1 scenario. | FLOAT | - |
  | y | Output | Represents the result corresponding to the quantized input x, corresponding to $P_i$ in the formula.<br>The shape is the same as that of the input x. | FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2 | ND |
  | scale1 | Output | Represents the quantization scale of each group along the -1 axis, corresponding to scale1 broadcast from scale in the formula.<br>The shape is the value of the -1 axis of x divided by 32 and rounded up, then padded to an even number, with a pad fill value of 0. | FLOAT8_E8M0 | ND |
  | scale2 | Output | Represents the quantization scale of each group along the -2 axis, corresponding to scale2 broadcast from scale in the formula. <br>The shape is the value of the -2 axis of x divided by 32 and rounded up, then padded to an even number, with a pad fill value of 0; <br>The scale2 output requires interleaving processing for every two rows of data. | FLOAT8_E8M0 | ND |

## Constraints

 - The shape constraints on x, scale1, and scale2 are as follows:
    - The dimensions of x should be 2 to 3.
    - rank(scale1) = rank(x) + 1.
    - rank(scale2) = rank(x) + 1.
    - scale1.shape[-2] = (ceil(x.shape[-1] / 32) + 2 - 1) / 2.
    - scale2.shape[-3] = (ceil(x.shape[-2] / 32) + 2 - 1) / 2.
    - scale1.shape[-1] = 2.
    - scale2.shape[-1] = 2.
    - The other dimensions are the same as those of the input x.
    - Example: when the shape of the input x is [B, M, N] and the destination data type is an FP8 type, the shape of the corresponding y is [B, M, N], the shape of scale1 is [B, M, ceil(N/64), 2], and the shape of scale2 is [B, ceil(M/64), N, 2].

## Invocation Description

| Invocation method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_dynamic_block_mx_quant](examples/arch35/test_aclnn_dynamic_block_mx_quant.cpp) | Invokes the DynamicBlockMxQuant operator through the [aclnnDynamicBlockMxQuant](docs/aclnnDynamicBlockMxQuant_en.md) API. |
| Graph mode | -  | Invokes the DynamicBlockMxQuant operator through graph construction using the [operator IR](op_graph/dynamic_block_mx_quant_proto.h).         |
