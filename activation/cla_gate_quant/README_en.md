# ClaGateQuant

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: This is a fused operator that implements the head-wise gate weighted merging of the two CLA (Cross-Layer Attention) paths and dynamic block quantization. It first applies the Sigmoid gating to the Attention outputs of the Global/CLA branch and the Local/SWA branch and merges them with weights, then reshapes the merged result to `[T, K]` (`K = N*D`) for block-based dynamic quantization, and outputs low-precision FP8/FP4 tensors and the corresponding E8M0 scaling factors. When `dst_type` is FLOAT8_E4M3FN or FLOAT8_E5M2, the value of `scale_alg` specifies the different algorithms for computing the scale. The quantization mode is controlled by `dual_axis_flag`: in single-axis mode (`dual_axis_flag=false`, default), only one set of quantization data is output in the K direction (`[1,32]` block, row-wise), and `col_data`/`col_scale` are empty outputs; in dual-axis mode (`dual_axis_flag=true`), col-wise quantization data is also output in the T direction (`[32,1]` block, col-wise).

- Formula:

  **Stage 1: CLA gate fusion**

  Apply the two-path Sigmoid gated weighting token by token and head by head, where $\sigma(x) = \dfrac{1}{1 + e^{-x}}$ is the Sigmoid function:

  $$
  s_g = \sigma(z_g), \qquad s_l = \sigma(z_l)
  $$

  $$
  merged[t, n, d] = s_g[t, n] \cdot O_g[t, n, d] + s_l[t, n] \cdot O_l[t, n, d]
  $$

  Logical matrix before quantization:

  $$
  X = \operatorname{reshape}(merged, [T, K]), \qquad K = N \times D
  $$

  **Stage 2: Dual-axis dynamic block quantization**

  - Scenario 1, when `scale_alg` is 0, that is, the OCP Microscaling Formats (Mx) Specification implementation:
    - **Row-wise quantization (K direction)**: Quantize X along the K dimension by groups of 32 numbers. A group of 32 numbers $\{\{V_i\}_{i=1}^{32}\}$ is quantized to $\{row\_scale, \{P_i\}_{i=1}^{32}\}$

      $$
      shared\_exp = floor(log_2(max_i(|V_i|))) - emax
      $$

      $$
      row\_scale = 2^{shared\_exp}
      $$

      $$
      P_i = cast\_to\_dst\_type(V_i/row\_scale, round\_mode), \space i\space from\space 1\space to\space 32
      $$

    - **Col-wise quantization (T direction, only when `dual_axis_flag=true`)**: Quantize X along the T dimension by groups of 32 numbers. A group of 32 numbers $\{\{V_j\}_{j=1}^{32}\}$ is quantized to $\{col\_scale, \{P_j\}_{j=1}^{32}\}$

      $$
      shared\_exp = floor(log_2(max_j(|V_j|))) - emax
      $$

      $$
      col\_scale = 2^{shared\_exp}
      $$

      $$
      P_j = cast\_to\_dst\_type(V_j/col\_scale, round\_mode), \space j\space from\space 1\space to\space 32
      $$

    - The $P_i$ values after row-wise quantization are placed at the positions of the corresponding $V_i$ values to form the output `row_data`, and $row\_scale$ is grouped along the corresponding K dimension to form the output `row_scale` (the scales of every two adjacent groups form a pair, stored in the last dimension; the shape is `[T, ceil(N*D/64), 2]`, and when the last group has fewer than two values, even padding with 0 is applied). The $P_j$ values after col-wise quantization are placed at the positions of the corresponding $V_j$ values to form the output `col_data`, and $col\_scale$ is grouped along the corresponding T dimension to form the output `col_scale` (the shape is `[ceil(T/64), N*D, 2]`, and even padding with 0 is applied likewise).
    - emax: the exponent of the largest normal number of the corresponding data type.

      |   DataType    | emax |
      | :-----------: | :--: |
      |  FLOAT4_E2M1  |  2   |
      |  FLOAT4_E1M2  |  0   |
      | FLOAT8_E4M3FN |  8   |
      |  FLOAT8_E5M2  |  15  |

  - Scenario 2, when `scale_alg` is 1, only FP8 types are involved (round-up algorithm):
    - **Row-wise quantization (K direction)**: Quantize X along the K dimension by groups of 32 numbers. Each group has a length of 32. Compute a block scaling factor $S_{fp32}^b$ for each group separately, and then map all elements in the group to the target low-precision type FP8 by using the same $S_{fp32}^b$. If the last group contains fewer than 32 elements, treat the missing values as 0 and process the group as a complete group.
      - Find the maximum absolute value of the numbers in the group:

        $$
        Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{32})
        $$

      - Map FP32 to the range representable by the target data type FP8, where $Amax(DType)$ is the maximum value representable by the target precision (448 for FLOAT8_E4M3FN and 57344 for FLOAT8_E5M2):

        $$
        S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
        $$

      - Convert the block scaling factor $S_{fp32}^b$ to a scaling value $S_{ue8m0}^b$ representable in the FP8 format: extract the unbiased exponent $E_{int}^b$ and the mantissa $M_{fixp}^b$ from $S_{fp32}^b$. To prevent overflow during quantization, round the exponent up, and keep it within the range representable by FP8:

        $$
        E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a normal number, and } E_{int}^b < 254 \text{ and } M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{if } S_{fp32}^b \text{ is a subnormal number, and } M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{otherwise} \end{cases}
        $$

      - Compute the block scaling factor: $row\_scale=S_{ue8m0}^b=2^{E_{int}^b}$
      - Compute the block conversion factor: $R_{fp32}^b=\dfrac{1}{fp32(S_{ue8m0}^b)}$
      - Apply the final step of quantization. For each element in the group, $P_i = cast\_to\_dst\_type(d_{fp32}^i \cdot R_{fp32}^b)$. The final quantization result of the row-wise output is $\left(row\_scale, [P_i]_{i=1}^{32}\right)$.
    - **Col-wise quantization (T direction, only when `dual_axis_flag=true`)**: Quantize X along the T dimension by groups of 32 numbers. Use the same round-up algorithm as row-wise quantization, compute the block scaling factor for each group independently, and quantize the group to obtain $col\_scale$ and $\{P_j\}_{j=1}^{32}$. The final quantization result of the col-wise output is $\left(col\_scale, [P_j]_{j=1}^{32}\right)$.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format | shape |
| :----- | :------------ | :--- | :------- | :------- | :---- |
| global_attn | Input | Attention output `O_g` of the Global/CLA branch | BFLOAT16, FLOAT16 | ND (TND) | `[T, N, D]` |
| local_attn | Input | Attention output `O_l` of the Local/SWA branch, with the same shape as global_attn | BFLOAT16, FLOAT16 | ND (TND) | `[T, N, D]` |
| global_gate_logits | Input | Pre-Sigmoid value `z_g` of the Global gate | BFLOAT16, FLOAT16 | ND | `[T, N]` |
| local_gate_logits | Input | Pre-Sigmoid value `z_l` of the Local gate | BFLOAT16, FLOAT16 | ND | `[T, N]` |
| row_data | Output | Row-wise quantization data `P_i` | FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2 | ND | `[T, N*D]` |
| row_scale | Output | E8M0 scaling factor of each `[1,32]` group in the row-wise direction (stored in pairs in the last dimension, with even padding of 0) | FLOAT8_E8M0 | ND | `[T, ceil(N*D/64), 2]` |
| col_data | Output | Col-wise quantization data `P_j`; empty output when `dual_axis_flag=false` | FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2 | ND | `[T, N*D]` |
| col_scale | Output | E8M0 scaling factor of each `[32,1]` group in the col-wise direction (with even padding of 0); empty output when `dual_axis_flag=false` | FLOAT8_E8M0 | ND | `[ceil(T/64), N*D, 2]` |
| dst_type | Optional attribute | Target quantization type: underlying attributes 35=FLOAT8_E5M2, 36=FLOAT8_E4M3FN, 40=FLOAT4_E2M1, 41=FLOAT4_E1M2, default 36; the PTA API registers by ScalarType int enumeration: 23=torch.float8_e5m2, 24=torch.float8_e4m3fn, 296=torch_npu.float4_e2m1fn_x2, 297=torch_npu.float4_e1m2fn_x2, and None is equivalent to 24=torch.float8_e4m3fn | INT64 | - | - |
| round_mode | Optional attribute | Rounding mode: FP8 supports only `"rint"`; FP4 supports `"rint"`/`"floor"`/`"round"`. Default `"rint"` | STRING | - | - |
| scale_alg | Optional attribute | Scale computation method: 1=cuBLAS, 0=OCP; FP4 supports only 0. Default 1 | INT64 | - | - |
| input_attn_layout | Optional attribute | Layout format of the inputs global_attn/local_attn. Currently only `"TND"` is supported. Default `"TND"` | STRING | - | - |
| dual_axis_flag | Optional attribute | Quantization mode: true=dual-axis quantization (outputs both row/col result sets at the same time); false=single-axis quantization (outputs only row-wise, and col_data/col_scale are empty outputs). Default false | BOOL | - | - |

> Note: The attribute registration order of the operator is `dst_type`, `round_mode`, `scale_alg`, `input_attn_layout`, and `dual_axis_flag`, which is consistent with the tiling and aclnn implementations; the default value of `dual_axis_flag` is `False` on both the operator side and the PTA side, and `dst_type` is an optional ScalarType int enumeration. When `None` is passed, the value is processed as `24=torch.float8_e4m3fn`.

## Constraints

- `global_attn` / `local_attn` must be 3-dimensional tensors with the same shape, namely `[T, N, D]`; `N ∈ [1,128]`, `D ∈ {128, 256}`.
- `global_gate_logits` / `local_gate_logits` must be 2-dimensional tensors `[T, N]`, and their data types must be the same as those of `global_attn`/`local_attn`.
- When `dst_type` is FP4 (40/41 on the underlying side and 296/297 on the PTA side), `K = N*D` must be divisible by 4, and `scale_alg` must be 0.
- The FP8 output types support only the `"rint"` rounding mode.
- When `dual_axis_flag=false`, `col_data` / `col_scale` are empty outputs (with the internal empty shape `[0]`), and the API does not access these outputs.

## Invocation Description

| Invocation method | Invocation sample | Description |
| :------- | :------- | :--- |
| aclnn invocation | [test_aclnn_cla_gate_quant](./examples/arch35/test_aclnn_cla_gate_quant.cpp) | Invoke the ClaGateQuant operator through the [aclnnClaGateQuant](./docs/aclnnClaGateQuant.md) API. |
| Graph mode | [cla_gate_quant_proto.h](./op_graph/cla_gate_quant_proto.h) | Invoke the ClaGateQuant operator by building a graph through the operator IR prototype [ClaGateQuant](./op_graph/cla_gate_quant_proto.h). |
| PyTorch API | [cla_gate_quant.py](./torch_extension/cla_gate_quant.py) | Invoke the ClaGateQuant operator through the Torch Extension API [cla_gate_quant](./docs/torchapi_cla_gate_quant.md) (both the single-operator mode and the TorchAir graph mode are supported). |
