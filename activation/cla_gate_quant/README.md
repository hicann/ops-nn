# ClaGateQuant

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：融合算子，实现CLA（Cross-Layer Attention）两路head-wise gate加权合并与动态块量化。先对Global/CLA分支与Local/SWA分支的Attention输出分别施加Sigmoid门控并加权合并，再将合并结果reshape为`[T, K]`（`K = N*D`）进行基于块的动态量化，输出低精度的FP8/FP4张量和对应的E8M0缩放因子。在`dst_type`为FLOAT8_E4M3FN、FLOAT8_E5M2时，根据`scale_alg`的取值来指定计算scale的不同算法。量化模式由`dual_axis_flag`控制：单轴模式（`dual_axis_flag=false`，默认）仅在K方向（`[1,32]` block，row-wise）输出一套量化数据，`col_data`/`col_scale`为空输出；双轴模式（`dual_axis_flag=true`）还会在T方向（`[32,1]` block，col-wise）输出col-wise量化数据。

- 计算公式：

  **阶段1：CLA gate融合**

  逐token、逐head做两路Sigmoid门控加权，其中$\sigma(x) = \dfrac{1}{1 + e^{-x}}$为Sigmoid函数：

  $$
  s_g = \sigma(z_g), \qquad s_l = \sigma(z_l)
  $$

  $$
  merged[t, n, d] = s_g[t, n] \cdot O_g[t, n, d] + s_l[t, n] \cdot O_l[t, n, d]
  $$

  量化前逻辑矩阵：

  $$
  X = \operatorname{reshape}(merged, [T, K]), \qquad K = N \times D
  $$

  **阶段2：双轴动态块量化**

  - 场景1，当`scale_alg`为0时，即OCP Microscaling Formats (Mx) Specification实现：
    - **Row-wise量化（K方向）**：将X在K维度上按32个数一组量化，一组32个数$\{\{V_i\}_{i=1}^{32}\}$量化为$\{row\_scale, \{P_i\}_{i=1}^{32}\}$

      $$
      shared\_exp = floor(log_2(max_i(|V_i|))) - emax
      $$

      $$
      row\_scale = 2^{shared\_exp}
      $$

      $$
      P_i = cast\_to\_dst\_type(V_i/row\_scale, round\_mode), \space i\space from\space 1\space to\space 32
      $$

    - **Col-wise量化（T方向，仅`dual_axis_flag=true`）**：将X在T维度上按32个数一组量化，一组32个数$\{\{V_j\}_{j=1}^{32}\}$量化为$\{col\_scale, \{P_j\}_{j=1}^{32}\}$

      $$
      shared\_exp = floor(log_2(max_j(|V_j|))) - emax
      $$

      $$
      col\_scale = 2^{shared\_exp}
      $$

      $$
      P_j = cast\_to\_dst\_type(V_j/col\_scale, round\_mode), \space j\space from\space 1\space to\space 32
      $$

    - Row-wise量化后的$P_i$按对应的$V_i$的位置组成输出`row_data`，$row\_scale$按对应的K维度上的分组组成输出`row_scale`（每两个相邻分组的scale组成一对，存于最后一维，shape为`[T, ceil(N*D/64), 2]`，尾组不足两个时偶数pad填充0）。Col-wise量化后的$P_j$按对应的$V_j$的位置组成输出`col_data`，$col\_scale$按对应的T维度上的分组组成输出`col_scale`（shape为`[ceil(T/64), N*D, 2]`，同样偶数pad填充0）。
    - emax: 对应数据类型的最大正则数的指数位。

      |   DataType    | emax |
      | :-----------: | :--: |
      |  FLOAT4_E2M1  |  2   |
      |  FLOAT4_E1M2  |  0   |
      | FLOAT8_E4M3FN |  8   |
      |  FLOAT8_E5M2  |  15  |

  - 场景2，当`scale_alg`为1时，只涉及FP8类型（向上取整算法）：
    - **Row-wise量化（K方向）**：将X在K维度上按32个数一组量化，每组长度为32，对每组单独计算一个块缩放因子$S_{fp32}^b$，再把组内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型FP8。如果最后一组不足32个元素，把缺失值视为0，按照完整组处理。
      - 找到该组中数值的最大绝对值：

        $$
        Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{32})
        $$

      - 将FP32映射到目标数据类型FP8可表示的范围内，其中$Amax(DType)$是目标精度能表示的最大值（FLOAT8_E4M3FN为448，FLOAT8_E5M2为57344）：

        $$
        S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
        $$

      - 将块缩放因子$S_{fp32}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$：从$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$，为保证量化时不溢出，对指数向上取整，且在FP8可表示的范围内：

        $$
        E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{否则} \end{cases}
        $$

      - 计算块缩放因子：$row\_scale=S_{ue8m0}^b=2^{E_{int}^b}$
      - 计算块转换因子：$R_{fp32}^b=\dfrac{1}{fp32(S_{ue8m0}^b)}$
      - 应用到量化的最终步骤，对于每个组内元素，$P_i = cast\_to\_dst\_type(d_{fp32}^i \cdot R_{fp32}^b)$，最终Row-wise输出的量化结果是$\left(row\_scale, [P_i]_{i=1}^{32}\right)$。
    - **Col-wise量化（T方向，仅`dual_axis_flag=true`）**：将X在T维度上按32个数一组量化，采用与Row-wise相同的向上取整算法，对每组独立计算块缩放因子并量化，得到$col\_scale$与$\{P_j\}_{j=1}^{32}$，最终Col-wise输出的量化结果是$\left(col\_scale, [P_j]_{j=1}^{32}\right)$。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 | shape |
| :----- | :------------ | :--- | :------- | :------- | :---- |
| global_attn | 输入 | Global/CLA分支Attention输出`O_g` | BFLOAT16、FLOAT16 | ND (TND) | `[T, N, D]` |
| local_attn | 输入 | Local/SWA分支Attention输出`O_l`，shape与global_attn一致 | BFLOAT16、FLOAT16 | ND (TND) | `[T, N, D]` |
| global_gate_logits | 输入 | Global gate的Sigmoid前值`z_g` | BFLOAT16、FLOAT16 | ND | `[T, N]` |
| local_gate_logits | 输入 | Local gate的Sigmoid前值`z_l` | BFLOAT16、FLOAT16 | ND | `[T, N]` |
| row_data | 输出 | Row-wise量化数据`P_i` | FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2 | ND | `[T, N*D]` |
| row_scale | 输出 | Row-wise每个`[1,32]`分组的E8M0缩放因子（两两一组存最后一维，偶数pad补0） | FLOAT8_E8M0 | ND | `[T, ceil(N*D/64), 2]` |
| col_data | 输出 | Col-wise量化数据`P_j`；`dual_axis_flag=false`时为空输出 | FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2 | ND | `[T, N*D]` |
| col_scale | 输出 | Col-wise每个`[32,1]`分组的E8M0缩放因子（偶数pad补0）；`dual_axis_flag=false`时为空输出 | FLOAT8_E8M0 | ND | `[ceil(T/64), N*D, 2]` |
| dst_type | 可选属性 | 目标量化类型：底层属性35=FLOAT8_E5M2、36=FLOAT8_E4M3FN、40=FLOAT4_E2M1、41=FLOAT4_E1M2，默认36；PTA接口按ScalarType int枚举注册：23=torch.float8_e5m2、24=torch.float8_e4m3fn、296=torch_npu.float4_e2m1fn_x2、297=torch_npu.float4_e1m2fn_x2，None等价24=torch.float8_e4m3fn | INT64 | - | - |
| round_mode | 可选属性 | 舍入模式：FP8仅支持`"rint"`；FP4支持`"rint"`/`"floor"`/`"round"`。默认`"rint"` | STRING | - | - |
| scale_alg | 可选属性 | scale计算方法：1=cuBLAS，0=OCP；FP4仅支持0。默认1 | INT64 | - | - |
| input_attn_layout | 可选属性 | 输入global_attn/local_attn的排布格式，当前仅支持`"TND"`。默认`"TND"` | STRING | - | - |
| dual_axis_flag | 可选属性 | 量化模式：true=双轴量化（同时输出row/col两套结果）；false=单轴量化（仅输出row-wise，col_data/col_scale为空输出）。默认false | BOOL | - | - |

> 说明：算子属性注册顺序为`dst_type`、`round_mode`、`scale_alg`、`input_attn_layout`、`dual_axis_flag`，与tiling、aclnn实现保持一致；算子与PTA侧`dual_axis_flag`默认均为`False`，`dst_type`为可选ScalarType int枚举，传`None`时按`24=torch.float8_e4m3fn`处理。

## 约束说明

- `global_attn` / `local_attn`必须为3维张量且shape一致，为`[T, N, D]`；`N ∈ [1,128]`，`D ∈ {128, 256}`。
- `global_gate_logits` / `local_gate_logits`必须为2维张量`[T, N]`，数据类型与`global_attn`/`local_attn`一致。
- 当`dst_type`为FP4（底层40/41，PTA侧为296/297）时，`K = N*D`必须可被4整除，且`scale_alg`必须为0。
- FP8输出类型仅支持`"rint"`舍入模式。
- `dual_axis_flag=false`时，`col_data` / `col_scale`为空输出（内部空形状`[0]`），接口不访问该输出。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :------- | :------- | :--- |
| aclnn调用 | [test_aclnn_cla_gate_quant](./examples/arch35/test_aclnn_cla_gate_quant.cpp) | 通过[aclnnClaGateQuant](./docs/aclnnClaGateQuant.md) 接口方式调用ClaGateQuant算子。 |
| 图模式 | [cla_gate_quant_proto.h](./op_graph/cla_gate_quant_proto.h) | 通过[ClaGateQuant](./op_graph/cla_gate_quant_proto.h) 算子IR原型构图方式调用ClaGateQuant算子。 |
| PyTorch API | [cla_gate_quant.py](./torch_extension/cla_gate_quant.py) | 通过[cla_gate_quant](./docs/torchapi_cla_gate_quant.md) Torch Extension接口调用ClaGateQuant算子（支持单算子模式与TorchAir图模式）。 |
