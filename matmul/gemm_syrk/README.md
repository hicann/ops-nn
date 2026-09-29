# GemmSyrk

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2系列产品 | × |
| Atlas 200I/500 A2推理产品 | × |
| Atlas推理系列产品 | × |
| Atlas训练系列产品 | × |

## 功能说明

- 算子功能：实现对称秩k更新（syrk，参考 cublas `?syrk`）计算。基于 Blaze 框架
  （`BlockMmadSyrk` + `GemmUniversal` syrk 特化 + `BlockEpilogueFmmWithScaleAdd`，
  MIX 1 AIC : 2 AIV）实现。

- 计算公式：

  $$
  C = \alpha \times (A @ A^T) + \beta \times C
  $$

  其中 $A$ 的shape为 $(\dots, m, k)$（2-6 维，前面为 batch 轴），$C$ 为 $(\dots, m, m)$
  的对称矩阵。

- transpose_x 属性为 true 时，$A$ 以转置的 $(\dots, k, m)$ 布局存储（2-6 维），
  计算 $C = \alpha \times (A^T @ A) + \beta \times C$（对应 cublas syrk 的 OP_T
  语义）。kernel 以 DNExt 视图绑定转置存储，单次搬运走 dn2nz 路径，产生与
  非转置场景字节级相同的双视图 L1 镜像（$NZ(X^T) \equiv ZN(X)$），L0A/L0B
  装配与双 fixpipe 输出链路完全复用。

- 单次搬运（GM→L1）：利用分形对偶性 $NZ(X)(m,k) \equiv ZN(X^T)(k,m)$（字节级相等），
  每个 A 行块每 k-chunk 仅做一次 `nd2nz CopyGM2L1`，同一份 L1 镜像以 NZ 视图供给
  `CopyL12L0A`（L0A）、以 ZN 视图供给 `CopyL12L0B`（L0B）；对角块（i==j）一次搬运
  同时喂两个 L0。kernel 仅遍历上三角槽位，总 GM→L1 搬运量为通用 matmul 组合的一半
  （下界 $nB^2$）。

- 计算减半（Mmad）：每个上三角槽位仅跑一条 Mmad 链产出 C[i,j]；镜像 tile
  C[j,i] = C[i,j]^T（依赖输入 C 对称——syrk 语义约定）由 AIV 侧
  `WriteTransposedTile` 转置写出：16×16 块分块跨步 GM→UB 回读（L2 热）→
  `asc_transpose`（3510 `vtranspose`，b16 位级 16×16 转置，half/bf16 通用）→
  分块跨步 UB→GM 写出；边角块与 CPU 仿真回退标量路径。cube 计算量 ≈ 通用 matmul
  的一半（$nB(nB+1)/2$）。

- 原地更新：`c` 的输入与输出为同一地址（def 中 Input("c") 与 Output("c") 同名，
  GE 将输出端口别名到输入内存），kernel 内 epilogue 分段读取 $\beta \times C$
  后原位写回，读写时序由 MTE2→V→MTE3 事件链保证安全。

- 完整输出：算子内部按 M×N 全量分块调度（上三角槽位成对计算，下三角由对偶 tile
  覆盖），将计算结果完整写出到整个对称矩阵，不单只输出上三角或下三角。

- $k = 0$ 场景由 aclnn 层路由为逐元素 $C = \beta \times C$（`l0op::Muls`），
  不进入 matmul kernel。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ---- | ---- | ---- | ---- | ---- |
| a | 输入 | 输入矩阵A，shape为(…, m, k)，2-6维，前面为batch轴；transpose_x为true时为(…, k, m)。 | FLOAT16, BFLOAT16 | ND |
| c | 输入&输出 | 对称矩阵C，shape为(…, m, m)（与a的batch轴一致），输入输出同地址原地更新。 | FLOAT16, BFLOAT16 | ND |
| alpha | 属性 | 矩阵乘结果的缩放系数，默认1.0。 | FLOAT | - |
| beta | 属性 | C的缩放系数，默认1.0。 | FLOAT | - |
| transpose_x | 属性 | 为true时按转置布局解读a，计算C = alpha * (A^T @ A) + beta * C。 | BOOL | - |
| fill_mode | 属性 | 输出区域模式："full"（完整对称矩阵，默认）/"up"（上三角）/"low"（下三角）。原型支持三值，当前 aclnn/tiling 仅实现 "full"，其余值报参数错误。 | STRING | - |

## 约束

- 仅支持 DAV_3510（Ascend 950 / 350）平台，要求 `aivNum == aicNum * 2`（MIX 1:2）。
- a 与 c 数据类型必须一致（FLOAT16 或 BFLOAT16），格式仅支持 ND。
- c 的最后两维必须相等且等于 a 的 m 轴（transpose_x 为 true 时 m 为 a 的最后一维）；
  batch 轴必须与 a 一致（原地更新不支持广播）。
- 暂不支持图模式直接调用（proto IR 已定义但 infer-shape 未注册，构图期输出 shape 无法推导）；
  请使用 aclnn 接口（k = 0 时自动路由为逐元素缩放）。
- `fill_mode` 原型可配置 "full"/"up"/"low"，当前仅支持 "full"（完整对称矩阵输出）；"up"/"low" 在
  aclnn/tiling 层均返回参数错误，待后续实现。
- 对称方块 tiling 契约（host 侧已强制）：`baseM == baseN == mL1 == nL1`
  （镜像 tile (j, i) 换轴后由同一对称块尺寸同时满足 epilogue 单 N-chunk 规则
  与 baseM/baseN 行 clamp，即 `mL1 >= baseM` 取等号；转置存储下 L1 槽位亦对称）、
  `mTailCnt == nTailCnt == 1`（不做核间尾块切分）、
  `baseM × baseN × 4B ≤ L0C_SIZE / 4` 且 16 对齐
  （单累加器位于一个 L0C 槽、槽粒度即 L0C/2，UB 需容纳 (i,j) ND 与 (j,i) DN
  两幅 fp32 镜像及两个 AIV 的 staging；对称块上界为
  floor16(sqrt(L0C/4/4B))，无需 L0CDB）。

## 接口

```c
aclnnStatus aclnnGemmSyrkGetWorkspaceSize(const aclTensor* a, aclTensor* cRef,
    const aclScalar* alphaOptional, const aclScalar* betaOptional, bool transposeX,
    const char* fillMode, uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnGemmSyrk(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
```

`cRef` 为原地更新的对称矩阵（[in/out]）；`alphaOptional`/`betaOptional` 为
`nullptr` 时取默认值 1.0；`transposeX` 为 true 时按转置布局解读 a；`fillMode`
为输出区域模式（"full"/"up"/"low"，当前仅支持 "full"，`nullptr` 默认 "full"）。
调用示例见 `examples/arch35/test_aclnn_gemm_syrk.cpp`。

torch 侧接口（cann_ops_nn 包）见 `docs/torchapi_gemm_syrk.md`：

```python
cann_ops_nn.gemm_syrk(a, c, *, alpha=None, beta=None, transpose_x=False, fill_mode="full")
    -> Tensor   # 原地更新c并返回c本身
```

## 目录结构

```
gemm_syrk/
├── CMakeLists.txt
├── op_graph/gemm_syrk_proto.h            # IR 定义（Input("c")/Output("c") 同名原地；图模式暂未支持）
├── op_api/                               # aclnn 接口 + l0op 层（无需 INFER_SHAPE：in-place cRef 自带输出描述符）
├── op_host/
│   ├── gemm_syrk_def.cpp                 # OpDef 注册
│   └── op_tiling/arch35/                  # tiling：独立 TilingBaseClass 阶段化实现（仅提取/校验/编排），
│       │                                  # DoTiling 经 tiling_strategy 优先级分发到 base_tiling
│       │                                  # （IsCapable + DoOpTiling：BMM ASW basic 计算 + syrk 钳制）
│       ├── compile_info / tiling_key
│       └── gemm_syrk_tiling_registry.cpp  # tiling 入口注册（950 分发 + TilingParse）
├── op_kernel/arch35/                     # Blaze kernel（MIX 1:2）
├── examples/arch35/                      # aclnn 调用示例
└── tests/                                # UT/ST
```
