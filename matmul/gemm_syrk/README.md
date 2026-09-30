# GemmSyrk

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | × |
| Atlas A2系列产品 | × |
| Atlas 200I/500 A2推理产品 | × |
| Atlas推理系列产品 | × |
| Atlas训练系列产品 | × |

## 功能说明

- 算子功能：实现对称秩k更新（syrk，参考 cublas `syrk`）计算：

  <div>
  C = α × (A @ A<sup>T</sup>) + β × C
  </div>

  其中 A 的shape为 (…, m, k)（2-6 维，前面为 batch 轴），C 为 (…, m, m)
  的对称矩阵，输入输出同地址原地更新。

- transpose_x 属性为 true 时，A 以转置的 (…, k, m) 布局存储（cublas syrk
  的 OP_T 语义），计算 C = α × (A<sup>T</sup> @ A) + β × C。

- 实现原理（基于 Blaze 框架，MIX 1 AIC : 2 AIV）：
  - 单次搬运：利用分形对偶性 NZ(X)(m,k) ≡ ZN(X<sup>T</sup>)(k,m)（字节级相等），
    每个 A 行块每 k-chunk 仅一次 GM→L1 搬运，同一份 L1 镜像同时供给 L0A 与 L0B
    两个 cube 输入，总搬运量为通用矩阵乘组合的一半；
  - 计算减半：仅遍历紧凑上三角槽位（`BlockSchedulerSyrkTriangular`），每槽位单条
    Mmad 链产出 C[i,j]，AIV sub0 以 nz2nd fixpipe 写出 (i,j)、AIV sub1 以
    nz2dn fixpipe（硬件转置）写出镜像 (j,i)，无 GM 回读、无软件转置，
    cube 计算量约为通用矩阵乘的一半；
  - 特殊路径：k = 0 或 α = 0 时由 aclnn 层路由为逐元素C = β × C（`l0op::Muls`），不启动 matmul kernel
    m 或 batch 为 0 时直接返回成功（空张量 no-op）。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| ---- | ---- | ---- | ---- | ---- |
| a | 输入 | 输入矩阵A，shape为(…, m, k)，2-6维，前面为batch轴；transpose_x为true时为(…, k, m)。 | FLOAT16, BFLOAT16 | ND |
| c | 输入&输出 | 对称矩阵C，shape为(…, m, m)（与a的batch轴一致），输入输出同地址原地更新。 | FLOAT16, BFLOAT16 | ND |
| alpha | 属性 | 矩阵乘结果的缩放系数，默认1.0。 | FLOAT | - |
| beta | 属性 | C的缩放系数，默认1.0。 | FLOAT | - |
| transpose_x | 属性 | 为true时按转置布局解读a，计算C = alpha * (A^T @ A) + beta * C。 | BOOL | - |
| fill_mode | 属性 | 输出区域模式："full"（完整对称矩阵，默认）/"up"（上三角）/"low"（下三角）。原型支持三值，当前 aclnn/tiling 仅实现 "full"，其余值报参数错误。 | STRING | - |

## 约束说明

- 仅支持 DAV_3510（Ascend 950PR/Ascend 950DT）平台，要求 `aivNum == aicNum * 2`（MIX 1:2）。
- a 与 c 数据类型必须一致（FLOAT16 或 BFLOAT16），格式仅支持 ND。
- c 的最后两维必须相等且等于 a 的 m 轴（transpose_x 为 true 时 m 为 a 的最后一维）；
  batch 轴必须与 a 一致（原地更新不支持广播）。
- m、k、n各维度及多维batch乘积的取值范围为 (0, 2147483647)。
- 暂不支持图模式直接调用（proto IR 已定义但 infer-shape 未注册，构图期输出 shape 无法推导）；
  请使用 aclnn 或 torch 接口。
- `fill_mode` 当前仅支持 "full"（完整对称矩阵输出）；"up"/"low" 在 aclnn/tiling 层
  均返回参数错误，待后续实现。
- 对称方块 tiling 契约（host 侧已强制）：`baseM == baseN == mL1 == nL1`
  （对称块尺寸，16 对齐）、每轴单尾块（不做核间尾块切分）、
  `baseM × baseN × 4B ≤ L0C_SIZE / 2`（单累加器位于一个 L0C 槽，槽粒度即 L0C/2；
  UB 需容纳 fp32 累加器镜像与两个 AIV 的 staging）。详细推导见
  [BlockMmadSyrk](https://gitcode.com/cann/ops-tensor/blob/master/docs/API/gemm/block/block_mmad_matmul_syrk.md) 文档。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn调用</td>
    <td><a href="./examples/arch35/test_aclnn_gemm_syrk.cpp">test_aclnn_gemm_syrk</a></td>
    <td>两段式接口：`aclnnGemmSyrkGetWorkspaceSize` + `aclnnGemmSyrk`，
    接口原型与参数说明详见<a href="./docs/aclnnGemmSyrk.md">aclnnGemmSyrk</a>。</td>
  </tr>
  <tr>
    <td>torch调用</td>
    <td><a href="./docs/torchapi_gemm_syrk.md">torchapi_gemm_syrk</a></td>
    <td>`cann_ops_nn.gemm_syrk(a, c, *, alpha=None, beta=None, transpose_x=False, fill_mode="full")`，
    原地更新 c 并返回 c。</td>
  </tr>
  <tr>
    <td>图模式调用</td>
    <td>-</td>
    <td>暂不支持（见约束说明）。</td>
  </tr>
</tbody>
</table>

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
