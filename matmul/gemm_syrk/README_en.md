# GemmSyrk

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | × |
| Atlas A2 products | × |
| Atlas 200I/500 A2 inference products | × |
| Atlas inference products | × |
| Atlas training products | × |

## Function

- Operator function: Implements the symmetric rank-k update (syrk, referring to cublas `syrk`) computation:

  <div>
  C = α × (A @ A<sup>T</sup>) + β × C
  </div>

  The shape of A is (…, m, k) (2-6 dimensions, with batch axes in front), C is a
  symmetric matrix of shape (…, m, m), and the input and output share the same
  address for in-place update.

- When the transpose_x attribute is true, A is stored in the transposed (…, k, m) layout (the OP_T
  semantics of cublas syrk), and the operator computes C = α × (A<sup>T</sup> @ A) + β × C.

- Implementation principle (based on the Blaze framework, MIX 1 AIC : 2 AIV):
  - Single data movement: uses the fractal duality NZ(X)(m,k) ≡ ZN(X<sup>T</sup>)(k,m) (byte-level equality);
    each A row block requires only one GM→L1 movement per k-chunk, and the same L1 copy feeds
    both cube inputs L0A and L0B at the same time, so the total movement volume is half of
    that of a general matrix multiplication combination;
  - Halved computation: only traverses the compact upper triangular slots (`BlockSchedulerSyrkTriangular`); each slot produces
    C[i,j] with a single Mmad chain; AIV sub0 writes out (i,j) with the nz2nd fixpipe, and AIV sub1 writes out
    the mirrored (j,i) with the nz2dn fixpipe (hardware transpose); there is no GM readback and no software transpose,
    and the cube computation volume is about half of that of a general matrix multiplication;
  - Special path: when k = 0 or α = 0, the aclnn layer routes the computation to element-wise C = β × C (`l0op::Muls`) and does not start the matmul kernel;
    when m or batch is 0, the operator directly returns success (no-op for empty tensors).
  - cuBLAS semantics alignment: when β = 0, C is not read (C does not need to be a valid input, and NaN/Inf in it does not propagate to the output),
    and the output is α × (A @ A^T); when k = 0 or α = 0 and β = 0, the output is all 0 (`l0op::ZerosLike`).

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| ---- | ---- | ---- | ---- | ---- |
| a | Input | Input matrix A, with shape (…, m, k), 2-6 dimensions, with batch axes in front; (…, k, m) when transpose_x is true. | FLOAT16, BFLOAT16 | ND |
| c | Input & Output | Symmetric matrix C, with shape (…, m, m) (consistent with the batch axes of a); the input and output share the same address for in-place update. | FLOAT16, BFLOAT16 | ND |
| alpha | Attribute | Scaling factor of the matrix multiplication result. Default value: 1.0. | FLOAT | - |
| beta | Attribute | Scaling factor of C. Default value: 1.0. | FLOAT | - |
| transpose_x | Attribute | When true, the operator interprets a in the transposed layout and computes C = alpha * (A^T @ A) + beta * C. | BOOL | - |
| fill_mode | Attribute | Output region mode: "full" (complete symmetric matrix, default)/"up" (upper triangle)/"low" (lower triangle). The prototype supports the three values; currently the aclnn/tiling implementation supports only "full", and other values report a parameter error. | STRING | - |

## Constraints

- Only the DAV_3510 (Ascend 950PR/Ascend 950DT) platform is supported, and `aivNum == aicNum * 2` (MIX 1:2) is required.
- The data types of a and c must be the same (FLOAT16 or BFLOAT16), and only the ND format is supported.
- The last two dimensions of c must be equal and must equal the m axis of a (m is the last dimension of a when transpose_x is true);
  the batch axes must be consistent with those of a (in-place update does not support broadcasting).
- The value range of each of the m, k, and n dimensions and of the multi-dimensional batch product is [0, 2147483647].
- Direct invocation in graph mode is temporarily not supported (the proto IR is defined but infer-shape is not registered, so the output shape cannot be deduced during graph building);
  use the aclnn or torch interface.
- `fill_mode` currently supports only "full" (complete symmetric matrix output); "up"/"low" both return a parameter error at the aclnn/tiling layer,
  pending later implementation.
- Symmetric block tiling contract (already enforced on the host side): `baseM == baseN == mL1 == nL1`
  (symmetric block size, 16-aligned), a single tail block per axis (no inter-core tail block splitting), and
  `baseM × baseN × 4B ≤ L0C_SIZE / 2` (a single accumulator occupies one L0C slot, and the slot granularity is L0C/2;
  the UB must hold the fp32 accumulator mirror and the staging of the two AIVs). For the detailed derivation, refer to
  the [BlockMmadSyrk](https://gitcode.com/cann/ops-tensor/blob/master/docs/API/gemm/block/block_mmad_matmul_syrk.md) documentation.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation method</th>
    <th>Invocation sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn invocation</td>
    <td><a href="./examples/arch35/test_aclnn_gemm_syrk.cpp">test_aclnn_gemm_syrk</a></td>
    <td>Two-phase interface: `aclnnGemmSyrkGetWorkspaceSize` + `aclnnGemmSyrk`.
    For the interface prototype and parameter description, refer to <a href="./docs/aclnnGemmSyrk.md">aclnnGemmSyrk</a>.</td>
  </tr>
  <tr>
    <td>torch invocation</td>
    <td><a href="./docs/torchapi_gemm_syrk.md">torchapi_gemm_syrk</a></td>
    <td>`cann_ops_nn.gemm_syrk(a, c, *, alpha=None, beta=None, transpose_x=False, fill_mode="full")`,
    which updates c in place and returns c.</td>
  </tr>
  <tr>
    <td>Graph-mode invocation</td>
    <td>-</td>
    <td>Temporarily not supported (see Constraints).</td>
  </tr>
</tbody>
</table>

## Directory Structure

```text
gemm_syrk/
├── CMakeLists.txt
├── op_graph/gemm_syrk_proto.h            # IR definition (Input("c")/Output("c") in-place with the same name; graph mode is not supported yet)
├── op_api/                               # aclnn interface + l0op layer (no INFER_SHAPE is required: the in-place cRef carries its own output descriptor)
├── op_host/
│   ├── gemm_syrk_def.cpp                 # OpDef registration
│   └── op_tiling/arch35/                  # tiling: phased implementation as an independent TilingBaseClass (extraction/validation/orchestration only),
│       │                                  # DoTiling dispatches to base_tiling by tiling_strategy priority
│       │                                  # (IsCapable + DoOpTiling: BMM ASW basic computation + syrk clamping)
│       ├── compile_info / tiling_key
│       └── gemm_syrk_tiling_registry.cpp  # tiling entry registration (950 dispatch + TilingParse)
├── op_kernel/arch35/                     # Blaze kernel (MIX 1:2)
├── examples/arch35/                      # aclnn invocation example
└── tests/                                # UT/ST
```
