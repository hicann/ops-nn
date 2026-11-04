# QuantMatmulActivationQuant Tests

English | [简体中文](./README.md)

Covers the MXFP8 and MXFP4 paths of GELU (tanh/erf), and the MXFP8 paths of SwiGLU and SiLU.

| Directory | Content |
| --- | --- |
| `ut/op_api` | ND and WeightNZ interface tests |
| `ut/op_host` | InferShape and Host tiling tests |
| `ut/op_kernel` | CSV-driven Kernel UT |
| `st/arch35` | TTK Kernel, ACLNN, and E2E cases |
| `assets` | Golden and comparators |

## Environment

- Linux, Ascend950, and the matching CANN and ops-tensor/Blaze; Kernel UT requires tikicpulib.
- Install the Host/API and Kernel operator packages of the current version. For details, see [Compilation and Installation Guide](../../../docs/zh/install/build.md).
- TTK requires `ml-dtypes` and `en-dtypes>=0.0.4`; ACLNN, E2E, and NPU Golden require the matching `torch` and `torch_npu`, and E2E also requires TorchAir and `cann_ops_nn`.
- TTK requires support for full-storage transfer of WeightNZ and a custom NPU Golden for non-native dtypes (FP4).

## Kernel UT

Run the following command in the root directory of ops-nn:

```bash
bash build.sh -u --opkernel --ops=quant_matmul_activation_quant --soc=ascend950
```

The cases are driven by [CSV](ut/op_kernel/test_quant_matmul_activation_quant.csv). When you modify inputs or computation parameters, update `expectedY` and `expectedYScale` at the same time.

## TTK ST

| Path | Subcommand | Case file | Fused operator device time column |
| --- | --- | --- | --- |
| Kernel | `kernel` | [Kernel CSV](st/arch35/ttk_kernel_quant_matmul_activation_quant_st.csv) | `dyn_perf_us` (dynamic compilation by default) |
| ACLNN ND | `aclnn` | [ND CSV](st/arch35/ttk_aclnn_quant_matmul_activation_quant_st.csv) | `op_perf_us` |
| ACLNN WeightNZ | `aclnn` | [WeightNZ CSV](st/arch35/ttk_aclnn_quant_matmul_activation_quant_weight_nz_st.csv) | `op_perf_us` |
| E2E | `e2e` | [E2E CSV](st/arch35/ttk_e2e_quant_matmul_activation_quant_st.csv) | `eager_device_perf_us` |

Run in the ops-test-kit directory, for example ACLNN ND:

```bash
cd /path/to/ops-test-kit
OP_TESTS=/path/to/ops-nn/matmul/quant_matmul_activation_quant/tests
python3 -m ttk aclnn \
  -i "$OP_TESTS/st/arch35/ttk_aclnn_quant_matmul_activation_quant_st.csv" \
  --plugin "$OP_TESTS/assets/spec.py" --seed 42
```

For the other paths, replace the subcommand and CSV according to the table. CPU Golden is used by default; check whether `precision_status` in the result CSV is `PASS`.

Case configuration conventions:

- Kernel NZ: fill `input_shapes` with the physical shape, `input_ori_shapes` with the logical shape, and `input_ori_formats` with `ND`.
- ACLNN NZ: fill `tensor_view_shapes` with the logical shape, and fill `tensor_storage_shapes` explicitly with the complete physical shape. When the logical shape of FP8 weights is `(*batch, R, C)`, the physical shape is `(*batch, ceil(C/32), ceil(R/16), 16, 32)`.
- E2E: the API is `cann_ops_nn.ops.quant_matmul_activation_quant`, and the tensor order is `x1,x2,x2_scale,x1_scale,bias`; keep the matrix views as `(...,M,K)` and `(...,K,N)`.
- Output dtype attribute: Kernel uses `y_dtype`, ACLNN uses `yDtype`, and E2E uses `output_dtype`. E2E FP4 uses `uint8` packing and sets `output_dtype/x1_dtype/x2_dtype=296`. For specific enumerations and layouts, refer to the CSV of each path.

## Concatenation Performance Comparison

Kernel, ACLNN (including WeightNZ), and E2E can all measure the following concatenation benchmarks through `--golden-mode NPU`:

| Activation | Quantization type | Concatenation benchmark |
| --- | --- | --- |
| SiLU | MXFP8 | QBMM(FP32) → SiLU(FP32) → Cast(BF16) → DynamicMxQuant |
| SwiGLU | MXFP8 | QBMM(BF16) → SwigluMxQuant (last-axis split, left-half activation, mode=0) |
| GELU (tanh/erf) | MXFP8, MXFP4 | QBMM(FP32) → GELU(FP32) → Cast(BF16) → DynamicMxQuant |

Cast counts toward the benchmark time. SwiGLU requires the environment to provide `torch_npu.npu_swiglu_mx_quant`. MXFP8 supports `scale_alg=0/1` and `round_mode="rint"`; MXFP4 supports `scale_alg=0/2` and `round_mode="rint"/"floor"/"round"`, and passes `dst_type_max`.

Run in the environment above, for example:

```bash
python3 -m ttk aclnn \
  -i "$OP_TESTS/st/arch35/ttk_aclnn_quant_matmul_activation_quant_st.csv" \
  --plugin "$OP_TESTS/assets/spec.py" -t silu_nd_e4m3_n65 \
  --golden-mode NPU --devices 0 --process-count 1 \
  --warmup true --run 1 --seed 42 -o silu_perf.csv
```

When you test SwiGLU or GELU, change `-t` to `swiglu_s0_y36`, `gelu_tanh_s0_y36`, or `gelu_tanh_fp4_nd_rint`. For other paths, replace the subcommand and CSV according to the table above, and select the cases of the corresponding activation.
Keep warmup enabled so that the NZ weight format preparation completes during the warmup phase.

The concatenation time is `golden_device_perf_us`, and the speedup ratio is **concatenation time / fused operator device time**.
Use only results with `golden_source=npu` and a positive time; `golden_kernel_details` provides the time and call count of each operator in the benchmark.
These columns measure the device execution time. The current NPU Golden samples once per case, and `--run` does not increase its sampling count; you can run serially multiple times, save the results separately, and take the median.

## Golden and Precision

Precision acceptance uses the CPU Golden with `--golden-mode Enable` (default). For the reference implementation and comparators, see [assets/spec.py](assets/spec.py). The precision results of `NPU` mode are based on the concatenation output; the SwiGLU benchmark rounds to BF16 before the activation, which differs from the rounding position of the fused implementation.

ST requires that at least 99.9% of the elements of `y` and `y_scale` have an error of no more than 1 ULP respectively; the shapes and dtypes must match, the NaN positions and Inf signs must match, and positive and negative zeros are treated as equal. FP4 is compared as unpacked elements. Kernel UT performs strict comparison on fixed inputs.
