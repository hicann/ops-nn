# QuantMatmulActivationQuant Tests

English | [简体中文](./README.md)

Covers MXFP8 and MXFP4 computation for GELU (tanh/erf), and MXFP8 computation for SwiGLU, SiLU, and sigmoid followed by elementwise multiplication with `x3`. For operator specifications, see the [README](../README_en.md).

| Directory | Test coverage |
| --- | --- |
| [ut/op_api](ut/op_api) | Parameter and layout validation for ND and NZ interfaces |
| [ut/op_host](ut/op_host) | Shape inference and tiling |
| [ut/op_kernel](ut/op_kernel) | CSV-driven Kernel UT |
| [st/arch35](st/arch35) | TTK Kernel, ACLNN, and E2E tests |
| [assets/spec.py](assets/spec.py) | Reference computation (Golden) and accuracy comparators |

## Environment

- Use Linux and Ascend 950 with compatible CANN and ops-tensor/Blaze versions, and install the Host/API and Kernel operator packages for the current version. See the [build and installation instructions](../../../docs/zh/install/build.md). Kernel UT also requires tikicpulib.
- TTK requires `ml-dtypes` and `en-dtypes>=0.0.4`. ACLNN, E2E, and NPU Golden require `torch` and `torch_npu`; E2E also requires `cann_ops_nn`.
- For NZ or FP4 tests, TTK must support passing the complete NZ storage and using a custom NPU Golden for FP4.

## ACLNN UT

Run the following command from the ops-nn repository root.

```bash
bash build.sh -u --opapi --ops=quant_matmul_activation_quant --soc=ascend950
```

Both the [ND cases](ut/op_api/test_aclnn_quant_matmul_activation_quant.csv) and the [WeightNZ cases](ut/op_api/test_aclnn_quant_matmul_activation_quant_weight_nz.csv) cover V1 and V2.

- `x1Stride`, `x2Stride`, `x1ScaleStride`, and `x2ScaleStride` describe non-contiguous views. The optional trailing columns `yStride` and `yScaleStride` verify that non-contiguous outputs return `ACLNN_ERR_PARAM_INVALID`. An empty stride field constructs a contiguous tensor.
- Transpose cases cover matching and mismatched views on either matrix and its scale, explicit transpose flags, equal-length scale axes, and FP4. Mismatched standard transpose views must return `ACLNN_ERR_PARAM_INVALID` from `GetWorkspaceSize`.
- Singleton-axis cases cover equivalent shapes such as `[2,1,2]` and `[1,2,2]`, ND and NZ weights, V1 and V2, explicit and view transposes, FP4, and sigmoid. Inputs with missing K groups, mismatched batch dimensions, or a last dimension other than 2 must be rejected.

## Kernel UT

Run the following command in the root directory of ops-nn:

```bash
bash build.sh -u --opkernel --ops=quant_matmul_activation_quant --soc=ascend950
```

The cases are driven by [CSV](ut/op_kernel/test_quant_matmul_activation_quant.csv). When you modify inputs or computation parameters, update `expectedY` and `expectedYScale` at the same time.

The CSV covers A/B transposes, FP8/FP4 ND and NZ inputs, `scaleAlg`, `roundMode` and `dstTypeMax`. FP4 `expectedY` contains one unpacked 4-bit code per logical element; the driver packs GM inputs and unpacks outputs.

Sigmoid cases use `x3Pattern=signed_coordinate`: the BF16 input `x3` cycles through `{-2, -0.5, 0, 0.5, 1, 2}` using `(3 * row + col) % 6`, exercising signs, zeros and row/column offsets. `expectedY` and `expectedYScale` contain every output row. Other activations use `x3Pattern=none`.

## TTK ST

| Path | Subcommand | Case file | Fused operator device time column |
| --- | --- | --- | --- |
| Kernel | `kernel` | [Kernel CSV](st/arch35/ttk_kernel_quant_matmul_activation_quant_st.csv) | `dyn_perf_us` (dynamic compilation by default) |
| ACLNN ND | `aclnn` | [ND CSV](st/arch35/ttk_aclnn_quant_matmul_activation_quant_st.csv) | `op_perf_us` |
| ACLNN NZ | `aclnn` | [NZ CSV](st/arch35/ttk_aclnn_quant_matmul_activation_quant_weight_nz_st.csv) | `op_perf_us` |
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

Sigmoid + mul uses 2D inputs. The shape of `x3` must match `y`. See [Sigmoid Scenario](../README_en.md#sigmoid-scenario) for the operator constraints.

- Kernel cases cover full tiles, multiple tiles with tails, single rows, E4M3FN/BLAS, and E5M2/OCP. They exercise UB buffer reuse for `x3` and masked tail computation.
- API and tiling cases check rejection of batch dimensions on either matrix (including size-1 dimensions), extra dimensions in `x3`, and `x3` supplied with other activations.
- Positive sigmoid ST cases use 2D matrices. Other activations continue to cover batches with `x3` omitted.

E2E delivery and acceptance for this operator cover only PyTorch eager mode and validate single-operator invocation:

```bash
python3 -m ttk e2e \
  -i "$OP_TESTS/st/arch35/ttk_e2e_quant_matmul_activation_quant_st.csv" \
  --plugin "$OP_TESTS/assets/spec.py" --golden-mode Enable --seed 42 \
  --eager=true -c=false -d=false -o e2e_results.csv
```

Check that `eager_precision` contains accuracy results and that `precision_status` is `PASS`. Static and dynamic graph modes are outside the delivery scope.

### Test Case Configuration

- Kernel NZ: set `input_shapes` to the physical shape, `input_ori_shapes` to the logical shape, and `input_ori_formats` to `ND`.
- ACLNN NZ: set `tensor_view_shapes` to the logical shape and `tensor_storage_shapes` to the complete physical shape. For example, an FP8 weight with a logical shape of `(batch,R,C)` has a physical shape of `(batch,ceil(C/32),ceil(R/16),16,32)`.
- E2E: the API name in the CSV is `cann_ops_nn.ops.quant_matmul_activation_quant`. Tensor fields follow `x1,x2,x2_scale,x1_scale,bias,x3`; scalar parameters are bound by name from `attributes`. Although `x3` follows `dst_type_max` in the Torch API, no scalar placeholders belong in the CSV tensor fields. The logical matrix views are `(...,M,K)` and `(...,K,N)`.
- E2E cases for existing activations omit the trailing `x3` and use five tensor slots. `silu_nz_optional_defaults_e5m2_e4m3` keeps a sixth slot set to `None` to cover an explicit null value. Sigmoid cases pass the BF16 `x3` in the sixth slot.
- Specify the output type using `y_dtype` in Kernel CSV files, `yDtype` in ACLNN CSV files, and `output_dtype` in E2E CSV files. These are test configuration fields; the ACLNN interface itself obtains the data type from the output tensor. Use the enumeration values required by each CSV.
- E2E FP4 uses uint8 packing with `x1_dtype=x2_dtype=output_dtype=296`. For the layout, see the [PyTorch interface documentation](../docs/torchapi_quant_matmul_activation_quant_en.md#layout-and-storage).

## Performance Comparison with Separate Operators

Use `--golden-mode NPU` to measure the following reference sequences. QBMM denotes quantized matrix multiplication, and the type in parentheses is the output type of that step.

| Activation | Quantization type | Reference sequence |
| --- | --- | --- |
| SiLU | MXFP8 | QBMM(FP32) → SiLU(FP32) → Cast(BF16) → DynamicMxQuant |
| sigmoid + mul | MXFP8 | QBMM(BF16) → sigmoid(BF16) → mul(BF16 input `x3`, BF16 output) → DynamicMxQuant |
| SwiGLU | MXFP8 | QBMM(BF16) → SwigluMxQuant (last-axis split, left-half activation, mode=0) |
| GELU (tanh/erf) | MXFP8, MXFP4 | QBMM(FP32) → GELU(FP32) → Cast(BF16) → DynamicMxQuant |

Cast is included in the reference sequence's execution time. SwiGLU requires `torch_npu.npu_swiglu_mx_quant`. The quantization algorithm and rounding mode follow the test case parameters.

For example, to test ND SiLU:

```bash
python3 -m ttk aclnn \
  -i "$OP_TESTS/st/arch35/ttk_aclnn_quant_matmul_activation_quant_st.csv" \
  --plugin "$OP_TESTS/assets/spec.py" -t silu_nd_e4m3_n65 \
  --golden-mode NPU --devices 0 --process-count 1 \
  --warmup true --run 1 --seed 42 -o silu_perf.csv
```

To test other activations, change `-t` to `swiglu_s0_y36`, `gelu_tanh_s0_y36`, or `gelu_tanh_fp4_nd_rint`. Keep warm-up enabled so that NZ weight format preparation completes during warm-up.

In the results, `golden_device_perf_us` is the device execution time of the reference sequence. See the table above for the fused operator's execution time column. Calculate the speedup as "reference sequence time / fused operator time", using only results where `golden_source=npu` and both times are positive. `golden_kernel_details` records the execution time and invocation count of each operator in the reference sequence.

NPU Golden currently takes one measurement per case. `--run` does not increase this count. To collect multiple measurements, run the test repeatedly in sequence, save each result separately, and take the median.

## Accuracy Criteria

For accuracy validation, use the default `--golden-mode Enable`, which calculates expected values with CPU Golden. NPU mode uses the output of separate operators as the reference: SwiGLU rounds to BF16 before activation, and the sigmoid sequence produces BF16 outputs after MatMul, sigmoid, and mul. The fused sigmoid path computes sigmoid and multiplication in FP32 and converts to BF16 only before quantization. Therefore, use NPU mode for performance comparison.

ST checks `y` and `y_scale` separately: shapes and dtypes must match; at least 99.9% of elements must have an error of no more than 1 ULP; NaN positions and Inf signs must match; positive and negative zero are treated as equal. FP4 elements are compared after unpacking. Kernel UT uses fixed inputs and requires exact agreement with expected results.
