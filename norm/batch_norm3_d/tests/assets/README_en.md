# BatchNorm3D Regression Tests

English | [简体中文](./README.md)

These assets are used for regression tests on <term>Ascend 950PR&950DT products</term>, using the `golden.py` of this operator.

| Asset | Purpose |
| --- | --- |
| [ttk_kernel_batch_norm3_d.csv](../st/arch35/ttk_kernel_batch_norm3_d.csv) | 18 small-shape cases of regular ST. Shared by kernel and GEIR. The total size of all inputs of a single case is less than 50KiB. |
| `generate_nonfinite_cases.py` | Generates 36 non-finite fixed-input cases on demand for the special diagnosis of AR, RA, and split-core reduction. They are not included in regular ST. |
| `generate_precision_cases.py` | Generates 50 fixed-input cases on demand for the special diagnosis of near-zero cancellation, low bits of the mean, extreme values, and the five inference paths. They are not included in regular ST. |

The archived CSV covers all 8 combinations of FP16/FP32, NCDHW/NDHWC, and training/inference, and includes omitted optional inputs and attributes, M=1, epsilon of 0, reduction and channel tails, the 4096 reduction blocking boundary, extreme-value constants, large variance, large scale, near-zero cancellation, and Inf propagation. Invalid format rejection, memory instrumentation, and the large-shape path require separate verification and are not counted in this positive set.

The two special generation scripts accept `--output` to specify the output directory, and generate fixed inputs and `cases.csv`. Regular ST does not need to run them. The special sets are large in scale. Before use, you need to separately evaluate the memory and duration.

## Preparation

The source code and `ops-test-kit` are in the same task directory. First, follow the [Compilation and Installation Guide](../../../../docs/zh/install/compile.md) to load the target CANN and the operator package to be tested. The effective parallelism of the build must not exceed 8, and TTK execution always explicitly uses `--pc=1`.

```bash
export BN3D_TASK_ROOT=/path/to/task
export BN3D_ASSETS="$BN3D_TASK_ROOT/ops-nn/norm/batch_norm3_d/tests/assets"
export BN3D_CASES="$BN3D_TASK_ROOT/ops-nn/norm/batch_norm3_d/tests/st/arch35/ttk_kernel_batch_norm3_d.csv"
export BN3D_RESULTS="$BN3D_TASK_ROOT/reports/batch_norm3d_st"
mkdir -p "$BN3D_RESULTS"
cd "$BN3D_TASK_ROOT/ops-test-kit"
export TILINGKEY_PAR_COMPILE=1 CMAKE_BUILD_PARALLEL_LEVEL=8 MAX_JOBS=8
```

In the CSV, `op_name=batch_norm3_d` corresponds to the installation entry and the golden registration key. `BatchNorm3D` is the GE operator type name.

## Execution

The following commands compare the NPU output with the CPU Golden using the two-sided `stat_rel_err`, covering all five outputs. The plugin declares `cross_check` by default, and the comparison method in the command line must be kept. The Golden uses float64 intermediate computation. epsilon is first converted to the float32 attribute value declared by the interface and then promoted to float64 to participate in the computation.

```bash
python3 -m ttk kernel --pc=1 --devices=0 --no-memory-check -d=false -b=release \
  --run 1 --plugin "$BN3D_ASSETS/golden.py" --compare stat_rel_err \
  -i "$BN3D_CASES" -o "$BN3D_RESULTS/kernel.csv"

python3 -m ttk geir --pc=1 --devices=0 --no-memory-check \
  --run 1 --plugin "$BN3D_ASSETS/golden.py" --compare stat_rel_err \
  -i "$BN3D_CASES" -o "$BN3D_RESULTS/geir.csv"
```

Precompiled package verification checks `bin_compile_s=BINARY_MATCH`, `bin_precision`, and non-empty precision metrics. GEIR checks `cst_compile_s`, `cst_precision`, and the actual output comparison. `COMPILE_ONLY`, `SUPPRESSED`, or only the process exit code cannot serve as the basis for numerical passing.

For cross-version comparison, save the inputs and golden first, and then replay the same data through `--manual-data-dirs`. `--no-prof` is only used for preparing frozen data and must be removed for acceptance execution. The UB sub-buffer safety of full_reduce still requires an instrumented JIT kernel and mssanitizer verification, and an ordinary GM sentinel check cannot replace them.
