# ConfidenceHeadFused Tests

English | [简体中文](./README.md)

## UT

| Directory | Number of test cases | Coverage |
| --- | --- | --- |
| [ut/op_api](ut/op_api/test_aclnn_confidence_head_fused.cpp) | 16 | FP16/BF16, Ascend950, optional inputs, zero K, empty outputs, as well as pointer, type, dimension, and shape validation. |
| [ut/op_host](ut/op_host/test_confidence_head_fused_tiling.cpp) | 12 | Two tiling keys, 2D/3D, on-chip capacity boundaries, AIC/AIV ratio, flatten overflow, and empty outputs. |

Run the following commands in the repository root directory:

```bash
bash build.sh --ophost --ops=confidence_head_fused -u
bash build.sh --opapi --ops=confidence_head_fused -u
```

## ST

The test cases follow the repository TTK CSV format, and the corresponding test entry loads the case tables under [st/arch35](st/arch35).

| Path | Number of test cases | Golden |
| --- | --- | --- |
| ACLNN | 16 | [confidence_head_fused_aclnn_golden.py](assets/confidence_head_fused_aclnn_golden.py) |
| Torch | 15 | [confidence_head_fused_e2e_golden.py](assets/confidence_head_fused_e2e_golden.py) |
| Kernel | 13 | [confidence_head_fused_kernel_golden.py](assets/confidence_head_fused_kernel_golden.py) |

The three paths retain 44 representative test cases in total, covering FP16/BF16, 2D/3D, N/K tail blocks, hidden/markov multi-round K, batch bias, markov-only, zero K, zero and negative temperature values, one-sided and two-sided truncation, and empty outputs. ACLNN and Torch additionally cover non-contiguous inputs, and ACLNN additionally covers non-contiguous outputs.

The common computation logic is located in [Kernel Golden](assets/confidence_head_fused_kernel_golden.py). The ACLNN and Torch Golden files reuse this logic: they first generate data in the input precision, and then compute the projection and post-processing in FP32. The bias regression test cases for multi-round K use zero features and a non-zero bias to check whether the bias is omitted or added repeatedly.

After temperature scaling and clamp, the device uses the default inexact Div; the existing Sigmoid of FusedMatMul uses the exact Div, and the two are not required to be bit-by-bit consistent. All three ST paths use `precision_tolerances=((0.001, 0.001),)` from the CSV, which represent `rtol=1e-3` and the allowed mismatch ratio `ptol=1e-3` respectively, and `absolute_precision=1e-4` in regular test cases represents `atol=1e-4`. Elements are compared one by one according to `abs(actual-golden) <= atol + rtol*abs(golden)`, and then the pass ratio is determined by `ptol`. The `high_only_large_exponent` test case retains `atol=1e-30` to check the execution order of the one-sided clamp.
