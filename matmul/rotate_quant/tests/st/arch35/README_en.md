# RotateQuant Normal-Mode TTK Regression

English | [简体中文](./README.md)

The Ascend950 normal regression in this directory covers only direct ACLNN invocation, direct kernel invocation, and single-operator Torch eager mode.
groupList is absent by default, and GEIR or graph mode is not involved. Sharing one scale per 32 elements is the normal MX quantization rule,
which does not belong to the M-axis grouping feature.

## Test Assets

| File | Content | Case count |
| --- | --- | ---: |
| ttk_kernel_rotate_quant_st.csv | Normal RotateQuant kernel | 30 |
| ttk_aclnn_rotate_quant_st.csv | aclnnRotateQuant V1 | 33 |
| ttk_e2e_rotate_quant_st.csv | 43 cases of the operator repository interface and 43 cases of the original torch_npu interface | 86 |
| ttk_kernel_rotate_quant_dav_920_series_st.csv | Normal MX comparison of DAV_9201/DAV_9202, executable on Ascend950 | 12 |
| ttk_aclnn_rotate_quant_dav_920_series_st.csv | Corresponding ACLNN V1 comparison | 12 |

The three types of golden entries are located in ../../assets:

- rotate_quant_kernel_golden.py: rotation, clamp, and MX reference computation, and the Y/scale comparison functions.
- rotate_quant_aclnn_golden.py: V1 parameter adaptation.
- rotate_quant_e2e_golden.py: adapts torch_npu.npu_rotate_quant and
  torch.ops.cann_ops_nn.rotate_quant respectively by api_name. The two share the mathematical computation and handle the output storage representation separately.

TTK registers and executes entries by the api_name in the CSV and the __spec__ with the same name in the golden; no operator-specific framework branch is required.
The two sets of Torch cases use the same specifications and are compared with the golden separately; different case names do not guarantee automatic sharing of the same input.

## Coverage

- FLOAT16/BFLOAT16 inputs, and MXFP4-E2M1, MXFP8-E4M3FN, MXFP8-E5M2 outputs.
- K=32/64/128/256, including K=256 of FP8-E5M2; shared and per-N-segment rotation.
- Representative 1D, 2D, 3D, and 7D scenarios of x, covering high-dimension flattening, tail blocks, and scale padding.
- alpha absent by default, valid clipping, boundary values, and valid roundMode/scaleAlg/dstTypeMax branches.
- ACLNN/E2E include M=0, N=0, and non-contiguous inputs; direct kernel invocation is designed under the contiguous-input contract.
- Torch includes FP4 packing, FP8 dtype aliases, optional attributes of None, and non-contiguous views.

The DAV_9201/DAV_9202 comparison tables use K=32/64/128, cover 2D/3D rotation,
M=17 and M=32769, and check multi-round scheduling and tail blocks. This platform does not use K=256 or grouped cases.
HiFloat4 is out of the coverage of this MX golden.

## Execution

The Linux test machine needs the matching CANN, the operator package that contains the current operator, torch/torch_npu, and cann_ops_nn installed.
The golden depends on NumPy, ml_dtypes, en_dtypes (at least 0.0.4), and a matching Torch that provides
torch.float8_e8m0fnu and other types. The original-interface cases also require torch_npu to provide npu_rotate_quant.

Run in the root directory of ops-test-kit, and replace OPS_NN_PATH with the absolute path of the operator repository:

~~~bash
export OPS_NN_PATH=/path/to/ops-nn
python3 -m ttk e2e \
  -i "$OPS_NN_PATH/matmul/rotate_quant/tests/st/arch35/ttk_e2e_rotate_quant_st.csv" \
  --plugin "$OPS_NN_PATH/matmul/rotate_quant/tests/assets" \
  --backend npu --devices 0 --pc 1 --seed 42 --task-prof false \
  --dump-on-fail -o rotate_quant_e2e_result.csv
~~~

ACLNN/kernel use the corresponding subcommands and the CSV files in the same directory. When you verify only the new interface, select the rows of the corresponding api_name.
--plugin points to the test assets, and the code under test comes from the currently active operator package and Torch extension.
Do not enable GEIR, torch.compile, or aclgraph; --no-prof skips device execution and cannot replace
--task-prof false. You must check the accuracy status in the result CSV and cannot rely only on the process exit code.

## Output and Accuracy

- The kernel input slots are x, rotation, alpha, and group_list, and group_list uses None as a placeholder.
  The V1 ACLNN and the original Torch interface have no group_list parameter.
- For Torch FP4, Y is uint8[...,N/2], and the low 4 bits store the first logical element; for FP8, Y keeps the original shape.
- The Torch scale of the operator repository is uint8[...,ceil(N/64),2] and stores the raw E8M0 encoding;
  the original Torch scale is compared by E8M0 semantics and is also compatible with its uint8 storage carrier.
- Y is compared against adjacent representable values of the target format; scale is compared exactly by encoding, including the trailing padding of odd blocks.
  The second item of the first group of precision_tolerances controls only the allowed bad-point ratio of Y.
- The reference computation keeps the order of FP32 matrix multiplication, Fixpipe BF16 rounding, BF16 clamp/reciprocal multiplication,
  the final FP4/FP8 conversion, and the E8M0 encoding.

The reference implementation does not simulate the MMAD parallel accumulation order. Conventional CSVs use finite inputs;
NaN/Inf, overflow, and special-value conversion still need to be verified on A5.

The golden self-test is located in tests/ut/golden, and you run it in the root directory of the operator repository:

~~~bash
python3 -m unittest discover -s matmul/rotate_quant/tests/ut/golden -p 'test_rotate_quant*.py' -v
~~~

The grouped regression has moved to ../grouped. Execute it with the explicit commands in its README;
it is not collected by the normal ttk_*.csv scan of this directory.
