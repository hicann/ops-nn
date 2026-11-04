# RotateQuant Grouped Special Regression

English | [简体中文](./README.md)

This directory retains the existing 20 aclnnRotateQuantV2 grouped test cases and the corresponding golden files.
It is independent of the regular RotateQuant regression under arch35 and is not included in the default collection scope of the regular CSV.
It is run explicitly only in the Ascend950 environment and is not used for DAV_9201/DAV_9202.

Run the following command in the root directory of ops-test-kit:

~~~bash
export OPS_NN_PATH=/path/to/ops-nn
python3 -m ttk aclnn \
  -i "$OPS_NN_PATH/matmul/rotate_quant/tests/st/grouped/arch35/ttk_aclnn_rotate_quant_v2_st.csv" \
  --plugin "$OPS_NN_PATH/matmul/rotate_quant/tests/st/grouped/assets" \
  --pc 1 --seed 42 --run 1 --task-prof false
~~~

The special golden files reuse the regular reference computation in tests/assets, and retain the original handling of count/cumsum,
3D/4D rotation, empty groups, and uncovered output rows. The CPU self-test has been moved to tests/ut/golden.
