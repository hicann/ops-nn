# EmbeddingHashTableExport ST (System Test)

Device-level regression cases for the export clamping behavior (`export_output_st.py`), covering:

- **Valid prefix**: the table is non-empty (lookup inserts N keys) and `table_sizes` is greater than N; assert that the output prefix is exactly the set of inserted keys (no missing, no extra)
- **Truncation**: `table_sizes` is smaller than the actual entry count; assert that only `table_sizes` entries are exported and the process survives

Requirements: an Ascend 950PR device, the CANN environment (`ge`/`es_nn` Python packages), and a user with device access.

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 tests/st/export_output_st.py
```

Note: the clamping logic lives in the kernel, which op_host UT does not cover; these cases are intended for device-environment regression runs.
