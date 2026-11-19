# EmbeddingHashTableExport ST（系统测试）

针对导出钳制行为的设备级回归用例（`export_output_st.py`），覆盖：

- **有效前缀**：表非空（lookup 插入 N 个 key）且 `table_sizes` 大于 N，断言输出前缀恰好为插入的 key 集合（无缺失、无多余）
- **截断**：`table_sizes` 小于实际条目数，断言只导出前 `table_sizes` 条且进程存活

运行依赖：Ascend 950PR 设备、CANN 环境（`ge`/`es_nn` Python 包）、root 或设备可用用户。

```sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 tests/st/export_output_st.py
```

说明：钳制逻辑位于 kernel，op_host UT 不覆盖执行路径，本用例供设备环境回归。
