# ScatterAddWithSorted 同步测试

## CPU 数值测试

在仓库根目录执行：

```bash
bash build.sh -u --opkernel --ops=scatter_add_with_sorted --soc=ascend910b -j8
```

`ICPU_RUN_KF` 用例检查数值、索引换块和跨核边界。CPU 仿真不模拟真机流水并发，同步检查使用下面的真机测试。

## 真机重复校验

测试环境为 Ascend 910B、PyTorch 和 torch_npu。先按[编译部署说明](../../../docs/zh/install/compile.md)构建并加载本分支的算子包。测试调用当前环境加载的 CANN 实现。

```bash
bash build.sh --pkg --soc=ascend910b --ops=scatter_add_with_sorted -j8
python index/scatter_add_with_sorted/tests/st/test_scatter_add_with_sorted_sync.py --device 0 --iterations 10000 --index-refill
```

脚本固定输入，使用 stride 为 `(1, 0)` 的行广播索引调用 `scatter_add_`。每次调用都与 CPU 参考逐位比较。输入覆盖非零初值、随机重复索引、全部索引指向一行，以及 800001 条更新的索引换块。出现差异时，脚本立即输出用例、调用次数、位置和两组数值并退出。

运行前用 profiling 确认调用链包含 `LinearIndex → SortAiCore → ScatterAddWithSorted`，并确认加载的是本分支编译产物。

## 流水竞争检查

构建带 msSanitizer 插桩的版本，再按编译部署说明加载该算子包：

```bash
bash build.sh --pkg --soc=ascend910b --ops=scatter_add_with_sorted --mssanitizer -j8
```

把短测试命令保存为 `run_scatter_sync.sh`：

```bash
#!/bin/bash
set -eu
python index/scatter_add_with_sorted/tests/st/test_scatter_add_with_sorted_sync.py --device 0 --iterations 1 --index-refill
```

在仓库根目录运行：

```bash
mssanitizer -t racecheck bash run_scatter_sync.sh
```

日志应显示对 `ScatterAddWithSorted` 启动 racecheck，并输出 `No error detected`。记录 RAW（写后读）、WAR（读后写）、WAW（写后写）的数量。`No active sanitizer` 表示没有执行竞争检查。

整数累加和 generic 覆盖写入分支还需分别检查 tiling key 13 和 1。输入使用重复索引，索引块长度小于每核处理数量，使同一缓存多次被重填。generic 用例让重复索引跨块和跨核，覆盖没有调用 `CopyLastUpdate` 的扫描路径。保存原始版本与修复版本的插桩日志、头文件和编译产物哈希。
