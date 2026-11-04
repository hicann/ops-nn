# ScatterAddWithSorted Synchronization Test

English | [简体中文](./README.md)

## CPU Numerical Test

Run the following command in the repository root directory:

```bash
bash build.sh -u --opkernel --ops=scatter_add_with_sorted --soc=ascend910b -j8
```

The `ICPU_RUN_KF` test cases check values, index block refilling, and cross-core boundaries. CPU simulation does not model the real-device pipeline concurrency, so use the real-device test below for synchronization checking.

## Real-Device Repeated Verification

The test environment is Ascend 910B, PyTorch, and torch_npu. First build and load the operator package of this branch by following the [compilation and deployment instructions](../../../docs/zh/install/compile.md). The test invokes the CANN implementation loaded in the current environment.

```bash
bash build.sh --pkg --soc=ascend910b --ops=scatter_add_with_sorted -j8
python index/scatter_add_with_sorted/tests/st/test_scatter_add_with_sorted_sync.py --device 0 --iterations 10000 --index-refill
```

The script fixes the inputs and calls `scatter_add_` with row-broadcast indices of stride `(1, 0)`. Each invocation is compared bit-by-bit with the CPU reference. The inputs cover non-zero initial values, randomly repeated indices, all indices pointing to one row, and 800001 updates with index block refilling. When a difference occurs, the script immediately prints the test case, the invocation count, the position, and the two groups of values, and then exits.

Before running, use profiling to confirm that the invocation chain contains `LinearIndex → SortAiCore → ScatterAddWithSorted`, and confirm that the build artifacts compiled from this branch are loaded.

## Pipeline Race Check

Build the version instrumented with msSanitizer, and then load the operator package by following the compilation and deployment instructions:

```bash
bash build.sh --pkg --soc=ascend910b --ops=scatter_add_with_sorted --mssanitizer -j8
```

Save the short test command as `run_scatter_sync.sh`:

```bash
#!/bin/bash
set -eu
python index/scatter_add_with_sorted/tests/st/test_scatter_add_with_sorted_sync.py --device 0 --iterations 1 --index-refill
```

Run the following command in the repository root directory:

```bash
mssanitizer -t racecheck bash run_scatter_sync.sh
```

The log should show that racecheck is started for `ScatterAddWithSorted` and output `No error detected`. Record the counts of RAW (read after write), WAR (write after read), and WAW (write after write). `No active sanitizer` means that the race check is not executed.

The integer accumulation branch and the generic overwriting-write branch also need to be checked separately for tiling key 13 and tiling key 1. The inputs use repeated indices, and the index block length is smaller than the number of elements processed by each core, so that the same cache is refilled multiple times. The generic test cases make the repeated indices cross blocks and cross cores, covering the scan path that does not call `CopyLastUpdate`. Save the instrumentation logs, header files, and build artifact hashes of both the original version and the fixed version.
