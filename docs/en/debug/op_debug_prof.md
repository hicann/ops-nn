# Operator Debugging and Tuning

## Debugging and Troubleshooting (AI Core Operators)

If an operator execution failure or accuracy anomaly occurs during operator execution, you can print information at each stage, such as Kernel intermediate results, for problem analysis and troubleshooting.

### 1. Host-Side Log Acquisition Method

* **plog acquisition**

   After program execution completes, you can view the logs by default in "$HOME/ascendc/log". The host log file storage path is as follows:

   ```bash
   $HOME/ascend/log/debug/plog/plog-pid_*.log
   ```

   Enable the environment variable ASCEND_SLOG_PRINT_TO_STDOUT to display log output directly on the screen (1: enable screen display, 0: disable screen display). The configuration example is as follows:

   ```bash
   export ASCEND_SLOG_PRINT_TO_STDOUT=1
   ```

   For log-related information, refer to [Log Reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/logreference/logreference_0001.html). For environment variable information, refer to [Environment Variable Reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0001.html).

* **aclnn exception error message acquisition**

   Obtain exception information during aclnn interface invocation through the aclGetRecentErrMsg interface (refer to [acl API (C)](https://www.hiascend.com/document/detail/en/canncommercial/latest/API/appdevgapi/aclcppdevg_03_0004.html)). The usage method is as follows:

   ```bash
   printf(aclGetRecentErrMsg());
   ```

   The printed error message example is as follows:

   ```bash
   [PID:646612] 2026-01-24-11:53:44.671.727 AclNN_Parameter_Error(EZ1001): Expected a proper Tensor but got null for argument addmmTensor.self.
   ```

### 2. Kernel Debugging

Common debugging methods are as follows:

* **printf**

  This interface supports printing Scalar-type data, such as integers, characters, and Boolean values. For detailed information, refer to [Ascend C API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0003.html) in "Operator Debugging API > printf".

  ```c++
  blockLength_ = tilingData->totalLength / AscendC::GetBlockNum();
  tileNum_ = tilingData->tileNum;
  tileLength_ = blockLength_ / tileNum_ / BUFFER_NUM;
  // Print the current core computation Block length
  AscendC::PRINTF("Tiling blockLength is %llu\n", blockLength_);
  ```

* **DumpTensor**

  This interface supports dumping the content of a specified Tensor and also supports printing custom additional information, such as the current line number. For detailed information, refer to [Ascend C API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0003.html) in "Operator Debugging API > DumpTensor".

  ```c++
  AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
  // Print zLocal Tensor information
  DumpTensor(zLocal, 0, 128);
  AscendC::DataCopy(outputGMZ[progress * tileLength_], zLocal, tileLength_);
  ```

For troubleshooting in complex scenarios, such as operator hangs or GM/UB access out-of-bounds, you can use **step-by-step debugging**. For specific operations, refer to the [msDebug](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/optool/docs/en/quick_start/msdebug_quick_start.md) operator debugging tool.

## Debugging and Troubleshooting (AI CPU Operators)

If an operator execution failure or accuracy anomaly occurs during operator execution, you can print information at each stage, such as Kernel intermediate results, for problem analysis and troubleshooting.

### 1. Host-Side Log Acquisition Method

   Refer to the AI Core operator [Host-Side Log Acquisition Method](#1-host-side-log-acquisition-method)

### 2. Kernel Debugging

Common debugging methods are as follows:

* **KERNEL_LOG macro**

  You can print log information during operator execution through the following macros, including DEBUG, INFO, WARN, and ERROR level logs.

  ```Cpp
  KERNEL_LOG_DEBUG(fmt, ...)      // The fmt parameter represents the format control string
  KERNEL_LOG_INFO(fmt, ...)
  KERNEL_LOG_WARN(fmt, ...)
  KERNEL_LOG_ERROR(fmt, ...)      // ERROR level logs are printed by default
  ```

  To print logs at non-ERROR levels, you need to configure the environment variable `ASCEND_GLOBAL_LOG_LEVEL` in advance. For specific usage, refer to [Environment Variable Reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0001.html).

  The printing example is as follows:

  ```c++
  Tensor* input0 = ctx.Input(kFirstInputIndex);
  Tensor* input1 = ctx.Input(kSecondInputIndex);
  Tensor* output = ctx.Output(0);

  if (input0 == nullptr || input1 == nullptr || output == nullptr) {
    // Print error information
    KERNEL_LOG_ERROR("Invalid argument");
    return kParamInvalid;
  }

  int64_t num_elements = input0->NumElements();
  // Print the number of input elements
  KERNEL_LOG_INFO("Num of elements is %ld", data_size);
  ```

## Performance Tuning

### Method 1 (For Atlas A2/A3 Series Products)

If execution accuracy degradation or abnormal memory usage occurs during operator execution, you can use the [msProf](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/optool/docs/en/quick_start/msopprof_quick_start.md) performance analysis tool to analyze the operator's performance metrics at each execution stage (such as throughput, memory usage, and latency), thereby identifying the root cause and performing targeted optimization.

This chapter uses the [AddExample custom operator](../../../examples/add_example/) as an example to introduce the two commonly used methods in operator tuning: on-board performance collection and pipeline simulation. By collecting the on-board running pipeline metrics of the operator, you can analyze the operator's Bound scenario. Understanding the simulation pipeline diagram helps optimize the operator's internal pipeline.

1. Prerequisites.

   After completing operator development and compilation, assuming the aclnn interface invocation method is used, the generated operator executable file (test_aclnn_add_example) is located in the `examples/add_example/examples/build/bin/` directory of this project.

2. Collect performance data.

   When you need to collect the on-board running pipeline metrics of the operator, navigate to the directory where the operator executable file is located and execute the following command:

   ```bash
   msprof op ./test_aclnn_add_example
   ```

   The collection results are in the `examples/add_example/examples/build/bin/OPPROF_*` directory of this project. After collection completes, the following information is printed:

    ``` text
    Op Name: AddExample_a1532827238e1555db7b997c7bce2928_high_performance_1
    Op Type: vector
    Task Duration(us): 97.861954
    Block Dim: 8
    Mix Block Dim:
    Device Id: 0
    Pid: 2776181
    Current Freq: 1800
    Rated Freq: 1800
    ```

   Task Duration is the current operator Kernel execution time, and Block Dim is the current operator execution core count.

   For detailed pipeline metrics of the operator, refer to the `ArithmeticUtilization` file under `OPPROF_*`, which contains the proportion of each pipeline. For specific descriptions, refer to the [msProf](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/optool/docs/en/quick_start/msopprof_quick_start.md) section "Performance Data Files > msprof op > ArithmeticUtilization (cube and vector type instruction latency and proportion)".

3. Collect simulation pipeline diagrams.

   Before using the msProf tool for operator simulation tuning, execute the following command to configure the environment variable.

   ```bash
   export LD_LIBRARY_PATH=${INSTALL_DIR}/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH
   ```

   Modify the above environment variable according to the actual CANN software package installation path and AI processor model.

   Then navigate to the directory where the operator executable file is located and execute the following command:

   ```bash
   msprof op simulator --output=$PWD/pipeline_auto --kernel-name"AddExample" ./test_aclnn_add_example
   ```

   The collection results are in the `$PWD/pipeline_auto/OPPROF_**` directory of this project.
   The pipeline-related file path is `OPPROF**/simulator/visualize_data.bin`, which can be viewed using the [mindStudio Insight](https://www.hiascend.com/document/detail/en/mindstudio/latest/visualization_tool/MindStudioInsight/docs/en/user_guide/overview.md) tool.

### Method 2 (For Ascend 950PR)

If execution accuracy degradation or abnormal memory usage occurs during operator development, you can use the [CANN Simulator](./cann_simulator.md) simulation tool to analyze the operator's instruction pipeline situation, thereby identifying the root cause and performing targeted optimization.

This chapter uses the [AddExample custom operator](../../../examples/add_example/) as an example to introduce the use of the simulation tool. It describes how to perform accuracy and performance tuning through the simulation tool.

1. Prerequisites.

   After completing operator development and compilation, assuming the aclnn interface invocation method is used, the generated operator executable file (test_aclnn_add_example) is located in the `examples/add_example/examples/build/bin/` directory of this project.

2. Execute the simulation command to generate simulation data.

   ```text
   cannsim record ./test_aclnn_add_example -s Ascend950 --gen-report
   ```

   The simulation results are in the `examples/add_example/examples/build/bin/cannsim_*` directory of this project. The pipeline-related file is:

   ```text
   trace_core0.json
   ```

3. Enter "chrome://tracing" in the Chrome browser and drag the generated instruction pipeline diagram file (trace_core0.json) to the blank area to open it. For specific parameter descriptions, refer to the [Simulation Result Analysis](./cann_simulator.md#simulation-result-analysis-instructions) section in CANN Simulator.
