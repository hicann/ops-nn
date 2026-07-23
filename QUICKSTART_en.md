# Quick Start: Based on ops-nn Repository

## Usage Notice

This guide aims to help you quickly get started with CANN and the `ops-nn` operator repository, providing simplified software installation and compilation and running guidance **based on WebIDE or Docker environment**. Note that WebIDE or Docker environment provides the **latest commercial release version of CANN software package** by default, which is currently CANN 8.5.0.

> **Note**: If you are manually installing the CANN package or experiencing the latest capabilities of the master branch in other scenarios, you can skip the quick start and refer to the guides below for detailed tutorials. These tutorials provide rich operational methods in different scenarios to meet diverse business requirements.
>
> 1. [Environment Deployment Guide](./docs/en/context/quick_install.md): Environment setup guidance in different scenarios, including Docker installation, manual CANN software package installation, and so on.
> 2. [Compile and Execute Operator Guide](./docs/en/invocation/quick_op_invocation.md): Operator package compilation and verification guidance in different scenarios, such as offline compilation, in-depth understanding of compilation parameters and invocation methods.
> 3. [Operator Development Guide](./docs/en/develop/aicore_develop_guide.md): Guide for custom development of standard operators, learning to create operator projects from scratch and implement Tiling and Kernel.
> 4. [Debugging and Tuning Guide](./docs/en/debug/op_debug_prof.md): Systematic debugging techniques and performance optimization methods in different scenarios.

The basic process of operator development and contribution is shown in the figure below. We welcome and encourage you to contribute operators in the community to jointly enrich the project ecosystem.

<!--![Operator Development Contribution Process](./docs/en/figures/operator_development_contribution_process.png "Operator Development Contribution Process Diagram")-->

To help you quickly understand the entire process of operator development, we will use the **AddExample** operator as a practical object. Its source files are located in `ops-nn/examples/add_example`. The specific operation steps are as follows:

1. **[Environment Installation](#i-environment-installation-choose-one-of-two)**: Set up the operator development and running environment.
2. **[Compilation and Deployment](#ii-compilation-and-deployment)**: Compile the custom operator package and deploy the installation to achieve quick operator invocation.
3. **[Operator Development](#iii-operator-development)**: Experience the complete loop of development, compilation, and verification by modifying the existing operator Kernel.
4. **[Operator Debugging](#iv-operator-debugging)**: Master the methods of operator printing and performance collection.
5. **[Operator Verification](#v-operator-verification)**: Learn how to modify operator example samples to verify the functional correctness of operators under different inputs.

## I. Environment Installation (Choose One of Two)

### 1. No Environment Scenario: WebIDE Development

For users without an environment, you can directly use the WebIDE development platform, that is, the "**Operator One-Stop Development Platform**". This platform provides you with an online directly runnable Ascend environment, where necessary software packages have been installed, without manual installation. For more introduction about the development platform, refer to [LINK](https://gitcode.com/org/cann/discussions/54).

1. Enter the ops-nn open-source project and click the "`Cloud Development`" button. Log in with a certified Huawei Cloud account. If you have not registered or certified, please register and certify according to the page prompts.

   <!--<img src="docs/en/figures/cloudIDE.png" alt="Cloud Platform"  width="750px" height="90px">-->

2. Create and start the cloud development environment according to the page prompts, and click "`Connect > WebIDE`" to enter the operator one-stop development platform. The resources of the open-source project are in the `/mnt/workspace` directory by default.

    <!--<img src="docs/en/figures/webIDE.png" alt="Cloud Platform"  width="1000px" height="150px">-->

3. Check whether the environment is complete.

    In the cloud platform terminal window, execute the following commands to verify whether the environment and driver are normal.

    - **Check NPU Device**

        Execute the following command. If driver-related information is returned, it means that the device has been successfully mounted.

        ```bash
        npu-smi info
        ```

    - **Check CANN Version**

        Execute the following command to view the CANN Toolkit version information.

        ```bash
        cat /home/developer/Ascend/ascend-toolkit/latest/opp/version.info
        ```

### 2. Existing Environment Scenario: Docker Installation

#### Prerequisites

* **Docker Environment**: Taking Atlas A2 product (910B) as an example, the Docker engine (version 1.11.2 or above) has been installed on the host machine in the environment.

* **Driver and Firmware**: The host machine has installed Ascend NPU [driver and firmware](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.0.RC3.alpha002&driver=1.0.26.alpha) Ascend HDK version 24.1.0 or above. For installation instructions, see the "[CANN Software Installation Guide](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/softwareinst/instg/instg_0107.html)".

    > **Note**: Use `npu-smi info` to view the corresponding driver and firmware version.

#### Download Image

Pull the image that has pre-integrated the CANN software package and `ops-nn` required dependencies.

1. Log in to the host machine as the root user.
2. Execute the pull command (select according to your host machine architecture):

    * ARM architecture:

        ```bash
        docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
        ```

    * X86 architecture:

        ```bash
        docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
        ```

> **Note**: Under normal network speed, the image download time is about 5-10 minutes.

#### Docker Run

Run docker according to the following command:

```bash
docker run --name cann_container --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops bash
```

The following are parameter descriptions that users need to pay attention to:

| Parameter | Description | Precautions |
| :--- | :--- | :--- |
| `--name cann_container` | Specify a name for the container for management. | Can be customized. |
| `--device /dev/davinci0` | Core: Map the host machine's NPU device card to the container. Multiple NPU device cards can be specified for mapping. | Must be adjusted according to the actual situation: `davinci0` corresponds to the 0th NPU card in the system. Please execute the `npu-smi info` command on the host machine first, and modify this number according to the device number displayed in the output (such as `NPU 0`, `NPU 1`).|
| `-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/` | Key mount: Map the host machine's NPU driver library to the container. | - |

#### Check Environment

After entering the container, verify whether the environment and driver are normal.

- **Check NPU Device**

    Execute the following command. If driver-related information is returned, it means that the device has been successfully mounted.

    ```bash
    npu-smi info
    ```

- **Check CANN Version**

    Execute the following command to view the CANN Toolkit version information.

    ```bash
    cat /usr/local/Ascend/ascend-toolkit/latest/opp/version.info
    ```

You now have an "out-of-the-box" operator development environment. Next, you need to verify the complete toolchain from source code to runnable operators in this environment.

## II. Compilation and Deployment

The purpose of this stage is to **quickly experience the project standard process** and verify whether the environment can successfully perform operator source code compilation, packaging, installation, and running.

### 1. Obtain Project Source Code

1. Obtain the project source code.

    Docker or WebIDE environment provides the latest commercial release version source code by default. If you need to obtain other version source code, you can download through the following command. ${tag_version} needs to be replaced with the target branch tag name. For the correspondence between branch tags and CANN versions, see the [release repository](https://gitcode.com/cann/release-management).

    ```bash
    git clone -b ${tag_version} https://gitcode.com/cann/ops-nn.git
    ```

    If "`fatal: destination path 'ops-nn' already exists and is not an empty directory.`" appears, it means that the project source code already exists. If you need to refresh the project code, you can use the `git pull` command.

2. Enter the project root directory. The command is as follows. Please distinguish between Docker and WebIDE scenarios.
    - Docker scenario:

      ```bash
      cd ops-nn
      ```

    - WebIDE scenario:

      ```bash
      cd /mnt/workspace/ops-nn
      ```

### 2. Compile AddExample Operator

Enter the project root directory and compile the specified operator. The general compilation command format: `bash build.sh --pkg --soc=<chip version> --ops=<operator name>`.

Taking the AddExample operator as an example, the compilation command is as follows:

```bash
bash build.sh --pkg --soc=ascend910b --ops=add_example -j16
```

If the following information is prompted, the compilation is successful.

```bash
Self-extractable archive "cann-ops-nn-custom-linux.${arch}.run" successfully created.
```

After successful compilation, the run package is stored in the build_out directory under the project root directory.

### 3. Install AddExample Operator Package

```bash
./build_out/cann-ops-nn-*linux*.run
```

`AddExample` is installed in the ```${ASCEND_HOME_PATH}/opp/vendors``` path. ```${ASCEND_HOME_PATH}``` indicates the CANN software installation directory.

### 4. Configure Environment Variables

Add the path of the custom operator package to the environment variables to ensure that it can be found at runtime.

```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_nn/op_api/lib:${LD_LIBRARY_PATH}
```

### 5. Quick Verification: Run Operator Sample

The general running command format: `bash build.sh --run_example <operator name> <running mode> <package mode>`.

Taking AddExample as an example, it provides a simple operator sample `add_example/examples/test_aclnn_add_example.cpp`. Run this sample to verify whether the operator function is normal.

```bash
bash build.sh --run_example add_example eager cust --vendor_name=custom
```

Expected output: Print the addition calculation result of the operator `AddExample`, indicating that the operator has been successfully deployed and executed correctly.

```bash
add_example first input[0] is: 1.000000, second input[0] is: 1.000000, result[0] is: 2.000000
add_example first input[1] is: 1.000000, second input[1] is: 1.000000, result[1] is: 2.000000
add_example first input[2] is: 1.000000, second input[2] is: 1.000000, result[2] is: 2.000000
add_example first input[3] is: 1.000000, second input[3] is: 1.000000, result[3] is: 2.000000
add_example first input[4] is: 1.000000, second input[4] is: 1.000000, result[4] is: 2.000000
add_example first input[5] is: 1.000000, second input[5] is: 1.000000, result[5] is: 2.000000
add_example first input[6] is: 1.000000, second input[6] is: 1.000000, result[6] is: 2.000000
add_example first input[7] is: 1.000000, second input[7] is: 1.000000, result[7] is: 2.000000
...
```

## III. Operator Development

The purpose of this stage is to try **modifying the kernel function code** for the successfully running AddExample operator.

### 1. Modify Kernel Implementation

Find the core kernel implementation file of the AddExample operator `ops-nn/examples/add_example/op_kernel/add_example.h`, and try to change the Add operation in the operator to a Mul operation:

```cpp
__aicore__ inline void AddExample<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
    // === Replace Add with Mul here ===
    // AscendC::Add(zLocal, xLocal, yLocal, tileLength_);
    AscendC::Mul(zLocal, xLocal, yLocal, tileLength_);
    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
}
```

### 2. Compile and Verify

Repeat steps 2 to 5 in the [Compilation and Deployment](#ii-compilation-and-deployment) section:

1. **Recompile**:
    First return to the project root directory. The compilation command is as follows:

    ```bash
    bash build.sh --pkg --soc=ascend910b --ops=add_example -j16
    ```

2. **Reinstall**:

    ```bash
    ./build_out/cann-ops-nn-*linux*.run
    ```

3. **Re-verify**:

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom
    ```

4. **Success Sign**: The output result becomes the multiplication result.

    ```bash
    add_example first input[0] is: 1.000000, second input[0] is: 1.000000, result[0] is: 1.000000
    add_example first input[1] is: 1.000000, second input[1] is: 1.000000, result[1] is: 1.000000
    add_example first input[2] is: 1.000000, second input[2] is: 1.000000, result[2] is: 1.000000
    add_example first input[3] is: 1.000000, second input[3] is: 1.000000, result[3] is: 1.000000
    add_example first input[4] is: 1.000000, second input[4] is: 1.000000, result[4] is: 1.000000
    add_example first input[5] is: 1.000000, second input[5] is: 1.000000, result[5] is: 1.000000
    add_example first input[6] is: 1.000000, second input[6] is: 1.000000, result[6] is: 1.000000
    add_example first input[7] is: 1.000000, second input[7] is: 1.000000, result[7] is: 1.000000
    ...
    ```

## IV. Operator Debugging

This stage takes AddExample as an example to add printing in the operator and collect operator performance data for subsequent problem analysis and positioning.

### 1. Printing

If the operator has execution failure, precision abnormality, or other problems, add printing for problem analysis and positioning.

Please modify the code in `examples/add_example/op_kernel/add_example.h`.

* **printf**

  This interface supports printing Scalar type data, such as integers, character type, Boolean type, and so on. For detailed introduction, see "Operator Debugging API > printf" in "[Ascend C API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0003.html)".

  ```c++
  blockLength_ = (tilingData->totalLength + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
  tileNum_ = tilingData->tileNum;
  tileLength_ = ((blockLength_ + tileNum_ - 1) / tileNum_ / BUFFER_NUM) ?
        ((blockLength_ + tileNum_ - 1) / tileNum_ / BUFFER_NUM) : 1;
  // Print the current kernel calculation Block length
  AscendC::PRINTF("Tiling blockLength is %llu\n", blockLength_);
  ```

* **DumpTensor**

  This interface supports dumping the content of the specified Tensor, and also supports printing custom additional information, such as the current line number. For detailed introduction, see "Operator Debugging API > DumpTensor" in "[Ascend C API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0003.html)".

  ```c++
  AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
  // Print zLocal Tensor information
  DumpTensor(zLocal, 0, 128);
  ```

### 2. Performance Collection

When the operator function verification is correct, you can collect operator performance data through the `msprof` tool.

- **Generate Executable File**

    Call the example sample of the AddExample operator to generate an executable file (test_aclnn_add_example), which is located in the project `ops-nn/build` directory.

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom
    ```

- **Collect Performance Data**

    Enter the AddExample operator executable file directory `ops-nn/build/` and execute the following command:

    ```bash
    msprof --application="./test_aclnn_add_example"
    ```

The collection result is in the project `ops-nn/build/` directory. After the msprof command is executed, it will automatically parse and export the performance data result file. For detailed content, see [msprof](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/T&ITools/Profiling/atlasprofiling_16_0110.html#ZH-CN_TOPIC_0000002504160251).

## V. Operator Verification

This stage verifies the functional correctness of the operator in multiple scenarios by modifying the input data of the AddExample operator example sample.

### 1. Modify Test Input

Find and edit the `ops-nn/examples/add_example/examples/test_aclnn_add_example.cpp` of `AddExample`, and modify the shape and numerical values of the input tensor.

**Modify Input/Output Data**: Modify the shape information of input and output, as well as the initialization data, and construct the corresponding input and output tensors.

```c++
int main() {
    // ... initialization code ...

    // === ① Modify selfX input ===
    // Before modification: shape = {32, 4, 4, 4}, all values are 1
    // After modification: change input shape to {8, 8, 8, 8}, and fill with different test data
    std::vector<int64_t> selfXShape = {8, 8, 8, 8};
    std::vector<float> selfXHostData(4096); // 4096 = 8 * 8 * 8 *8
    // You can use a loop to fill more distinguishable data, such as an increasing sequence
    for (int i = 0; i < 4096; ++i) {
        selfXHostData[i] = static_cast<float>(i % 10); // Fill with cyclic values of 0-9
    }
    // === ② Refer to selfX, similarly modify selfY and selfZ inputs ===

    // ... subsequent execution code ...
}
```

### 2. Recompile and Verify

1. Since only the example test code is modified, there is no need to recompile the operator package.

2. Re-execute the verification command:

    ```bash
    bash build.sh --run_example add_example eager cust --vendor_name=custom
    ```

3. Observe whether the operator output result meets expectations.

## VI. Development Contribution

After experiencing the above operations, you have basically completed an operator development. You can contribute the operator to the `experimental` directory of this project. For the contribution process, refer to the [Contribution Guide](CONTRIBUTING_en.md). During the process, any questions can be consulted through the Issue method.
