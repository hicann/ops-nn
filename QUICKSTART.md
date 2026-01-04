# 算子开发快速入门：基于ops-nn仓
算子开发贡献流程如下：
![算子开发贡献流程](./docs/zh/figures/算子开发贡献流程.png "算子开发贡献流程图")

本指南旨在帮助你快速上手基于CANN和`ops-nn`算子仓的使用，最简化地完成环境安装、编译部署及算子运行。

## 目录导读
1.  **[环境安装](#一环境安装)**：通过Docker快速搭建算子开发环境。
2.  **[编译部署](#二编译部署)**：编译自定义算子包并进行安装，快速调用算子样例验证。
3.  **[算子开发](#三算子开发)**：通过修改现有算子Kernel，体验开发、编译、验证的完整闭环。
4.  **[算子调试能力](#四算子调试能力)**：掌握算子打印和性能采集方法。
5.  **[算子验证](#五算子验证)**：学习如何修改样例，以验证算子在不同输入下的功能正确性。

完成以上步骤，你将对算子开发的全流程有一个基础实践认知。

## 一、环境安装
### 1. 有环境场景：Docker安装

Docker安装环境以Atlas A2产品（910B）为例。
**前提条件**：
*   **Docker环境**：宿主机已安装Docker引擎（版本1.11.2及以上）。
*   **驱动与固件**：宿主机已安装昇腾NPU的[驱动与固件](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.0.RC3.alpha002&driver=1.0.26.alpha)Ascend HDK 24.1.0版本以上。安装指导详见《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_0005.html?Mode=PmIns&OS=openEuler&Software=cannToolKit)》。
    > **注意**：使用`npu-smi info`查看对应的驱动与固件版本。

#### 下载镜像
拉取已预集成CANN软件包及`ops-nn`所需依赖的镜像。

*   **操作步骤**：
    1.  以root用户登录宿主机。
    2.  执行拉取命令（请根据你的宿主机架构选择）：
        * ARM架构：
        ```bash
        docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
        ```
        * X86架构：
        ```bash
        docker pull --platform=amd64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops
        ```
        > **注意**：正常网速下，镜像下载时间约为5-10分钟。

#### Docker运行
请根据以下命令运行docker：

```bash
docker run --name cann_container --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-ubuntu22.04-py3.10-ops bash
```
以下为用户需关注的参数说明：
| 参数 | 说明 | 注意事项 |
| :--- | :--- | :--- |
| `--name cann_container` | 为容器指定名称，便于管理。 | 可自定义。 |
| `--device /dev/davinci0` | 核心：将宿主机的NPU设备卡映射到容器内，可指定映射多张NPU设备卡。 | 必须根据实际情况调整：`davinci0`对应系统中的第0张NPU卡。请先在宿主机执行 `npu-smi info`命令，根据输出显示的设备号（如`NPU 0`, `NPU 1`）来修改此编号。|
| `-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/` | 关键挂载：将宿主机的NPU驱动库映射到容器内。 | |

#### 检查环境
进入容器后，验证环境和驱动是否正常。

*   **检查NPU设备**：
    ```bash
    # 运行npu-smi，若返回驱动相关信息说明已成功挂载。
    npu-smi info
    ```
*   **检查CANN安装**：
    ```bash
    # 查看CANN Toolkit版本信息，是否为对应的8.5.0版本
    cat /usr/local/Ascend/ascend-toolkit/latest/opp/version.info
    ```

你已经拥有了一个“开箱即用”的算子开发环境。接下来，需要在这个环境里验证从源码到可运行算子的完整工具链。

### 2. 无环境场景：WebIDE开发（建设中）
对于无环境的用户，提供WebIDE开发方式，目前本方式正在建设中。

## 二、编译部署

本阶段的目的是**快速走通标准流程**，验证你的开发环境能否成功地将算子源代码编译、打包、安装并运行。我们以仓库内置的`AddExample`算子作为实践对象。

### 1. 拉取ops-nn仓库代码

在容器内获取算子源代码。
```bash
git clone https://gitcode.com/cann/ops-nn.git
cd ops-nn
```

### 2. 编译AddExample算子

以AddExample算子为例，进入项目根目录，编译指定的AddExample算子。
```bash
# 编译命令格式：bash build.sh --pkg --soc=<芯片版本> --ops=<算子名>
bash build.sh --pkg --soc=ascend910b --ops=add_example
```

若提示如下信息，说明编译成功。
```bash
Self-extractable archive "cann-ops-nn-custom-linux.${arch}.run" successfully created.
```
编译成功后，run包存放于项目根目录的build_out目录下。

### 3. 安装AddExample算子包
```bash
./build_out/cann-ops-nn-*linux*.run
```
`AddExample`安装在`${ASCEND_HOME_PATH}/opp/vendors`路径中，`${ASCEND_HOME_PATH}`表示CANN软件安装目录。

### 4. 配置环境变量

将自定义算子包的路径加入环境变量，确保运行时能够找到。
```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_nn/op_api/lib:${LD_LIBRARY_PATH}
```

### 5.快速验证：运行算子样例

`AddExample`目录下提供了简单的算子样例`add_example/examples/test_aclnn_add_example.cpp`，运行该样例来验证算子功能是否正常。

```bash
# 运行命令格式：bash build.sh --run_example <算子名> <运行模式> <包模式>
bash build.sh --run_example add_example eager cust --vendor_name=custom
```
> 预期输出：打印算子`AddExample`的加法计算结果，表明算子已成功部署并正确执行。
```
add_example result[0] is: 2.000000
add_example result[1] is: 2.000000
add_example result[2] is: 2.000000
add_example result[3] is: 2.000000
add_example result[4] is: 2.000000
add_example result[5] is: 2.000000
add_example result[6] is: 2.000000
add_example result[7] is: 2.000000
```

成功运行AddExample算子后，我们尝试修改这个算子的核函数代码。

## 三、算子开发

### 1. 修改kernel实现
找到AddExample算子的核心kernel实现文件`ops-nn/examples/add_example/op_kernel/add_example.h`，尝试在Init函数中增加打印：

```
__aicore__ inline void AddExample<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AddExampleTilingData* tilingData)
{
    // === 在此处添加一行调试打印 ===
    AscendC::PRINTF("This is AddExample Kernel Init.\n");

    blockLength_ = (tilingData->totalLength + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
    // ... 后续原有代码 ...
}
```
保存修改。

### 2.重新编译、部署与验证

重复[编译部署](#二编译部署)章节中的第2至第5步：
1. **重新编译**：回到根目录，`bash build.sh --pkg --soc=ascend910b --ops=add_example`。
2. **重新安装**：`./build_out/cann-ops-nn-*linux*.run`
3. **重新验证**：`bash build.sh --run_example add_example eager cust --vendor_name=custom`。
4. **成功标志**：在运行结果中，除了计算结果，你应能看到打印的"This is AddExample Kernel Init."信息。

## 四、算子调试能力
以`AddExample`算子为例，在算子中添加打印和采集性能数据。

### 1. 打印
算子如果出现执行失败、精度异常等问题，添加打印进行问题分析和定位。

请在`examples/add_example/op_kernel/add_example.h`中进行代码修改调试：

* **printf**

  该接口支持打印Scalar类型数据，如整数、字符型、布尔型等，详细介绍请参见[《Ascend C API》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中“算子调测API > printf”。
  
  ```c++
  blockLength_ = (tilingData->totalLength + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
  tileNum_ = tilingData->tileNum;
  tileLength_ = ((blockLength_ + tileNum_ - 1) / tileNum_ / BUFFER_NUM) ?
        ((blockLength_ + tileNum_ - 1) / tileNum_ / BUFFER_NUM) : 1;
  // 打印当前核计算Block长度
  AscendC::PRINTF("Tiling blockLength is %llu\n", blockLength_);
  ```
* **DumpTensor**

  该接口支持Dump指定Tensor的内容，同时支持打印自定义附加信息，比如当前行号等，详细介绍请参见[《Ascend C API》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中“算子调测API > DumpTensor”。
  
  ```c++
  AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
  // 打印zLocal Tensor信息
  DumpTensor(zLocal, 0, 128);
  ```
### 2. 性能采集

当算子功能正确后，可通过`msprof`工具采集算子性能数据。

* **生成可执行文件**：
    ```
    bash build.sh --run_example add_example eager cust --vendor_name=custom
    ```

    调用`AddExample`的example样例，生成算子可执行文件（test_aclnn_add_example）。该文件位于本项目的`ops-nn/build/`目录。

* **采集性能数据**：

    进入`AddExample`算子可执行文件目录`ops-nn/build/`，执行如下命令：

    ```bash
    msprof --application="./test_aclnn_add_example"
    ```
采集结果在本项目`ops-nn/build/`目录，msprof命令执行完成后，会自动解析并导出性能数据结果文件，详细内容请参见：[msprof](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/T&ITools/Profiling/atlasprofiling_16_0110.html#ZH-CN_TOPIC_0000002504160251)

掌握了调试手段，你的算子开发能力将更加全面。最后，为了确保算子的通用性，需要学习如何构建不同的测试用例对其进行验证。

## 五、算子验证

通过修改example样例中的输入数据，可以验证算子在多种场景下的功能正确性。

### 1. 修改测试输入
找到并编辑`AddExample`的`ops-nn/examples/add_example/examples/test_aclnn_add_example.cpp`，修改输入张量的形状和数值。

- **修改输入、输出数据**：
修改输入、输出的shape信息，以及初始化数据，构造相应的输入、输出tensor。
```c++
int main() {
    // ... 初始化代码 ...
    
    // 修改前：shape = {32, 4, 4, 4}, 数值全为1
    // 修改后：将输入shape改为 {8, 8, 8, 8}，并填充不同的测试数据
    std::vector<int64_t> selfXShape = {8, 8, 8, 8};
    std::vector<float> selfXHostData(4096); // 4096 = 8 * 8 * 8 *8
    // 可使用循环填充更有区分度的数据，例如递增序列
    for (int i = 0; i < 4096; ++i) {
        selfXHostData[i] = static_cast<float>(i % 10); // 填充0-9的循环值
    }
    // 同理修改selfY的输入...
    
    // ... 后续执行代码 ...
}
```
### 2. 重新编译并运行验证

1. 由于只修改了example测试代码，无需重新编译算子包。

2. 直接重新运行验证命令即可：`bash build.sh --run_example add_example eager cust --vendor_name=custom`

3. 观察输出结果是否符合运算的预期。

## 开发贡献

开发完成后，可以将完成的算子贡献到算子仓，以下为贡献流程：[贡献指南](CONTRIBUTING.md)。

若需深入每个环节，请参考以下详细指南：

- [环境部署](./docs/zh/context/quick_install.md)：更详细的环境搭建说明。
- [编译部署及算子调用](./docs/zh/invocation/quick_op_invocation.md)：深入了解编译参数与调用方式。
- [算子开发](./docs/zh/develop/aicore_develop_guide.md)：学习如何从零创建算子工程，实现Tiling和Kernel。
- [调试调优](./docs/zh/debug/op_debug_prof.md)：掌握更系统的调试技巧与性能优化方法。

