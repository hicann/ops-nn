# AI Core算子开发指南

> 说明：  
> 1. 算子开发过程中涉及的基本概念如Tiling、Kernel、Ascend C接口等，详细介绍请参考[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)。  
> 2. AI CORE算子使用Ascend C语言开发，运行在AI CORE硬件单元，AI CPU算子使用C++语言开发，运行在AI CPU硬件单元，如果你想贡献aicpu算子，请参考[AI CPU算子开发指南](./aicpu_develop_guide.md)。

开发指南以`AddExample`算子开发为例，介绍新算子开发流程以及涉及的交付件，流程图如下，完整样例代码请访问项目`examples`目录。

1. [工程创建](#工程创建)：开发算子前，需完成环境部署并创建算子目录，方便后续算子的编译和部署。

2. [算子定义](#算子定义)：算子功能说明与原型定义

3. [Tiling实现](#Tiling实现)：实现Host侧算子Tiling函数。

4. [Kernel实现](#Kernel实现)：实现Device侧算子核函数。

5. [aclnn适配](#aclnn适配)：自定义算子推荐aclnn接口调用，需完成二进制发布。如需入图，请参考[附录](#附录)。

6. [编译部署](#编译部署)：通过工程编译脚本完成自定义算子的编译和安装。 

7. [算子验证](#算子验证)：通过常见算子调用方式，验证自定义算子功能。  

## 工程创建
**1. 环境部署**

开发算子前，请参考[环境准备](../invocation/quick_op_invocation.md#环境准备)完成环境搭建。

**2. 目录创建**

目录创建是算子开发的重要步骤，为后续代码编写、编译构建和调试提供统一的目录结构和文件组织方式。

本项目`build.sh`，支持快速创建算子目录。进入项目根目录，执行以下命令：

```bash
# 创建指定算子目录，如bash build.sh --genop=examples/add_example
# ${op_class}表示算子类型，如matmul类。
# ${op_name}表示算子名的小写下划线形式，如`AddExample`算子对应为add_example。
bash build.sh --genop=${op_class}/${op_name}
```

如果命令执行成功，会看到如下提示信息：

```bash
Create the initial directory for ${op_name} under ${op_class} success
```
创建完成后，目录结构如下所示：

```
${op_name}                              # 替换为实际算子名的小写下划线形式
├── examples                            # 算子调用示例
│   ├── test_aclnn_${op_name}.cpp       # 算子aclnn调用示例
├── op_host                             # Host侧实现
│   ├── ${op_name}_def.cpp              # 算子信息库，定义算子基本信息，如名称、输入输出、数据类型等
│   ├── ${op_name}_infershape.cpp       # InferShape实现，实现算子形状推导，在运行时推导输出shape
│   ├── ${op_name}_tiling.cpp           # Tiling实现，将张量划分为多个小块，区分数据类型进行并行计算
│   └── CMakeLists.txt                  # Host侧cmakelist文件
└── op_kernel                           # Device侧Kernel实现
│   ├── ${op_name}_tiling_key.h         # Tilingkey文件，定义Tiling策略的Key，标识不同的划分方式
│   ├── ${op_name}_tiling_data.h        # Tilingdata文件，存储Tiling策略相关的配置数据，如块大小、并行度
│   ├── ${op_name}.cpp                  # Kernel入口文件，包含主函数和调度逻辑
│   └── ${op_name}.h                    # Kernel实现文件，定义Kernel头文件，包含函数声明、结构定义、逻辑实现
├── tests                               # UT实现
│   ├── ut                              # tiling/kernel/aclnn UT实现
└── CMakeLists.txt                      # 算子cmakelist入口
```

使用上述命令行创建算子工程后，若要手动删除新创建出的算子工程，需要同时删除与算子工程同目录CMakeLists.txt中新添加的add_subdirectory(${op_class})
## 算子定义
算子定义需要完成两个交付件：`README.md` `${op_name}_def.cpp`

**交付件1：README.md**

开发算子前需要先确定目标算子的功能和计算逻辑。

以自定义`AddExample`算子说明为例，请参考[AddExample算子说明](../../examples/add_example/README.md)。

**交付件2：${op_name}_def.cpp**

算子信息库。

以自定义`AddExample`算子说明为例，请参考[AddExample算子信息库](../../examples/add_example/op_host/add_example_def.cpp)。
## Tiling实现

### Tiling简介

因NPU中AI Core内部存储空间有限，无法一次性将整个张量数据加载到计算单元中处理，因此需要将输入张量切分为多个小块（Tile），逐块进行计算，这一过程称为Tiling。

用于指导数据切分的算法称为Tiling策略或Tiling算法，其决定了如何将输入数据切分为多个计算块，并指导Kernel如何分配内存、调度计算任务。Tiling与Kernel之间通过`TilingData`结构体进行信息传递。

### 代码实现

Tiling一共需要三个交付件：`${op_name}_tiling.cpp` `${op_name}_tiling_key.h` `${op_name}_tiling_data.h`

**交付件1：${op_name}_tiling.cpp**

Tiling主要切分逻辑。

如需查看详细实现，请参考[add_example_tiling.cpp](../../examples/add_example/op_host/add_example_tiling.cpp)。

```CPP
// ${op_name}_tiling.cpp
// 1.Tiling需要获取运行环境信息，包括可用核数、UB(Unified Buffer)大小，并将获取到的信息传递给`CompileInfo`。 由于自动生成的aclnn接口实现不调用该函数，直接返回`ge::GRAPH_SUCCESS`即可。
static ge::graphStatus TilingParse(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;

    // 若手写aclnn接口，可以按照下面步骤完善parse函数
    // // 1.1获取环境信息
    // auto compileInfo = context->GetCompiledInfo<CompileInfo>();
    // OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    // auto platformInfo = context->GetPlatformInfo();
    // auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    // // 1.2获取可用核数

    // compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    // // 1,3获取UB大小
    // uint64_t ubSizePlatForm;
    // ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    // compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);
    // ...
    // return ge::GRAPH_SUCCESS;
}

// 2.Tiling计算主入口
static ge::graphStatus TilingFunc(gert::TilingContext* context){
    // 2.1获取平台信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    
    // 2.2获取输入信息
    // 获取输入张量shape信息
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);

    // 如果输入shape是标量，转换为{1}，否则保持原shape不变
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());

    // 获取输入张量的描述信息
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    // 获取数据类型
    dataType = inputDesc->GetDataType();

    // 2.3计算Tiling参数（根据算子功能不同自行设计）
    ...

    // 2.4设置TilingData信息
    ${op_name}TilingData* tiling = context->GetTilingData<${op_name}TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(${op_name}TilingData), 0, sizeof(${op_name}TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->totalLength = totalIdx;
    tiling->tileNum = TILE_NUM;

    // 2.5设置WorkspaceSize（可选）
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
}

// 3.Tiling注册入口
IMPL_OP_OPTILING(${op_name}).Tiling(TilingFunc).TilingParse<CompileInfo>(TilingParse);
```
**交付件2：${op_name}_tiling_key.h**

TilingKey是一个算子内为了区分不同的实现而将kernel代码进行区分的方法，kernel侧可以通过TilingKey来选择不同的算法逻辑。

如需查看详细实现，请参考[add_example_tiling_key.h](../../examples/add_example/op_kernel/add_example_tiling_key.h)。

```CPP
// ${op_name}_tiling_key.h
ASCENDC_TPL_ARGS_DECL(
    ${op_name},
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, ELEMENTWISE_TPL_SCH_MODE_0, ELEMENTWISE_TPL_SCH_MODE_1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, ELEMENTWISE_TPL_SCH_MODE_0, ELEMENTWISE_TPL_SCH_MODE_1)));
```
**交付件3：${op_name}_tiling_data.h**

切分算法相关的参数，比如总数据量大小、每个核数据切块数量，通过结构体存储。

如需查看详细实现，请参考[add_example_tiling_data.h](../../examples/add_example/op_kernel/add_example_tiling_data.h)。

```CPP
// ${op_name}_tiling_data.h
struct ${op_name}TilingData {
    int64_t totalLength;
    int64_t tileNum;
};
```

如需实现复杂参数组合完成分支选择（涉及多TilingKey场景），请参考[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)中“算子实现 > 工程化算子开发 > Host侧Tiling实现 > Tiling模板编程”。

## Kernel实现

### Kernel简介
Kernel是算子在NPU执行的核心部分，负责张量数据的加载、计算和存储，是算子功能实现的最终载体。Kernel的实现需要与Tiling策略紧密配合，根据Tiling提供的`TilingData`、`TilingKey`信息进行内存分配和计算调度。

Kernel实现包括如下步骤，整个流程通过`Process`函数串联，实现完整的算子流程。

```mermaid
graph LR
    H([核函数定义]) --> A([定义Kernel类])
    A --> B([初始化函数<br>Init])
    B --> C([主处理函数<br>Process])
    
    subgraph C [主处理函数 Process]
        D([数据搬入<br>CopyIn]) --> E([计算<br>Compute]) --> F([数据搬出<br>CopyOut])
    end
    
    F --> G([Kernel执行完成])
```



### 代码实现

Kernel需要两个交付件：`${op_name}.cpp` `${op_name}.h`

**交付件1：${op_name}.cpp**

Kernel入口文件，包含主函数和调度逻辑。

如需查看详细实现，请参考[add_example.cpp](../../examples/add_example/op_kernel/add_example.cpp)。

```CPP
// 1、核函数定义
// schMode是一个模板参数，用于支持不同数据类型（如float和int32）的计算路径
// __global__ __aicore__表示该函数是个全局函数，可以在AI Core上执行
template <uint32_t schMode>
__global__ __aicore__ void add_example(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling){
    ....
    // Tiling注册入口
    REGISTER_TILING_DEFAULT(AddExampleTilingData);

    // 宏方式获取TilingData
    GET_TILING_DATA_WITH_STRUCT(AddExampleTilingData, tilingData, tiling);

    // 根据TilingKey实例化Kernel对象并完成计算
    if constexpr (schMode == static_cast<uint32_t>(AddExampleTilingKey::TILING_KEY_EXAMPLE_FLOAT)) { // float数据类型走该分支
        NsAddExample::AddExample<float> op;     // 算子Kernel实例获取
        op.Init(x, y, z, &tilingData);          // 算子Kernel实例初始化
        op.Process();                           // 算子Kernel实例执行
    }
    ....
}
```
**交付件2：${op_name}.h**

定义Kernel头文件，包含函数声明、结构定义、逻辑实现等。

如需查看详细实现，请参考[add_example.h](../../examples/add_example/op_kernel/add_example.h)。

```C++
// 2、定义Kernel类
template <typename T>
class AddExample
{
public:
    // 默认构造函数，__aicore__表示该函数在AI Core上运行
    __aicore__ inline AddExample(){};     
    // 初始化函数，用于设置输入输出地址和Tiling切分信息计算
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AddExampleTilingData* tilingData);
    // 主处理函数，执行数据拷贝和计算
    __aicore__ inline void Process();

private:
    // 数据从GM拷贝到LM的函数
    __aicore__ inline void CopyIn(int32_t progress);
    // 数据从LM拷贝到GM的函数
    __aicore__ inline void CopyOut(int32_t progress);
    // 执行计算的函数，datalength表示当前处理的数据长度
    __aicore__ inline void Compute(const int32_t dataLength);

private:
    // 管道对象，用于管理数据流（拷贝和计算的流水线）
    TPipe pipe;
    // 输入队列X，从GM拷贝到LM，BUFFER_NUM表示buffer数量，开启double buff达到流水并行，为2
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    // 输入队列Y，从GM拷贝到LM，BUFFER_NUM表示buffer数量，开启double buff达到流水并行，为2
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    // 输出队列Z，从LM拷贝到GM，BUFFER_NUM表示 buffer数量，这里开启double buff达到流水并行，为2
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;

    // 输入X的GM地址
    GlobalTensor<T> inputGMX;
    // 输入Y的GM地址
    GlobalTensor<T> inputGMY;
    // 输出Z的GM地址
    GlobalTensor<T> outputGMZ;

    // 总数据长度
    int64_t blockLength_ = 0;
    // 每个block被划分多少块
    int64_t tileNum_ = 0;
    // 每个tile处理数据长度
    int64_t tileLength_ = 0;
    ...
};

// 3、初始化函数Init
template <typename T>
__aicore__ inline void AddExample<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AddExampleTilingData* tilingData)
{
    // 3.1 初始化成员变量
    blockLength_ = tilingData->totalLength / AscendC::GetBlockNum();
    ...
    // 3.2 初始化GM地址
    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockLength_ * AscendC::GetBlockIdx(), blockLength_);
    ...
    // 3.3 初始化队列长度
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, tileLength_ * sizeof(T));
    ...
}

// 4、主处理函数Process
template <typename T>
__aicore__ inline void AddExample<T>::Process()
{
    // 计算当前核处理数据循环次数
    int32_t loopCount = tileNum_ * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
        CopyIn(i);              // 数据搬入
        Compute(i);             // 计算
        CopyOut(i);             // 数据搬出
    }
}
...
```
## aclnn适配

通常算子开发和编译完成后，会自动生成aclnn接口（一套基于C 的API），可直接在应用程序中调用aclnn接口实现调用算子。

为实现该调用方式，需提前生成算子对应的二进制包，增加二进制编译json文件，以`AddExample`算子为例：

1. 在`scripts/kernel/binary_config`目录[ascendc_config.json](../../scripts/kernel/binary_config/ascendc_config.json)中，注册算子的NPU型号和实现模式，示例如下，输入实际name和compute_units即可。如果未生成aclnn接口，请检查`examples/add_example/op_host`下的CMakeLists，ACLNNTYPE为aclnn生成aclnn接口，ACLNNTYPE为aclnn_exclude不生成接口。

    ```json
    {"name":"AddExample", "compute_units": ["${soc_version}"], "auto_sync":true, "impl_mode" : "high_performance"},
    ```

## 编译部署

算子开发完成后，需对算子工程进行编译，生成自定义算子安装包\*\.run，详细的编译操作如下：

1. **准备工作。**

    参考[前提条件](#前提条件)完成基础环境搭建，同时检查算子开发交付件是否完备，是否在对应算子分类目录下。

2. **编译自定义算子包。**

    以`AddExample`算子为例，假设开发交付件在`examples`目录，完整代码参见[add_example](../../examples/add_example)目录。

    进入项目根目录，执行如下编译命令：

    ```bash
    # 编译指定算子，如--ops=add_example
    bash build.sh --pkg --soc=${soc_version} --vendor_name=${vendor_name} --ops=${op_list}
    ```

    若提示如下信息，说明编译成功：

    ```bash
    Self-extractable archive "cann-ops-nn-${vendor_name}_linux-${arch}.run" successfully created.
    ```

3. **安装自定义算子包。**

    执行以下命令进行安装：
    
    ```bash
    # 安装run包
    ./build_out/cann-ops-nn-${vendor_name}_linux-${arch}.run
    ```
    自定义算子包安装在`${ASCEND_HOME_PATH}/opp/vendors`路径中，`${ASCEND_HOME_PATH}`表示CANN软件安装目录，可提前在环境变量中配置。自定义算子包不支持卸载。
    

## 算子验证
```bash
    # 执行前需要导入环境变量
    export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/${vendor_name}_nn/op_api/lib:${LD_LIBRARY_PATH}
```

1. **UT验证。**

    算子开发过程中，可通过UT验证（如tiling/kernel UT）方式进行快速验证，方法请参考[本地验证](../invocation/quick_op_invocation.md#本地验证)中执行算子UT的部分。

2. **aclnn调用验证。**

    开发好的算子完成编译部署后，可通过aclnn方式验证功能，方法请参考[算子调用方式](../invocation/op_invocation.md)。

## 附录

自定义算子如需运行图模式，不需要aclnn适配，做如下交付件适配，详细内容请参考[图模式开发指南](./graph_develop_guide.md)。
