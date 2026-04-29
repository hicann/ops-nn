# 算子调用

## 使用须知

- 前提说明：算子调用前，请参考本项目README完成环境准备和源码下载，再完成[源码构建](../install/compile.md)，此处不再赘述。
- 调用算子：支持调用非experimental目录算子（内置算子，清单详见[算子列表](../op_list.md)）和experimental目录算子（贡献算子）。

## 调用算子样例

如何快速调用项目中的算子，本章介绍最简调用方法，通过build.sh命令执行算子样例。

> **说明**：对于Ascend 950PR产品，可通过Simulator仿真工具执行算子样例，详见[仿真指导](../../../docs/zh/debug/op_debug_prof.md#方式二-仿真流水图采集)。

- 基于**自定义算子包**执行算子样例，包安装后，执行如下命令：

    ```bash
    bash build.sh --run_example ${op} ${mode} ${pkg_mode} [--example_name=${example_name}] [--vendor_name=${vendor_name}] [--soc=${soc_version}] [--simulator] [--experimental]
    # TransposeBatchMatMul算子执行test_aclnn_transpose_batch_mat_mul.cpp为例
    # bash build.sh --run_example transpose_batch_mat_mul eager cust --example_name=transpose_batch_mat_mul --vendor_name=custom
    ```

    - \$\{op\}：表示待执行算子，算子名小写下划线形式，如transpose_batch_mat_mul。
    - \$\{mode\}：表示调用方式，目前支持eager（aclnn调用）、graph（图模式调用）。
    - \$\{pkg_mode\}：表示包模式，目前仅支持cust，即自定义算子包。
    - \$\{example_name\}（可选）：表示待执行样例名，名称为各个算子examples文件夹下的文件名称，去掉`test_aclnn_`前缀和`.cpp`后缀。
    - \$\{vendor\_name\}（可选）：与构建的自定义算子包设置一致，默认名为custom。
    - \$\{soc_version\}（可选）：表示NPU型号。当设置为"ascend950"时会额外运行"arch35"目录下的示例文件。
    - \$\{simulator\}（可选）：表示仿真模式，目前仅支持eager（aclnn调用）场景下使用。仿真模式下，会根据soc_version链接对应的仿真库。
    - \$\{experimental\}（可选）：表示执行用户保存在experimental贡献目录下的算子。
    
    说明：\$\{mode\}为graph时，不指定\$\{pkg_mode\}和\$\{vendor\_name\}

- 基于**ops-nn包**执行算子样例，安装后，执行如下命令：

    ```bash
    bash build.sh --run_example ${op} ${mode} [--soc=${soc_version}] [--simulator]
    # 以TransposeBatchMatMul算子example执行为例
    # bash build.sh --run_example transpose_batch_mat_mul eager
    ```

    - \$\{op\}：表示待执行算子，算子名为小写下划线形式，如transpose_batch_mat_mul。
    - \$\{mode\}：表示算子执行模式，目前支持eager（aclnn调用）、graph（图模式调用）。
    - \$\{soc_version\}（可选）：表示NPU型号。当设置为"ascend950"时会额外运行"arch35"目录下的示例文件。
    - \$\{simulator\}（可选）：表示仿真模式，目前仅支持eager（aclnn调用）场景下使用。仿真模式下，会根据soc_version链接对应的仿真库。

- 基于**ops-nn静态库**执行算子样例：

    1. **前提条件**

        ops-nn静态库依赖于ops-legacy静态库和ops-math静态库，将上述静态库准备好，解压并将所有lib64、include目录移动至统一目录\$\{static\_lib\_path\}下。

        > 说明：ops-legacy静态库```cann-${soc_name}-ops-legacy-static_${cann_version}_linux-${arch}.tar.gz```可通过单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-release/software/master)获取， ops-nn静态库、ops-math静态库暂未提供软件包，请通过本地编译生成。

    2. **创建run.sh**

        在待执行算子`examples/test_aclnn_${op_name}.cpp`同级目录下创建run.sh文件。

        以TransposeBatchMatMul算子执行test_aclnn_transpose_batch_mat_mul.cpp为例，示例如下:

        ```bash
        # 静态库文件路径
        static_lib_path=""

        # 环境变量生效
        if [ -n "$ASCEND_INSTALL_PATH" ]; then
            _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
        elif [ -n "$ASCEND_HOME_PATH" ]; then
            _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
        else
            _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
        fi

        source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash

        # 编译可执行文件
        g++ test_aclnn_transpose_batch_mat_mul.cpp \
        -I ${static_lib_path}/include \
        -L ${static_lib_path}/lib64 \
        -I ${_ASCEND_INSTALL_PATH}/include \
        -I ${_ASCEND_INSTALL_PATH}/include/aclnnop \
        -L ${_ASCEND_INSTALL_PATH}/lib64 \
        -Wl,--allow-multiple-definition \
        -Wl,--start-group -lcann_nn_static -lcann_math_static -lcann_legacy_static -Wl,--end-group -lgraph -lgraph_base \
        -lpthread -lmmpa -lmetadef -lascendalog -lregister -lopp_registry -lops_base -lascendcl -ltiling_api -lplatform \
        -ldl -lc_sec -lnnopbase -lruntime -lerror_manager -lunified_dlog \
        -o test_aclnn_transpose_batch_mat_mul   # 替换为实际算子可执行文件名

        # 执行程序
        ./test_aclnn_transpose_batch_mat_mul
        ```

        \$\{static\_lib\_path\}表示静态库统一放置路径； \$\{ASCEND\_INSTALL\_PATH\}已通过环境变量配置，表示CANN toolkit包安装路径； 最终可执行文件名请替换为**实际算子可执行文件名**。  
        
        其中lcann\_nn\_static、lcann\_math\_static、lcann\_legacy\_static表示算子依赖的静态库文件，从静态库统一放置路径\$\{static\_lib\_path\}中获取； 
        lgraph、lmetadef等表示算子依赖的底层库文件，可在CANN toolkit包获取。 
        
    3. **执行run.sh**

        ```bash
        bash run.sh
        ```

无论上述哪种方式，算子样例执行后会打印结果，以TransposeBatchMatMul算子为例：

```bash
result[0] is: 0.000000
result[1] is: 0.000000
result[2] is: 0.000000
result[3] is: 0.000000
...
```

## 调用方式

通过build.sh执行算子时，其底层调用算子的原理目前包括如下几种方式：

- PyTorch调用**（建设中）**：通过`<<<>>>`方式启动算子Kernel，流程极简，实现PyTorch方式调用NPU算子。

- aclnn调用 **（推荐）**：Host侧提供算子对应的C语言API（前缀aclnn），无需提供IR定义，实现aclnn API方式调用算子。

- 图模式调用：提供IR（Intermediate Representation）定义，实现构图方式调用算子。

### PyTorch调用（建设中）

该方式提供一套基于Ascend Extension for PyTorch（torch_npu）框架调用NPU算子的方法，具体调用原理和过程请参考[examples/fast_kernel_launch_example](../../../examples/fast_kernel_launch_example/README.md)，内容仍在建设和优化中，欢迎您提问和建议。

### aclnn调用

#### 调用流程

该方式也称为“单算子API调用”，通过提供一套基于C的API（以aclnn为前缀API）实现算子调用，无需提供算子IR（Intermediate Representation）定义。aclnn API的调用流程如下：

![原理图](../figures/aclnn调用.png)

aclnn API调用示例以`AddExample`算子为例，代码示例如下，仅供参考，全量代码参见[test_aclnn_add_example.cpp](../../../examples/add_example/examples/test_aclnn_add_example.cpp)。调用前，请按照环境安装的提示信息设置环境变量。

> 注意：如需调用项目已实现的算子，可访问目标算子`examples`目录下test\_aclnn\_\$\{op\_name\}.cpp，\$\{op\_name\}表示算子名。

```Cpp
int main()
{
    // 1. 调用acl进行device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    aclTensor* selfX = nullptr;
    void* selfXDeviceAddr = nullptr;
    std::vector<int64_t> selfXShape = {32, 4, 4, 4};
    std::vector<float> selfXHostData(2048, 1);
    ret = CreateAclTensor(selfXHostData, selfXShape, &selfXDeviceAddr, aclDataType::ACL_FLOAT, &selfX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* selfY = nullptr;
    void* selfYDeviceAddr = nullptr;
    std::vector<int64_t> selfYShape = {32, 4, 4, 4};
    std::vector<float> selfYHostData(2048, 1);
    ret = CreateAclTensor(selfYHostData, selfYShape, &selfYDeviceAddr, aclDataType::ACL_FLOAT, &selfY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* out = nullptr;
    void* outDeviceAddr = nullptr;
    std::vector<int64_t> outShape = {32, 4, 4, 4};
    std::vector<float> outHostData(2048, 1);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 4. 调用aclnnAddExample第一段接口
    ret = aclnnAddExampleGetWorkspaceSize(selfX, selfY, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddExampleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. 调用aclnnAddExample第二段接口
    ret = aclnnAddExample(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddExample failed. ERROR: %d\n", ret); return ret);

    // 6. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(outShape, &outDeviceAddr);

    // 8. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(selfX);
    aclDestroyTensor(selfY);
    aclDestroyTensor(out);

    // 9. 释放device资源
    aclrtFree(selfXDeviceAddr);
    aclrtFree(selfYDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);

    // 9. acl去初始化
    aclFinalize();
    return 0;
}
```

#### 编译与运行

>说明：对于本项目内已实现的算子（非自定义算子），可通过根目录下build.sh直接运行算子，操作请参考[调用算子样例](#调用算子样例)。

1. 前提条件。
   请参考本项目[源码构建](../install/compile.md)完成目标算子的编译部署。

2. 创建CMakeLists.txt文件。

   在test\_aclnn\_\$\{op\_name\}.cpp同级目录下创建CMakeLists.txt文件，以`AddExample`算子为例，示例如下，请根据实际情况自行修改。

    ```bash
    cmake_minimum_required(VERSION 3.14)
    # 设置工程名
    project(ACLNN_EXAMPLE)

    # 设置C++编译标准
    add_compile_options(-std=c++11)

    # 设置编译输出目录为当前目录下的bin文件夹
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")    

    # 设置调试和发布模式的编译选项
    set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

    # 添加可执行文件（请替换为实际算子可执行文件），指定算子调用的*.cpp文件
    add_executable(test_aclnn_add_example              
    test_aclnn_add_example.cpp)         

    # ASCEND_PATH（CANN软件包目录，请根据实际路径修改）
    if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")      
        set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
    else()
        set(ASCEND_PATH "/usr/local/Ascend/cann")
    endif()

    # 获取自定义算子包名称，存在多个自定义算子包时，只会使用其中一个
    set(VENDORS_DIR "${ASCEND_PATH}/opp/vendors")
    file(GLOB CUSTOM_DIRS "${VENDORS_DIR}/*")
    foreach(CUSTOM_DIR ${CUSTOM_DIRS})
        if(IS_DIRECTORY ${CUSTOM_DIR})
            set(TARGET_SUBDIR ${CUSTOM_DIR})
        endif()
    endforeach()

    if(NOT DEFINED TARGET_SUBDIR)
        message(FATAL_ERROR "在路径${ASCEND_PATH}中未找到自定义算子包") 
    endif()

    # 设置头文件路径
    set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
    include_directories(
        ${INCLUDE_BASE_DIR}
        ${TARGET_SUBDIR}/op_api/include    # 仅自定义算子需要
        # ${INCLUDE_BASE_DIR}/aclnn                                   # 仅内置算子需要
    )
    include_directories(
        ${INCLUDE_BASE_DIR}
    )

    # 链接所需的动态库
    target_link_libraries(test_aclnn_add_example PRIVATE             # 替换实际算子可执行文件
        ${ASCEND_PATH}/lib64/libascendcl.so
        ${ASCEND_PATH}/lib64/libnnopbase.so
        ${TARGET_SUBDIR}/op_api/lib/libcust_opapi.so   # 仅自定义算子需要
        # ${ASCEND_PATH}/lib64/libopapi_nn.so    # 仅内置算子需要
    )
    target_link_options(test_aclnn_add_example PRIVATE
        "-Wl,-rpath,${TARGET_SUBDIR}/op_api/lib" # 仅自定义算子需要
    )

    # 安装目标文件到bin目录  
    install(TARGETS test_aclnn_add_example DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    ```

3. 创建run.sh文件。
   
    在test\_aclnn\_\$\{op\_name\}.cpp同级目录下创建run.sh文件，以`AddExample`算子为例，示例如下，请根据实际情况自行修改。
    
    ```bash
    if [ -n "$ASCEND_INSTALL_PATH" ]; then                      # 实际CANN包安装路径
        _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
    elif [ -n "$ASCEND_HOME_PATH" ]; then
        _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
    else
        _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
    fi

    source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
    
    rm -rf build
    mkdir -p build 
    cd build
    cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE  # 执行构建命令
    make
    cd bin
    ./test_aclnn_add_example            # 替换为实际算子可执行文件名
    ```
    
4. 运行run.sh文件。
    在run.sh文件所在路径执行如下命令：

   ```bash
   bash run.sh
   ```
   
    默认在当前执行路径 `/build/bin`下生成可执行文件test\_aclnn\_add\_example，运行结果如下：

   ```bash
   mean result[2046] is 2.000000
   mean result[2047] is 2.000000
   ```

### 图模式调用

#### 调用流程

该方式采用算子IR（Intermediate Representation）构图方式调用算子，调用流程如下：

![原理图](../figures/IR调用.png)

调用示例以`AddExample`算子为例，代码示例如下，仅供参考，全量代码参见[test_geir_add_example.cpp](../../../examples/add_example/examples/test_geir_add_example.cpp)。调用前，请按照环境安装的提示信息设置环境变量。

> 说明：如需调用项目已实现算子，可访问目标算子`examples`目录下test\_geir\_\$\{op\_name\}.cpp，$\{op\_name\}表示算子名。

```CPP
int main() {
    // 1. 创建图对象
    Graph graph(graphName);

    // 2. 图全局编译选项初始化
    Status ret = ge::GEInitialize(globalOptions);

    // 3. 创建AddExample算子实例
    auto add1 = op::AddExample("add1");

    // 4. 定义图输入输出向量
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    // 5. 准备输入数据
    std::vector<int64_t> xShape = {32,4,4,4};
    // 宏展开方式处理变量赋值
    ADD_INPUT(1, x1, inDtype, xShape);
    ADD_INPUT(2, x2, inDtype, xShape);
    ADD_OUTPUT(1, y, inDtype, xShape);

    outputs.push_back(add1);

    // 6. 设置图对象的输入算子和输出算子
    graph.SetInputs(inputs).SetOutputs(outputs);

    // 7. 创建session对象
    ge::Session* session = new Session(buildOptions);

    // 8. session添加图
    ret = session->AddGraph(graphId, graph, graphOptions);

    // 9. 运行图
    ret = session->RunGraph(graphId, input, output);

    // 10. 释放资源
    GEFinalize();

    return 0;
}
```

#### 编译与运行

>说明：对于本项目内已实现的算子（非自定义算子），可通过根目录下build.sh直接运行算子，操作请参考[调用算子样例](#调用算子样例)。

1. 前提条件。
   请参考本项目[源码构建](../install/compile.md)完成目标算子的编译部署。

2. 创建CMakelist文件。

   在test\_geir\_\$\{op\_name\}.cpp同级目录下创建CMakelist文件，以`AddExample`算子为例，示例如下，请根据实际情况自行修改。

    ```bash
   cmake_minimum_required(VERSION 3.14)
    
   # 设置工程名
   project(GE_IR_EXAMPLE)
   
   if(NOT "$ENV{ASCEND_OPP_PATH}" STREQUAL "")
       get_filename_component(ASCEND_PATH $ENV{ASCEND_OPP_PATH} DIRECTORY)
   elseif(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
       set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
   else()
       set(ASCEND_PATH "/usr/local/Ascend/cann")
   endif()
   
   set(FWK_INCLUDE_DIR "${ASCEND_PATH}/compiler/include")
   
   message(STATUS "ASCEND_PATH: ${ASCEND_PATH}")
   
   file(GLOB files CONFIGURE_DEPENDS
        test_geir_add_example.cpp         
   )
   
   # 添加可执行文件（请替换为实际算子可执行文件）
   add_executable(test_geir_add_example ${files})      
   
   find_library(GRAPH_LIBRARY_DIR libgraph.so "${ASCEND_PATH}/compiler/lib64/stub")
   find_library(GE_RUNNER_LIBRARY_DIR libge_runner.so "${ASCEND_PATH}/compiler/lib64/stub")
   find_library(GRAPH_BASE_LIBRARY_DIR libgraph_base.so "${ASCEND_PATH}/compiler/lib64")
   
   # 链接所需的动态库
   target_link_libraries(test_geir_add_example PRIVATE      
        ${GRAPH_LIBRARY_DIR}
        ${GE_RUNNER_LIBRARY_DIR}
        ${GRAPH_BASE_LIBRARY_DIR}
   )
   
   # 设置头文件路径
   target_include_directories(test_geir_add_example PRIVATE       
        ${FWK_INCLUDE_DIR}/graph/
        ${FWK_INCLUDE_DIR}/ge/
        ${ASCEND_PATH}/opp/built-in/op_proto/inc/
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ASCEND_PATH}/compiler/include
   )
    ```

3. 创建run.sh脚本。

   在test\_geir\_\$\{op\_name\}.cpp同级目录下创建run.sh文件，以`AddExample`算子为例，示例如下，请根据实际情况自行修改。

    ```bash
    if [ -n "$ASCEND_INSTALL_PATH" ]; then                      # 实际CANN包安装路径
        _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
    elif [ -n "$ASCEND_HOME_PATH" ]; then
        _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
    else
        _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
    fi

    source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash               

    rm -rf build                 
    mkdir -p build 
    cd build
    cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE  # 执行构建命令
    make
    ./test_geir_add_example                  # 替换为实际算子可执行文件名
    ```

4. 运行run.sh脚本。
    在run.sh文件所在路径执行如下命令：
   
    ```bash
    bash run.sh
    ```
   
    默认在当前执行路径 `/build/bin`下生成可执行文件test\_geir\_add\_example，运行结果如下：
   
    ```bash
    INFO - [XIR]: Finalize ir graph session success
    ```
