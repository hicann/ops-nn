# aclnnAdvanceStepV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：

  vLLM是一个高性能的LLM推理和服务框架，专注于优化大规模语言模型的推理效率。它的核心特点包括PageAttention和高效内存管理。advance_step算子的主要作用是推进推理步骤，即在每个生成步骤中更新模型的状态并生成新的inputTokens、inputPositions、seqLens和slotMapping，为vLLM的推理提升效率。

- 计算公式：

  $$
  blockIdx是当前代码被执行的核的index。
  $$

  $$
  blockTablesStride = blockTables.stride(0)
  $$

  $$
  inputTokens[blockIdx] = sampledTokenIds[blockIdx]
  $$

  $$
  inputPositions[blockIdx] = seqLens[blockIdx]
  $$

  $$
  seqLens[blockIdx] = seqLens[blockIdx] + 1
  $$

  $$
  slotMapping[blockIdx] = (blockTables[blockIdx] + blockTablesStride * blockIdx) * blockSize + (seqLens[blockIdx] \% blockSize)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnAdvanceStepV2GetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnAdvanceStepV2”接口执行计算。
```Cpp
aclnnStatus aclnnAdvanceStepV2GetWorkspaceSize(
  const aclTensor *inputTokens,
  const aclTensor *sampledTokenIds,
  const aclTensor *inputPositions,
  const aclTensor *seqLens,
  const aclTensor *slotMapping,
  const aclTensor *blockTables,
  const aclTensor *specToken,
  const aclTensor *acceptedNum,
  int64_t          numSeqs,
  int64_t          numQueries,
  int64_t          blockSize,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAdvanceStepV2(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```


## aclnnAdvanceStepV2GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1330px"><colgroup>
  <col style="width: 101px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 250px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>inputTokens</td>
      <td>输入/输出</td>
      <td>待进行AdvanceStepV2计算的入参/出参，公式中的输出inputTokens，用于更新vLLM模型中的token值。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape第一维长度与numSeqs一致，第二维长度为1+specNum。</li><li>取值范围是大于0的正整数。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sampledTokenIds</td>
      <td>输入</td>
      <td>待进行AdvanceStepV2计算的入参，用于储存tokenID，公式中的输入sampledTokenIds。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape第一维长度与numSeqs一致，第二维长度为1+specNum。</li><li>取值范围是大于0的正整数。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
     <tr>
      <td>inputPositions</td>
      <td>输入/输出</td>
      <td>待进行AdvanceStepV2计算的入参/出参，公式中的输出inputPositions，用于记录token的index。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape长度与numSeqs一致。</li><li>取值范围是大于0的正整数。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>seqLens</td>
      <td>输入/输出</td>
      <td>待进行AdvanceStepV2计算的入参/出参，用于记录不同blockIdx下seq的长度，公式中的输入/输出seqLens。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape长度与numSeqs一致。</li><li>取值范围是大于0的正整数。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
      <tr>
      <td>slotMapping</td>
      <td>输入/输出</td>
      <td>待进行AdvanceStepV2计算的入参/出参，公式中的输出slotMapping，用于将token值在序列中的位置映射到物理位置。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape长度与numSeqs一致。</li><li>取值范围是大于0的正整数。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockTables</td>
      <td>输入</td>
      <td>待进行AdvanceStepV2计算的入参，用于记录不同blockIdx下block的大小，公式中的输入blockTables。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape长度与numSeqs一致，第二维大于（seqLens中的最大值）/blockSize。</li><li>取值范围是大于0的正整数。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
      <tr>
      <td>specToken</td>
      <td>输入</td>
      <td>待进行AdvanceStepV2计算的入参，用于记录当前投机模型的token的index。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape第一维长度与numSeqs一致，第二维长度为specNum。</li><li>取值范围是大于0的正整数。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
      <tr>
      <td>acceptedNum</td>
      <td>输入</td>
      <td>待进行AdvanceStepV2计算的入参，用于记录每个request接受的投机的数量。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape长度与numSeqs一致。</li><li>取值范围是大于0的正整数。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
      <tr>
      <td>numSeqs</td>
      <td>输入</td>
      <td>记录输入的seq数量，大小与seqLens的长度一致。</td>
      <td><ul><li>取值范围是大于0的正整数。</li><li>numSeqs的值大于输入numQueries的值。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>numQueries</td>
      <td>输入</td>
      <td>记录输入的Query的数量，大小与sampledTokenIds第一维的长度一致。</td>
      <td>取值范围是大于0的正整数。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>blockSize</td>
      <td>输入</td>
      <td>每个block的大小，对应公式中的blockSize。</td>
      <td>取值范围是大于0的正整数。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
  </tbody>
  </table>


- **返回值：**
aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。
第一段接口会完成入参校验，出现以下场景时报错：
<table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
<col style="width: 319px">
<col style="width: 108px">
<col style="width: 621px">
</colgroup>
<thead>
  <tr>
    <th>返回码</th>
    <th>错误码</th>
    <th>描述</th>
  </tr></thead>
<tbody>
  <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>传入的inputTokens、sampledTokenIds、inputPositions、seqLens、slotMapping、blockTables、specToken、accepetdNum是空指针。</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>inputTokens、sampledTokenIds、inputPositions、seqLens、slotMapping、blockTables、specToken、accepetdNum的数据类型不在支持的范围之内。</td>
  </tr>
  <tr>
    <td rowspan="5">aclnnAdvanceStepV2GetWorkspaceSize failed</td>
    <td rowspan="5">561002</td>
    <td>输入inputTokens、inputPositions、seqLens、slotMapping、blockTables、specToken、accepetdNum的shape的第一维长度与numSeqs不一致。</td>
  </tr>
  <tr>
    <td>输入sampledTokenIds的shape的第一维长度与numQueries不一致，或者shape的第二维长度不为1。</td>
  </tr>
  <tr>
    <td>输入inputTokens的shape的第二维长度不为1+specNum。</td>
  </tr>
    <tr>
    <td>输入specToken的shape的第二维长度不为specNum。</td>
  </tr>
    <tr>
    <td>输入numSeqs的值不等于输入numQueries的值。</td>
  </tr>
</tbody>
</table>


## aclnnAdvanceStepV2

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAdvanceStepV2GetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

无。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
#include <unistd.h>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_advance_step_V2.h"//不确定头文件名字
#define CHECK_RET(cond, return_expr) \
    do {                               \
    if (!(cond)) {                   \
        return_expr;                   \
    }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
    printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
    shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
    auto size = GetShapeSize(shape);
    std::vector<int64_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                        *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %ld\n", i, resultData[i]);
    }
}

int Init(int64_t deviceId, aclrtStream* stream) {
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> inputShape = {8,1};
    std::vector<int64_t> input2Shape = {8,3};
    std::vector<int64_t> input3Shape = {8,2};
    std::vector<int64_t> inputHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int64_t> input2HostData = {{0, 1, 2, 3, 4, 5, 6, 7},
                                           {0, 1, 2, 3, 4, 5, 6, 7},
                                           {0, 1, 2, 3, 4, 5, 6, 7}};
    std::vector<int64_t> input3HostData = {{0, 1, 2, 3, 4, 5, 6, 7},
                                           {0, 1, 2, 3, 4, 5, 6, 7}};
    void* input1DeviceAddr = nullptr;
    aclTensor* input1 = nullptr;
    void* input2DeviceAddr = nullptr;
    aclTensor* input2 = nullptr;
    void* input3DeviceAddr = nullptr;
    aclTensor* input3 = nullptr;
    void* input4DeviceAddr = nullptr;
    aclTensor* input4 = nullptr;
    void* input5DeviceAddr = nullptr;
    aclTensor* input5 = nullptr;
    void* input6DeviceAddr = nullptr;
    aclTensor* input6 = nullptr;
    void* input7DeviceAddr = nullptr;
    aclTensor* input7 = nullptr;
    void* input8DeviceAddr = nullptr;
    aclTensor* input8 = nullptr;
    // 创建input aclTensor
    ret = CreateAclTensor(input2HostData, input2Shape, &input1DeviceAddr, aclDataType::ACL_INT64, &input1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input2HostData, input2Shape, &input2DeviceAddr, aclDataType::ACL_INT64, &input2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(inputHostData, inputShape, &input3DeviceAddr, aclDataType::ACL_INT64, &input3);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(inputHostData, inputShape, &input4DeviceAddr, aclDataType::ACL_INT64, &input4);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(inputHostData, inputShape, &input5DeviceAddr, aclDataType::ACL_INT64, &input5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(inputHostData, inputShape, &input6DeviceAddr, aclDataType::ACL_INT64, &input6);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input3HostData, input3Shape, &input5DeviceAddr, aclDataType::ACL_INT64, &input7);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(inputHostData, inputShape, &input6DeviceAddr, aclDataType::ACL_INT64, &input8);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int64_t numseq = 8;
    int64_t specnum = 2;
    int64_t blocksize = 2;

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 16 * 1024 * 1024;
    aclOpExecutor* executor;

    // 调用aclnnAdvanceStepV2第一段接口
    ret = aclnnAdvanceStepV2GetWorkspaceSize(
    input1,input2,input3,input4,input5,input6,input7,input8,
    numseq,numseq,blocksize,
    &workspaceSize,
    &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdvanceStepV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnAdvanceStepV2第二段接口
    ret = aclnnAdvanceStepV2(
    workspaceAddr,
    workspaceSize,
    executor,
    stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdvanceStepV2 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    PrintOutResult(inputShape, &input1DeviceAddr);
    PrintOutResult(inputShape, &input3DeviceAddr);
    PrintOutResult(inputShape, &input4DeviceAddr);
    PrintOutResult(inputShape, &input5DeviceAddr);

    // 6. 释放aclTensor和aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(input1);
    aclDestroyTensor(input2);
    aclDestroyTensor(input3);
    aclDestroyTensor(input4);
    aclDestroyTensor(input5);
    aclDestroyTensor(input6);
    aclDestroyTensor(input7);
    aclDestroyTensor(input8);

    // 7.释放device资源，需要根据具体API的接口定义修改
    aclrtFree(input1DeviceAddr);
    aclrtFree(input2DeviceAddr);
    aclrtFree(input3DeviceAddr);
    aclrtFree(input4DeviceAddr);
    aclrtFree(input5DeviceAddr);
    aclrtFree(input6DeviceAddr);
    aclrtFree(input7DeviceAddr);
    aclrtFree(input8DeviceAddr);

    if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```