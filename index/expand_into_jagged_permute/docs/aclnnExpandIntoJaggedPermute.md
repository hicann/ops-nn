# aclnnExpandIntoJaggedPermute

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/index/expand_into_jagged_permute)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |     √   |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> | √ |


## 功能说明

- **接口功能**：将稀疏数据置换索引从表维度扩展到批次维度，适用于稀疏特征在不同rank中具有不同批次大小的情况。
- **计算公式**：

$$
len = outputOffset[i+1] - outputOffset[i]
$$

$$
outputPermuteOut[outputOffset[i]:outputOffset[i+1]] = arange(inputOffset[permute[i]],inputOffset[permute[i]]+len)
$$


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnExpandIntoJaggedPermuteGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnExpandIntoJaggedPermute”接口执行计算。

```c++
aclnnStatus aclnnExpandIntoJaggedPermuteGetWorkspaceSize(
    const aclTensor *permute,
    const aclTensor *inputOffset,
    const aclTensor *outputOffset,
    int64_t          outputSize,
    const aclTensor *outputPermuteOutOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```
```c++
aclnnStatus aclnnExpandIntoJaggedPermute(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnExpandIntoJaggedPermuteGetWorkspaceSize

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 187px">
    <col style="width: 121px">
    <col style="width: 287px">
    <col style="width: 387px">
    <col style="width: 187px">
    <col style="width: 187px">
    <col style="width: 187px">
    <col style="width: 146px">
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
        <td>permute</td>
        <td>输入</td>
        <td>公式中的输入permute，代表表级别的置换索引。</td>
        <td>要求为一维的Tensor，取值范围为[0,inputLen-1]。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>inputLen</td>
        <td>√</td>
    </tr>
    <tr>
        <td>inputOffset</td>
        <td>输入</td>
        <td>公式中的inputOffset，代表表级别长度的互斥偏移量。</td>
        <td>要求为一维的Tensor，Tensor的值严格单调递增且最后一个值等于outputSize。</td>
        <td>与permute一致。</td>
        <td>ND</td>
        <td>inputLen+1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>outputOffset</td>
        <td>输入</td>
        <td>公式中的outputOffset，表示第一个共享专家。</td>
        <td>要求为一维的Tensor，Tensor的值严格单调递增且最后一个值等于outputSize。</td>
        <td>与permute一致。</td>
        <td>ND</td>
        <td>inputLen+1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>outputSize</td>
        <td>输入</td>
        <td>代表输出结果的长度。</td>
        <td>-</td>
        <td>INT32、INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>outputPermute</td>
        <td>输出</td>
        <td>公式中的输出。</td>
        <td>shape为(outputSize)。</td>
        <td>与permute一致。</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
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
    </tbody></table>

- **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
        <col style="width: 267px">
        <col style="width: 124px">
        <col style="width: 775px">
        </colgroup>
        <thead>
            <tr>
            <th>返回值</th>
            <th>错误码</th>
            <th>描述</th>
            </tr>
        </thead>
        <tbody>
            <tr>
            <td> ACLNN_ERR_PARAM_NULLPTR </td>
            <td> 161001 </td>
            <td>传入的必选输入、必选输出或者必选属性，是空指针。</td>
            </tr>
            <tr>
            <td> ACLNN_ERR_PARAM_INVALID </td>
            <td> 161002 </td>
            <td>输入和输出的数据类型和数据格式不在支持的范围之内。</td>
            </tr>
            <tr>
            <td rowspan="2"> ACLNN_ERR_INNER_TILING_ERROR </td>
            <td rowspan="2"> 561002 </td>
            <td>多个输入tensor之间的shape信息不匹配。</td>
            </tr>
            <tr>
            <td>输入属性和输入tensor之间的shape信息不匹配。</td>
            </tr>
        </tbody></table>

## aclnnExpandIntoJaggedPermute

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
        <col style="width: 173px">
        <col style="width: 133px">
        <col style="width: 860px">
        </colgroup>
            <thead>
                <tr>
                <th>参数名</th>
                <th>输入/输出</th>
                <th>描述</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                <td>workspace</td>
                <td>输入</td>
                <td>在Device侧申请的workspace内存地址。</td>
                </tr>
                <tr>
                <td>workspaceSize</td>
                <td>输入</td>
                <td>在Device侧申请的workspace大小，由第一段接口aclnnExpandIntoJaggedPermuteGetWorkspaceSize获取。</td>
                </tr>
                <tr>
                <td>executor</td>
                <td>输入</td><td> op执行器，包含了算子计算流程。</td>
                </tr>
                <tr>
                <td>stream</td>
                <td>输入</td>
                <td> 指定执行任务的Stream。</td>
                </tr>
            </tbody>
        </table>

- **返回值**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

1.inputOffset、outputOffset的shape要相同。

2.permute、inputOffset、outputOffset、outputPermuteOut的数据类型需要相同。

3.outputOffset的值要求严格单调递增且最后一个值等于outputSize。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include "aclnnop/aclnn_expand_into_jagged_permute.h"
#include <iostream>
#include <vector>
#include <sys/stat.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cassert>
#include <iomanip>
#include <unistd.h>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}


template <typename T>
bool ReadFile(const std::string &filePath, std::vector<int64_t> shape, std::vector<T>& hostData)
{
    size_t fileSize = 1;
    for (int64_t i : shape){
        fileSize *= i; 
    }
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
    // 获取文件大小
    file.seekg(0, std::ios::end);
    file.seekg(0, std::ios::beg);
    hostData.reserve(fileSize);
    if (file.read(reinterpret_cast<char*>(hostData.data()), fileSize * sizeof(T))) {
    } else {
        std::cerr << "读取文件失败" << std::endl;
        return 1;
    }
    file.close();
    return true;
}

template <typename T>
bool WriteFile(const std::string &filePath, int64_t size, std::vector<T>& hostData)
{
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        LOG_PRINT("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    size_t writeSize = write(fd, reinterpret_cast<char*>(hostData.data()), size * sizeof(T));
    (void)close(fd);
    if (writeSize != size * sizeof(T)) {
        LOG_PRINT("Write file Failed.");
        return false;
    }

    return true;
}
void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
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
  // 1. （固定写法）device/stream初始化, 参考acl API
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. 构造输入与输出, 需要根据API的接口自定义构造
  std::vector<int64_t> permuteShape = {3};
  std::vector<int64_t>inputOffsetsShape = {4};
  std::vector<int64_t> outputOffsetsShape = {4};
  std::vector<int64_t> outputPermuteShape= {6};
  void* permuteDeviceAddr = nullptr;
  void* inputOffsetsDeviceAddr = nullptr;
  void* outputOffsetsDeviceAddr = nullptr;
  void* outputPermuteDeviceAddr = nullptr;

  aclTensor* permute = nullptr;
  aclTensor* inputOffsets = nullptr;
  aclTensor* outputOffsets = nullptr;
  aclTensor* outputPermute = nullptr;
  int64_t outputSize = 6;


  std::vector<int32_t> permuteHostData = {1, 0, 2};
  std::vector<int32_t> inputOffsetsHostData = {0, 3, 5, 8};
  std::vector<int32_t>outputOffsetsHostData = {0, 2, 4, 6};
  std::vector<int32_t> outputPermuteHostData = {3, 4, 0, 1, 5, 6};

  ret = CreateAclTensor(permuteHostData, permuteShape,
                        &permuteDeviceAddr, aclDataType::ACL_INT32,
                        &permute);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(inputOffsetsHostData, inputOffsetsShape, &inputOffsetsDeviceAddr,
                      aclDataType::ACL_INT32, &inputOffsets);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outputOffsetsHostData, outputOffsetsShape, &outputOffsetsDeviceAddr,
                      aclDataType::ACL_INT32, &outputOffsets);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(outputPermuteHostData , outputPermuteShape , &outputPermuteDeviceAddr,
                      aclDataType::ACL_INT32, &outputPermute);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API, 需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 调用aclnnExpandIntoJaggedPermute第一段接口
  ret = aclnnExpandIntoJaggedPermuteGetWorkspaceSize(permute, inputOffsets, outputOffsets, 
                                               outputSize, outputPermute, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnExpandIntoJaggedPermuteGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // 调用aclnnExpandIntoJaggedPermute第二段接口
  ret = aclnnExpandIntoJaggedPermute(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnExpandIntoJaggedPermute failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.获取输出的值,将device侧内存上的结果拷贝至host侧,需要根据具体API的接口定义修改
  PrintOutResult(outputPermuteShape, &outputPermuteDeviceAddr);

  // 6. 释放aclTensor和aclScalar,需要根据具体API的接口定义修改
  aclDestroyTensor(permute);
  aclDestroyTensor(inputOffsets);
  aclDestroyTensor(outputOffsets);
  aclDestroyTensor(outputPermute);

  // 7. 释放device资源
  aclrtFree(permuteDeviceAddr);
  aclrtFree(inputOffsetsDeviceAddr);
  aclrtFree(outputOffsetsDeviceAddr);
  aclrtFree(outputPermuteDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```