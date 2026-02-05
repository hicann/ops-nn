# aclnnL1LossBackward

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/loss/l1_loss_grad)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnL1LossBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnL1LossBackward”接口执行计算。

- `aclnnStatus aclnnL1LossBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, int64_t reduction, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnL1LossBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能说明

- 算子功能：计算[aclnnL1Loss](../../lp_loss/docs/aclnnL1Loss.md)的反向传播。reduction指定损失函数的计算方式，支持 'none'、'mean'、'sum'。'none' 表示不应用减少，'mean' 表示输出的总和将除以输出中的元素数，'sum' 表示输出将被求和。

## aclnnL1LossBackwardGetWorkspaceSize

- **参数说明：**
  
  - gradOutput (aclTensor*，计算输入)：梯度反向输入，公式中的输入`gradOutput`，Device侧的aclTensor。数据类型与self、target的数据类型满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。shape需要与self、target满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)，且小于9维。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。
  - self (aclTensor*，计算输入)：公式中的输入`self`，Device侧的aclTensor。数据类型与gradOutput、target满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。shape需要与gradOutput、target满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)，且小于9维。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。
  - target (aclTensor*，计算输入)：公式中的输入`target`，Device侧的aclTensor。数据类型要与self、gradOutput满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。shape需要与gradOutput、self满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)，且小于9维。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。
  - reduction (int64_t, 计算输入)：用于指定损失函数的计算方式，Host侧的整型。取值支持 0('none') | 1('mean') | 2('sum')。'none' 表示不应用减少，'mean' 表示输出的总和将除以输出中的元素数，'sum' 表示输出将被求和。
  - gradInput (aclTensor*，计算输出)：公式中的输出`gradInput`，Device侧的aclTensor。数据类型要与self满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。shape需要与target、self、gradOutput满足broadcast关系[broadcast关系](../../../docs/zh/context/broadcast关系.md)，且小于9维。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。
  - workspaceSize (uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。
  - executor (aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。
  
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的gradOutput、self、target或gradInput是空指针时。
  返回161002 (ACLNN_ERR_PARAM_INVALID): 1. self、gradOutput、target的数据类型不满足数据类型推导规则。
                                       2. gradOutput、target、self做数据类型推导后的结果不在支持的范围之内。
                                       3. gradOutput、self和target的shape无法做broadcast。
                                       4. gradOutput、self和target的shape超过8维。
                                       5. gradOutput、self和target做broadcast后的shape与gradInput不一致。
                                       6. reduction值不在0~2范围之内。
                                       7. 计算结果无法cast到gradInput的dtype。
  ```

## aclnnL1LossBackward

- **参数说明：**

  - workspace(void*，入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnL1LossBackwardGetWorkspaceSize获取。
  - executor(aclOpExecutor*，入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream，入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性计算： 
    - aclnnL1LossBackward默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_l1_loss_backward.h"

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

int Init(int32_t deviceId, aclrtStream* stream) {
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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造gradOutput
  std::vector<int64_t> gradOutputShape = {2, 2};
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> targetShape = {2, 2};
  std::vector<int64_t> gradInputShape = {2, 2};
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* gradInput = nullptr;
  std::vector<float> gradOutputHostData = {0, 1, 2, 3};
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> targetHostData = {1, 1, 1, 1};
  std::vector<float> gradInputHostData(4, 0);
  // 创建gradOutput aclTensor
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr,
                        aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建target aclTensor
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gradInput aclTensor
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建reduction
  int64_t reduction = 1;

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnL1LossBackward第一段接口
  ret = aclnnL1LossBackwardGetWorkspaceSize(gradOutput, self, target, reduction, gradInput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnL1LossBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnL1LossBackward第二段接口
  ret = aclnnL1LossBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnL1LossBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(gradInputShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(gradInput);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(gradInputDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
