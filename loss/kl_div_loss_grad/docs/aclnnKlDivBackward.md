# aclnnKlDivBackward
## 产品支持情况
[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/loss/kl_div_loss_grad)


| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

算子功能：进行[aclnnKlDiv](https://gitcode.com/cann/ops-math/blob/master/math/kl_div_v2/docs/aclnnKlDiv.md) api的结果的反向计算。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnKlDivBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnKlDivBackward”接口执行计算。

- `aclnnStatus aclnnKlDivBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, int64_t reduction, bool logTarget, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnKlDivBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnKlDivBackwardGetWorkspaceSize

- **参数说明：**

  - gradOutput(aclTensor*, 计算输入)：Device侧的aclTensor，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。shape需要与self满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT、FLOAT16。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - self(aclTensor*, 计算输入)：Device侧的aclTensor。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT、FLOAT16。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - target(aclTensor*, 计算输入)：Device侧的aclTensor，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。shape需要与self满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT、FLOAT16。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - reduction(int64_t, 计算输入)：Host侧的int64_t，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。指定要应用到输出的缩减。支持0（‘none’）| 1（‘mean’）| 2（‘sum’）|3（‘batchmean’）。‘none’表示不应用减少，‘mean’表示输出的总和将除以输出中的元素数，‘sum’表示输出将被求和，‘batchmean’表示输出的总和将除以batch的个数。
  - logTarget(bool, 计算输入)：Host侧的BOOL类型，是否对target进行log空间转换。
  - out(aclTensor*, 计算输出)：Device侧的aclTensor。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT、FLOAT16。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - workspaceSize(uint64_t*, 出参): 返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor**, 出参): 返回op执行器，包含了算子计算流程。
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_INVALID）: 1. 传入的gradOutput、self、target和out是空指针时。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. gradOutput、self、target和out的数据类型不在支持的范围内时。
                                        2. gradOutput、self、target的数据类型不一致。
                                        3. gradOutput的shape不能向self或者target broadcast。
                                        4. target的shape和self的shape不满足broadcast关系。
                                        5. self的shape与out的shape不相同。
                                        6. gradOutput、self、target或者out维度大于8。
  ```

## aclnnKlDivBackward

- **参数说明：**

  * workspace（void\*, 入参）：在Device侧申请的workspace内存地址。
  * workspaceSize（uint64\_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnKlDivBackwardGetWorkspaceSize获取。
  * executor（aclOpExecutor\*, 入参）：op执行器，包含了算子计算流程。
  * stream（aclrtStream, 入参）：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性计算： 
    - aclnnKlDivBackward默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_kl_div_backward.h"

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

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> gradOutputShape = {2, 2};
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> targetShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> gradOutputHostData = {2, 3, 5, 8};
  std::vector<float> selfHostData = {2, 3, 5, 8};
  std::vector<float> targetHostData = {2, 3, 5, 8};
  std::vector<float> outHostData = {2, 3, 5, 8};
  int64_t reduction = 0;
  bool logTarget = false;
  // 创建gradOutput aclTensor
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建target aclTensor
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnKlDivBackward第一段接口
  ret = aclnnKlDivBackwardGetWorkspaceSize(gradOutput, self, target, reduction, logTarget, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKlDivBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnKlDivBackward第二段接口
  ret = aclnnKlDivBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnKlDivBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("resultData[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

