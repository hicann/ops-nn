# aclnnSmoothL1Loss

## 产品支持情况
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>。

## 功能说明

- 算子功能：计算SmoothL1损失函数。
- 计算公式:

  Batch为N的损失函数，当`reduction`为none时，此函数定义为：

  $$
  \ell(self,target) = L = \{l_1,\dots,l_N\}^\top
  $$

  其中的$l_n$为：

  $$
  l_n = \begin{cases}
  0.5(self_n-target_n)^2/beta, & if |self_n-target_n| < beta \\
  |self_n-target_n| - 0.5*beta, &  otherwise
  \end{cases}
  $$

  如果`reduction`为`mean`或`sum`时，

  $$
  \ell(self,target)=\begin{cases}
  mean(L), & \text{if reduction} = \text{mean}\\
  sum(L), & \text{if reduction} = \text{sum}
  \end{cases}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnSmoothL1LossGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSmoothL1Loss”接口执行计算。

- `aclnnStatus aclnnSmoothL1LossGetWorkspaceSize(const aclTensor* self, const aclTensor* target, int64_t reduction, float beta, aclTensor* result, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnSmoothL1Loss(void* workspace, uint64_t workspaceSize,  aclOpExecutor* executor, aclrtStream stream)`

## aclnnSmoothL1LossGetWorkspaceSize
- **参数说明：**

  - self（aclTensor*，计算输入）：公式中的`self`，Device侧的aclTensor。 shape需要与target满足[broadcast关系](../../../docs/context/broadcast关系.md)且最高支持8维，数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)），支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)， [数据格式](../../../docs/context/数据格式.md)支持ND、NCL、NCHW、NHWC。

    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。

  - target（aclTensor*，计算输入）：公式中的`target`，Device侧的aclTensor。shape需要与self满足[broadcast关系](../../../docs/context/broadcast关系.md)且最高支持8维，数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)），支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)， [数据格式](../../../docs/context/数据格式.md)支持ND、NCL、NCHW、NHWC。

    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。

  - reduction（int64_t，计算输入）：用于指定要应用到输出的缩减公式中的输入，公式中的`reduction`，Host侧的整型。取值支持0('none')|1('mean')|2('sum')，其中'none'表示不应用减少，'mean'表示输出的总和将除以输出中的元素数，'sum'表示输出将被求和。

  - beta（float，计算输入）：数据类型支持FLOAT。

  - result（aclTensor*，计算输出）：公式中输出的损失函数$\ell$，当`reduction`为`none`时，shape与self和target的broadcast结果一致，当`reduction`为`mean`或`sum`时为[ ]，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND、NCL、NCHW、NHWC。

    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。

  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。

  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：传入的self、target或result为空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. self、target或result的数据类型不在支持的范围之内。
                                        2. self、target或result的shape不符合约束。
                                        3. reduction不符合约束。
                                        4. beta不符合约束。
                                        5. self和target的数据类型不满足数据类型推导规则。
                                        6. self和target的shape不满足broadcast关系。
                                        7. reduction=0时，result的shape与self和target的broadcast的shape不一致。
  ```
## aclnnSmoothL1Loss

- **参数说明：**

  - workspace（void*，入参）：在Device侧申请的workspace内存地址。

  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnSmoothL1LossGetWorkspaceSize获取。

  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。

  - stream（aclrtStream，入参）：指定执行任务的Stream。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)

## 约束说明
无

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_smooth_l1_loss.h"

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
  std::vector<int64_t> selfShape = {2, 2, 7, 7};
  std::vector<int64_t> targetShape = {2, 2, 7, 7};
  std::vector<int64_t> resultShape = {2, 2, 7, 7};

  // 创建self aclTensor
  std::vector<float> selfData(GetShapeSize(selfShape)* 2, 1);
  aclTensor* self = nullptr;
  void *selfDeviceAddr = nullptr;
  ret = CreateAclTensor(selfData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建target aclTensor
  std::vector<float> targetData(GetShapeSize(targetShape)* 2, 1);
  aclTensor* target = nullptr;
  void *targetDeviceAddr = nullptr;
  ret = CreateAclTensor(targetData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT16, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建result aclTensor
  std::vector<float> resultData(GetShapeSize(resultShape)* 2, 1);
  aclTensor* result = nullptr;
  void *resultDeviceAddr = nullptr;
  ret = CreateAclTensor(resultData, resultShape, &resultDeviceAddr, aclDataType::ACL_FLOAT16, &result);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnSmoothL1Loss第一段接口
  int64_t reduction = 0;
  float beta = 1.0;
  ret = aclnnSmoothL1LossGetWorkspaceSize(self, target, reduction, beta, result, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSmoothL1LossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnSmoothL1Loss第二段接口
  ret = aclnnSmoothL1Loss(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSmoothL1Loss failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(resultShape);
  std::vector<float> resultOutData(size, 0);
  ret = aclrtMemcpy(resultOutData.data(), resultOutData.size() * sizeof(resultOutData[0]), resultDeviceAddr,
                    size * sizeof(resultOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultOutData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(result);
  // 7. 释放device资源，需要根据具体API的接口定义参数
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(resultDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

