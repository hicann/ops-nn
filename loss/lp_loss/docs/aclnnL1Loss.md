# aclnnL1Loss

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/loss/lp_loss)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnL1LossGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnL1Loss”接口执行计算。

- `aclnnStatus aclnnL1LossGetWorkspaceSize(const aclTensor* self, const aclTensor* target, int64_t reduction, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnL1Loss(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## 功能说明

- 算子功能：计算输入self和目标target中每个元素之间的平均绝对误差（Mean Absolute Error，简称MAE）。reduction指定要应用到输出的缩减，支持 'none'、'mean'、'sum'。'none' 表示不应用缩减，'mean' 表示输出的总和将除以输出中的元素数，'sum' 表示输出将被求和。

- 计算公式：

  当`reduction`为`none`时
  
  $$
  \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
  l_n = \left| x_n - y_n \right|,
  $$
  
  其中$x$是self，$y$是target，$N$是batch的大小。如果`reduction`不是`'none'`, 那么
  
  $$
  \ell(x, y) =
  \begin{cases}
      \operatorname{mean}(L), &  \text{if reduction} = \text{'mean';}\\
      \operatorname{sum}(L),  &  \text{if reduction} = \text{'sum'.}
  \end{cases}
  $$

## aclnnL1LossGetWorkspaceSize

- **参数说明：**

  - self (aclTensor*，计算输入)：公式中的输入`self`，Device侧的aclTensor。数据类型与target的数据类型满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。shape支持0-8维，shape需要与target满足[broadcast规则](../../../docs/zh/context/broadcast关系.md)。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT16、FLOAT32、INT64。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32、INT64。
  - target (aclTensor*，计算输入)：公式中的输入`target`，Device侧的aclTensor。数据类型与self的数据类型满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)）。shape支持0-8维，shape需要与self满足[broadcast规则](../../../docs/zh/context/broadcast关系.md)。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT16、FLOAT32、INT64。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32、INT64。
  - reduction (int64_t，计算输入，为属性)：用于指定要应用到输出的缩减公式中的输入，公式中的`reduction`，Host侧的整型。支持 0('none') | 1('mean') | 2('sum')。'none' 表示不应缩减，'mean' 表示输出的总和将除以输出中的元素数，'sum' 表示输出将被求和。
  - out (aclTensor*，计算输出)：公式中的`out`，Device侧的aclTensor。且数据类型需要是self与target推导之后可转换的数据类型（参见[互转换关系](../../../docs/zh/context/互转换关系.md)）。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。当reduction的值为0时，out的shape与self和target broadcast后的shape一致；当reduction的值不为0时，out的shape支持0维。
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT16、FLOAT32、INT64、COMPLEX64、COMPLEX128。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32、INT64、COMPLEX64、COMPLEX128。
  - workspaceSize (uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。
  - executor (aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR) : 1. 传入的self、target或out是空指针时。
  返回161002 (ACLNN_ERR_PARAM_INVALID) : 1. self和target的数据类型不满足数据类型推导规则，或者推导后的dtype不在支持的范围之内。
                                        2. self和target进行数据类型推导后的类型无法cast为out的数据类型。
                                        3. self或target的维度大于8。
                                        4. self和target的shape不满足broadcast规则。
                                        5. reduction值不在0~2范围之内。
                                        6. 当reduction的值为0时，self和target broadcast后的shape与out的shape不一致。
                                        7. 当reduction的值不为0时，out的维度大于0。
                                        8. 当reduction为none、self不是浮点数时，那target也不能是浮点数。
                                        9. 当reduction为mean时，self和target至少有一个是浮点数。
  ```

## aclnnL1Loss

- **参数说明：**

  - workspace(void*，入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t，入参)：在Device侧申请的workspace大小，由第一段接口aclnnL1LossGetWorkspaceSize获取。
  - executor(aclOpExecutor*，入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream，入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性计算： 
    - aclnnL1Loss默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_l1_loss.h"

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
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> targetShape = {2, 2};
  std::vector<int64_t> outShape = {};
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> targetHostData = {1, 1, 1, 1};
  std::vector<float> outHostData = {0};
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建target aclTensor
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建reduction
  int64_t reduction = 1;

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnL1Loss第一段接口
  ret = aclnnL1LossGetWorkspaceSize(self, target, reduction, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnL1LossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnL1Loss第二段接口
  ret = aclnnL1Loss(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnL1Loss failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
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
