# aclnnSmoothL1LossBackward

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/loss/smooth_l1_loss_grad_v2)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |      √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |   ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |
## 功能说明

- 算子功能：计算[aclnnSmoothL1Loss](../../smooth_l1_loss_v2/docs/aclnnSmoothL1Loss.md) api的反向传播。
- 计算公式：

  SmoothL1Loss的反向传播可以通过求导计算。对于SmoothL1Loss的第一种情况，即|x-y|<1，其导数为：
  $$
  \frac{\partial SmoothL1Loss(x,y)}{\partial x} = x - y
  $$

  对于SmoothL1Loss的第二种情况，即|x-y|≥1，其导数为：
  $$
  \frac{\partial SmoothL1Loss(x,y)}{\partial x} = sign(x - y)
  $$
  
  其中sign(x)表示x的符号函数，即：
  $$
  sign(x) =\begin{cases}
  1,& if　x>0 \\
  0,& if　x=0 \\
  -1,& if　x<0
  \end{cases}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnSmoothL1LossBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSmoothL1LossBackward”接口执行计算。

- `aclnnStatus aclnnSmoothL1LossBackwardGetWorkspaceSize(const aclTensor* gradOut, const aclTensor* self, const aclTensor* target, int64_t reduction, float beta, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnSmoothL1LossBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnSmoothL1LossBackwardGetWorkspaceSize

- **参数说明：**

  - gradOut（aclTensor*，计算输入）：梯度反向输入，公式中的`SmoothL1Loss`，Device侧的aclTensor。shape需要与self，target满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)，数据类型与self、target的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)），支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)， [数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT16、FLOAT32。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。

  - self（aclTensor*，计算输入）：公式中的`x`，Device侧的aclTensor。shape需要与gradOut，target满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)，数据类型与gradOut、target的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)），支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)， [数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT16、FLOAT32。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。

  - target（aclTensor*，计算输入）：公式中的`y`，Device侧的aclTensor。shape需要与gradOut，self满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)，数据类型与gradOut、target的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)），支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)， [数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT16、FLOAT32。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。

  - reduction（int64_t，计算输入）：计算属性，指定要应用到输出的缩减，Host侧的整型。取值支持0('none')|1('mean')|2('sum')。'none'表示不应用减少，'mean'表示输出的总和将除以输出中的元素数，'sum'表示输出将被求和。

  - beta（float，计算输入）：计算属性，指定在L1和L2损失之间更改的数值，数据类型为float，该值必须是非负的。

  - gradInput（aclTensor*，计算输出）：shape为gradOut，self，target的broadcast结果，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：数据类型支持FLOAT16、FLOAT32。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。

  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。

  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：1.传入的self、target、gradOut或gradInput为空指针
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. self、target、gradOut或gradInput的数据类型不在支持的范围之内
                                        2. self、target、gradOut或gradInput的shape不符合约束
                                        3. reduction不符合约束
                                        4. beta不符合约束
  ```
## aclnnSmoothL1LossBackward

- **参数说明：**

  - workspace（void*，入参）：在Device侧申请的workspace内存地址。

  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnSmoothL1LossBackwardGetWorkspaceSize获取。

  - executor(aclOpExecutor *，入参)：op执行器，包含了算子计算流程。
  
  - stream(aclrtStream，入参)：指定执行任务的Stream。


- **返回值：**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性计算： 
    - aclnnSmoothL1LossBackward默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_smooth_l1_loss_backward.h"

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
  std::vector<int64_t> gradOutShape = {4, 2};
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> targetShape = {4, 2};
  std::vector<int64_t> gradInputShape = {4, 2};
  int64_t reduction = 0;
  float beta = 1.0;
  void* gradOutDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* gradInput = nullptr;
  std::vector<float> gradOutHostData = {1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> targetHostData = {1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<float> gradInputHostData(8, 0);
  // 创建gradOut aclTensor
  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
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

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnSmoothL1LossBackward第一段接口
  ret = aclnnSmoothL1LossBackwardGetWorkspaceSize(gradOut, self, target, reduction, beta, gradInput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSmoothL1LossBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnSmoothL1LossBackward第二段接口
  ret = aclnnSmoothL1LossBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSmoothL1LossBackward failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(gradOut);
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(gradInput);
  // 7. 释放device资源，需要根据具体API的接口定义参数
  aclrtFree(gradOutDeviceAddr);
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

