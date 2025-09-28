# aclnnBinaryCrossEntropy

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：计算self和target的二元交叉熵。

- 计算公式：

  $$
  \ell(self, target)= L = \{l_{1},...,l_{n}\}^{T}, \ell_{n} = -  weight_{n}[target_{n}·log(self_{n}) + (1 - target_{n}·log(1-self_{n}))]
  $$

  当reduction不为None时：

  $$
  \ell(self, target)
  \begin{cases}
  mean(L), & if\ reduction = mean \\
  sum(L), & if\ reduction = sum \\
  \end{cases}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnBinaryCrossEntropyGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnBinaryCrossEntropy”接口执行计算。

  - `aclnnStatus aclnnBinaryCrossEntropyGetWorkspaceSize(const aclTensor* self, const aclTensor* target, const aclTensor* weight, int64_t reduction, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnBinaryCrossEntropy(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnBinaryCrossEntropyGetWorkspaceSize

- **参数说明：**
  - self（aclTensor*, 计算输入）：表示预测的概率值，公式中的输入`self`，Device侧的Tensor。不支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)。[数据格式](../../../docs/context/数据格式.md)支持ND，取值在0~1之间。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。
  - target（aclTensor*, 计算输入）：表示目标张量，公式中的输入`target`，Device侧的Tensor。不支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)。shape需要与`self`一致，[数据格式](../../../docs/context/数据格式.md)支持ND，取值在0~1之间。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型与`self`一致，支持BFLOAT16、FLOAT16、FLOAT32。
  - weight（aclTensor*, 计算输入）：表示权重张量，公式中的输入`weight`，Device侧的Tensor。不支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)。shape需要与`self`一致，[数据格式](../../../docs/context/数据格式.md)支持ND，weight可以是nullptr，等价于所有权重值都是1。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型与`self`一致，支持BFLOAT16、FLOAT16、FLOAT32。
  - reduction（int64_t, 计算输入）：表示规约方式，公式中的输入`reduction`，输出规约的枚举值，支持三种枚举值：当取值为0，即为Reduction::None；当取值为1，即为Reduction::Mean；当取值为2，即为Reduction::Sum。
  - out（aclTensor \*, 计算输出）：表示计算输出，公式中的$\ell(self,target)$，如果reduction = None，shape与`self`一致，其他情况shape为[1]，[数据格式](../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型与`self`一致，支持BFLOAT16、FLOAT16、FLOAT32。
  - workspaceSize（uint64_t \*, 出参）：返回需要在Device侧申请的workspace大小。
- executor（aclOpExecutor \*\*, 出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的self、target或out为空指针。
  161002 (ACLNN_ERR_PARAM_INVALID)：1. self、target、weight或out的数据类型不在支持的范围内。
                                    2. out shape与实际不匹配。
  ```

## aclnnBinaryCrossEntropy

- **参数说明：**
  - workspace（void \*, 入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnBinaryCrossEntropyGetWorkspaceSize获取。
  - executor（aclOpExecutor \*, 入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream, 入参）：指定执行任务的Stream。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明
无

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_binary_cross_entropy.h"

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
  // 调用aclrtMalloc申请Device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
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
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> targetShape = {2, 2};
  std::vector<int64_t> weightShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0.3, 0.3, 0.3, 0.3};
  std::vector<float> targetHostData = {0.5, 0.5, 0.5, 0.5};
  std::vector<float> weightHostData = {1, 1, 1, 1};
  std::vector<float> outHostData = {0, 0, 0, 0};
  int64_t reduction = 0;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建target aclTensor
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建weight aclTensor
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnBinaryCrossEntropy接口调用示例
  // 3. 调用CANN算子库API，需要修改为具体的API名称
  // 调用aclnnBinaryCrossEntropy第一段接口
  ret = aclnnBinaryCrossEntropyGetWorkspaceSize(self, target, weight, reduction, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBinaryCrossEntropyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnBinaryCrossEntropy第二段接口
  ret = aclnnBinaryCrossEntropy(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBinaryCrossEntropy failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
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
  aclDestroyTensor(weight);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(weightDeviceAddr);
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
