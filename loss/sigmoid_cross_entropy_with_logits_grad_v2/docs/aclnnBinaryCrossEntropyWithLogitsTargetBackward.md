# aclnnBinaryCrossEntropyWithLogitsTargetBackward

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：
将输入self执行logits计算，将得到的值与标签值target一起进行[BECLoss](../../sigmoid_cross_entropy_with_logits_v2/docs/aclnnBinaryCrossEntropyWithLogits.md)关于target的反向传播计算。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnBinaryCrossEntropyWithLogitsTargetBackward”接口执行计算。

  * `aclnnStatus aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target, const aclTensor *weightOptional, const aclTensor *posWeightOptional, int64_t reduction, aclTensor *gradTarget, uint64_t *workspaceSize, aclOpExecutor **executor)`
  * `aclnnStatus aclnnBinaryCrossEntropyWithLogitsTargetBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize

- **参数说明：**
  * gradOutput(aclTensor \*, 计算输入): 网络反向传播前一步的梯度值，Device侧的aclTensor，数据类型支持FLOAT16、FLOAT、BFLOAT16，shape需要可以[broadcast](../../../docs/zh/context/broadcast关系.md)到self的shape，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  * self(aclTensor \*, 计算输入): 网络正向前一层的计算结果，Device侧的aclTensor，数据类型支持FLOAT16、FLOAT、BFLOAT16，维度小于等于8，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  * target(aclTensor \*, 计算输入): lable标签值，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16，shape必须与self的shape一致，维度小于等于8，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  * weightOptional(aclTensor \*, 计算输入): 二分交叉熵权重，Device侧的aclTensor，数据类型支持FLOAT16、FLOAT、BFLOAT16，shape需要可以[broadcast](../../../docs/zh/context/broadcast关系.md)到self的shape。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，当weightOptional为空时，将以self的shape创建一个全1的Tensor。
  * posWeightOptional(aclTensor \*, 计算输入): 正类的权重，Device侧的aclTensor，数据类型支持FLOAT16、FLOAT、BFLOAT16，shape可以[broadcast](../../../docs/zh/context/broadcast关系.md)到self的shape。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，当weightOptional为空时，将以self的shape创建一个全1的Tensor。
  * reduction(int64_t, 计算输入): 表示对二元交叉熵反向求梯度计算结果做的reduce操作，Host侧的整型值，数据类型支持INT64，仅支持0,1,2三个值，0表示不做任何操作；1表示对结果取平均值；2表示对结果求和。
  * gradTarget(aclTensor\*, 计算输出): 存储梯度计算结果，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16、BFLOAT16，数据类型需要与target相同。shape必须与self的shape一致，维度小于等于8。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，且[数据格式](../../../docs/zh/context/数据格式.md)需要与self一致。
  * workspaceSize(uint64_t \*, 出参): 返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor \*\*, 出参): 返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的gradOutput、self、target、gradTarget为空指针。
  161002（ACLNN_ERR_PARAM_INVALID）：1. gradOutput、self、target、gradTarget的数据类型和数据格式不在支持的范围内。
                                    2. 当weightOptional和posWeightOptional不为空指针，其数据类型和数据格式不在支持的范围内。
                                    3. self、target、gradTarget的shape不一致。
                                    4. 当weightOptional和posWeightOptional不为空指针，其shape不能broadcast到self的shape。
                                    5. gradOutput不能broadcast到self的shape。
                                    6. reduction的值非0, 1, 2三值之一。
                                    7. gradOutput、self、target、gradTarget维度大于8。
  ```

## aclnnBinaryCrossEntropyWithLogitsTargetBackward

- **参数说明：**
  * workspace(void \*, 入参): 在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参): 在Device侧申请的workspace大小，由第一段接口aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize获取。
  * executor(aclOpExecutor \*, 入参): op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参): 指定执行任务的Stream。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性计算：
  - aclnnBinaryCrossEntropyWithLogitsTargetBackward默认确定性实现。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_binary_cross_entropy_with_logits_target_backward.h"

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
  std::vector<int64_t> gradOutputShape = {2, 2};
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> targetShape = {2, 2};
  std::vector<int64_t> weightShape = {2, 2};
  std::vector<int64_t> posWeightShape = {2, 2};
  std::vector<int64_t> gradTargetShape = {2, 2};
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* posWeightDeviceAddr = nullptr;
  void* gradTargetDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* gradTarget = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* posWeight = nullptr;
  std::vector<float> gradOutputHostData = {0, 1, 2, 3};
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> targetHostData = {0.1, 0.1, 0.1, 0.1};
  std::vector<float> weightHostData = {0, 1, 2, 3};
  std::vector<float> posWeightHostData = {0, 1, 2, 3};
  std::vector<float> gradTargetHostData = {0, 0, 0, 0};
  int64_t reduction = 0;

  // 创建gradOutput aclTensor
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建target aclTensor
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建weight aclTensor
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建posWeight aclTensor
  ret = CreateAclTensor(posWeightHostData, posWeightShape, &posWeightDeviceAddr, aclDataType::ACL_FLOAT, &posWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gradTarget aclTensor
  ret = CreateAclTensor(gradTargetHostData, gradTargetShape, &gradTargetDeviceAddr, aclDataType::ACL_FLOAT, &gradTarget);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnBinaryCrossEntropyWithLogitsTargetBackward接口调用示例
  // 3. 调用CANN算子库API，需要修改为具体的API名称
  // 调用aclnnBinaryCrossEntropyWithLogitsTargetBackward第一段接口
  ret = aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize(gradOutput, self, target, weight, posWeight,
      reduction, gradTarget, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize failed. ERROR: %d\n",
                                          ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnBinaryCrossEntropyWithLogitsTargetBackward第二段接口
  ret = aclnnBinaryCrossEntropyWithLogitsTargetBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBinaryCrossEntropyWithLogitsTargetBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(gradTargetShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradTargetDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(weight);
  aclDestroyTensor(posWeight);
  aclDestroyTensor(gradTarget);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(posWeightDeviceAddr);
  aclrtFree(gradTargetDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
