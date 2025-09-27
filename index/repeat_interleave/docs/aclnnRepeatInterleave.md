# aclnnRepeatInterleave

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明
  - 算子功能：将tensor self进行flatten后，重复Tensor repeats中的相应次数。

  - 示例：
    假设input tensor是 ([[a, b], [c, d], [e, f]]), repeats为([1, 2, 2, 1, 1, 1])。
    那么最后生成的tensor为 tensor([a, b, b, c, c, d, e, f])，输出tensor的元素个数为8，与repeats中所有元素之和相同
    将tensor进行flatten后，input转变为 ([a, b, c, d, e, f])。该tensor与repeats一一对应进行复制，a重复1次、b重复2次、c重复2次，以此类推。

    假设input tensor是 ([[a, b], [c, d], [e, f]]), repeats为([2])。 那么最后生成的tensor为 tensor([a, a, b, b, c, c, d, d, e, e, f, f])。
    将tensor进行flatten后，input转变为 ([a, b, c, d, e, f])。该tensor中的每个元素复制repeats中的元素次数，也就是每个元素复制2次。
    注意：该场景等效于 repeats为(2)。

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnRepeatInterleaveGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnRepeatInterleave”接口执行计算。

  - `aclnnStatus aclnnRepeatInterleaveGetWorkspaceSize(const aclTensor* self, const aclTensor* repeats, int64_t outputSize, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnRepeatInterleave(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnRepeatInterleaveGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*, 计算输入)：Device侧的aclTensor，功能说明中待被数据复制的输入tensor。支持空tensor，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)。[数据格式](../../../docs/context/数据格式.md)支持ND，shape支持0-8维。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、BFLOAT16、FLOAT类型。

  - repeats(aclTensor*, 计算输入)：Device侧的aclTensor，重复的次数。数据类型支持INT32、INT64。repeats只能为0D / 1D Tensor。如果为1D Tensor，那么repeats的size必须为1或self的元素个数。支持空tensor，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)。[数据格式](../../../docs/context/数据格式.md)支持ND。

  - outputSize(int64_t, 计算输入)：进行重复后的tensor最终大小。Host侧的int64_t类型。当repeats中只有一个元素时，outputSize = self的元素个数 * repeats的值。当repeats中有多个值时，outputSize = repeats的值之和。

  - out(aclTensor*, 计算输出)：Device侧的aclTensor，功能说明中数据复制完成的输出tensor。数据类型需要与self一致，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)。[数据格式](./../../../docs/context/数据格式.md)支持ND，shape为1D Tensor，shape大小与outputSize相同。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、BFLOAT16、FLOAT类型。

  - workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的self、repeats或out是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID): 1. self、repeats的数据类型不在支持范围内。
                                        2. self、out的数据类型不一样。
                                        3. repeats不为0D / 1D tensor。
                                        4. 当repeats为1D tensors，repeats的size不为1，不为self的元素个数。
                                        5. self的维度数超过8。
                                        6. 当self为空tensor时，repeats不为空tensor，不为0维1元素，不为1维1元素。
                                        7. 当self不为空tensor，但repeats为空tensor时。
  ```

## aclnnRepeatInterleave

- **参数说明：**

  - workspace(void*, 入参)：在Device侧申请的workspace内存地址。

  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnRepeatInterleaveGetWorkspaceSize获取。

  - executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。

  - stream(aclrtStream, 入参)：指定执行任务的Stream。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

在计算时需要满足以下要求：
  - repeats只能为0D / 1D tensor。
    如果为1D tensor，那么repeats的size必须为1或self的元素个数。
    repeats tensor中的值必须为自然数。
  - outputSize的值必须符合以下计算结果：
    当repeats中只有一个元素时，outputSize = self的元素个数 * repeats的值。
    当repeats中有多个值时，outputSize = repeats的值之和。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_repeat_interleave.h"

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
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> repeatsShape = {6};
  std::vector<int64_t> outShape = {21};
  void* selfDeviceAddr = nullptr;
  void* repeatsDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* repeats = nullptr;
  aclTensor* out = nullptr;
  int64_t output_size = 21;
  std::vector<float> selfHostData = {3, 4, 5, -3, -4, -5};
  std::vector<int64_t> repeatsHostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建repeats aclTensor
  ret = CreateAclTensor(repeatsHostData, repeatsShape, &repeatsDeviceAddr, aclDataType::ACL_INT64, &repeats);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnRepeatInterleave第一段接口
  ret = aclnnRepeatInterleaveGetWorkspaceSize(self, repeats, output_size, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleaveGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnRepeatInterleave第二段接口
  ret = aclnnRepeatInterleave(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleave failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(repeats);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(repeatsDeviceAddr);
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

