# aclnnMaxUnpool3dBackward

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：MaxPool3d的逆运算[aclnnMaxUnpool3d](../../scatter_elements/docs/aclnnMaxUnpool3d.md)的反向传播，根据indices索引在out中填入gradOutput的元素值。
- 计算公式：
  - 输入为4维时，各维度含义分别为(N, D, H, W)：

  $$
  out[N][i] = gradOutput[N][indices[N][i]]
  $$

  - 输入为5维时，各维度含义分别为(N, C, D, H, W)：

  $$
  out[N][C][i] = gradOutput[N][C][indices[N][C][i]]
  $$

  其中out、gradOutput、indices是最后两轴合为一轴，经过reshape得到的，i ∈ [0, D * H * W)。

## 函数原型

  每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMaxUnpool3dBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMaxUnpool3dBackward”接口执行计算。

  - `aclnnStatus aclnnMaxUnpool3dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclIntArray* outputSize, const aclIntArray* stride, const aclIntArray* padding, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
  - `aclnnStatus aclnnMaxUnpool3dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnMaxUnpool3dBackwardGetWorkspaceSize

- **参数说明**

  - gradOutput（aclTensor\*, 计算输入）：公式中的输入`gradOutput`，Device侧的aclTensor。数据类型支持FLOAT、FLOAT16、INT16、INT32、INT64、INT8、UINT8和DOUBLE，且数据类型需要与self、out一致，维度支持4-5维，各维度含义分别为(N, outputSize[0], outputSize[1], outputSize[2])或(N, C, outputSize[0], outputSize[1], outputSize[2])，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND和NCDHW。
  - self（aclTensor\*, 计算输入）：Device侧的aclTensor。数据类型支持FLOAT、FLOAT16、INT16、INT32、INT64、INT8、UINT8和DOUBLE，且数据类型需要与gradOutput、out一致，维度支持4-5维，各维度含义分别为(N, D, H, W)或(N, C, D, H, W)，维度需要与gradOutput一致，shape需要与indices、out一致，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)， [数据格式](../../../docs/zh/context/数据格式.md)支持ND和NCDHW。
  - indices（aclTensor\*, 计算输入）：表示输入gradOutput的元素在输出结果中的索引位置，公式中的输入`indices`，Device侧的aclTensor。数据类型支持INT64，且shape需要与self一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)， [数据格式](../../../docs/zh/context/数据格式.md)支持ND和NCDHW。
  - outputSize（aclIntArray\*, 计算输入）：表示输出结果在D、H和W维度上的空间大小，Host侧的aclIntArray，数据类型支持INT64，size大小为3。
  - stride（aclIntArray\*, 计算输入）：表示最大池化窗口在D、H和W维度上的步长大小，Host侧的aclIntArray，数据类型支持INT64，size大小为3。
  - padding（aclIntArray\*, 计算输入）：表示最大池化窗口在D、H和W维度上的填充值，Host侧的aclIntArray，数据类型支持INT64，size大小为3。
  - out（aclTensor\*, 计算输出）：公式中的`out`，Device侧的aclTensor。数据类型支持FLOAT、FLOAT16、INT16、INT32、INT64、INT8、UINT8和DOUBLE，且数据类型与gradOutput、self一致，shape需要与self、indices一致。[数据格式](../../../docs/zh/context/数据格式.md)支持ND和NCDHW。
  - workspaceSize（uint64_t\*, 出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**, 出参）：返回op执行器，包含了算子计算流程。

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的gradOutput、self、indices、outputSize、stride、padding或out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. out为不连续的Tensor。
                                        2. gradOutput、self、indices或out的数据类型不在支持的范围之内。
                                        3. gradOutput、self或out的数据类型不一致。
                                        4. self的维度不为4维或者5维。
                                        5. self、indices或out的维度不一致
                                        6. self、indices或out的shape不一致。
                                        7. self在除N维度外各维度的size大小不大于0。
                                        8. outputSize、stride或padding的size大小不等于3。
                                        9. outputSize或stride的元素值不大于0。
                                       10. outputSize的三个元素乘积值小于self在D、H和W维度上的size乘积值。
                                       11. gradOutput在D、H、W维度上的size大小与outputSize中的三个元素值不相等。
                                       12. gradOutput与self的维度不同。
                                       13. 在self为4维时，gradOutput与self在N维度上的size大小不一致。
                                       14. 在self为5维时，gradOutput与self在C或N维度上的size大小不一致。
  ```

## aclnnMaxUnpool3dBackward

- **参数说明**

  - workspace（void\*, 入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnMaxUnpool3dBackwardGetWorkspaceSize获取。
  - executor（aclOpExecutor\*, 入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream, 入参）: 指定执行任务的Stream。

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnMaxUnpool3dBackward默认确定性实现。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_unpool3d_backward.h"

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
  std::vector<int64_t> selfShape = {1, 1, 4, 4};
  std::vector<int64_t> indicesShape = {1, 1, 4, 4};
  std::vector<int64_t> gradShape = {1, 1, 4, 4};
  std::vector<int64_t> outShape = {1, 1, 4, 4};
  void* gradDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* grad = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* indices = nullptr;
  std::vector<float> gradHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> outHostData = {0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0};
  std::vector<int64_t> indicesHostData = {0, 0, 0, 3, 0, 0, 0, 8, 0, 0, 0, 11, 0, 0, 0, 13};
  // 创建grad aclTensor
  ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建indices aclTensor
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> arraySize1 = {1, 4, 4};
  const aclIntArray *outputSize = aclCreateIntArray(arraySize1.data(), arraySize1.size());
  CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  std::vector<int64_t> arraySize2 = {1, 2, 3};
  const aclIntArray *stride = aclCreateIntArray(arraySize2.data(), arraySize2.size());
  CHECK_RET(stride != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  const aclIntArray *padding = aclCreateIntArray(arraySize2.data(), arraySize2.size());
  CHECK_RET(padding != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. 调用CANN算子库API，需要修改为具体的API名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnMaxUnpool3dBackward第一段接口
  ret = aclnnMaxUnpool3dBackwardGetWorkspaceSize(grad, self, indices, outputSize, stride, padding, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool3dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnMaxUnpool3dBackward第二段接口
  ret = aclnnMaxUnpool3dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool3dBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr, size * sizeof(outData[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out[%ld] is: %f\n", i, outData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(grad);
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(indices);
  aclDestroyIntArray(outputSize);
  aclDestroyIntArray(stride);
  aclDestroyIntArray(padding);

  // 7. 释放divice 资源
  aclrtFree(gradDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(indicesDeviceAddr);
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
