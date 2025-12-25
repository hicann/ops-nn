# aclnnIndexPutImpl

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：根据索引 indices 将输入 self 对应坐标的数据与输入 values 进行替换或累加。
- 计算公式：

  - accumulate = False:

    $$
    self[indices] = values
    $$

  - accumulate = True:
    
    $$
    self[indices]  = self[indices]  + values
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnIndexPutImplGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnIndexPutImpl”接口执行计算。

- `aclnnStatus aclnnIndexPutImplGetWorkspaceSize(aclTensor* selfRef, const aclTensorList* indices, const aclTensor* values, const bool accumulate, const bool unsafe, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnIndexPutImpl(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnIndexPutImplGetWorkspaceSize

* **参数说明**：
  
  * selfRef（aclTensor*，计算输入/输出）：公式中的 $self$，Device侧的aclTensor，且数据类型和values一致，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，数据维度支持1-8维。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、BFLOAT16。

  * indices（aclTensorList*，计算输入）：公式中的 $indices$，Device侧的aclTensorList，数据类型支持INT32、INT64、BOOL。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。

  * values（aclTensor*，计算输入）：公式中的 $values$，Device侧的aclTensor，且数据类型和selfRef一致，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、BFLOAT16。

  * accumulate（bool，计算输入）：累加或更新的操作类型标志位，True为累加，False为更新，Host侧的布尔值。
  * unsafe（bool，计算输入）: 检查索引是否在有效范围内标志位。Host侧的布尔值。unsafe为True时，索引越界会直接报错退出执行；当unsafe为False时，如果出现了索引越界，就可能出现运行时异常。

  * workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。

  * executor（aclOpExecutor\**，出参）：返回op执行器，包含了算子计算流程。

* **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：传入的selfRef、indices、values是空指针时。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. selfRef和values的数据类型不在支持的范围之内。
                                        2. selfRef和values数据类型不同。
                                        3. selfRef和values数据格式不同。
  ```

## aclnnIndexPutImpl

- **参数说明**：
  
  - workspace（void\*，入参）：在Device侧申请的workspace内存地址。

  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnIndexPutImplGetWorkspaceSize获取。

  - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。

  - stream（aclrtStream, 入参）: 指定执行任务的Stream。

- **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnIndexPutImpl默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

- 输入参数selfRef, indices, values一般有以下约束：
  - 1.indices中的Tensor个数不能超过selfRef的维度。
  - 2.values的维度需满足以下公式或广播后满足以下公式：
      - values.Dims() = indices[i].Dims() + (selfRef.Dims() - indices.size())
      - 其意义是values前一半维度需要与indices中的Tensor维度相同（indices中的Tensor会广播成相同shape），后一半维度需要与selfRef维度扣除indices中Tensor个数后相同。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_index_put_impl.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
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
  // 1. （固定写法）device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {3, 4};
  std::vector<int64_t> indexShape = {1};
  std::vector<int64_t> valueShape = {1};
  void* selfDeviceAddr = nullptr;
  void* indexOneDeviceAddr = nullptr;
  void* indexTwoDeviceAddr = nullptr;
  void* valueDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* indexOne = nullptr;
  aclTensor* indexTwo = nullptr;
  aclTensor* value= nullptr;
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> valueHostData = {1};
  std::vector<int64_t> indexOneHostData = {0};
  std::vector<int64_t> indexTwoHostData = {2};
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建index aclTensor
  ret = CreateAclTensor(indexOneHostData, indexShape, &indexOneDeviceAddr, aclDataType::ACL_INT64, &indexOne);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(indexTwoHostData, indexShape, &indexTwoDeviceAddr, aclDataType::ACL_INT64, &indexTwo);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* indexs[] = {indexOne, indexTwo};
  auto indexTensorList = aclCreateTensorList(indexs, 2);
  // value aclTensor
  ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &value);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnIndexPutImpl接口调用示例
  // 3.调用CANN算子库API，需要修改为具体的算子接口
  // 调用aclnnIndexPutImpl第一段接口
  ret = aclnnIndexPutImplGetWorkspaceSize(self, indexTensorList, value, true, false, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexPutImplGetWorkspaceSizefailed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnIndexPutImpl第二段接口
  ret = aclnnIndexPutImpl(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexPutImplfailed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(selfShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensorList(indexTensorList);
  aclDestroyTensor(value);

  // 7.释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(indexOneDeviceAddr);
  aclrtFree(indexTwoDeviceAddr);
  aclrtFree(valueDeviceAddr);
  if (workspaceSize > 0) {
   aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
