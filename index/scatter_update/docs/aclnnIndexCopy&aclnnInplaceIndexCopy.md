# aclnnIndexCopy&aclnnInplaceIndexCopy

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

算子功能:将index张量中元素值作为索引，针对指定轴dim，把source中元素复制到selfRef的对应位置上。

## 函数原型

- aclnnIndexCopy和aclnnInplaceIndexCopy实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。
  - aclnnIndexCopy：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceIndexCopy：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
- 每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnIndexCopyGetWorkspaceSize”或者“aclnnInplaceIndexCopyGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnIndexCopy”或者“aclnnInplaceIndexCopy”接口执行计算。
  * `aclnnStatus aclnnIndexCopyGetWorkspaceSize(aclTensor* selfRef, int64_t dim, const aclTensor* index, const aclTensor* source, aclTensor* outRef, uint64_t* workspaceSize, aclOpExecutor** executor)`
  * `aclnnStatus aclnnIndexCopy(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`
  * `aclnnStatus aclnnInplaceIndexCopyGetWorkspaceSize(aclTensor* selfRef, int64_t dim, const aclTensor* index, const aclTensor* source, uint64_t* workspaceSize, aclOpExecutor** executor)`
  * `aclnnStatus aclnnInplaceIndexCopy(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnIndexCopyGetWorkspaceSize

- **参数说明**：

  * selfRef(aclTensor*, 计算输入)：Device侧的aclTensor，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
     * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：FLOAT、BFLOAT16、FLOAT16、INT32、INT64、INT16、INT8、UINT8、DOUBLE、BOOL、COMPLEX128、COMPLEX64
  * dim(int64_t,计算输入): host侧的int64类型，在[-selfRef.dim(), selfRef.dim() - 1]范围内。
  * index(aclTensor*, 计算输入)：Device侧的aclTensor，数据类型支持INT32、INT64，维度不能大于1，元素个数要求和张量source中dim轴的大小相同。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND，当index有重复索引值的时候，结果不保序。index中的索引数据不支持越界。
  * source(aclTensor*, 计算输入): Device侧的aclTensor，数据类型与维度和selfRef一致，dim轴的大小和index的元素数量相同，其他维度同selfRef的shape，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  * out(aclTensor*, 计算输出)：Device侧的aclTensor，数据类型与形状与selfRef相同
     * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：FLOAT、BFLOAT16、FLOAT16、INT32、INT64、INT16、INT8、UINT8、DOUBLE、BOOL、COMPLEX128、COMPLEX64。
  * workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  (161001)ACLNN_ERR_PARAM_NULLPTR: 1. 传入的selfRef、index、source、out是空指针。

  (161002)ACLNN_ERR_PARAM_INVALID: 1. selfRef和index的数据类型不在支持的范围之内。
                                   2. selfRef和source的数据类型不同。
                                   3. source在轴dim上的大小和index的元素数量不等时。
                                   4. selfRef和source的shape在非dim轴上的大小不同时。
                                   5. index的维度大于1时。
                                   6. 当source为scalar时，index元素数量不为1。
                                   7. 当source和selfRef不是scalar时，二者维度不同。
                                   8. selfRef与out形状或数据类型不同。
                                   9. dim越界时。
  ```

## aclnnIndexCopy

- **参数说明**：

  * workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnIndexCopyGetWorkspaceSize获取。
  * executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## aclnnInplaceIndexCopyGetWorkspaceSize

- **参数说明**：

  * selfRef(aclTensor*, 计算输入)：Device侧的aclTensor，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
     * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：FLOAT、BFLOAT16、FLOAT16、INT32、INT64、INT16、INT8、UINT8、DOUBLE、BOOL、COMPLEX128、COMPLEX64
  * dim(int64_t,计算输入): host侧的int64类型，在[-selfRef.dim(), selfRef.dim() - 1]范围内。
  * index(aclTensor*, 计算输入)：Device侧的aclTensor，数据类型支持INT32、INT64，维度不能大于1，元素个数要求和张量source中dim轴的大小相同。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND，当index有重复索引值的时候，结果不保序。
  * source(aclTensor*, 计算输入): Device侧的aclTensor，数据类型与维度和selfRef一致，dim轴的大小和index的元素数量相同，其他维度同selfRef的shape，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  * workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。
  * executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  (161001)ACLNN_ERR_PARAM_NULLPTR: 1. 传入的selfRef、index、source是空指针。

  (161002)ACLNN_ERR_PARAM_INVALID: 1. selfRef和index的数据类型不在支持的范围之内。
                                   2. selfRef和source的数据类型不同。
                                   3. source在轴dim上的大小和index的元素数量不等时。
                                   4. selfRef和source的shape在非dim轴上的大小不同时。
                                   5. index的维度大于1时。
                                   6. 当source为scalar时，index元素数量不为1。
                                   7. 当source和selfRef不是scalar时，二者维度不同。
                                   8. dim越界时。
  ```

## aclnnInplaceIndexCopy

- **参数说明**：

  * workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnInplaceIndexCopyWorkspaceSize获取。
  * executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

无

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_index_copy.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> indexShape = {4};
  std::vector<int64_t> sourceShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* sourceDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* index = nullptr;
  aclTensor* source = nullptr;
  aclTensor* out = nullptr;
  int64_t dim=0;
  std::vector<float> selfHostData = {0,0,0,0,0,0,0,0};
  std::vector<int64_t> indexHostData = {3,2,1,0};
  std::vector<float> sourceHostData={1,2,3,4,5,6,7,8};
  std::vector<float> outHostData = {0,0,0,0,0,0,0,0};
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT32, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建index aclTensor
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建source aclTensor
  ret = CreateAclTensor(sourceHostData, sourceShape, &sourceDeviceAddr, aclDataType::ACL_INT32, &source);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);


  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnIndexCopy第一段接口
  ret = aclnnIndexCopyGetWorkspaceSize(self, dim, index, source, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnIndexCopy第二段接口
  ret = aclnnIndexCopy(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexCopy failed. ERROR: %d\n", ret); return ret);

  uint64_t inplaceWorkspaceSize = 0;
  aclOpExecutor* inplaceExecutor;
  // 调用aclnnInplaceIndexCopy第一段接口
  ret = aclnnInplaceIndexCopyGetWorkspaceSize(self, dim, index, source, &inplaceWorkspaceSize, &inplaceExecutor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceIndexCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* inplaceWorkspaceAddr = nullptr;
  if (inplaceWorkspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, inplaceWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnInplaceIndexCopy第二段接口
  ret = aclnnInplaceIndexCopy(inplaceWorkspaceAddr, inplaceWorkspaceSize, inplaceExecutor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceIndexCopy failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(selfShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("aclnnIndexCopy result[%ld] is: %f\n", i, resultData[i]);
  }

  auto inplaceSize = GetShapeSize(selfShape);
  std::vector<float> inplaceResultData(inplaceSize, 0);
  ret = aclrtMemcpy(inplaceResultData.data(), inplaceResultData.size() * sizeof(inplaceResultData[0]), selfDeviceAddr,
                    inplaceSize * sizeof(inplaceResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < inplaceSize; i++) {
    LOG_PRINT("aclnnInplaceIndexCopy result[%ld] is: %f\n", i, inplaceResultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(index);
  aclDestroyTensor(source);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(indexDeviceAddr);
  aclrtFree(sourceDeviceAddr);
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

