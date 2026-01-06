# aclnnTopk


## 产品支持情况
| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |

## 功能说明

算子功能：返回输入Tensor在指定维度上的k个极值及索引。

## 函数原型
每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnTopkGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnTopk”接口执行计算。

- `aclnnStatus aclnnTopkGetWorkspaceSize(const aclTensor *self, int64_t k, int64_t dim, bool largest, bool sorted, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnTopk(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnTopkGetWorkspaceSize

- **参数说明**

  - self（aclTensor\*, 计算输入）：Device侧的aclTensor。shape支持1-8维度，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md), [数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、FLOAT32、DOUBLE、BFLOAT16。
  - k（int64_t, 计算输入）：Host侧的整型。表示计算维度上输出的极值个数。取值范围为[0, self.size(dim)]。
  - dim（int64_t, 计算输入）：Host侧的整型。表示计算维度。取值范围为[-self.dim(), self.dim())。
  - largest（bool, 计算输入）：Host侧的布尔型。True表示计算维度上的结果应由大到小输出，False表示计算维度上的结果由小到大输出。
  - sorted（bool, 计算输入）：Host侧的布尔型。True表示输出结果排序（若largest为True则结果从大到小排序，否则结果从小到大排序），False表示输出结果不排序，按输入时的数据顺序输出。注意：当前该参数仅支持取True,暂不支持取False。
  - valuesOut（aclTensor\*, 计算输出）：Device侧的aclTensor，数据类型与self保持一致。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md), [数据格式](../../../docs/zh/context/数据格式.md)支持ND。shape排序轴与k一致，非排序轴与self一致。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持INT8、UINT8、INT16、INT32、INT64、FLOAT16、FLOAT32、DOUBLE、BFLOAT16。
  - indicesOut（aclTensor\*, 计算输出）：Device侧的aclTensor，数据类型支持INT64。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md), [数据格式](../../../docs/zh/context/数据格式.md)支持ND。shape排序轴与k一致，非排序轴与self一致。
  - workspaceSize（uint64_t\*, 出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**, 出参）：返回op执行器，包含了算子计算流程。

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 传入的self、valuesOut或indicesOut是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. self、valuesOut或indicesOut的数据类型和数据格式不在支持的范围之内。
                                        2. dim不在输入self的合理维度范围内。
                                        3. k小于0或者k大于输入self在dim维度上的size大小。
  ```

## aclnnTopk

- **参数说明**

  - workspace（void\*, 入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnTopkGetWorkspaceSize获取。
  - executor（aclOpExecutor\*, 入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream, 入参）：指定执行任务的Stream。

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnTopk默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_topk.h"

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
  std::vector<int64_t> selfShape = {2, 4};
  std::vector<int64_t> outShape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* indices = nullptr;
  std::vector<float> selfHostData = {0, 1, 2.5, 3, 4, 5.5, 6, 7};
  std::vector<float> outHostData = {0, 1, 2, 3.1};
  std::vector<int64_t> indicesHostData = {0, 1, 2, 3};

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建indices aclTensor
  ret = CreateAclTensor(indicesHostData, outShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API名称
  int64_t k = 2;
  int64_t dim = 1;
  bool largest = true;
  bool sorted = true;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnTopk第一段接口
  ret = aclnnTopkGetWorkspaceSize(self, k, dim, largest, sorted, out, indices, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTopkGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnTopk第二段接口
  ret = aclnnTopk(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTopk failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto outSize = GetShapeSize(outShape);
  std::vector<float> outData(outSize, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    outSize * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < outSize; i++) {
    LOG_PRINT("out[%ld] is: %f\n", i, outData[i]);
  }

  auto indicesSize = GetShapeSize(outShape);
  std::vector<int64_t> indicesData(indicesSize, 0);
  ret = aclrtMemcpy(indicesData.data(), indicesData.size() * sizeof(indicesData[0]), indicesDeviceAddr,
                    indicesSize * sizeof(indicesData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < indicesSize; i++) {
    LOG_PRINT("indices[%ld] is: %ld\n", i, indicesData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(indices);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```