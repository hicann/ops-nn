# aclnnRepeatInterleaveGrad

## 支持的产品型号


## 功能说明
  - 算子功能：算子repeatInterleave的反向, 将yGrad tensor的axis维度按repeats进行ReduceSum。

  - 示例：
    假设tensor yGrad是 ([[a<sub>1</sub>, b<sub>1</sub>, c<sub>1</sub>, d<sub>1</sub>, e<sub>1</sub>, f<sub>1</sub>], [a<sub>2</sub>, b<sub>2</sub>, c<sub>2</sub>, d<sub>2</sub>, e<sub>2</sub>, f<sub>2</sub>]]), repeats为([1, 2, 2, 1])，axis为1。
    那么最后生成的tensor为 tensor([[a<sub>1</sub>, b<sub>1</sub> + c<sub>1</sub>, d<sub>1</sub> + e<sub>1</sub>, f<sub>1</sub>], [a<sub>2</sub>, b<sub>2</sub> + c<sub>2</sub>, d<sub>2</sub> + e<sub>2</sub>, f<sub>2</sub>]])。将tensor yGrad的axis轴按照repeats做ReduceSum。

    假设tensor yGrad是 ([[a<sub>1</sub>, b<sub>1</sub>, c<sub>1</sub>, d<sub>1</sub>, e<sub>1</sub>, f<sub>1</sub>], [a<sub>2</sub>, b<sub>2</sub>, c<sub>2</sub>, d<sub>2</sub>, e<sub>2</sub>, f<sub>2</sub>]]), repeats为([2])，axis为1。
    那么最后生成的tensor为 tensor([[a<sub>1</sub> + b<sub>1</sub>， c<sub>1</sub> + d<sub>1</sub>， e<sub>1</sub> + f<sub>1</sub>], [a<sub>2</sub> + b<sub>2</sub>, c<sub>2</sub> + d<sub>2</sub>, e<sub>2</sub> + f<sub>2</sub>]])。将tensor yGrad的axis轴按照repeats的值，每2根轴做ReduceSum。
    注意：该场景等效于 repeats为(2)。

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnRepeatInterleaveGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnRepeatInterleaveGrad”接口执行计算。

  - `aclnnStatus aclnnRepeatInterleaveGradGetWorkspaceSize(const aclTensor *yGrad, const aclTensor *repeats, int64_t axis, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnRepeatInterleaveGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnRepeatInterleaveGradGetWorkspaceSize

- **参数说明：**

  - yGrad(aclTensor*, 计算输入)：Device侧的aclTensor，功能说明中待被ReduceSum的输入tensor。数据类型支持FLOAT16、BFLOAT16、FLOAT类型。不支持空tensor，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)。[数据格式](../../../docs/context/数据格式.md)支持ND，shape支持1-8维。

  - repeats(aclTensor*, 计算输入)：Device侧的aclTensor，重复的次数。数据类型支持INT32、INT64。repeats只能为0D / 1D Tensor。如果repeats为1D Tensor且size为1，那么repeats支持广播。如果repeats为1D Tensor且size大于1，那么repeats的元素之和等于yGrad的axis维数。不支持空tensor，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)。[数据格式](../../../docs/context/数据格式.md)支持ND。

  - axis(int64_t, 计算输入)：ReduceSum作用的维度。Host侧的INT64类型。axis的取值范围为`[-n, n)`, 其中`n`为yGrad的维度。

  - out(aclTensor*, 计算输出)：Device侧的aclTensor，功能说明中ReduceSum完成的输出tensor。数据类型需要与yGrad一致。[数据格式](./../../../docs/context/数据格式.md)支持ND。shape约束参见约束说明。

  - workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的yGrad、repeats或out是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID): 1. yGrad、repeats的数据类型不在支持范围内。
                                        2. yGrad、out的数据类型不一样。
  返回561002 (ACLNN_ERR_INNER_TILING_ERROR): 1. yGrad的维度数超过8。
  ```

## aclnnRepeatInterleaveGrad

- **参数说明：**

  - workspace(void*, 入参)：在Device侧申请的workspace内存地址。

  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnRepeatInterleaveGradGetWorkspaceSize获取。

  - executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。

  - stream(aclrtStream, 入参)：指定执行任务的AscendCL Stream流。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

在计算时需要满足以下要求：
  - 如果repeats为0D tensor或者size为1的1D tensor，那么repeats的元素值必须为yGrad在axis上维数的约数。
    如果为1D tensor且size大于1，那么repeats的元素之和必须为yGrad在axis上的维数。
    repeats tensor中的值必须为自然数。
  - out的shape大小满足yGrad的axis轴ReduceSum后的shape大小。
    例如，yGrad的shape为[64]、repeat为[2]、axis为0，out的shape可以为[32]、[2, 16]、[2, 4, 4]等，只需满足out的shape大小为32即可。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_repeat_interleave_grad.h"

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
  // 固定写法，AscendCL初始化
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
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> yGradShape = {4, 3};
  std::vector<int64_t> repeatsShape = {2};
  std::vector<int64_t> outShape = {2, 3};
  void* yGradDeviceAddr = nullptr;
  void* repeatsDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* yGrad = nullptr;
  aclTensor* repeats = nullptr;
  aclTensor* out = nullptr;
  int64_t axis = 0;
  std::vector<float> yGradHostData = {3, 4, 5, 3, 4, 5, -3, -4, -5, -3, -4, -5};
  std::vector<int64_t> repeatsHostData = {2, 2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0};

  // 创建yGrad aclTensor
  ret = CreateAclTensor(yGradHostData, yGradShape, &yGradDeviceAddr, aclDataType::ACL_FLOAT, &yGrad);
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
  // 调用aclnnRepeatInterleaveGrad第一段接口
  ret = aclnnRepeatInterleaveGradGetWorkspaceSize(yGrad, repeats, axis, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleaveGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnRepeatInterleaveGrad第二段接口
  ret = aclnnRepeatInterleaveGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleaveGrad failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(yGrad);
  aclDestroyTensor(repeats);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(yGradDeviceAddr);
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

