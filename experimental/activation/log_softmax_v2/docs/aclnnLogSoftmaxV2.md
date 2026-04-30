
### aclnnLogSoftmaxV2

#### 产品支持情况

| 产品                                                               | 是否支持 |
| :----------------------------------------------------------------- | :------- |
| Atlas A2 训练系列产品/Atlas 800I A2                                | √       |

#### 功能说明

**算子功能**: 对输入Tensor沿着指定维度（dim）进行LogSoftmax运算。

**计算公式**：
算子的核心功能是沿着指定轴进行归约（Reduce）计算。为保证数值计算稳定性，实际计算采用减去最大值的优化方法，其实现步骤涉及Reduce和Broadcast操作，最终等效于以下公式：
`logsoftmax(x_i) = x_i - log(Σe^(x_j))`

#### 函数原型

`aclnnLogSoftmaxV2` 实现对张量的LogSoftmax计算。

该算子为两段式接口，必须先调用 “aclnnLogSoftmaxV2GetWorkspaceSize” 接口获取workspace大小和执行器，再调用 “aclnnLogSoftmaxV2” 接口执行计算。

```c
aclnnStatus aclnnLogSoftmaxV2GetWorkspaceSize(
    const aclTensor *self,
    int64_t dim,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor
);

aclnnStatus aclnnLogSoftmaxV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream
);
```

---

#### aclnnLogSoftmaxV2GetWorkspaceSize

**参数说明**：

| 参数          | 说明                                                                                                                                                                                                                                                                   |
| :------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| self          | **(aclTensor\*, 计算输入)**：Device侧的aclTensor，数据格式支持ND，维度不超过8维，支持非连续的Tensor。 `<br>` **Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件**：数据类型支持FLOAT32、FLOAT16、BFLOAT16。                            |
| dim           | **(int64_t, 计算输入)**：指定进行LogSoftmax运算的维度，取值范围为 `[-rank, rank-1]`，其中 `rank` 是输入Tensor `self` 的维度。                                                                                                                              |
| out           | **(aclTensor\*, 计算输出)**：Device侧的aclTensor，数据格式支持ND，shape必须和 `self` 一样，维度不超过8维，支持非连续的Tensor。 `<br>` **Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件**：数据类型支持FLOAT32、FLOAT16、BFLOAT16。 |
| workspaceSize | **(uint64_t\*, 出参)**：返回需要在Device侧申请的workspace大小。                                                                                                                                                                                                  |
| executor      | **(aclOpExecutor \*\*, 出参)**：返回op执行器，包含了算子计算流程。                                                                                                                                                                                               |

**返回值**：

`aclnnStatus`：返回状态码，具体参见aclnn返回码。

**第一段接口完成入参校验，出现如下场景时报错**：

* **161001 (ACLNN_ERR_PARAM_NULLPTR)**：
  1. 传入的 `self` 或 `out` 是空指针。
* **161002 (ACLNN_ERR_PARAM_INVALID)**：
  1. `self` 和 `out` 的数据类型不在支持的范围之内。
  2. `self` 和 `out` 的shape不一致。
  3. `self`、`out` 的维度超过8。
  4. `dim` 参数的值超出有效范围 `[-rank, rank-1]`。

---

#### aclnnLogSoftmaxV2

**参数说明**：

| 参数          | 说明                                                                                                                 |
| :------------ | :------------------------------------------------------------------------------------------------------------------- |
| workspace     | **(void\*, 入参)**：在Device侧申请的workspace内存地址。                                                        |
| workspaceSize | **(uint64_t, 入参)**：在Device侧申请的workspace大小，由第一段接口 `aclnnLogSoftmaxV2GetWorkspaceSize` 获取。 |
| executor      | **(aclOpExecutor\*, 入参)**：op执行器，包含了算子计算流程。                                                    |
| stream        | **(aclrtStream, 入参)**：指定执行任务的Stream。                                                                |

**返回值**：

`aclnnStatus`：返回状态码，具体参见aclnn返回码。

---

#### 使用示例

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
// 实际使用时，请包含正确的头文件
// #include "aclnn_log_softmax_v2.h" 

// 以下为示例代码，函数签名仅为示意
aclnnStatus aclnnLogSoftmaxV2GetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *out,
                                              uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnLogSoftmaxV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);


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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("logsoftmax result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> outShape = {2, 3};
  int64_t dim = 1; // 沿着第1个维度（每行）做LogSoftmax

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {0, 1, 2, 5, 2, 1}; // 输入数据
  std::vector<float> outHostData(selfHostData.size(), 0); // 输出数据，仅用于占位

  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnLogSoftmaxV2第一段接口
  ret = aclnnLogSoftmaxV2GetWorkspaceSize(self, dim, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogSoftmaxV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnLogSoftmaxV2第二段接口
  ret = aclnnLogSoftmaxV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLogSoftmaxV2 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值
  // 预期输出: {-2.4076, -1.4076, -0.4076, -0.0659, -3.0659, -4.0659}
  PrintOutResult(outShape, &outDeviceAddr);

  // 6. 释放aclTensor
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. 释放device资源
  aclrtFree(selfDeviceAddr);
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
