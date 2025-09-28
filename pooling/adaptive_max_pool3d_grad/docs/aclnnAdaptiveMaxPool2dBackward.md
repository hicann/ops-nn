# aclnnAdaptiveMaxPool2dBackward

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：
  正向自适应最大池化的反向传播，将梯度回填到每个自适应窗口最大值的坐标处，相同坐标处累加。
- 正向计算公式：
  对于输入self维度$[N,C,H,W]$，N（Batch）表示批量大小、C（Channels）表示特征图通道、H（Height）表示特征图高度、W（Width）表示特征图宽度，outputSize值为$[H_o,W_o]$的场景，其输出output维度为$[N,C,H_o,W_o]$，索引indices维度为$[N,C,H_o,W_o]$，相应tensor中每个元素的计公式如下：
  $$
  H_{left}^m = \lfloor(m*H)/H_o\rfloor \\
  H_{right}^m = \lceil(m*H)/H_o\rceil  \\
  W_{left}^n = \lfloor(n*W)/W_o\rfloor \\
  W_{right}^n = \lceil(n*W)/W_o\rceil  \\
  output(N,C,l,m,n) = \mathop{\max}\limits_{i \in ,j\in[H_{left}^m,H_{right}^m],k\in[W_{left}^n,W_{right}^n]} input(N,C,i,j,k) \\
  indices(N,C,l,m,n) = \mathop{\arg\max}\limits_{i \in ,j\in[H_{left}^m,H_{right}^m],k\in[W_{left}^n,W_{right}^n]} input(N,C,i,j,k)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAdaptiveMaxPool2dBackward”接口执行计算。

- `aclnnStatus aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnAdaptiveMaxPool2dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize

- **参数说明：**

  - gradOutput（aclTensor \*， 计算输入）：梯度Tensor，Device侧aclTensor。数据类型支持FLOAT32、FLOAT16、BFLOAT16，和正向的输出shape一致。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND， 当输入是4维时，内部按照NCHW处理，当输入是3维时，在0维度处补1，内部按照NCHW处理。

  - self（aclTensor \*， 计算输入）：正向的输入Tensor，Device侧aclTensor。数据类型支持FLOAT32、FLOAT16、BFLOAT16。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND， 当输入是4维时，内部按照NCHW处理，当输入是3维时，在0维度处补1，内部按照NCHW处理，与gradOutput一致。

  - indices（aclTensor \*， 计算输入）：输入Tensor，是Device侧aclTensor。数据类型支持INT32，INT64。正向输入中最大元素的索引位置。[数据格式](../../../docs/context/数据格式.md)与gradOutput保持一致。shape与gradOutput一致。

  - gradInput（aclTensor \*，计算输出）：反向输出Tensor，是Device侧aclTensor。数据类型支持FLOAT32、FLOAT16、BFLOAT16，shape与self保持一致。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)与self保持一致。

  - workspaceSize(uint64_t \*， 出参)：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor \*\*， 出参）：返回op执行器，包含了算子计算流程。
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的gradOutput、self或indices是空指针。
  161002（ACLNN_ERR_PARAM_INVALID）：1. gradOutput、self、indices、gradInput的数据类型不在支持的范围内。
                                   2. gradOutput、self、indices、gradInput的数据格式不在支持的范围内。
                                   3. 输入输出的shape不是3维或者4维。
                                   4. gradOutput与indices的shape不一致，self和gradInput的shape不一致。
                                   5. height * width > max int32，超出了indices的表达范围。
  ```

## aclnnAdaptiveMaxPool2dBackward

- **参数说明：**

  - workspace（void \*， 入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t， 入参）：在Device侧申请的workspace大小，由第一段接口aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize获取。
  - executor（aclOpExecutor \*， 入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream， 入参）：指定执行任务的Stream。
- **返回值：**
  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明
非整除场景下（self的后两个维度的维度值对gradOutput后两个维度的维度值取余不为0），shape不超过2的24次方。整除场景下，没有这个限制。
举例整除场景：self=[a,b,c,d]，gradOutput=[a,b,e,f]。 c%e为0并且e%f为0就是整除场景。
## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
#include <unistd.h>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_adaptive_max_pool2d_backward.h"

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
  std::vector<int64_t> gradOutShape = {1, 1, 1, 1};
  std::vector<int64_t> selfShape = {1, 1, 2, 2};
  std::vector<int64_t> indicesShape = {1, 1, 1, 1};
  std::vector<int64_t> gradInShape = {1, 1, 2, 2};
  void* gradOutDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* gradInDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* self = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* gradIn = nullptr;
  std::vector<float> gradOutHostData = {0.4757};
  std::vector<float> selfHostData = {0.0850, -0.5147, -0.0212, -0.5654};
  std::vector<int32_t> indicesHostData = {6};
  std::vector<float> gradInHostData = {0, 0, 0, 0};

  // 创建gradOut aclTensor
  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建indices aclTensor
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gradIn aclTensor
  ret = CreateAclTensor(gradInHostData, gradInShape, &gradInDeviceAddr, aclDataType::ACL_FLOAT, &gradIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnAdaptiveMaxPool2dBackward接口调用示例
  // 3. 调用CANN算子库API，需要修改为具体的API名称
  // 调用aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize第一段接口
  ret = aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize(gradOut, self, indices, gradIn, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnAdaptiveMaxPool2dBackward第二段接口
  ret = aclnnAdaptiveMaxPool2dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveMaxPool2dBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(gradInShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradIn result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(gradOut);
  aclDestroyTensor(self);
  aclDestroyTensor(indices);
  aclDestroyTensor(gradIn);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradOutDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  aclrtFree(gradInDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  _exit(0);
}
```