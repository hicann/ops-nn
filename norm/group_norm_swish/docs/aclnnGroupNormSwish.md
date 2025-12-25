# aclnnGroupNormSwish

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 接口功能：计算输入x的组归一化结果out，均值meanOut，标准差的倒数rstdOut，以及swish的输出。
- 计算公式：
  - **GroupNorm:**
    记 $E[x] = \bar{x}$代表$x$的均值，$Var[x] = \frac{1}{n} * \sum_{i=1}^n(x_i - E[x])^2$代表$x$的方差，则
    
    $$
    \left\{
    \begin{array} {rcl}
    yOut& &= \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta \\
    meanOut& &= E[x]\\
    rstdOut& &= \frac{1}{\sqrt{Var[x] + eps}}\\
    \end{array}
    \right.
    $$

  - **Swish:**
    
    $$
    yOut = \frac{x}{1+e^{-scale * x}}
    $$
    
    当activateSwish为True时，会计算Swish， 此时swish计算公式的x为GroupNorm公式得到的out。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGroupNormSwishGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGroupNormSwish”接口执行计算。

- `aclnnStatus aclnnGroupNormSwishGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta, int64_t numGroups, char *dataFormatOptional, double eps, bool activateSwish, double swishScale, const aclTensor *yOut, const aclTensor *meanOut, const aclTensor *rstdOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnGroupNormSwish(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnGroupNormSwishGetWorkspaceSize

- **参数说明：**

  * x（aclTensor*， 计算输入）：表示待组归一化的目标张量，`yOut`计算公式中的$x$，维度需大于一维，要求x第0维和第1维大于0，第1维要求能被group整除，Device侧的aclTensor。数据类型支持FLOAT32、FLOAT16、BFLOAT16，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
  * gamma（aclTensor*， 计算输入）：表示组归一化中的 gamma 参数，`yOut`计算公式中的$\gamma$，维度为一维，元素数量需与输入$x$的第1维度相同，gamma与beta的数据类型必须保持一致，且数据类型与x相同或者为FLOAT，Device侧的aclTensor。数据类型支持FLOAT32、FLOAT16、BFLOAT16，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
  * beta（aclTensor*， 计算输入）：表示组归一化中的 beta 参数，`yOut`计算公式中的$\beta$，维度为一维，元素数量需与输入$x$的第1维度相同，gamma与beta的数据类型必须保持一致，且数据类型与x相同或者为FLOAT，Device侧的aclTensor。数据类型支持FLOAT32、FLOAT16、BFLOAT16，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
  * numGroups（int64\_t， 计算输入）：表示将输入$x$的第1维度分为group组，值范围大于0，Host侧的整数。
  * dataFormatOptional（char*，计算输入）：表示数据格式，当前版本只支持输入NCHW，Host侧的字符类型。
  * eps（double， 计算输入）：用于防止产生除0的偏移，`yOut`和`rstdOut`计算公式中的$eps$值，值范围大于0，Host侧的DOUBLE类型。
  * activateSwish（bool， 计算输入）：表示是否支持swish计算。如果设置为true，则表示groupnorm计算后继续swish计算，Host侧的BOOL类型。
  * swishScale（double， 计算输入）：表示Swish计算时的$scale$值，Host侧的DOUBLE类型。
  * yOut（aclTensor*， 计算输出）：表示组归一化结果，数据类型和shape与$x$相同，Device 侧的aclTensor。数据类型支持FLOAT32、FLOAT16、BFLOAT16，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
  * meanOut（aclTensor*， 计算输出）：表示x分组后的均值，数据类型与$gamma$相同，shape为`(N， numGroups)`，其中`N`表示$x$第0维度的大小，`numGroups`为计算输入，表示将输入$x$的第1维度分为group组，Device 侧的aclTensor。数据类型支持FLOAT32、FLOAT16、BFLOAT16，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
  * rstdOut（aclTensor*， 计算输出）：表示x分组后的标准差的倒数，数据类型与$gamma$相同，shape为`(N， numGroups)`，其中`N`表示$x$第0维度的大小，`numGroups`为计算输入，表示将输入$x$的第1维度分为group组，Device 侧的aclTensor。数据类型支持FLOAT32、FLOAT16、BFLOAT16，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
  * workspaceSize（uint64_t\*， 出参）：返回需要在Device侧申请的workspace大小。
  * executor（aclOpExecutor **， 出参）：返回op执行器，包含算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
161001 ACLNN_ERR_PARAM_NULLPTR：1. 传入的x、gamma、beta、yOut、meanOut、rstdOut是空指针时。
161002 ACLNN_ERR_PARAM_INVALID：1. x、gamma、beta、yOut、meanOut、rstdOut数据类型不在支持的范围之内。
```

## aclnnGroupNormSwish

- **参数说明：**

  * workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnGroupNormSwishGetWorkspaceSize获取。
  * executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算
  - aclnnGroupNormSwish默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_group_norm_swish.h"

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
  // 1. （固定写法）device/stream初始化, 参考acl对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> xShape = {2, 3, 4};
  std::vector<int64_t> gammaShape = {3};
  std::vector<int64_t> betaShape = {3};
  std::vector<int64_t> outShape = {2, 3, 4};
  std::vector<int64_t> meanOutShape = {2, 1};
  std::vector<int64_t> rstdOutShape = {2, 1};
  void* xDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* betaDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* meanOutDeviceAddr = nullptr;
  void* rstdOutDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* beta = nullptr;
  aclTensor* yOut = nullptr;
  aclTensor* meanOut = nullptr;
  aclTensor* rstdOut = nullptr;
  std::vector<float> xHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                     13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> gammaHostData = {2.0, 2, 2};
  std::vector<float> betaHostData = {2.0, 2, 2};
  std::vector<float> outHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                    13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> meanOutHostData = {2.0, 2};
  std::vector<float> rstdOutHostData = {2.0, 2};

  int64_t numGroups = 1;
  double eps = 0.00001;
  bool activateSwish = true;
  double scale = 1.0;
  char* dataFormatOptional = "NCHW";
  // 创建x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gamma aclTensor
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建beta aclTensor
  ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &yOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建meanOut aclTensor
  ret = CreateAclTensor(meanOutHostData, meanOutShape, &meanOutDeviceAddr, aclDataType::ACL_FLOAT, &meanOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建rstdOut aclTensor
  ret = CreateAclTensor(rstdOutHostData, rstdOutShape, &rstdOutDeviceAddr, aclDataType::ACL_FLOAT, &rstdOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnGroupNormSwish第一段接口
  ret = aclnnGroupNormSwishGetWorkspaceSize(x, gamma, beta, numGroups, dataFormatOptional, eps, activateSwish, scale, yOut, meanOut, rstdOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSwishGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnGroupNormSwish第二段接口
  ret = aclnnGroupNormSwish(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSwish failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> outResultData(size, 0);
  ret = aclrtMemcpy(outResultData.data(), outResultData.size() * sizeof(outResultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("outResultData[%ld] is: %f\n", i, outResultData[i]);
  }

  size = GetShapeSize(meanOutShape);
  std::vector<float> meanResultData(size, 0);
  ret = aclrtMemcpy(meanResultData.data(), meanResultData.size() * sizeof(meanResultData[0]), meanOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("meanResultData[%ld] is: %f\n", i, meanResultData[i]);
  }

  size = GetShapeSize(rstdOutShape);
  std::vector<float> rstdResultData(size, 0);
  ret = aclrtMemcpy(rstdResultData.data(), rstdResultData.size() * sizeof(rstdResultData[0]), rstdOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("rstdResultData[%ld] is: %f\n", i, rstdResultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(gamma);
  aclDestroyTensor(beta);
  aclDestroyTensor(yOut);
  aclDestroyTensor(meanOut);
  aclDestroyTensor(rstdOut);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(xDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(betaDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(meanOutDeviceAddr);
  aclrtFree(rstdOutDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```

