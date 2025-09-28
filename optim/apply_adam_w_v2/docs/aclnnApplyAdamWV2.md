# aclnnApplyAdamWV2
## 支持的产品型号
|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |


## 功能说明

- **算子功能：** 实现adamW优化器功能。

- **计算公式：**
  $$
  m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t} \\
  $$
  $$
  v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}
  $$
  $$
  \hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}} \\
  $$
  $$
  \hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}} \\
  $$
  $$
  \theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}-\eta \cdot \lambda \cdot \theta_{t-1}
  $$

## 函数原型
每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnApplyAdamWV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnApplyAdamWV2”接口执行计算。

* `aclnnStatus aclnnApplyAdamWV2GetWorkspaceSize(aclTensor *varRef, aclTensor *mRef, aclTensor *vRef, aclTensor *maxGradNormOptionalRef, const aclTensor *grad, const aclTensor *step, float lr, float beta1, float beta2, float weightDecay, float eps, bool amsgrad, bool maximize, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnApplyAdamWV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnApplyAdamWV2GetWorkspaceSize

- **参数说明：**

  * varRef（aclTensor\*, 计算输入/计算输出）：待计算的权重输入同时也是输出，公式中的theta，Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  * mRef（aclTensor\*, 计算输入/计算输出）：adamw优化器中m参数，公式中的m，Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32，shape、dtype要求与“varRef”参数一致。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  * vRef（aclTensor\*, 计算输入/计算输出）：adamw优化器中v参数，公式中的v，Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32，shape、dtype要求与“varRef”参数一致，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  * maxGradNormOptionalRef（aclTensor\*, 计算输入/计算输出）：保存v参数的最大值，公式中的v，Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32，shape要求与“varRef”参数一致，此参数在amsgrad参数为true时必选，在amsgrad参数为false时可选。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  * grad（aclTensor*, 计算输入）：梯度数据，公式中的gt，Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16、FLOAT32，shape要求与“varRef”参数一致。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  * step（aclTensor*, 计算输入）：迭代次数，公式中的t，Device侧的aclTensor，数据类型支持INT64、FLOAT32，元素个数为1，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  * lr（float\*, 计算输入）：学习率，公式中的eta，数据类型支持FLOAT。
  * beta1（float\*, 计算输入）：beta1参数，数据类型支持FLOAT。
  * beta2（float\*, 计算输入）：beta2参数，数据类型支持FLOAT。
  * weightDecay（float\*, 计算输入）：权重衰减系数，数据类型支持FLOAT。
  * eps（float\*, 计算输入）：防止除数为0，数据类型支持FLOAT。
  * amsgrad（bool, 计算输入）：是否使用算法的AMSGrad变量，数据类型为BOOL。
  * maximize（bool, 计算输入）：是否最大化参数，数据类型为BOOL。
  * workspaceSize（uint64_t\*, 出参）：返回需要在Device侧申请的workspace大小。
  * executor（aclOpExecutor\*\*, 出参）：内存地址。

- **返回值：**

  aclnnStatus： 返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的varRef、mRef、vRef、maxGradNormOptionalRef、grad、step是空指针时。
  161002 (ACLNN_ERR_PARAM_INVALID)：1. varRef、mRef、vRef、maxGradNormOptionalRef、grad、step的数据类型不在支持的范围内时。
                                    2. varRef、mRef、vRef、maxGradNormOptionalRef、grad、step的数据格式不在支持的范围内时。
                                    3. mRef、vRef、grad和varRef的shape不一致时。
                                    4. 当amsgrad为true时，maxGradNormOptionalRef和varRef的shape不一致时
                                    5. step的shape大小不为1时。
  ```

## aclnnApplyAdamWV2

- **参数说明：**

  * workspace(void \*, 入参): 在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参): 在Device侧申请的workspace大小，由第一段接口aclnnApplyAdamWV2GetWorkspaceSize获取。
  * executor(aclOpExecutor \*, 入参): op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参): 指定执行任务的AscendCL Stream流。

- **返回值：**

  aclnnStatus： 返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明
- 输入张量中varRef、mRef、vRef的数据类型一致时，数据类型支持FLOAT16、BFLOAT16、FLOAT32。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。
```Cpp
#include <unistd.h>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_adam_w_v2.h"

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
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> varShape = {2, 2};
  std::vector<int64_t> mShape = {2, 2};
  std::vector<int64_t> vShape = {2, 2};
  std::vector<int64_t> gradShape = {2, 2};
  std::vector<int64_t> maxgradShape = {2, 2};
  std::vector<int64_t> stepShape = {1};
  void* varDeviceAddr = nullptr;
  void* mDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* gradDeviceAddr = nullptr;
  void* maxgradDeviceAddr = nullptr;
  void* stepDeviceAddr = nullptr;
  aclTensor* var = nullptr;
  aclTensor* m = nullptr;
  aclTensor* v = nullptr;
  aclTensor* grad = nullptr;
  aclTensor* maxgrad = nullptr;
  aclTensor* step = nullptr;
  std::vector<float> varHostData = {0, 1, 2, 3};
  std::vector<float> mHostData = {0, 1, 2, 3};
  std::vector<float> vHostData = {0, 1, 2, 3};
  std::vector<float> gradHostData = {0, 1, 2, 3};
  std::vector<float> maxgradHostData = {0, 1, 2, 3};
  std::vector<float> stepHostData = {1};
  bool amsgrad = true;
  bool maximize = true;
  float lr = 1e-3;
  float beta1 = 0.9;
  float beta2 = 0.999;
  float weightDecay = 1e-2;
  float eps = 1e-8;
  // 创建var aclTensor
  ret = CreateAclTensor(varHostData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建m aclTensor
  ret = CreateAclTensor(mHostData, mShape, &mDeviceAddr, aclDataType::ACL_FLOAT, &m);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建v aclTensor
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建grad aclTensor
  ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建maxgrad aclTensor
  ret = CreateAclTensor(maxgradHostData, maxgradShape, &maxgradDeviceAddr, aclDataType::ACL_FLOAT, &maxgrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建step aclTensor
  ret = CreateAclTensor(stepHostData, stepShape, &stepDeviceAddr, aclDataType::ACL_FLOAT, &step);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnApplyAdamWV2第一段接口
  ret = aclnnApplyAdamWV2GetWorkspaceSize(var, m, v, maxgrad, grad, step, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamWV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnApplyAdamWV2第二段接口
  ret = aclnnApplyAdamWV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamWV2 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(varShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), varDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(var);
  aclDestroyTensor(m);
  aclDestroyTensor(v);
  aclDestroyTensor(grad);
  aclDestroyTensor(maxgrad);
  aclDestroyTensor(step);

  // 7. 释放device 资源
  aclrtFree(varDeviceAddr);
  aclrtFree(mDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(gradDeviceAddr);
  aclrtFree(maxgradDeviceAddr);
  aclrtFree(stepDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  _exit(0);
}
```
