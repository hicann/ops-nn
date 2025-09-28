# aclnnAdaptiveMaxPool3d
## 支持的产品型号
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>。


## 功能说明

- 算子功能： 根据输入的outputSize计算每次kernel的大小，对输入self进行3维最大池化操作，输出池化后的值outputOut和索引indicesOut。aclnnAdaptiveMaxPool3d与aclnnMaxPool3d的区别在于，只需指定outputSize大小，并按outputSize的大小来划分pooling区域。

- 计算公式：
  outputOut tensor中对于DHW轴上每个位置为$(l,m,n)$的元素来说，其计算公式为：

  $$
  D^{l}_{left} = floor((l*D)/D_o)
  $$

  $$
  D^{l}_{right} = ceil(((l+1)*D)/D_o)
  $$

  $$
  H^{m}_{left} = floor((m*H)/H_o)
  $$

  $$
  H^{m}_{right} = ceil(((m+1)*H)/H_o)
  $$

  $$
  W^{n}_{left} = floor((n*W)/W_o)
  $$

  $$
  W^{n}_{right} = ceil(((n+1)*W)/W_o)
  $$

  $$
  outputOut(N,C,l,m,n)=\underset {i \in [D^{l}_{left}, D^{l}_{right}],j\in [H^m_{left},H^m_{right}], k \in [W^n_{left},W^n_{right}] }{max} input(N,C,i,j,k)
  $$

  $$
  indicesOut(N,C,l,m,n)=\underset {i \in [D^{l}_{left}, D^{l}_{right}],j\in [H^m_{left},H^m_{right}], k \in [W^n_{left},W^n_{right}] }{argmax} input(N,C,i,j,k)
  $$


## 函数原型
每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnAdaptiveMaxPool3dGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAdaptiveMaxPool3d”接口执行计算。

- `aclnnStatus aclnnAdaptiveMaxPool3dGetWorkspaceSize(const aclTensor* self, const aclIntArray* outputSize, aclTensor* outputOut, aclTensor* indicesOut, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnAdaptiveMaxPool3d(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnAdaptiveMaxPool3dGetWorkspaceSize

- **参数说明：**
  - self（aclTensor*，计算输入）：输入Tensor，Device侧的aclTensor。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持NCDHW，当输入是4维时，在0维度处补1，内部按照NCDHW处理。D轴H轴W轴3个维度的乘积$D*H*W$不能大于INT32的最大表示。数据类型支持BFLOAT16、FLOAT16、FLOAT32，且数据类型与outputOut的数据类型保持一致。
  - outputSize（aclIntArray\*，计算输入）：Host侧的aclIntArray，size大小为3。表示输出结果在$D_o$，$H_o$，$W_o$维度上的空间大小。数据类型支持INT32和INT64。outputSize中元素值不可小于0。
  - outputOut（aclTensor\*，计算输出）：输出Tensor，是Device侧的aclTensor。池化后的结果。shape与indicesOut保持一致。[数据格式](../../../docs/context/数据格式.md)支持NCDHW。数据类型支持BFLOAT16、FLOAT16、FLOAT32，且数据类型与self的数据类型一致。
  - indicesOut（aclTensor\*，计算输出）：输出Tensor，是Device侧的aclTensor。indicesOut表示outputOut元素在输入self中的索引位置。shape与outputOut保持一致，[数据格式](../../../docs/context/数据格式.md)支持NCDHW。数据类型支持INT32。
  - workspaceSize（uint64_t*，出参）：返回用户需要在npu Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的self、outputSize、outputOut或indicesOut是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID)：1. self、outputOut、indicesOut的数据类型、shape、format、参数取值不在支持的范围之内。
                                        2. outputSize的shape、参数取值不在支持的范围内
                                        3. self和outputOut数据类型不一致
                                        4. outputOut和indicesOut shape不一致
                                        5. 平台不支持
  ```

## aclnnAdaptiveMaxPool3d

- **参数说明：**

  - workspace（void*，入参）：在npu device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在npu device侧申请的workspace大小，由第一段接口aclnnAdaptiveMaxPool3dGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明
Shape描述：
  - self.shape = (N, C, Din, Hin, Win)
  - outputSize = [Din, Hout, Wout]
  - outputOut.shape = (N, C, Din, Hout, Wout)
  - indicesOut.shape = (N, C, Din, Hout, Wout)

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
#include <unistd.h>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_adaptive_max_pool3d.h"

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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW,
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
  std::vector<int64_t> selfShape = {1, 1, 1, 4, 4};
  std::vector<int64_t> outShape = {1, 1, 1, 2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* indDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* indices = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4.1, 5, 6, 7,
                                     8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> outHostData = {0, 0, 0, 0.0};
  std::vector<int64_t> indicesHostData = {0, 0, 0, 0};

  //创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  //创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  //创建indices aclTensor
  ret = CreateAclTensor(indicesHostData, outShape, &indDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> arraySize = {1, 2, 2};
  const aclIntArray *outputSize = aclCreateIntArray(arraySize.data(), arraySize.size());
  CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. 调用CANN算子库API，需要修改为具体的API名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnAdaptiveMaxPool3d第一段接口
  ret = aclnnAdaptiveMaxPool3dGetWorkspaceSize(self, outputSize, out, indices, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveMaxPool3dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnAdaptiveMaxPool3d第二段接口
  ret = aclnnAdaptiveMaxPool3d(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveMaxPool3d failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  std::vector<int64_t> indicesData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(indicesData.data(), indicesData.size() * sizeof(indicesData[0]), indDeviceAddr,
                    size * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out[%ld] is: %f\n", i, outData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(indices);
  aclDestroyIntArray(outputSize);


  // 7. 释放Device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(indDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  _exit(0);
}
```

