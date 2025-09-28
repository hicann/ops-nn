# aclnnBaddbmm&aclnnInplaceBaddbmm

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：
计算α与batch1、batch2的矩阵乘结果的乘积，再与β和self的乘积求和。
注意：batch1、batch2必须是三维Tensor，两个shape不支持做broadcast；
self必须要支持和batch1@batch2的结果做broadcast。（broadcast，广播机制，是指较小的shape扩展至较大的shape，使两者shape互相兼容，当前仅支持（1，n）的broadcast，即两个Tensor对应的每一维度必须相同或其中一个为1。）

- 计算公式：

  $$
  out = βself+α(batch1@batch2)
  $$
  注意：如果β为0，则self会被忽略，不参与计算。

- 示例：

  self的shape是[1, M, K], batch1@batch2的shape是[A, M, 1]，计算输出out的shape是[A, M, K]。每一维度的数字需要相同或其中一个为1。此处若self的shape是[2, M, K]，则不满足broadcast条件，报错。

## 函数原型

- aclnnBaddbmm和aclnnInplaceBaddbmm实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。
  - aclnnBaddbmm：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceBaddbmm：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
- 每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnBaddbmmGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnBaddbmm”接口执行计算。

  * `aclnnStatus aclnnBaddbmmGetWorkspaceSize(const aclTensor* self, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta, const aclScalar* alpha, aclTensor* out, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)`
  * `aclnnStatus aclnnBaddbmm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`
  * `aclnnStatus aclnnInplaceBaddbmmGetWorkspaceSize(const aclTensor* selfRef, const aclTensor* batch1, const aclTensor* batch2, const aclScalar* beta, const aclScalar* alpha, int8_t cubeMathType, uint64_t* workspaceSize, aclOpExecutor** executor)`
  * `aclnnStatus aclnnInplaceBaddbmm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnBaddbmmGetWorkspaceSize

- **参数说明：**
  - self(aclTensor*, 计算输入): Device侧的aclTensor, 数据类型需要与batch1@batch2后的二维数据满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)和[约束说明](#约束说明)）。 shape需要与batch1@batch2满足[broadcast关系](../../../docs/context/broadcast关系.md)。（注意：只能是self通过broadcast变成和batch1@batch2一样的shape，不可以batch1@batch2通过broadcast变成和self的shape一样。举例：self：[2, 3, 5]，batch1@batch2：[1, 1, 1]，会报错。）支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，支持空Tensor传入，[数据格式](../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32.
  - batch1(aclTensor*, 计算输入): Device侧的aclTensor，数据类型与self、batch2的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)和[约束说明](#约束说明)），shape需要与batch2满足bmm输入约束关系。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，支持空Tensor传入，[数据格式](../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32.
  - batch2(aclTensor*, 计算输入): Device侧的aclTensor，数据类型与self、batch2的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)和[约束说明](#约束说明)），shape需要与batch1满足bmm输入约束关系。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，支持空Tensor传入，[数据格式](../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32.
  - beta(aclScalar*,计算输入): 公式中的输入`β`，Host侧的aclScalar。数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL,且数据类型需要可转换成self与batch1@batch2推导后的数据类型（参见[互推导关系](../../../docs/context/互推导关系.md)和[约束说明](#约束说明)）。
  - alpha(aclScalar*,计算输入):公式中的输入`α`，Host侧的aclScalar。数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL，且数据类型需要可转换成self与batch1@batch2推导后的数据类型（参见[互推导关系](../../../docs/context/互推导关系.md)和[约束说明](#约束说明)）。
  - out(aclTensor \*, 计算输出): Device侧的aclTensor，数据类型与self、batch1@batch2推导后的数据类型满足数据类型推导原则（参见[互推导关系](../../../docs/context/互推导关系.md)和[约束说明](#约束说明)），[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND，format需要与self、batch1@batch2保持一致。out的shape需要与batch1@batch2的结果shape保持一致。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32.
  - cubeMathType（INT8，计算输入）：用于指定Cube单元的计算逻辑，Host侧的整型。数据类型支持INT8。注意：如果输入的数据类型存在互推导关系，该参数默认对互推导后的数据类型进行处理。支持的枚举值如下：
    * 0：KEEP_DTYPE，保持输入的数据类型进行计算。
    * 1：ALLOW_FP32_DOWN_PRECISION，支持将输入数据降精度计算。
      * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入数据类型为FLOAT32时，会转换为HFLOAT32计算。当输入为其他数据类型时不做处理。
    * 2：USE_FP16，支持将输入降精度至FLOAT16计算。
      * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入数据类型为BFLOAT16时不支持该选项。
    * 3：USE_HF32，支持将输入降精度至数据类型HFLOAT32计算。
      * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入数据类型为FLOAT32时，会转换为HFLOAT32计算。当输入为其他数据类型时不支持该选项。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的self、batch1或batch2是空指针。
  161002(ACLNN_ERR_PARAM_INVALID)：1. self、batch1、batch2或out的数据类型不在支持的范围内。
                                   2. self、batch1、batch2或out的数据格式不在支持的范围内。
                                   3. self、batch1、batch2或out的数据类型不一致。
                                   4. self、batch1、batch2或out的数据格式不一致。
                                   5. self不能与batch1@batch2做broadcast操作。
                                   6. self与batch1@batch2必须同时为空tensor或同时不为空tensor。
                                   7. self、batch1、batch2或out的数据类型不满足推导规则
  ```

## aclnnBaddbmm

- **参数说明：**
  * workspace(void \*, 入参): 在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参): 在Device侧申请的workspace大小，由第一段接口aclnnBaddbmmGetWorkspaceSize获取。
  * executor(aclOpExecutor \*, 入参): op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参): 指定执行任务的Stream。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## aclnnInplaceBaddbmmGetWorkspaceSize

- **参数说明：**
  * selfRef(aclTensor*, 计算输入/输出): Device侧的aclTensor, 数据类型需要与batch1@batch2后的二维数据满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)和[约束说明](#约束说明)）。shape需要与batch1@batch2一致，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，支持空Tensor传入，[数据格式](../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32.
  * batch1(aclTensor*, 计算输入): Device侧的aclTensor，数据类型与self、batch2的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)和[约束说明](#约束说明)），shape需要与batch1满足bmm输入约束关系。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，支持空Tensor传入，[数据格式](../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32.
  * batch2(aclTensor*, 计算输入): Device侧的aclTensor，数据类型与self、batch2的数据类型需满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)和[约束说明](#约束说明)），shape需要与batch2满足bmm输入约束关系。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，支持空Tensor传入，[数据格式](../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32.
  - beta(aclScalar*,计算输入): 公式中的输入`β`，Host侧的aclScalar。数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL,且数据类型需要可转换成self与batch1@batch2推导后的数据类型（参见[互推导关系](../../../docs/context/互推导关系.md)和[约束说明](#约束说明)）。
  - alpha(aclScalar*,计算输入):公式中的输入`α`，Host侧的aclScalar。数据类型支持FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL，且数据类型需要可转换成self与batch1@batch2推导后的数据类型（参见[互推导关系](../../../docs/context/互推导关系.md)和[约束说明](#约束说明)）。
  - out(aclTensor \*, 计算输出): Device侧的aclTensor，数据类型与self、batch1@batch2推导后的数据类型满足数据类型推导原则（参见[互推导关系](../../../docs/context/互推导关系.md)和[约束说明](#约束说明)），[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND，format需要与self、batch1@batch2保持一致。out的shape需要与batch1@batch2的结果shape保持一致。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32.
  - cubeMathType（INT8，计算输入）：用于指定Cube单元的计算逻辑，Host侧的整型。数据类型支持INT8。注意：如果输入的数据类型存在互推导关系，该参数默认对互推导后的数据类型进行处理。支持的枚举值如下：
    * 0：KEEP_DTYPE，保持输入的数据类型进行计算。
    * 1：ALLOW_FP32_DOWN_PRECISION，支持将输入数据降精度计算。
      * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入数据类型为FLOAT32时，会转换为HFLOAT32计算。当输入为其他数据类型时不做处理。
    * 2：USE_FP16，支持将输入降精度至FLOAT16计算。
      * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入数据类型为BFLOAT16时不支持该选项。
    * 3：USE_HF32，支持将输入降精度至数据类型HFLOAT32计算。
      * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入数据类型为FLOAT32时，会转换为HFLOAT32计算。当输入为其他数据类型时不支持该选项。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

```
第一段接口完成入参校验，出现以下场景时报错：
161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的selfRef、batch1或batch2是空指针。
161002(ACLNN_ERR_PARAM_INVALID)：1. selfRef、batch1或batch2的数据类型不在支持的范围内。
                                 2. selfRef、batch1或batch2的数据格式不在支持的范围内。
                                 3. selfRef、batch1或batch2的数据格式不一致。
                                 4. selfRef不能与batch1@batch2做broadcast操作。
                                 5. batch1和batch2的第一维度不相等。
                                 6. batch1和batch2的维度不是三维。
                                 7. batch1的最后一维和batch2的倒数第二维不相等。
```

## aclnnInplaceBaddbmm

- **参数说明：**
  * workspace(void \*, 入参): 在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参): 在Device侧申请的workspace大小，由第一段接口aclnnInplaceBaddbmmGetWorkspaceSize获取。
  * executor(aclOpExecutor \*, 入参): op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参): 指定执行任务的Stream。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持batch1和batch2两输入其中一个输入为BFLOAT16, 另一个输入为FLOAT16的数据类型推导。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_baddbmm.h"

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
  std::vector<int64_t> selfShape = {1, 2, 4};
  std::vector<int64_t> batch1Shape = {1, 2, 3};
  std::vector<int64_t> batch2Shape = {1, 3, 4};
  std::vector<int64_t> outShape = {1, 2, 4};
  void* selfDeviceAddr = nullptr;
  void* batch1DeviceAddr = nullptr;
  void* batch2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* batch1 = nullptr;
  aclTensor* batch2 = nullptr;
  aclScalar* alpha = nullptr;
  aclScalar* beta = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> batch1HostData = {1, 1, 1, 2, 2, 2};
  std::vector<float> batch2HostData = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};
  std::vector<float> outHostData(8, 0);
  int8_t cubeMathType = 1;
  float alphaValue = 1.2f;
  float betaValue = 1.0f;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建batch1 aclTensor
  ret = CreateAclTensor(batch1HostData, batch1Shape, &batch1DeviceAddr, aclDataType::ACL_FLOAT, &batch1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建batch2 aclTensor
  ret = CreateAclTensor(batch2HostData, batch2Shape, &batch2DeviceAddr, aclDataType::ACL_FLOAT, &batch2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建alpha aclScalar
  alpha = aclCreateScalar(&alphaValue,aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);
  // 创建beta aclScalar
  beta = aclCreateScalar(&betaValue,aclDataType::ACL_FLOAT);
  CHECK_RET(beta != nullptr, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  // aclnnBaddbmm接口调用示例
  // 3. 调用CANN算子库API
  // 调用aclnnBaddbmm第一段接口
  ret = aclnnBaddbmmGetWorkspaceSize(self, batch1, batch2, alpha, beta, out, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBaddbmmGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnBaddbmm第二段接口
  ret = aclnnBaddbmm(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBaddbmm failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // aclnnInplaceBaddbmm接口调用示例
  // 3. 调用CANN算子库API
  LOG_PRINT("\ntest aclnnInplaceBaddbmm\n");
  // 调用aclnnInplaceBaddbmm第一段接口
  ret = aclnnInplaceBaddbmmGetWorkspaceSize(self, batch1, batch2, alpha, beta, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceBaddbmmGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* inplaceWorkspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&inplaceWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnInplaceBaddbmm第二段接口
  ret = aclnnInplaceBaddbmm(inplaceWorkspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceBaddbmm failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(batch1);
  aclDestroyTensor(batch2);
  aclDestroyScalar(alpha);
  aclDestroyScalar(beta);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(batch1DeviceAddr);
  aclrtFree(batch2DeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtFree(inplaceWorkspaceAddr);
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
