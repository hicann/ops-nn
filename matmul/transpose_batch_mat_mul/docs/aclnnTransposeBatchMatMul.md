# aclnnTransposeBatchMatMul

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √    |

## 功能说明

- 算子功能：完成张量x1与张量x2的矩阵乘计算。仅支持三维的Tensor传入。Tensor支持转置，转置序列根据传入的数列进行变更。permX1代表张量x1的转置序列，permX2代表张量x2的转置序列，序列值为0的是batch维度，其余两个维度做矩阵乘法。

- 示例：
  - x1的shape是[B, M, K]，x2的shape是[B, K, N]，scale为None，batchSplitFactor等于1时，计算输出out的shape是[M, B, N]。
  - x1的shape是[B, M, K]，x2的shape是[B, K, N]，scale不为None，batchSplitFactor等于1时，计算输出out的shape是[M, 1, B*N]。
  - x1的shape是[B, M, K]，x2的shape是[B, K, N]，scale为None，batchSplitFactor大于1时，计算输出out的shape是[M, batchSplitFactor, B*N/batchSplitFactor]。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnTransposeBatchMatMulGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnTransposeBatchMatMul”接口执行计算。

- `aclnnStatus aclnnTransposeBatchMatMulGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* scale, const aclIntArray* permX1, const aclIntArray* permX2, const aclIntArray* permY, int8_t cubeMathType, const int32_t batchSplitFactor, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnTransposeBatchMatMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnTransposeBatchMatMulGetWorkSpaceSize

- **参数说明：**
  - x1（aclTensor*，计算输入）：表示矩阵乘的第一个矩阵，Device侧aclTensor。数据类型需要与x2满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)和[约束说明](#约束说明)）。[数据格式](../../../docs/zh/context/数据格式.md)支持ND，shape维度支持3维{B, M, K}或者{M，B，K}, B的取值范围为[1, 65536)。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。同时x1的-1轴（末轴）<=65535。
  - x2（aclTensor*，计算输入）：表示矩阵乘的第二个矩阵，Device侧aclTensor，数据类型需要与x1满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)和[约束说明](#约束说明)）。[数据格式](../../../docs/zh/context/数据格式.md)支持ND，shape维度支持3维，{B, K, N}, N的取值范围为[1, 65536)。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。x2的Reduce维度需要与x1的Reduce维度大小相等。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。同时x2的-1轴（末轴）<=65535。
  - bias（aclTensor*，可选计算输入）：表示矩阵乘的偏置矩阵，Device侧aclTensor，数据类型需要与x1满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)和[约束说明](#约束说明)）。[数据格式](../../../docs/zh/context/数据格式.md)支持ND，shape维度支持3维。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，目前不支持该输入。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32。
  - scale（aclTensor*，可选计算输入）：表示量化输入，Device侧aclTensor，数据类型需要与x1满足数据类型推导规则（参见[互推导关系](../../../docs/zh/context/互推导关系.md)和[约束说明](#约束说明)）。[数据格式](../../../docs/zh/context/数据格式.md)支持ND，shape维度支持1维(B * N)。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持INT64、UINT64。
  - permX1（aclIntArray*, 计算输入）：表示矩阵乘的第一个矩阵的转置序列，host侧的aclIntArray，size大小为3，数据类型为INT64，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。支持{0, 1, 2}、{1, 0, 2}。
  - permX2（aclIntArray*, 计算输入）：表示矩阵乘的第二个矩阵的转置序列，host侧的aclIntArray，size大小为3，数据类型为INT64，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。仅支持{0, 1, 2}。
  - permY（aclIntArray*, 计算输入）：表示矩阵乘输出矩阵的转置序列，host侧的aclIntArray，size大小为3，数据类型为INT64，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。仅支持{1, 0, 2}。
  - cubeMathType（int8_t，计算输入）：用于指定Cube单元的计算逻辑，Host侧的整型。数据类型支持INT8。注意：如果输入的数据类型存在互推导关系，该参数默认对互推导后的数据类型进行处理。支持的枚举值如下：
    * 0：KEEP_DTYPE，保持输入的数据类型进行计算。
    * 1：ALLOW_FP32_DOWN_PRECISION，支持将输入数据降精度计算。
      * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入数据类型为FLOAT32时，会转换为HFLOAT32计算。当输入为其他数据类型时不做处理。
    * 2：USE_FP16，支持将输入降精度至FLOAT16计算。
      * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入数据类型为BFLOAT16时不支持该选项。
    * 3：USE_HF32，支持将输入降精度至数据类型HFLOAT32计算。
      * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入数据类型为FLOAT32时，会转换为HFLOAT32计算。当输入为其他数据类型时不支持该选项。
  - batchSplitFactor（int32_t，计算输入）：用于指定矩阵乘输出矩阵中N维的切分大小，Host侧的整型。数据类型支持INT32。取值范围为[1, N]且能被N整除。
      * 当scale不为空时，batchSplitFactor只能等于1。
  - out（aclTensor *，计算输出）：表示矩阵乘的输出矩阵，公式中的out，Device侧aclTensor。数据类型需要与x1与x2推导之后的数据类型保持一致（参见[互推导关系](../../../docs/zh/context/互推导关系.md)和[约束说明](#约束说明)）。[数据格式](../../../docs/zh/context/数据格式.md)支持ND，shape维度支持3维，{M, B, N}。
    * 当scale有值时，输出shape为{M, 1, B*N}。
    * 当batchSplitFactor大于1时，out的输出shape为{batchSplitFactor, M, B/batchSplitFactor*N}。      
      * 示例一: M, K, N, B = 32, 512, 128, 16；batchSplitFactor = 2时，out的输出shape大小为{2, 32, 1024}。
      * 示例二: M, K, N, B = 32, 512, 128, 16；batchSplitFactor = 4时，out的输出shape大小为{4, 32, 512}。  
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32、INT8。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor **，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的x1、x2或out是空指针。
  161002(ACLNN_ERR_PARAM_INVALID)：1. x1、x2或out的数据类型不在支持的范围内。
                                   2. x1的第二维和x2的第一维度不相等。
                                   3. x1或x2的维度大小不等于3。
                                   4. x2的第二维或x2的第三维不能被128整除。
                                   5. scale的数据类型不在支持的范围内。
                                   6. batchSplitFactor的数值大小不在支持的范围内。
  ```

## aclnnTransposeBatchMatMul

- **参数说明：**
  * workspace(void \*, 入参): 在Device侧申请的workspace内存地址。
  * workspaceSize(uint64_t, 入参): 在Device侧申请的workspace大小，由第一段接口aclnnTransposeBatchMatMulGetWorkSpaceSize获取。
  * executor(aclOpExecutor \*, 入参): op执行器，包含了算子计算流程。
  * stream(aclrtStream, 入参): 指定执行任务的Stream。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
  
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  * 不支持两个输入分别为BFLOAT16和FLOAT16的数据类型推导。
  * 不支持两个输入分别为BFLOAT16和FLOAT32的数据类型推导。
  * 当scale不为空时，B与N的乘积小于65536。
  * 当scale不为空时，仅支持输入为FLOAT16的类型推导。
  * 当scale不为空时，仅支持输出为INT8的类型推导。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_transpose_batch_mat_mul.h"

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
  int32_t M = 32;
  int32_t K = 512;
  int32_t N = 128;
  int32_t Batch = 16;
  std::vector<int64_t> x1Shape = {M, Batch, K};
  std::vector<int64_t> x2Shape = {Batch, K, N};
  std::vector<int64_t> outShape = {M, Batch, N};
  std::vector<int64_t> permX1Series = {1, 0, 2};
  std::vector<int64_t> permX2Series = {0, 1, 2};
  std::vector<int64_t> permYSeries = {1, 0, 2};
  void* x1DeviceAddr = nullptr;
  void* x2DeviceAddr = nullptr;
  void* scaleDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* x1 = nullptr;
  aclTensor* x2 = nullptr;
  aclTensor* scale = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> x1HostData(GetShapeSize(x1Shape));
  std::vector<float> x2HostData(GetShapeSize(x2Shape));
  std::vector<float> outHostData(GetShapeSize(outShape));
  int8_t cubeMathType = 1;
  int8_t batchSplitFactor = 1;

  // 创建x1 aclTensor
  ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建x2 aclTensor
  ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  aclIntArray *permX1 = aclCreateIntArray(permX1Series.data(), permX1Series.size());
  aclIntArray *permX2 = aclCreateIntArray(permX2Series.data(), permX2Series.size());
  aclIntArray *permY = aclCreateIntArray(permYSeries.data(), permYSeries.size());
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  // aclnnTransposeBatchMatMul接口调用示例
  // 3. 调用CANN算子库API，需要修改为具体的API名称
  // 调用aclnnTransposeBatchMatMul第一段接口
  ret = aclnnTransposeBatchMatMulGetWorkspaceSize(x1, x2, (const aclTensor*)nullptr, (const aclTensor*)nullptr,
                                                  permX1, permX2, permY, cubeMathType, batchSplitFactor, out,
                                                  &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeBatchMatMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnTransposeBatchMatMul第二段接口
  ret = aclnnTransposeBatchMatMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeBatchMatMul failed. ERROR: %d\n", ret); return ret);

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

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x1);
  aclDestroyTensor(x2);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(x1DeviceAddr);
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
