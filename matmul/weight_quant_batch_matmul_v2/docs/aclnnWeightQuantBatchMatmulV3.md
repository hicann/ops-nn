# aclnnWeightQuantBatchMatmulV3

## 产品支持情况

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

## 功能说明

- **算子功能**：完成一个输入为伪量化场景的矩阵乘计算，并可以实现对于输出的量化计算。相较于aclnnWeightQuantBatchMatmulV2接口，此接口变化点为：

  新增innerPrecise参数，用于支持高精度或者高性能计算模式选择。A16W4 per_group场景在batchSize<=16的场景下可设置为1， 提升性能。
- **计算公式**：

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

  公式中的$weight$为伪量化场景的输入，其反量化公式$ANTIQUANT(weight)$为

  $$
  ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
  $$

  当客户配置quantScaleOptional输入时, 会对输出进行量化处理, 其量化公式为

  $$
  \begin{aligned}
  y &= QUANT(x @ ANTIQUANT(weight) + bias) \\
  &= (x @ ANTIQUANT(weight) + bias) * quantScale + quantOffset \\
  \end{aligned}
  $$

  当客户配置quantScaleOptional输入为nullptr, 则直接输出:

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

## 函数原型

每个算子分为[两段式接口](./../../../docs/context/两段式接口.md)，必须先调用“aclnnWeightQuantBatchMatmulV3GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnWeightQuantBatchMatmulV3”接口执行计算。
  - `aclnnStatus aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *antiquantScale, const aclTensor *antiquantOffsetOptional, const aclTensor *quantScaleOptional, const aclTensor *quantOffsetOptional, const aclTensor *biasOptional, int antiquantGroupSize, int innerPrecise, const aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnWeightQuantBatchMatmulV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnWeightQuantBatchMatmulV3GetWorkspaceSize

- **参数说明**

  - x(aclTensor\*, 计算输入)：公式中的输入`x`，数据类型支持FLOAT16、BFLOAT16，[数据格式](./../../../docs/context/数据格式.md)支持ND。[非连续的Tensor](./../../../docs/context/非连续的Tensor.md)仅支持transpose场景。维度支持2维，shape支持(m, k)，其中Reduce维度k需要与`weight`的Reduce维度k大小相等。
  - weight(aclTensor\*，计算输入)：公式中的输入`weight`，数据类型支持INT8、INT4、INT32（当weight数据格式为FRACTAL_NZ且数据类型为INT4/INT32时，或者当weight数据格式为ND且数据类型为INT32时，仅在INT4Pack场景支持，需要配合aclnnConvertWeightToINT4Pack接口完成从INT32到INT4Pack的转换，以及从ND到FRACTAL_NZ的转换），若数据类型为INT4，则weight的内轴应为偶数。[非连续的Tensor](./../../../docs/context/非连续的Tensor.md)仅支持transpose场景。shape支持(k, n)。
    对于不同伪量化算法模式，weight的数据格式为FRACTAL_NZ仅在如下场景下支持:
    - per_channel模式：
      - weight的数据类型为INT8，y的数据类型为非INT8。
      - weight的数据类型为INT4/INT32，weight转置，y的数据类型为非INT8。
    - per_group模式：weight的数据类型为INT4/INT32，weight非转置，x非转置，antiquantGroupSize为64或128，k为antiquantGroupSize对齐，n为64对齐，y的数据类型为非INT8。

  - antiquantScale(aclTensor\*, 计算输入)：实现输入反量化计算的反量化scale参数，反量化公式中的输入`antiquantScale`，数据类型支持FLOAT16、BFLOAT16、UINT64、INT64。当数据类型为FLOAT16、BFLOAT16时，要求和输入`x`的数据类型保持一致。当数据类型为UINT64、INT64时，x仅支持FLOAT16，不转置，weight仅支持int8，ND转置，模式仅支持per_channel，quantScaleOptional和quantOffsetOptional必须为空，m仅支持[1, 96]，k和n要求64对齐，需要首先配合aclnnCast接口完成FLOAT16到FLOAT32的转换，再配合aclnnTransQuantParamV2接口完成FLOAT32到UINT64的转换。[数据格式](./../../../docs/context/数据格式.md)支持ND。[非连续的Tensor](./../../../docs/context/非连续的Tensor.md)仅支持transpose场景。
    对于不同伪量化算法模式，antiquantScale支持的shape如下:
    - per_tensor模式：输入shape为(1,)或(1, 1)。
    - per_channel模式：输入shape为(1, n)或(n,)。
    - per_group模式：输入shape为(ceil(k, group_size), n)。
  - antiquantOffsetOptional（aclTensor\*, 计算输入）：实现输入反量化计算的反量化offset参数，反量化公式中的输入`antiquantOffset`，数据类型支持FLOAT16、BFLOAT16、INT32。当数据类型为FLOAT16、BFLOAT16时，要求和输入`x`的数据类型保持一致。当数据类型为INT32时，数据范围限制为[-128, 127]，x仅支持FLOAT16，weight仅支持int8，antiquantScale仅支持UINT64/INT64。可选输入，当不需要时为空指针；存在时shape要求与`antiquantScale`一致。[数据格式](./../../../docs/context/数据格式.md)支持ND。[非连续的Tensor](./../../../docs/context/非连续的Tensor.md)仅支持transpose场景。

  - quantScaleOptional（aclTensor\*, 计算输入）：实现输出量化计算的量化参数，由量化公式中的quantScale和quantOffset的数据通过`aclnnTransQuantParam`接口转化得到。数据类型支持UINT64，[数据格式](./../../../docs/context/数据格式.md)支持ND。不支持[非连续的Tensor](./../../../docs/context/非连续的Tensor.md)。可选输入，当不需要时为空指针；对于不同的伪量化算法模式，支持的shape如下:
    - per_tensor模式：输入shape为(1,)或(1, 1)。
    - per_channel模式：输入shape为(1, n)或(n,)。
  - quantOffsetOptional（aclTensor\*, 计算输入）：实现输出量化计算的量化offset参数，量化公式中的输入`quantOffset`，数据类型支持FLOAT，[数据格式](./../../../docs/context/数据格式.md)支持ND。可选输入，当不需要时为空指针；存在时shape要求与`quantScaleOptional`一致。不支持[非连续的Tensor](./../../../docs/context/非连续的Tensor.md)。
  - biasOptional（aclTensor\*, 计算输入）：偏置输入，公式中的输入`bias`。当输入`x`的数据类型为BFLOAT16时，数据类型要求为FLOAT；当输入`x`的数据类型为FLOAT16时，数据类型要求为FLOAT16。可选输入，当不需要时为空指针；存在输入时支持1维或2维，shape支持(n,)或(1, n)。支持ND。不支持[非连续的Tensor](./../../../docs/context/非连续的Tensor.md)。
  - antiquantGroupSize（int, 计算输入）：表示伪量化per_group算法模式下，对输入`weight`进行反量化计算的groupSize输入，描述一组反量化参数对应的待反量化数据量在Reduce方向的大小。当伪量化算法模式不为per_group时传入0；当伪量化算法模式为per_group时传入值的范围为[32, k-1]且值要求是32的倍数。
  - innerPrecise (int, 计算输入)：表示伪量化是高精度还是高性能计算模式（仅支持传入0或1）。A16W4 per_group场景在batchSize<=16的场景下可设置为1，并且weight数据格式设为FRACTAL_NZ，来提升性能。其他场景不建议使用，建议传入0。
    - 0：代表高精度模式。
    - 1：代表高性能模式。

  - y（aclTensor\*, 计算输出）：计算输出，公式中的`y`。当`quantScaleOptional`存在时，数据类型为INT8；当`quantScaleOptional`不存在时，数据类型支持FLOAT16、BFLOAT16，且与输入`x`的数据类型一致。维度支持2维，shape支持(m, n)。[数据格式](./../../../docs/context/数据格式.md)支持ND。不支持[非连续的Tensor](./../../../docs/context/非连续的Tensor.md)。
  - workspaceSize（uint64_t\*, 出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**, 出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](./../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 (ACLNN_ERR_PARAM_NULLPTR)：如果必选参数传入的是空指针。
  161002 (ACLNN_ERR_PARAM_INVALID)：
    - 传入x、weight、antiquantScale、antiquantOffsetOptional、quantScaleOptional、quantOffsetOptional、biasOptional、y的shape维度不符合要求。
    - 传入x、weight、antiquantScale、antiquantOffsetOptional、quantScaleOptional、quantOffsetOptional、biasOptional、y的数据类型不在支持的范围之内。
    - x、weight的Reduce维度(k)不相等。
    - antiquantOffsetOptional存在输入时，shape与antiquantScale不相同。
    - quantOffsetOptional存在输入时，shape与quantScale不相同。
    - biasOptional的shape不符合要求。
    - antiquantGroupSize值不符合要求。
    - innerPrecise值不符合要求。
    - quantOffsetOptional存在时，quantScaleOptional是空指针。
    - 输入的k、n值不在[1, 65535]范围内；
    - x矩阵为非转置时，m不在[1, 2^31-1]范围内；转置时，m不在[1, 65535]范围内
    - 不支持空tensor场景。
    - 输入tensor的数据格式不在支持范围内。
    - 传入x、weight、antiquantScale、antiquantOffsetOptional、quantScaleOptional、quantOffsetOptional、biasOptional、y的连续性不符合要求。
  361001(ACLNN_ERR_RUNTIME_ERROR): 产品型号不支持。
  ```
## aclnnWeightQuantBatchMatmulV3

- **参数说明**

  - workspace(void\*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnWeightQuantBatchMatmulV3GetWorkspaceSize获取。
  - executor(aclOpExecutor\*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](./../../../docs/context/aclnn返回码.md)。


## 约束说明

- per_channel模式：为提高性能，推荐使用transpose后的weight输入。m范围为[65, 96]时，推荐使用数据类型为UINT64/INT64的antiquantScale。

- per_group模式：在A16W4，batchSize<=16的场景下可设置innerPrecise参数为1，并且weight数据格式设为FRACTAL_NZ，来提升性能，但会存在一定的精度下降。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_weight_quant_batch_matmul_v3.h"

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
  std::vector<int64_t> xShape = {16, 32};
  std::vector<int64_t> weightShape = {32, 16};
  std::vector<int64_t> yShape = {16, 16};
  void* xDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* y = nullptr;
  int32_t innerPrecise = 1;
  std::vector<float> xHostData(512, 1);
  std::vector<int8_t> weightHostData(512, 1);
  std::vector<float> yHostData(256, 0);

  std::vector<int64_t> antiquantScaleShape = {16};
  void* antiquantScaleDeviceAddr = nullptr;
  aclTensor* antiquantScale = nullptr;
  std::vector<float> antiquantScaleHostData(16, 1);


  // 创建x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other aclTensor
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建y aclTensor
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建antiquantScale aclTensor
  ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT, &antiquantScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建xFp16 aclTensor
  void* xFp16DeviceAddr = nullptr;
  aclTensor* xFp16 = nullptr;
  ret = CreateAclTensor(xHostData, xShape, &xFp16DeviceAddr, aclDataType::ACL_FLOAT16, &xFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建antiquantScale aclTensor
  void* antiquantScaleFp16DeviceAddr = nullptr;
  aclTensor* antiquantScaleFp16 = nullptr;
  ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleFp16DeviceAddr, aclDataType::ACL_FLOAT16, &antiquantScaleFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建yFp16 aclTensor
  void* yFp16DeviceAddr = nullptr;
  aclTensor* yFp16 = nullptr;
  ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  void* workspaceAddr = nullptr;

  // 调用cast生成FP16的输入
  ret = aclnnCastGetWorkspaceSize(x, aclDataType::ACL_FLOAT16, xFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize0 failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast0 failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  ret = aclnnCastGetWorkspaceSize(antiquantScale, aclDataType::ACL_FLOAT16, antiquantScaleFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize1 failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast1 failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 调用aclnnWeightQuantBatchMatmulV3第一段接口
  ret = aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(xFp16, weight, antiquantScaleFp16, nullptr, nullptr, nullptr, nullptr, 0, innerPrecise, yFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnWeightQuantBatchMatmulV3第二段接口
  ret = aclnnWeightQuantBatchMatmulV3(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV3 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

 // 将输出转为FP32
  ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize2 failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(yShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(weight);
  aclDestroyTensor(antiquantScale);
  aclDestroyTensor(y);
  aclDestroyTensor(xFp16);
  aclDestroyTensor(antiquantScaleFp16);
  aclDestroyTensor(yFp16);

  // 7. 释放device资源
  aclrtFree(xDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(antiquantScaleDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(xFp16DeviceAddr);
  aclrtFree(antiquantScaleFp16DeviceAddr);
  aclrtFree(yFp16DeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```