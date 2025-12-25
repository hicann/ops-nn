# aclnnQuantMatmulWeightNz

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √    |

## 功能说明

- 算子功能：完成量化的矩阵乘计算。相似接口有aclnnMm（仅支持2维Tensor作为输入的矩阵乘）和aclnnBatchMatMul（仅支持三维的矩阵乘，其中第一维是Batch维度）。支持T-C、T-T、K-C、 K-T[量化模式](../../../docs/zh/context/量化介绍.md)。

- 计算公式：

  - 无x1Scale无bias：

  $$
  out = x1@x2 * x2Scale + x2Offset
  $$

  - bias INT32：

  $$
  out = (x1@x2 + bias) * x2Scale + x2Offset
  $$

  - bias BFLOAT16/FLOAT32（此场景无x2Offset）：

  $$
  out = x1@x2 * x2Scale + bias
  $$

  - x1Scale无bias：

  $$
  out = x1@x2 * x2Scale * x1Scale
  $$

  - x1Scale， bias INT32(此场景无x2Offset)：

  $$
  out = (x1@x2 + bias) * x2Scale * x1Scale
  $$

  - x1Scale， bias BFLOAT16/FLOAT16/FLOAT32（此场景无x2Offset）：

  $$
  out = x1@x2 * x2Scale * x1Scale + bias
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnQuantMatmulWeightNzGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnQuantMatmulWeightNz”接口执行计算。

- `aclnnStatus aclnnQuantMatmulWeightNzGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale, const aclTensor *x2Scale, const aclTensor *yScale, const aclTensor *x1Offset, const aclTensor *x2Offset, const aclTensor *yOffset, const aclTensor *bias, bool transposeX1, bool transposeX2, int64_t groupSize, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`

- `aclnnStatus aclnnQuantMatmulWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnQuantMatmulWeightNzGetWorkspaceSize

- **参数说明：**

  - x1（aclTensor*，计算输入）：公式中的输入x1，device侧的aclTensor。数据类型支持INT8，支持最后两根轴转置情况下的非连续tensor，其他场景的[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)不支持，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，shape支持2~6维。
    - 在transposeX1为false情况下各个维度表示：（batch，m，k），batch可不存在。
    - 在transposeX1为true情况下各个维度表示：（batch，k，m），batch可不存在。
  - x2（aclTensor*，计算输入）：公式中的输入x2，device侧的aclTensor。数据类型支持INT8，[数据格式](../../../docs/zh/context/数据格式.md)支持AI处理器亲和数据排布格式。shape支持4~8维。
    - 在transposeX2为true情况下各个维度表示：（batch，k1，n1，n0，k0），batch可不存在，其中k0 = 32， n0 = 16， x1 shape中的k和x2 shape中的k1需要满足以下关系：ceil（k / 32） = k1, x2 shape中的n1与out的n满足以下关系: ceil(n / n0) = n1。
    - 在transposeX2为false情况下各个维度表示：（batch，n1，k1，k0，n0），batch可不存在，其中k0 = 16， n0 = 32， x1 shape中的k和x2 shape中的k1需要满足以下关系：ceil（k / 16） = k1, x2 shape中的n1与out的n满足以下关系: ceil(n / n0) = n1。
    - 可使用aclnnCalculateMatmulWeightSizeV2接口以及aclnnTransMatmulWeight接口完成输入Format从ND到AI处理器亲和数据排布格式的转换。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持非连续的tensor。
  - x1Scale（aclTensor*，计算输入）：公式中的输入x1Scale，device侧的aclTensor。可选的量化参数。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT32，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，shape是1维（t，），t = m，其中m与x1的m一致。
  - x2Scale（aclTensor*，计算输入）：公式中的输入x2Scale，device侧的aclTensor。量化参数，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，shape是1维（t，），t = 1或n，其中n与x2的n一致。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持UINT64、INT64、FLOAT32、BFLOAT16。
    - 当原始输入类型不满足[约束说明](#约束说明)中组合时，需提前调用TransQuantParamV2算子的aclnn接口来将scale转成INT64、UINT64数据类型。
  - yScale（aclTensor*，计算输入）：预留参数，当前版本不支持，需要传入nullptr或者空tensor。
  - x1Offset（aclTensor*，计算输入）：预留参数，当前版本不支持，需要传入nullptr或者空tensor。
  - x2Offset（aclTensor*，计算输入）：公式中的输入x2Offset，device侧的aclTensor。可选量化参数，数据类型支持FLOAT32，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，shape是1维（t，），t = 1或n，其中n与x2的n一致。
  - yOffset（aclTensor*，计算输入）：预留参数，当前版本不支持，需要传入nullptr或者空tensor。
  - bias（aclTensor*，计算输入）：公式中的输入bias，device侧的aclTensor。可选参数。[数据格式](../../../docs/zh/context/数据格式.md)支持ND，shape支持1维（n，）或3维（batch，1，n），n与x2的n一致。当out的shape为2、4、5、6维时，bias的shape只支持1维（n，）。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持INT32、BFLOAT16、FLOAT16、FLOAT32。
  - transposeX1（bool，计算输入）：表示x1的输入shape是否包含transpose，在transposeX1为false情况下各个维度表示：（batch，m，k），在transposeX1为true情况下各个维度表示：（batch，k，m），batch可不存在。
  - transposeX2（bool，计算输入）：表示x2的输入shape是否包含transpose，在transposeX2为true情况下各个维度表示：（batch，k1，n1，n0，k0），batch可不存在，其中k0 = 32，n0 = 16，x1 shape中的k和x2 shape中的k1需要满足以下关系：ceil（k / 32） = k1。在transposeX2为false情况下各个维度表示：（batch，n1，k1，k0，n0），batch可不存在，其中k0 = 16，n0 = 32，x1 shape中的k和x2 shape中的k1需要满足以下关系：ceil（k / 16） = k1。
  - groupSize（int64_t，计算输入）：预留参数，当前版本不支持，需要传入0。
  - out（aclTensor*，计算输出）：公式中的输出out，device侧的aclTensor。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，shape支持2~6维，（batch，m，n），batch可不存在，支持x1与x2的batch维度broadcast，输出batch与broadcast之后的batch一致，m与x1的m一致，n与x2的n1以及n0满足ceil(n / n0) = n1的关系。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、INT8、BFLOAT16、INT32。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的x1、x2、x2Scale或out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. x1、x2、bias、x2Scale、x2Offset或out的数据类型和数据格式不在支持的范围之内。
                                        2. x1、x2、bias、x2Scale、x2Offset或out的shape不满足校验条件。
                                        3. x1、x2、bias、x2Scale、x2Offset或out是空tensor。
                                        4. x1与x2的最后一维大小超过65535，x1的最后一维指transposeX1为true时的m或transposeX1为false时的k，x2的最后一维指transposeX2为true时的k或transposeX2为false时的n。
                                        5. 输入的yScale、x1Offset和yOffset不是nullptr并且不是空tensor。
                                        6. groupSize不为0。
  ```

## aclnnQuantMatmulWeightNz

- **参数说明：**

  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnQuantMatmulWeightNzGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性计算
  - aclnnQuantMatmulWeightNz默认确定性实现。
  
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持调用本接口前，通过[aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md)对format为ND的x2处理得到AI处理器亲和数据排布格式。
输入和输出支持以下数据类型组合：

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

  | x1   | x2   | x1Scale      | x2Scale          | x2Offset      | bias                        | out      |
  | ---- | ---- | ------------ | ---------------- | ------------- | --------------------------- | -------- |
  | INT8 | INT8 | null         | UINT64/INT64     | null          | null/INT32                  | FLOAT16  |
  | INT8 | INT8 | null         | UINT64/INT64     | null/FLOAT32  | null/INT32                  | INT8     |
  | INT8 | INT8 | null/FLOAT32 | FLOAT32/BFLOAT16 | null          | null/INT32/BFLOAT16/FLOAT32 | BFLOAT16 |
  | INT8 | INT8 | FLOAT32      | FLOAT32          | null          | null/INT32/FLOAT16/FLOAT32  | FLOAT16  |
  | INT8 | INT8 | null         | FLOAT32/BFLOAT16 | null          | null/INT32                  | INT32    |

以下数据类型组合在x1Scale为null时，支持T-C && T-T[量化模式](../../../docs/zh/context/量化介绍.md)，
在x1Scale不为null时支持K-C量化 && K-T[量化模式](../../../docs/zh/context/量化介绍.md)。

  | x1   | x2   | x1Scale      | x2Scale          | x2Offset      | bias                        | out      |
  | ---- | ---- | ------------ | ---------------- | ------------- | --------------------------- | -------- |
  | INT8 | INT8 | null         | UINT64/INT64     | null          | null/INT32                  | FLOAT16/BFLOAT16  |
  | INT8 | INT8 | null         | UINT64/INT64     | null/FLOAT32  | null/INT32                  | INT8     |
  | INT8 | INT8 | null/FLOAT32 | FLOAT32/BFLOAT16 | null          | null/INT32/BFLOAT16/FLOAT32 | BFLOAT16 |
  | INT8 | INT8 | FLOAT32      | FLOAT32          | null          | null/INT32/FLOAT16/FLOAT32  | FLOAT16  |

## 调用示例

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：x2为AI处理器亲和数据排布格式场景下的示例代码如下(transposeX2=false)，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_permute.h"
  #include "aclnnop/aclnn_quant_matmul_weight_nz.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"
  #include "aclnnop/aclnn_trans_quant_param_v2.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
  {
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
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
  {
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

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  template <typename T>
  int CreateAclTensorX2(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(shape));

      const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret);
                return ret);
      size *= sizeof(T);

      // 调用aclrtMalloc申请device侧内存
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      return 0;
  }

  int aclnnQuantMatmulWeightNzTest(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> x1Shape = {5, 32};
      std::vector<int64_t> x2Shape = {32, 32};
      std::vector<int64_t> biasShape = {32};
      std::vector<int64_t> offsetShape = {32};
      std::vector<int64_t> scaleShape = {32};
      std::vector<int64_t> outShape = {5, 32};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *quantParamDeviceAddr = nullptr;
      void *offsetDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *scale = nullptr;
      aclTensor *quantParam = nullptr;
      aclTensor *offset = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(5 * 32, 1);
      std::vector<int8_t> x2HostData(32 * 32, 1);
      std::vector<int32_t> biasHostData(32, 1);
      std::vector<float> scaleHostData(32, 1);
      std::vector<float> offsetHostData(32, 1);
      std::vector<uint16_t> outHostData(5 * 32, 1);  // 实际上是float16半精度方式
      // 创建x1 aclTensor
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建AI处理器亲和数据排布格式的x2 aclTensor
      ret = CreateAclTensorX2(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2HPTensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2HPDeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建scale aclTensor
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建quantParam aclTensor
      ret = CreateAclTensor(scaleHostData, scaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> quantParamTensorPtr(quantParam,
                                                                                        aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建offset aclTensor
      ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建bias aclTensor
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建out aclTensor
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;

      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      // 调用aclnnTransMatmulWeight第一段接口
      ret = aclnnTransMatmulWeightGetWorkspaceSize(x2, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrTrans(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrTrans.reset(workspaceAddr);
      }
      // 调用aclnnTransMatmulWeight第二段接口
      ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

      // FLOAT数据类型的scale需要提前调用TransQuantParamV2算子的aclnn接口
      // 调用aclnnTransQuantParamV2第一段接口
      ret = aclnnTransQuantParamV2GetWorkspaceSize(scale, offset, quantParam, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV2(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV2.reset(workspaceAddr);
      }
      // 调用aclnnTransQuantParamV2第二段接口
      ret = aclnnTransQuantParamV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);

      // 调用aclnnQuantMatmulWeightNz第一段接口
      workspaceSize = 0;
      ret = aclnnQuantMatmulWeightNzGetWorkspaceSize(x1, x2, nullptr, quantParam, nullptr, nullptr, nullptr, nullptr,
                                                    bias, transposeX1, transposeX2, 0, out, &workspaceSize,
                                                    &executor);

      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存

      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrNZ(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrNZ.reset(workspaceAddr);
      }
      // 调用aclnnQuantMatmulWeightNz第二段接口
      ret = aclnnQuantMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // C语言中无法直接打印fp16的数据，需要用uint16读出来，自行通过二进制转成fp16
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. （固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulWeightNzTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzTest failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

