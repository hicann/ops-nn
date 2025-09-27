# aclnnQuantMatmulReduceSumWeightNz

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：完成量化的分组矩阵计算，然后所有组的矩阵计算结果相加后输出

- 计算公式：

$$
out = \sum_{i=0}^{batch}(x1_i @ x2_i) * x1Scale * x2Scale
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnQuantMatmulReduceSumWeightNz”接口执行计算。

- `aclnnStatus aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale, const aclTensor *x2Scale, const aclTensor *yScale, const aclTensor *x1Offset, const aclTensor *x2Offset, const aclTensor *yOffset, const aclTensor *bias, bool transposeX1, bool transposeX2, int64_t groupSize, const aclIntArray *dims, bool keepDims, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`

- `aclnnStatus aclnnQuantMatmulReduceSumWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize

- **参数说明：**

  | 参数名  | 输入/输出 | 描述   | 使用说明    | 数据类型    | 数据格式 | 维度（shape） |
  |--------|-----------|--------|------------|------------|---------|---------------|
  | x1 | 输入 | 公式中的输入x1，device侧的aclTensor。 | 不支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)。 | INT8 | ND | 3维，(batch, m, k)。 |
  | x2 | 输入 | 公式中的输入x2，device侧的aclTensor。 | 不支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)。 | INT8 | NZ | 5维，由3维(batch, k, n)的ND格式转换为NZ得到。 |
  | x1Scale | 输入 | 公式中的输入x1Scale，device侧的aclTensor。 |   | FLOAT32 | ND | 2维，(batch, m)，实际计算时被广播为(batch, m, n)。 |
  | x2Scale | 输入 | 公式中的输入x2Scale，device侧的aclTensor。 |   | BFLOAT16 | ND | 1维，(n,)，实际计算时会被广播为(batch, m, n)。 |
  | yScale  | 输入 | 预留参数。 | 当前版本不支持，需传入nullptr。 |  |  |  |
  | x1Offset  | 输入 | 预留参数。 | 当前版本不支持，需传入nullptr。 |  |  |  |
  | x2Offset  | 输入 | 预留参数。 | 当前版本不支持，需传入nullptr。 |  |  |  |
  | yOffset  | 输入 | 预留参数。 | 当前版本不支持，需传入nullptr。 |  |  |  |
  | bias  | 输入 | 预留参数。 | 当前版本不支持，需传入nullptr。 |  |  |  |
  | transposeX1  | 输入 | x1的输入shape是否包含transpose。 | 当前版本仅支持false，表示x1的输入shape意义不变。 |  |  |  |
  | transposeX2  | 输入 | x2的输入shape是否包含transpose。 | 当前版本仅支持false，表示x2的输入shape意义不变。 |  |  |  |
  | groupSize  | 输入 | 预留参数。 | 当前版本不支持，需要传入0。 |  |  |  |
  | dims  | 输入 | 指定reduce维度，host侧的aclIntArray。 | 当前版本仅支持填[0]，表示在第0维（batch维）做ReduceSum。 | INT64 |  |  |
  | keepDims  | 输入 | 指定是否在输出张量中保留输入张量的维度。 | 当前版本仅支持false。 |  |  |  |
  | out  | 输出 | 公式中的输出out，device侧的aclTensor。 | 支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)。 | BFLOAT16 | ND | 2维，(m, n)。 |
  | workspaceSize  | 输出 | 返回需要在Device侧申请的workspace大小。 |  |  |  |  |
  | executor  | 输出 | 返回op执行器，包含了算子计算流程。 |  |  |  |  |

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。
  <table>
  <tr>
  <td align="center">返回值</td>
  <td align="center">错误码</td>
  <td align="center">描述</td>
  </tr>
  <tr>
  <td align="left">ACLNN_ERR_PARAM_NULLPTR</td>
  <td align="left">161001</td>
  <td align="left">传入的x1、x2、x1Scale、x2Scale或out是空指针。</td>
  </tr>
  <tr>
  <td rowspan="3" align="left">ACLNN_ERR_PARAM_INVALID</td>
  <td rowspan="3" align="left">161002</td>
  <td align="left">x1、x2、x1Scale、x2Scale或out的数据类型和数据格式不在支持的范围之内。</td>
  </tr>
  <tr><td align="left">x1、x2、x1Scale、x2Scaleout的shape不满足校验条件。</td></tr>
  <tr><td align="left">x1、x2、x1Scale、x2Scale或out是空tensor。</td></tr>
  </table>

## aclnnQuantMatmulReduceSumWeightNz

- **参数说明：**

  | 参数名         | 输入/输出    | 描述                                                                          |
  |---------------|------------- |------------------------------------------------------------------------------|
  | workspace     | 输入         | 在Device侧申请的workspace内存地址。                                             |
  | workspaceSize | 输入         | 在Device侧申请的workspace大小，由第一段接口aclnnIsFiniteGetWorkspaceSize获取。   |
  | executor      | 输入         | op执行器，包含了算子计算流程。                                                  |
  | stream        | 输入         | 指定执行任务的Stream。                                                         |

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

输入和输出支持以下数据类型组合：

| x1   | x2   | x1Scale  | x2Scale  | yScale | x1Offset | x2Offset | yOffset | bias  | out     |
|------|------|----------|----------|--------|----------|----------|---------|-------|---------|
| INT8 | INT8 | FLOAT32  | BFLOAT16 | null   | null     | null     | null    | null  |BFLOAT16 |

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_permute.h"
  #include "aclnnop/aclnn_quant_matmul_weight_nz.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"
  #include "aclnnop/aclnn_trans_quant_param_v2.h"
  #include "aclnnop/aclnn_quant_matmul_reduce_sum.h"

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
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      return 0;
  }

  int aclnnQuantMatmulWeightNzTest(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      int64_t b = 8;
      int64_t m = 2048;
      int64_t k = 1024;
      int64_t n = 7168;
      // 创建x1 aclTensor
      std::vector<int64_t> x1Shape = {b, m, k};
      void *x1DeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      std::vector<int8_t> x1HostData(b * m * k, 1);
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建昇腾亲和数据排布格式的x2 aclTensor
      std::vector<int64_t> x2Shape = {b, k, n};
      void *x2DeviceAddr = nullptr;
      aclTensor *x2 = nullptr;
      std::vector<int8_t> x2HostData(b * k * n, 1);
      ret = CreateAclTensorX2(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2HPTensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2HPDeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x1Scale aclTensor
      std::vector<int64_t> x1ScaleShape = {b, m};
      void *x1ScaleDeviceAddr = nullptr;
      std::vector<float> x1ScaleHostData(b * m, 1);
      aclTensor *x1Scale = nullptr;
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x2Scale aclTensor
      std::vector<int64_t> x2ScaleShape = {n};
      void *x2ScaleDeviceAddr = nullptr;
      aclTensor *x2Scale = nullptr;
      std::vector<uint16_t> x2ScaleHostData(n, 1);  // 实际上是bfloat16半精度方式
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_BF16, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建out aclTensor
      std::vector<int64_t> outShape = {m, n};
      void *outDeviceAddr = nullptr;
      aclTensor *out = nullptr;
      std::vector<uint16_t> outHostData(m * n, 1);  // 实际上是bfloat16半精度方式
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BF16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;
      // 创建dims aclIntArray
      std::vector<int64_t> dimsData = {0};
      aclIntArray *dims = nullptr;
      dims = aclCreateIntArray(dimsData.data(), dimsData.size());
      CHECK_RET(dims != nullptr, return ret);

      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;
      // 调用 aclnnQuantMatmulReduceSumWeightNz 第一段接口
      ret = aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize(
        x1, x2, x1Scale, x2Scale, nullptr, nullptr, nullptr, nullptr, nullptr, transposeX1, transposeX2, 0,
        dims, false, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // 调用 aclnnQuantMatmulReduceSumWeightNz 第二段接口
      ret = aclnnQuantMatmulReduceSumWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulReduceSumWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // C语言中无法直接打印bfloat16的数据，需要用uint16读出来，自行通过二进制转成fp16
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < 5; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
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
