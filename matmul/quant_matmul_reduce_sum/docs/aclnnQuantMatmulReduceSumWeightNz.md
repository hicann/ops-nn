# aclnnQuantMatmulReduceSumWeightNz

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品 </term>    |    ×     |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：完成量化的分组矩阵计算，然后所有组的矩阵计算结果相加后输出

- 计算公式：

$$
out = \sum_{i=0}^{batch}(x1_i @ x2_i) * x1Scale * x2Scale
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnQuantMatmulReduceSumWeightNz”接口执行计算。

```Cpp
aclnnStatus aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize(
    const aclTensor   *x1, 
    const aclTensor   *x2, 
    const aclTensor   *x1Scale, 
    const aclTensor   *x2Scale, 
    const aclTensor   *yScale, 
    const aclTensor   *x1Offset, 
    const aclTensor   *x2Offset, 
    const aclTensor   *yOffset, 
    const aclTensor   *bias, 
    bool               transposeX1, 
    bool               transposeX2, 
    int64_t            groupSize, 
    const aclIntArray *dims,
    bool               keepDims, 
    aclTensor         *out, 
    uint64_t          *workspaceSize, 
    aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnQuantMatmulReduceSumWeightNz(
    void          *workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```

## aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize

- **参数说明：**
  - x1（aclTensor*，计算输入）：公式中的输入x1，device侧的aclTensor。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，shape支持3维，形状为（batch, m, k）。数据类型支持INT8。

  - x2（aclTensor*，计算输入）：公式中的输入x2，device侧的aclTensor。数据类型支持INT8，不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持AI处理器亲和数据排布格式。shape支持5维。
    - 在transposeX2为false情况下各个维度表示：（batch，n1，k1，k0，n0），其中k0 = 16， n0 = 32， x1 shape中的k和x2 shape中的k1需要满足以下关系：ceil（k / 16） = k1, x2 shape中的n1与out的n满足以下关系: ceil(n / n0) = n1。
    - 可使用aclnnCalculateMatmulWeightSizeV2接口以及aclnnTransMatmulWeight接口完成输入Format从ND到AI处理器亲和数据排布格式的转换。原始的ND格式的shape为(batch, k, n)。

  - x1Scale（aclTensor*，计算输入）：公式中的输入x1Scale，device侧的aclTensor。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。shape支持2维，形状为（batch，m）。数据类型支持FLOAT32。
    - 在实际计算时，x1Scale会被广播为(batch，m，n)。
    
  - x2Scale（aclTensor*，计算输入）：表示量化参数，公式中的输入x2Scale，device侧的aclTensor。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。shape支持1维，形状为（n，）其中n与x2的n一致。数据类型支持BFLOAT16。
    - 在实际计算时，x2Scale会被广播为(batch，m，n)。

  - yScale（aclTensor*，计算输入）：预留参数，当前版本不支持，需要传入nullptr。
  
  - x1Offset（aclTensor*，计算输入）：预留参数，当前版本不支持，需要传入nullptr。
  
  - x2Offset（aclTensor*，计算输入）：预留参数，当前版本不支持，需要传入nullptr。

  - yOffset（aclTensor*，计算输入）：预留参数，当前版本不支持，需要传入nullptr。
  
  - bias（aclTensor*，计算输入）：预留参数，当前版本不支持，需要传nullptr。
  
  - transposeX1（bool，计算输入）：表示x1的输入shape是否包含transpose。当前版本仅支持false，表示x1的输入shape意义不变。

  - transposeX2（bool，计算输入）：表示x2的输入shape是否包含transpose，当前版本仅支持false，表示x2的输入shape意义不变。

  - groupSize（int64_t，计算输入）：预留参数，当前版本不支持，需要传入0。
  
  - dims（aclIntArray *）：Host侧的aclIntArray，指定reduce维度，数据类型支持INT64。当前版本仅支持填[0]，表示在第0维（batch维）做ReduceSum。

  - keepDims（bool, 计算输入）：指定是否在输出张量中保留输入张量的维度。当前版本仅支持false。

  - out（aclTensor*，计算输出）：公式中的输出out，device侧的aclTensor。[数据格式](../../../docs/zh/context/数据格式.md)支持ND。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，数据类型支持BFLOAT16。shape支持2维，形状为（m, n）。

  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 281px">
    <col style="width: 119px">
    <col style="width: 749px">
    </colgroup>
    <thead>
    <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>传入的x1、x2、x1Scale、x2Scale或out是空指针。</td>
    </tr>
    <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>x1、x2、x1Scale、x2Scale或out的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
        <td>x1、x2、x1Scale、x2Scale或out的shape不满足校验条件。</td>
    </tr>
    <tr>
        <td>x1、x2、x1Scale、x2Scale或out是空tensor。</td>
    </tr>
    </tbody>
    </table>


## aclnnQuantMatmulReduceSumWeightNz

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
    </colgroup>
    <thead>
    <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize获取。</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream。</td>
    </tr>
    </tbody>
    </table>

- **返回值：**

  aclnnStatus： 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性说明：
  - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：aclnnQuantMatmulReduceSumWeightNz默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
  
输入和输出支持以下数据类型组合：

| x1   | x2   | x1Scale  | x2Scale  | yScale | x1Offset | x2Offset | yOffset | bias  | out     |
|------|------|----------|----------|--------|----------|----------|---------|-------|---------|
| INT8 | INT8 | FLOAT32  | BFLOAT16 | null   | null     | null     | null    | null  |BFLOAT16 |

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

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
      // 创建AI处理器亲和数据排布格式的x2 aclTensor
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
