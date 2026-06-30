# aclnnQuantMatmulReduceSumWeightNz

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/matmul/quant_matmul_reduce_sum)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |    ×     |
|  <term>Atlas training series products</term>   |     ×    |

## Function

- Description: Performs quantized group matrix computation, adds the matrix computation results of all groups, and outputs the result.

- Formula:

$$
out = \sum_{i=0}^{batch}(x1_i @ x2_i) * x1Scale * x2Scale
$$


## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnQuantMatmulReduceSumWeightNz** is called to perform computation.

- `aclnnStatus aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale, const aclTensor *x2Scale, const aclTensor *yScale, const aclTensor *x1Offset, const aclTensor *x2Offset, const aclTensor *yOffset, const aclTensor *bias, bool transposeX1, bool transposeX2, int64_t groupSize, const aclIntArray *dims, bool keepDims, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`

- `aclnnStatus aclnnQuantMatmulReduceSumWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize

- **Parameters:**
  - **x1** (aclTensor*, compute input): input **x1** in the formula, aclTensor on the device. The [data format](../../../docs/en/context/data_formats.md) can be ND. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are not supported. The shape can be three-dimensional (batch, m, k). The data type can be INT8.

  - **x2** (aclTensor*, compute input): input **x2** in the formula, aclTensor on the device. The data type can be INT8. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are not supported. The [data format](../../../docs/en/context/data_formats.md) supports the AI processor affinity data layout format. The shape can be five-dimensional.
    - When **transposeX2** is **false**, the shape is represented by (batch, n1, k1, k0, n0), where k0 = 16 and n0 = 32. k in the shape of **x1** and k1 in the shape of **x2** must meet the following relationship: ceil(k/16) = k1. n1 in the shape of **x2** and n in the shape of **out** must meet the following relationship: ceil(n/n0) = n1.
    - **aclnnCalculateMatmulWeightSizeV2** and **aclnnTransMatmulWeight** can be used to convert the input format from ND to AI processor affinity format. The original shape in ND format is (batch, k, n).

  - **x1Scale** (aclTensor*, compute input): input **x1Scale** in the formula, aclTensor on the device. The [data format](../../../docs/en/context/data_formats.md) can be ND. The shape can be two-dimensional (batch, m). The data type can be FLOAT32.
    - During actual computation, **x1Scale** is broadcast as (batch, m, n).
    
  - **x2Scale** (aclTensor*, compute input): quantization parameter, **x2Scale** in the formula, aclTensor on the device. The [data format](../../../docs/en/context/data_formats.md) can be ND. The shape can be one-dimensional (n,), where n is the same as that of **x2**. The data type can be BFLOAT16.
    - During actual computation, **x2Scale** is broadcast as (batch, m, n).

  - **yScale** (aclTensor*, compute input): reserved parameter. This parameter is not supported in the current version. It must be set to **nullptr**.
  
  - **x1Offset** (aclTensor*, compute input): reserved parameter. This parameter is not supported in the current version. It must be set to **nullptr**.
  
  - **x2Offset** (aclTensor*, compute input): reserved parameter. This parameter is not supported in the current version. It must be set to **nullptr**.

  - **yOffset** (aclTensor*, compute input): reserved parameter. This parameter is not supported in the current version. It must be set to **nullptr**.
  
  - **bias** (aclTensor*, compute input): reserved parameter. This parameter is not supported in the current version. It must be set to **nullptr**.
  
  - **transposeX1** (bool, compute input): whether the input shape of **x1** is transposed. In the current version, only **false** is supported, indicating that the meaning of the input shape of **x1** remains unchanged.

  - **transposeX2** (bool, compute input): whether the input shape of **x2** is transposed. In the current version, only **false** is supported, indicating that the meaning of the input shape of **x2** remains unchanged.

  - **groupSize** (int64_t, compute input): reserved parameter. This parameter is not supported in the current version. It must be set to **0**.
  
  - **dims** (aclIntArray *): aclIntArray on the host, which specifies the reduce dimension. The data type can be INT64. In the current version, only **[0]** is supported, indicating that ReduceSum is performed on the 0th dimension (batch dimension).

  - **keepDims** (bool, compute input): whether to retain the dimensions of the input tensor in the output tensor. In the current version, only **false** is supported.

  - **out** (aclTensor*, compute output): output **out** in the formula, aclTensor on the device. The [data format](../../../docs/en/context/data_formats.md) can be ND. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The data type can be BFLOAT16. The shape can be two-dimensional (m, n).

  - **workspaceSize** (uint64_t*, output): size of the workspace to be allocated on the device.
  - **executor** (aclOpExecutor**, output): operator executor, containing the operator computation process.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed x1, x2, x1Scale, x2Scale, or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of x1, x2, x1Scale, x2Scale, or out is not supported.
                                    2. The shape of x1, x2, x1Scale, x2Scale, or out does not meet the verification condition.
                                    3. x1, x2, x1Scale, x2Scale, or out is an empty tensor.
  ```

## aclnnQuantMatmulReduceSumWeightNz

- **Parameters:**

  - **workspace** (void *, input): address of the workspace to be allocated on the device.

  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize**.

  - **executor** (aclOpExecutor*, input): operator executor, containing the operator computation process.

  - **stream** (aclrtStream, input): stream for executing the task.


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
- Deterministic description:
  - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: **aclnnQuantMatmulReduceSumWeightNz** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic compute.
  

The input and output support the following data type combinations:

| x1   | x2   | x1Scale  | x2Scale  | yScale | x1Offset | x2Offset | yOffset | bias  | out     |
|------|------|----------|----------|--------|----------|----------|---------|-------|---------|
| INT8 | INT8 | FLOAT32  | BFLOAT16 | null   | null     | null     | null    | null  |BFLOAT16 |


## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).


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
      // (Fixed writing) Initialize resources.
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
      // Call aclrtMalloc to allocate memory on the device.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy the data on the host to the memory on the device.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call aclCreateTensor to create an aclTensor.
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

      // Call aclrtMalloc to allocate memory on the device.
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy the data on the host to the memory on the device.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // Call aclCreateTensor to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      return 0;
  }

  int aclnnQuantMatmulWeightNzTest(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output based on the API.
      int64_t b = 8;
      int64_t m = 2048;
      int64_t k = 1024;
      int64_t n = 7168;
      // Create an x1 aclTensor.
      std::vector<int64_t> x1Shape = {b, m, k};
      void *x1DeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      std::vector<int8_t> x1HostData(b * m * k, 1);
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2 aclTensor in AI processor affinity format.
      std::vector<int64_t> x2Shape = {b, k, n};
      void *x2DeviceAddr = nullptr;
      aclTensor *x2 = nullptr;
      std::vector<int8_t> x2HostData(b * k * n, 1);
      ret = CreateAclTensorX2(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2HPTensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2HPDeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x1Scale aclTensor.
      std::vector<int64_t> x1ScaleShape = {b, m};
      void *x1ScaleDeviceAddr = nullptr;
      std::vector<float> x1ScaleHostData(b * m, 1);
      aclTensor *x1Scale = nullptr;
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2Scale aclTensor.
      std::vector<int64_t> x2ScaleShape = {n};
      void *x2ScaleDeviceAddr = nullptr;
      aclTensor *x2Scale = nullptr;
      std::vector<uint16_t> x2ScaleHostData(n, 1);  // The output data is actually in bfloat16 half-precision mode.
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_BF16, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an out aclTensor.
      std::vector<int64_t> outShape = {m, n};
      void *outDeviceAddr = nullptr;
      aclTensor *out = nullptr;
      std::vector<uint16_t> outHostData(m * n, 1);  // The output data is actually in bfloat16 half-precision mode.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BF16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;
      // Create a dims aclIntArray.
      std::vector<int64_t> dimsData = {0};
      aclIntArray *dims = nullptr;
      dims = aclCreateIntArray(dimsData.data(), dimsData.size());
      CHECK_RET(dims != nullptr, return ret);

      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;
      // Call the first-phase API of aclnnQuantMatmulReduceSumWeightNz.
      ret = aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize(
        x1, x2, x1Scale, x2Scale, nullptr, nullptr, nullptr, nullptr, nullptr, transposeX1, transposeX2, 0,
        dims, false, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnQuantMatmulReduceSumWeightNz.
      ret = aclnnQuantMatmulReduceSumWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulReduceSumWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // The bfloat16 data cannot be directly printed in the C language. The data needs to be read by using uint16 and converted into fp16 in binary mode.
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
      // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
      // Set the device ID in use.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulWeightNzTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulWeightNzTest failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }

```
