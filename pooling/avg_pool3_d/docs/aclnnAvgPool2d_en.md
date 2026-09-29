# aclnnAvgPool2d

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    √     |
| <term>Atlas training products</term>                             |    √     |

## Function

- This API performs 2D average pooling on the input tensor with the kernel size of $kH * kW$ and the stride of $sH * sW$. $k$ is kernelSize, indicating the size of the pooling window. $s$ is stride, indicating the stride of the pooling operation.
- Formula:
  The relationship among input ($N,C,H,W$), output out ($N,C,H_{out},W_{out}$), pooling step ($strides$), and pooling window size ($kH,kW$) is as follows:

  $$
  H_{out}=\lfloor \frac{H_{in}+2*paddings[0]-kH}{strides[0]}+1 \rfloor
  $$

  $$
  W_{out}=\lfloor \frac{W_{in}+2*paddings[1]-kW}{strides[1]}+1 \rfloor
  $$

  $$
  out(N_i,C_i,h,w)=\frac{1}{kH*kW}\sum_{m=0}^{kH-1}\sum_{n=0}^{kW-1}input(N_i,C_i,strides[0]*h+m,strides[1]*w+n)
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnAvgPool2dGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnAvgPool2d** is called to perform computation.

```Cpp
aclnnStatus aclnnAvgPool2dGetWorkspaceSize(
  const aclTensor   *self,
  const aclIntArray *kernelSize,
  const aclIntArray *strides,
  const aclIntArray *paddings,
  const bool         ceilMode,
  const bool         countIncludePad,
  const int64_t      divisorOverride,
  const int8_t       cubeMathType,
  aclTensor         *out,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnAvgPool2d(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAvgPool2dGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1478px"><colgroup>
  <col style="width: 149px">
  <col style="width: 121px">
  <col style="width: 264px">
  <col style="width: 253px">
  <col style="width: 262px">
  <col style="width: 148px">
  <col style="width: 135px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Tensor to be converted, which is the input in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW, NCL</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>Input</td>
      <td>Size of the pooling window, which is k in the formula.</td>
      <td>The value is 1 (kH = kW) or 2 (kH, kW), and must be greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Input</td>
      <td>Stride of the pooling operation, which is strides in the formula.</td>
      <td>The value is 1 (sH = sW) or 2 (sH, sW), and must be greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>paddings</td>
      <td>Input</td>
      <td>Number of zero-padded layers in the H and W directions of the input, which is paddings in the formula.</td>
      <td>The value is 1 (padH = padW) or 2 (padH, padW), and cannot be less than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceilMode</td>
      <td>Input</td>
      <td>Whether the derived output shape is rounded up.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>countIncludePad</td>
      <td>Input</td>
      <td>Whether to include the padding 0s when calculating average pooling.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divisorOverride</td>
      <td>Input</td>
      <td>Divisor for average calculation.</td>
      <td>If this parameter is set to 0, the function is disabled.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>compute logic of the Cube unit.</td>
      <td>-</td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor, which is out in the formula.</td>
      <td>-</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW, NCL</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>
  For <term>Atlas inference products</term> and <term>Atlas training products</term>, the data types of the self and out parameters do not support BFLOAT16.

  cubeMathType: The following enumerated values are supported:
    * 0: KEEP_DTYPE. The input data type is retained for computation.
      * <term>Atlas training products</term> and <term>Atlas inference products</term>: This option is not supported when the input data type is FLOAT32.
    * 1: ALLOW_FP32_DOWN_PRECISION. The input data can be computed with reduced precision.
      * <term>Atlas training products</term> and <term>Atlas inference products</term>: If the input data type is FLOAT32, it will be converted to FLOAT16 for computation. When the input is of other data types, it is not processed.
      * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: This option is not supported when the input data type is FLOAT32. When the input is of other data types, it is not processed.
    * 2: USE_FP16. The input data can be downgraded to FLOAT16 for computation.
      * <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: This option is not supported when the input data type is BFLOAT16.
    * 3: USE_HF32. This option is not supported.
- **Returns:**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
  </colgroup>
  <thead>
    <tr>
      <th>Return code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input self, kernelSize, strides, paddings, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data type or format of the input self or out is not supported.</td>
    </tr>
    <tr>
      <td>The data type or format of the input self is inconsistent with that of the output out.</td>
    </tr>
    <tr>
      <td>The value of a dimension of the input self, paddings, or out is less than 0.</td>
    </tr>
    <tr>
      <td>The value of a dimension of the input kernelSize or strides is less than or equal to 0.</td>
    </tr>
    <tr>
      <td>The length of the input kernelSize or paddings is less than 1.</td>
    </tr>
    <tr>
      <td>The output shape calculated based on the average pooling semantics is inconsistent with the specified shape.</td>
    </tr>
    <tr>
      <td>The padding attribute exceeds half of the kernel size. For example, paddingH = 2, kernelSizeH = 2, and paddingH>kernelSizeH*1/2.</td>
    </tr>
  </tbody>
  </table>

## aclnnAvgPool2d

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnAvgPool2dGetWorkspaceSize API.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Returns:**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - **aclnnAvgPool2d** defaults to a deterministic implementation.

- <term>Atlas training products</term>: The Cube unit does not support FLOAT32 computation. The input data type FLOAT32 can be converted to FLOAT16 in the API for computation by setting **cubeMathType** to **1** (**ALLOW_FP32_DOWN_PRECISION**).

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_avgpool2d.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                                     \
      if (!(cond)) {                       \
          Finalize(deviceId, stream);      \
          return_expr;                     \
      }                                    \
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy the data from the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream& stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnAvgPool2dTest(int32_t deviceId, aclrtStream& stream) {
  auto ret = Init(deviceId, &stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  int64_t divisorOverride = 0;
  bool countIncludePad = true;
  bool ceilMode = false;
  int8_t cubeMathType = 2;

  std::vector<int64_t> selfShape = {1, 16, 4, 4};
  std::vector<int64_t> outShape = {1, 16, 1, 1};

  std::vector<int64_t> kernelDims = {4, 4};
  std::vector<int64_t> strideDims = {1, 1};
  std::vector<int64_t> paddingDims = {0, 0};

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData(256, 2);
  std::vector<float> outHostData(16, 0);

  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> selfTensorPtr(self, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> selfDeviceAddrPtr(selfDeviceAddr, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create a kernel aclIntArray.
  aclIntArray *kernelSize = aclCreateIntArray(kernelDims.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> kernelSizePtr(kernelSize, aclDestroyIntArray);
  CHECK_FREE_RET(kernelSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // Create a stride aclIntArray.
  aclIntArray *strides = aclCreateIntArray(strideDims.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> stridesPtr(strides, aclDestroyIntArray);
  CHECK_FREE_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // Create a paddings aclIntArray.
  aclIntArray *paddings = aclCreateIntArray(paddingDims.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> paddingsPtr(paddings, aclDestroyIntArray);
  CHECK_FREE_RET(paddings != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnAvgPool2d.
  ret = aclnnAvgPool2dGetWorkspaceSize(self,
                                       kernelSize,
                                       strides,
                                       paddings,
                                       ceilMode,
                                       countIncludePad,
                                       divisorOverride,
                                       cubeMathType,
                                       out,
                                       &workspaceSize,
                                       &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAvgPool2dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // Call the second-phase API of aclnnAvgPool2d.
  ret = aclnnAvgPool2d(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAvgPool2d failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out result[%ld] is: %f\n", i, outData[i]);
  }
  return ACL_SUCCESS;
}

int main() {
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnAvgPool2dTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAvgPool2dTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
