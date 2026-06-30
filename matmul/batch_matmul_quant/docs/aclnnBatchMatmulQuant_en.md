# aclnnBatchMatmulQuant

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/matmul/batch_matmul_quant)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    ×     |

## Function

- Description:
Performs matrix multiplication with float16 input tensor and int8 output tensor.

- Formula:

  $$
  out = Quant(x1@x2 + bias)
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnBatchMatmulQuantGetWorkspaceSize** is called to obtain the input parameters and compute the required workspace size based on the process. Then, **aclnnBatchMatmulQuant** is called to perform computation.
```cpp
aclnnStatus aclnnBatchMatmulQuantGetWorkspaceSize(
  const aclTensor* x1, 
  const aclTensor* x2, 
  const aclTensor* quantParam, 
  const aclTensor* bias, 
  bool             transposeX1, 
  bool             transposeX2, 
  aclTensor*       out, 
  uint64_t*        workspaceSize, 
  aclOpExecutor**  executor)
```
```cpp
aclnnStatus aclnnBatchMatmulQuant(
  void*             workspace, 
  uint64_t          workspaceSize, 
  aclOpExecutor*    executor, 
  const aclrtStream stream)
```

## aclnnBatchMatmulQuantGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1587px"><colgroup>
  <col style="width: 159px">
  <col style="width: 127px">
  <col style="width: 230px">
  <col style="width: 400px">
  <col style="width: 249px">
  <col style="width: 117px">
  <col style="width: 117px">
  <col style="width: 153px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Precaution</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>Input x1 in the formula.</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>-</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input x2 in the formula.</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>-</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td>Input bias in the formula.</td>
      <td>The size of shape is equal to that of the last dimension of tensor out, and the input can be empty.</td>
      <td>FLOAT16</td>
      <td>-</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantParam</td>
      <td>Input</td>
      <td>Hardware quantization parameter, which can be obtained by calling aclnnTransQuantParam.</td>
      <td>The size of shape (number of elements) must meet any of the following conditions:<ul><li>The size of shape is 1.</li>
      <li>The size of shape is equal to that of the last dimension of tensor out aligned upwards to a multiple of 16.</li></ul></td>
      <td>UINT64</td>
      <td>NC1HWC0</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>transposeX1</td>
      <td>Input</td>
      <td>Whether to transpose x1.</td>
      <td>-</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX2</td>
      <td>Input</td>
      <td>Whether to transpose x2.</td>
      <td>-</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor, which is an aclTensor on the device.</td>
      <td>-</td>
      <td>INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace required to be allocated on the device.</td>
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

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 749px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed x1, x2, quantParam, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type of x1, x2, quantParam, or out is not supported.</td>
    </tr>
    <tr>
      <td>The data format of x1, x2, quantParam, or out is not supported.</td>
    </tr>
    <tr>
      <td>The dimension value of quantParam is not 1 or is not the size of the last dimension of tensor out aligned upwards to a multiple of 16.</td>
    </tr>
    <tr>
      <td>After the input shapes of x1 and x2 are processed based on the input transpose description, they do not meet the matrix multiplication relationship.</td>
    </tr>
    <tr>
      <td>The shape has 0, that is, an empty tensor.</td>
    </tr>
    <tr>
      <td>When bias exists, the size of the bias shape is inconsistent with the size of the last dimension of out.</td>
    </tr>
  </tbody>
  </table>

## aclnnBatchMatmulQuant

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API aclnnBatchMatmulQuantGetWorkspaceSize.</td>
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

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
- Determinism:
  - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: **aclnnBatchMatmulQuant** defaults to a deterministic implementation.
## Example
The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).
```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include "acl/acl.h"
#include "aclnnop/aclnn_batchmatmul_quant.h"
#include "aclnnop/aclnn_trans_quant_param.h"
#include "aclnnop/aclnn_cast.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      Finalize(deviceId, stream);\
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

void Finalize(int32_t deviceId, aclrtStream &stream) {
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnBatchMatmulQuantTest(int32_t deviceId, aclrtStream &stream) {
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> fMapShape = {2, 2};
  std::vector<int64_t> wtsShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  int64_t N = 2;
  void* fMapDeviceAddr = nullptr;
  void* fMapFp16DeviceAddr = nullptr;
  void* wtsDeviceAddr = nullptr;
  void* quantParamDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  std::vector<float> fMapHostData = {1, 1, 1, 1};
  std::vector<float> wtsHostData = {1, 1, 1, 1};
  std::vector<int8_t> outHostData = {0, 0, 0, 0};

  bool transposeX1 = false;
  bool transposeX2 = false;

  std::cout<<"host_side data processing..."<<std::endl;

  float quantOffset = 0;
  float quantScale = 1;
  std::vector<float>OffsetHostData = {0.0, 0.0};
  float* OffsetData = OffsetHostData.data();
  uint64_t OffsetSize = 2;
  std::vector<float>ScaleHostData = {1.0, 1.0};
  float* ScaleData = ScaleHostData.data();
  uint64_t ScaleSize = 2;

  // Get quantParam
  uint64_t quantParamSize = 0;
  uint64_t *quantParamData = nullptr;
  ret = aclnnTransQuantParam(ScaleData, ScaleSize, OffsetData, OffsetSize, &quantParamData, &quantParamSize);

  for (int64_t i = 0; i < quantParamSize; i++) {
    if (quantParamData == nullptr) {
        printf("ERROR: quantParamData[*ld] = nullptr", i);
        return ACL_SUCCESS;
    } else {
        printf("quantParamData[%ld] = %lu\n", i, quantParamData[i]);
    }
  }
  std::vector<uint64_t> quantParamHostData(quantParamData, quantParamData + quantParamSize);
  std::vector<int64_t> quantParamShape = {quantParamSize};
  std::cout<<"host_side data processing finish"<<std::endl;

  // create aclTensor
  aclTensor* fMap = nullptr;
  aclTensor* wts = nullptr;
  aclTensor* quantParam = nullptr;
  aclTensor* out = nullptr;
  aclTensor* fmapFp16 = nullptr;
  aclTensor* wtsFp16 = nullptr;

  // fmap aclTensor
  ret = CreateAclTensor(fMapHostData, fMapShape, &fMapDeviceAddr, aclDataType::ACL_FLOAT, &fMap);
  std::unique_ptr<void, aclError (*)(void *)> fMapDeviceAddrPtr(fMapDeviceAddr, aclrtFree);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> fMapPtr(fMap, aclDestroyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // fmapFp16 aclTensor
  ret = CreateAclTensor(fMapHostData, fMapShape, &fMapFp16DeviceAddr, aclDataType::ACL_FLOAT16, &fmapFp16);
  std::unique_ptr<void, aclError (*)(void *)> fMapFp16DeviceAddrPtr(fMapFp16DeviceAddr, aclrtFree);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> fmapFp16Ptr(fmapFp16, aclDestroyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // wts aclTensor
  ret = CreateAclTensor(wtsHostData, wtsShape, &wtsDeviceAddr, aclDataType::ACL_FLOAT, &wts);
  std::unique_ptr<void, aclError (*)(void *)> wtsDeviceAddrPtr(wtsDeviceAddr, aclrtFree);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> wtsPtr(wts, aclDestroyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // wtsFp16 aclTensor
  void* wtsFp16DeviceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> wtsFp16DeviceAddrPtr(wtsFp16DeviceAddr, aclrtFree);
  ret = CreateAclTensor(wtsHostData, wtsShape, &wtsFp16DeviceAddr, aclDataType::ACL_FLOAT16, &wtsFp16);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> wtsFp16Ptr(wtsFp16, aclDestroyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // quantPre aclTensor
  ret = CreateAclTensor(quantParamHostData, quantParamShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
  std::unique_ptr<void, aclError (*)(void *)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> quantParamPtr(quantParam, aclDestroyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
  std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outPtr(out, aclDestroyTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::cout<<"CreateAclTensor finish"<<std::endl;

  // 3. CANN API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  // aclnnCastfp16
  // fmap
  ret = aclnnCastGetWorkspaceSize(fMap, aclDataType::ACL_FLOAT16, fmapFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  void* fmapCastWorkspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> fmapCastWorkspacePtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&fmapCastWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    fmapCastWorkspacePtr.reset(fmapCastWorkspaceAddr);
  }
  ret = aclnnCast(fmapCastWorkspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // wts
  workspaceSize = 0;
  executor = nullptr;
  ret = aclnnCastGetWorkspaceSize(wts, aclDataType::ACL_FLOAT16, wtsFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  void *wtsCastWorkspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> wtsCastWorkspacePtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&wtsCastWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    wtsCastWorkspacePtr.reset(wtsCastWorkspaceAddr);
  }
  ret = aclnnCast(wtsCastWorkspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  std::cout<<"cast fp16 input finish"<<std::endl;

  workspaceSize = 0;
  executor = nullptr;
  ret = aclnnBatchMatmulQuantGetWorkspaceSize(fmapFp16, wtsFp16, quantParam, nullptr, transposeX1, transposeX2, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatmulQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  void *mmWorkspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> mmWorkspacePtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&mmWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    mmWorkspacePtr.reset(mmWorkspaceAddr);
  }
  ret = aclnnBatchMatmulQuant(mmWorkspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatmulQuant failed. ERROR: %d\n", ret); return ret);


  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnBatchMatmulQuantTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatmulQuantTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
