# aclnnSwiGluQuant

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/quant/swi_glu_quant)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     ×    |
|  <term>Atlas training series products</term>   |     ×    |

## Function

- Description: Adds quantization after the SwiGlu activation function to implement SwiGluQuant computation of the input x.
- Operator support: Currently, SwiGluQuant **only supports the MoE scenario**. The input x and group_index of SwiGluQuant come from the outputs of the GroupedMatMul operator and MoeInitRouting. group_index is used to implement dynamic MoE group quantization, static per_tensor quantization, and static per_channel quantization.
- Dynamic quantization formula: 
  
  $$
    Act = SwiGLU(x) = Swish(A)*B \\
    Y_{tmp}[0\colon g[0],\colon] = Act[0\colon g[0],\colon] * smooth\_scales[0,\colon], i=0 \\
    Y_{tmp}[g[i]\colon g[i+1], \colon] = Act[g[i]\colon g[i+1], \colon] *  smooth\_scales[i+1, \colon], i \in (0, G) \cap \mathbb{Z}\\
    scale=row\_max(abs(Y_{tmp}))/127
  $$

  $$
    Y = Cast(Mul(Y_{tmp}, Scale))
  $$
     A indicates the first half of the input x, B indicates the second half of the input x, g indicates group_index, and G indicates the number of groups of group_index.
     
- Static quantization formula:
  
  $$
    Act = SwiGLU(x) = Swish(A)*B \\
    Y_{tmp}[0\colon g[0],\colon] = Act[0\colon g[0],\colon] * smooth\_scales[0,\colon] + offsets[0,\colon], i=0 \\
    Y_{tmp}[g[i]\colon g[i+1], \colon] = Act[g[i]\colon g[i+1], \colon] *  smooth\_scales[i+1, \colon] + offsets[i+1, \colon], i \in (0, G) \cap \mathbb{Z}\\
  $$

  $$
    Y = Cast(Y_{tmp})
  $$

  A indicates the first half of the input x, B indicates the second half of the input x, g indicates group_index, and G indicates the number of groups of group_index.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnSwiGluQuantGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnSwiGluQuant** is called to perform computation.

```Cpp
aclnnStatus aclnnSwiGluQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *smoothScalesOptional,
  const aclTensor *offsetsOptional,
  const aclTensor *groupIndexOptional,
  bool             activateLeft,
  char            *quantModeOptional,
  const aclTensor *yOut,
  const aclTensor *scaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSwiGluQuant(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```


## aclnnSwiGluQuantGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1480px"><colgroup>
  <col style="width: 201px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 300px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
      <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input data to be processed, corresponding to x in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The last dimension of x must be a multiple of 2, and the number of dimensions of x must be greater than 1. Currently, the length of the last dimension of the input x cannot exceed 8,192.</li></ul></td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScalesOptional</td>
      <td>Input</td>
      <td>Quantization smooth_scales, corresponding to smooth_scales in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape supports [G, N] and [G, ], where G indicates the number of groupIndex groups, and N indicates half of the size of the last dimension of the input x. In the current version, passing a null pointer is not supported.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
     <tr>
      <td>offsetsOptional</td>
      <td>Input</td>
      <td>offsets in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>This parameter does not take effect in dynamic quantization scenarios. You can pass a null pointer to it. </li><li>In static quantization scenarios, the data type can be FLOAT. </li><li>In per_channel mode, the shape can be [G, N]. </li><li>In per_tensor mode, the shape can be [G, ], and the data type and shape must be the same as those of smoothScalesOptional.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndexOptional</td>
      <td>Input</td>
      <td>group_index required for MoE grouping, corresponding to group_index in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape can be [G, ]. The elements in group_index must be non-decreasing, and the maximum value cannot exceed the product of the sizes of all dimensions except the last dimension of the input x. The value of G cannot exceed the product of the sizes of all dimensions except the last dimension of the input x.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
      <tr>
      <td>activateLeft</td>
      <td>Input</td>
      <td>Input for computation.</td>
      <td>true indicates that the first half of the input x is to be activated, and false indicates that the second half of the input x is to be activated.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>quantModeOptional</td>
      <td>Input</td>
      <td>Input for computation.</td>
      <td>static indicates static quantization, dynamic indicates dynamic quantization, and dynamic_msd indicates dynamic MSD quantization. Currently, only dynamic quantization and static quantization are supported. In terms of static quantization, only per-tensor quantization and per-channel quantization are supported. A passed null pointer indicates dynamic quantization.</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>The size of the last dimension of the shape of the output yOut is half of the last dimension of the input x, while the other dimensions are the same as those of x.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
      <tr>
      <td>scaleOut</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>Compared with the input x, the shape of the output scaleOut does not have the last dimension, while the other dimensions are the same as those of the input x.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
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
  </tbody>
  </table>
  
- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed x or yOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The input or output data type is not supported.</td>
    </tr>
    <tr>
      <td>The input or output parameter dimension is not supported.</td>
    </tr>
      <tr>
      <td>quantModeOptional is not within the specified range.</td>
    </tr>
  </tbody></table>


## aclnnSwiGluQuant

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnSwiGluQuantGetWorkspaceSize.</td>
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

- Deterministic compute:
  - **aclnnSwiGluQuant** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swi_glu_quant.h"

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

int main() {
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  std::vector<int64_t> xShape = {3, 32};
  std::vector<int64_t> smoothScalesShape = {2, 16};
  std::vector<int64_t> groupIndexShape = {2};
  std::vector<int64_t> outShape = {3, 16};
  std::vector<int64_t> scaleShape = {3};
  void* xDeviceAddr = nullptr;
  void* smoothScalesDeviceAddr = nullptr;
  void* groupIndexDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* scaleDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* smoothScales = nullptr;
  aclTensor* groupIndex = nullptr;
  aclTensor* out = nullptr;
  aclTensor* scale = nullptr;
  std::vector<float> xHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                  43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 
                                  63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 
                                  83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};
  std::vector<float> smoothScalesHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                                      1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<float> groupIndexHostData = {1, 3};
  std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> scaleHostData = {0, 0, 0};
  
  // Create an x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
   // Create a scale aclTensor.
  ret = CreateAclTensor(smoothScalesHostData, smoothScalesShape, &smoothScalesDeviceAddr, aclDataType::ACL_FLOAT, &smoothScales);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
   // Create a groupIndex aclTensor.
  ret = CreateAclTensor(groupIndexHostData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT32, &groupIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a scale aclTensor.
  ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnSwiGluQuant.
  ret = aclnnSwiGluQuantGetWorkspaceSize(x, smoothScales, nullptr, groupIndex, false, "dynamic", out, scale, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwiGluQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnSwiGluQuant.
  ret = aclnnSwiGluQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwiGluQuant failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }
  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(smoothScales);
  aclDestroyTensor(groupIndex);
  aclDestroyTensor(out);
  aclDestroyTensor(scale);
  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(xDeviceAddr);
  aclrtFree(smoothScalesDeviceAddr);
  aclrtFree(groupIndexDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(scaleDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
