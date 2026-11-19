# aclnnEinsum

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:44:05.327Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Performs tensor computation using the Einstein summation convention in the form of "term1, term2 -> output-term". The output tensor is generated according to the following equation, where reduce-sum sums over all indices that appear in the input terms (term1, term2) but not in the output term.
- Formula:

  $$
  output[output-term] = reduce-sum(input1[term1] * input2[term2])
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnEinsumGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnEinsum" API to execute the computation.

```cpp
aclnnStatus aclnnEinsumGetWorkspaceSize(
  const aclTensorList *tensors,
  const char          *equation,
  aclTensor           *output,
  uint64_t            *workspaceSize,
  aclOpExecutor       **executor)
```

```cpp
aclnnStatus aclnnEinsum(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnEinsumGetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1508px"><colgroup>
  <col style="width: 151px">
  <col style="width: 121px">
  <col style="width: 250px">
  <col style="width: 220px">
  <col style="width: 260px">
  <col style="width: 111px">
  <col style="width: 111px">
  <col style="width: 111px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>tensors</td>
      <td>Input</td>
      <td>Contains two tensors. tensors[0] denotes term1 in the formula, and tensors[1] denotes term2 in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT16, FLOAT, INT16, UINT16, INT32, UINT32, INT64, UINT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>equation</td>
      <td>Input</td>
      <td>Represents the shorthand formula of the Einstein summation convention.</td>
      <td>Letter names do not participate in matching: the equation is first normalized by the order in which letters first appear, and then compared with the registry. Currently, the registry contains "abcd,abced-&gt;abce" and "a,b-&gt;ab"; any other label arrangement returns 161002. Expression string on the host.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Output</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT16, FLOAT, INT16, UINT16, INT32, UINT32, INT64, UINT64</td>
      <td>ND</td>
      <td>-</td>
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
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>: tensors and output do not support the FLOAT data type.

  <!-- end id7 -->

- **Return Value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 828px"><colgroup>
  <col style="width: 300px">
  <col style="width: 200px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Values</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>Pointers to tensors and output are null.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of tensors and output are not within the supported range.</td>
    </tr>
    <tr>
      <td>equation (expandable) is not in the registry.</td>
    </tr>
    <tr>
      <td>When equation=='abcd,abced-&gt;abce':
      <ul><li>tensors contains 2 tensors (i.e. tensors[0] & tensors[1]);</li>
      <li>The data types of tensors[0], tensors[1], and output must be the same;</li>
      <li>tensors[0] must be a 4D tensor;</li>
      <li>tensors[1] must be a 5D tensor;</li>
      <li>The first 3 dimensions of tensors[0] must be the same as the first 3 dimensions of tensors[1];</li>
      <li>The 4th dimension of tensors[0] must be the same as the 5th dimension of tensors[1].</li></ul>
      </td>
    </tr>
    <tr>
      <td>When equation=='a,b-&gt;ab':
      <ul><li>tensorList contains 2 tensors (i.e. tensors[0] & tensors[1]);</li>
      <li>The data types of tensors[0], tensors[1], and output must be the same;</li>
      <li>tensors[0] must be a 1D tensor;</li>
      <li>tensors[1] must be a 1D tensor.</li></ul>
      </td>
    </tr>
  </tbody>
  </table>

## aclnnEinsum

- **Parameters**:
  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 250px">
  <col style="width: 150px">
  <col style="width: 750px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace memory size allocated on the Device, obtained by the first-phase API aclnnEinsumGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream where tasks are executed.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Determinism description: aclnnEinsum defaults to a deterministic implementation.

- The equation is matched by first normalizing letters in the order of first appearance and then comparing with the registry. Therefore `a,b->ab` is equivalent to `i,j->ij` and `x,y->xy`, while `a,b->ba` (different output label order) is not in the registry and returns 161002.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_einsum.h"

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

template <typename T>
int64_t GetShapeSize(const std::vector<T>& shape) {
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}
int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate, resource initialization
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
  // Call aclrtMalloc to apply for device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host-side data to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction according to the API.
  std::vector<int64_t> selfShape1 = {1, 2, 3, 4};
  std::vector<int64_t> selfShape2 = {1, 2, 3, 5, 4};
  std::vector<int64_t> outShape = {1, 2, 3, 5};
  void* input1DeviceAddr = nullptr;
  void* input2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* input1 = nullptr;
  aclTensor* input2 = nullptr;
  aclTensor* out = nullptr;
  std::vector<int32_t> input1HostData = {0, 1, 2, 6, 5, 1, 6, 4, 4, 8, 0, 3, 5, 2, 2, 6, 9, 9, 9, 2, 0, 8, 0, 9};
  std::vector<int32_t> input2HostData = {4, 7, 1, 6, 9, 6, 6, 1, 3, 7, 1, 3, 5, 0, 0, 7, 6, 3, 3, 7, 2, 0, 5, 0,
                                       0, 7, 9, 3, 7, 2, 3, 3, 5, 1, 9, 0, 0, 9, 8, 9, 4, 3, 1, 2, 8, 3, 0, 5,
                                       5, 0, 1, 5, 4, 6, 6, 0, 5, 5, 2, 6, 4, 8, 2, 1, 7, 7, 9, 8, 9, 3, 9, 9,
                                       5, 5, 8, 1, 5, 8, 9, 1, 8, 6, 6, 9, 9, 6, 7, 9, 1, 8, 5, 2, 0, 2, 3, 1,
                                       5, 3, 7, 9, 6, 2, 5, 3, 6, 6, 4, 9, 8, 7, 6, 5, 0, 0, 9, 2, 6, 1, 0, 6};
  std::vector<int32_t> outHostData(30, 0);

  // Create the input1 aclTensor.
  ret = CreateAclTensor(input1HostData, selfShape1, &input1DeviceAddr, aclDataType::ACL_INT32, &input1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the input2 aclTensor.
  ret = CreateAclTensor(input2HostData, selfShape2, &input2DeviceAddr, aclDataType::ACL_INT32, &input2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<aclTensor*> tmp{input1, input2};
  aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());

  const char equation[] = "abcd,abced->abce";

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API aclnnEinsum.
  ret = aclnnEinsumGetWorkspaceSize(tensorList, equation, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEinsumGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory using the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API aclnnEinsum.
  ret = aclnnEinsum(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEinsum failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value, copy the result from the device side to the host side, and modify it based on the specific API definition.
  auto size = GetShapeSize(outShape);
  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(outHostData[0]),
                    outDeviceAddr, size * sizeof(outHostData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outHostData from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %i\n", i, outHostData[i]);
  }

  // 6. Release aclTensor and aclScalar, and modify them based on the specific API definition.
  aclDestroyTensorList(tensorList);
  aclDestroyTensor(out);


  // 7. Release device resources, and modify them based on the specific API definition.
  aclrtFree(input1DeviceAddr);
  aclrtFree(input2DeviceAddr);
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
