# aclnnEinsum

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/matmul/batch_mat_mul_v3)

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    √     |
| <term>Atlas training series products</term>                             |    ×     |

## Function
- Description: Performs tensor computation using the Einstein summation convention, following the pattern **term1, term2 -> output-term**. The output tensor is generated according to the following equation, where **reduce-sum** is applied to all indices that appear in the input terms **(term1, term2)** but not in the output term.
- Formula:

  $$
  output[output-term] = reduce-sum(input1[term1] * input2[term2])
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnEinsumGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, **aclnnEinsum** is called to perform computation.
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
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>tensors</td>
      <td>Input</td>
      <td>Two tensors are included. tensors[0] corresponds to term1 in the formula, and tensors[1] corresponds to term2 in the formula.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT16, FLOAT, INT16, UINT16, INT32, UINT32, INT64, UINT64</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>equation</td>
      <td>Input</td>
      <td>String expression of the Einstein summation convention.</td>
      <td>Currently, only "abcd,abced-&gt;abce" and "a,b-&gt;ab" are supported. This is a host-side expression string.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
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
      <td>Operator executor, containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  - <term>Atlas inference series products</term>: The FLOAT data type is not supported for **tensors** and **output**.

- **Returns**
  
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
      <td>The passed tensors or output is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of tensors or output is not supported.</td>
    </tr>
    <tr>
      <td>The equation (extensible) is not in the registry.</td>
    </tr>
    <tr>
      <td>When equation=='abcd,abced-&gt;abce':
      <ul><li>tensors contains two tensors (tensors[0] and tensors[1]).</li>
      <li>The data types of tensors[0], tensors[1], and output must be consistent.</li>
      <li>tensors[0] must be 4-dimensional.</li>
      <li>tensors[1] must be 5-dimensional.</li>
      <li>The first three dimensions of tensors[0] must equal those of tensors[1].</li>
      <li>The fourth dimension of tensors[0] must equal the fifth dimension of tensors[1].</li></ul>
      </td>
    </tr>
    <tr>
      <td>When equation=='a,b-&gt;ab':
      <ul><li>tensorList contains two tensors (tensors[0] and tensors[1]).</li>
      <li>The data types of tensors[0], tensors[1], and output must be consistent.</li>
      <li>tensors[0] must be 1-dimensional.</li>
      <li>tensors[1] must be 1-dimensional.</li></ul>
      </td>
    </tr>
  </tbody>
  </table>

## aclnnEinsum

- **Parameters**

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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnEinsumGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Returns**
  
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
- Deterministic description:
  - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: **aclnnEinsum** defaults to a deterministic implementation.

- Currently, **equation** must match exactly for the corresponding function to be selected.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

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
  // (Boilerplate) Initialize resources.
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

  // Calculate the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID (deviceId) based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on API definitions.
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

  // Create an input1 aclTensor.
  ret = CreateAclTensor(input1HostData, selfShape1, &input1DeviceAddr, aclDataType::ACL_INT32, &input1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an input2 aclTensor.
  ret = CreateAclTensor(input2HostData, selfShape2, &input2DeviceAddr, aclDataType::ACL_INT32, &input2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<aclTensor*> tmp{input1, input2};
  aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());

  const char equation[] = "abcd,abced->abce";

  // 3. Call the CANN operator library API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnEinsum.
  ret = aclnnEinsumGetWorkspaceSize(tensorList, equation, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEinsumGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second-phase API of aclnnEinsum.
  ret = aclnnEinsum(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEinsum failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize the stream and wait for task completion.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  ret = aclrtMemcpy(outHostData.data(), outHostData.size() * sizeof(outHostData[0]),
                    outDeviceAddr, size * sizeof(outHostData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outHostData from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %i\n", i, outHostData[i]);
  }

  // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensorList(tensorList);
  aclDestroyTensor(out);


  // 7. Release device resources. Modify the code based on the API definition.
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
