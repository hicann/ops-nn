# aclnnScatterAdd

📄 [View Source Code](https://gitcode.com/cann/ops-nn/tree/master/index/scatter_elements_v2)

## Supported Products

| Product| Supported|
| :--- | :---: |
| Ascend 950PR/Ascend 950DT| √ |
| <term>Atlas A3 training products/Atlas A3 inference products</term>| √ |
| <term>Atlas A2 training products/Atlas A2 inference products</term>| √ |
| <term>Atlas 200I/500 A2 inference products</term>| × |
| <term>Atlas inference products</term>| × |
| <term>Atlas training products</term>| × |

## Function

- This API is used to fill the values in the src tensor into the self tensor one by one according to the specified axis and the position relationship in the index tensor. If more than one src value is filled into the same position of the self tensor, these values will be accumulated at this position.

  Example:
  For a 3D tensor, self is updated according to the following rules:

  ```text
  self[index[i][j][k]][j][k] += src[i][j][k] # if dim == 0
  self[i][index[i][j][k]][k] += src[i][j][k] # if dim == 1
  self[i][j][index[i][j][k]] += src[i][j][k] # if dim == 2
  ```

  The following conditions must be met during computation:
  - **self**, **index**, and **src** must have the same number of dimensions.
  - For each dimension d, index.size(d) <= src.size(d).
  - For each dimension d, if d!= dim, index.size(d) <= self.size(d).
  - The value range of dim is [–self.dim(), self.dim() – 1].

- Example:

  Input tensor $self = \begin{bmatrix} [1&2&3] \\ [4&5&6] \\ [7&8&9] \end{bmatrix}$,
  Index tensor $index = \begin{bmatrix} [0&2&1] \\ [0&0&1] \end{bmatrix}$, dim = 1,
  Source tensor $src = \begin{bmatrix} [10&11&12] \\ [13&14&15] \end{bmatrix}$,
  Output tensor $output = \begin{bmatrix} [11&14&14] \\ [31&20&6] \\ [7&8&9] \end{bmatrix}$

  dim = 1 indicates that scatter_add accumulates values along the columns of the tensor according to $index$.

  $output[0][0] = self[0][0] + src[0][0]$ = 1 + 10,

  $output[0][1] = self[0][1] + src[0][2]$ = 2 + 12,

  $output[0][2] = self[0][2] + src[0][1]$ = 3 + 11,

  $output[1][0] = self[1][0] + src[1][0] + src[1][1]$ = 4 + 13 + 14,

  $output[1][1] = self[1][1] + src[1][2]$ = 5 + 15,

  $output[1][2] = self[1][2]$ = 6,

  $output[2][0] = self[2][0]$ = 7,

  $output[2][1] = self[2][1]$ = 8,

  $output[2][2] = self[2][2]$ = 9.

  The number of dimensions of $self$, $index$, and $src$ are all 2. The size of each dimension of $index$ {2, 3} is not greater than the corresponding dimension size {2, 3} of $src$. On the dimension dim!= 1 (dim = 0), the dimension size {2} of $index$ is not greater than the corresponding dimension size {3} of $self$. The maximum value {2} in $index$ is less than the size {3} of $self$ in the dim = 1 dimension.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnScatterAddGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnScatterAdd** is called to perform computation.

```Cpp
aclnnStatus aclnnScatterAddGetWorkspaceSize(
  const aclTensor* self,
  int64_t          dim,
  const aclTensor* index,
  const aclTensor* src,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnScatterAdd(
  void*               workspace,
  uint64_t            workspaceSize,
  aclOpExecutor*      executor,
  const aclrtStream   stream)
```

## aclnnScatterAddGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 280px">
    <col style="width: 320px">
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 140px">
    <col style="width: 140px">
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
        <td>self (aclTensor*)</td>
        <td>Input</td>
        <td>Input `self` in the formula, which is the target tensor of scatter.</td>
        <td>The number of dimensions must be the same as that of index and src.<br>The data type is the same as that of **src**.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE, INT64, INT32, INT16, INT8, UINT8, BOOL, COMPLEX64, COMPLEX128</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim (int64_t)</td>
        <td>Input</td>
        <td>Input `dim` in the formula.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index (aclTensor*)</td>
        <td>Input</td>
        <td>Input `index` in the formula.</td>
        <td>The number of dimensions must be the same as that of src.</td>
        <td>INT32 or INT64</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>src (aclTensor*) </td>
        <td>Input</td>
        <td>Input `src` in the formula, source tensor.</td>
        <td>The number of dimensions must be the same as that of index.<br>The data type must be the same as that of **self**.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE, INT64, INT32, INT16, INT8, UINT8, BOOL, COMPLEX64, COMPLEX128</td>
        <td>ND</td>
        <td>-</td>
        <td>×</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td> `output` in the formula.</td>
        <td>The shape must be the same as that of self.<br>The data type must be the same as that of **self**.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32, DOUBLE, INT64, INT32, INT16, INT8, UINT8, BOOL, COMPLEX64, COMPLEX128</td>
        <td>ND</td>
        <td>-</td>
        <td>×</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Size of the workspace to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
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

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 300px">
    <col style="width: 134px">
    <col style="width: 716px">
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
        <td>The input self, index, src, or out is a null pointer. </td>
      </tr>
      <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The data type of self, index, src, or out is not supported.</td>
      </tr>
      <tr>
        <td>The shapes of self and out are inconsistent.</td>
      </tr>
      <tr>
        <td>The shape of src or index is invalid.</td>
      </tr>
    </tbody></table>

## aclnnScatterAdd

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
    <col style="width: 200px">
    <col style="width: 130px">
    <col style="width: 770px">
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
        <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnScatterAddGetWorkspaceSize.</td>
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
    </tbody></table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic compute:
  - **aclnnScatterAdd** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter_add.h"

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
  std::vector<int64_t> selfShape = {4, 4};
  std::vector<int64_t> indexShape = {3, 4};
  std::vector<int64_t> srcShape = {4, 4};
  std::vector<int64_t> outShape = {4, 4};
  int64_t dim = 0;
  void* selfDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* srcDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* index = nullptr;
  aclTensor* src = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData(16, 0);
  std::vector<int64_t> indexHostData = {0, 1, 2, 1, 0, 1, 2, 0, 2, 2, 1, 0};
  std::vector<float> srcHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> outHostData(16, 0);
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an index aclTensor.
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an src aclTensor.
  ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, aclDataType::ACL_FLOAT, &src);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnScatterAdd.
  ret = aclnnScatterAddGetWorkspaceSize(self, dim, index, src, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnScatterAdd.
  ret = aclnnScatterAdd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterAdd failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(index);
  aclDestroyTensor(src);
  aclDestroyTensor(out);
  // 7. Release device resources. Set the parameters based on the API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(indexDeviceAddr);
  aclrtFree(srcDeviceAddr);
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
