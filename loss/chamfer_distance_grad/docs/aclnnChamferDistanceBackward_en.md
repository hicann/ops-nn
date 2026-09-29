# aclnnChamferDistanceBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/loss/chamfer_distance_grad)

## Product Support

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     ×    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- Function: This API is the backward operator of ChamferDistance. It calculates the gradient of the input based on the contribution of the forward input to the output and the initial gradient.
- Formula:

  Assume there are two point sets: xyz1=[B,N,2], xyz2=[B,M,2]

  - Forward ChamferDistance formula:

    $dist1_i=Min((x_{1_i}-x_2)^2+(y_{1_i}-y_2)^2), x_2, y_2∈xyz2$
    $dist2_i=Min((x_{2_i}-x_1)^2+(y_{2_i}-y_1)^2), x_1, y_1∈xyz1$

  - Backward operator (derivative) formula:
    - The derivative of $dist1_i$ to $x_{1_i}$ is $2*grad_dist1*(x_{1_i}-x_2)$.

      Where $x_{1_i}∈xyz1$, and $x_2$ denotes the x-coordinate of the nearest point in xyz2 indexed by the forward output id1. The single-point derivative formula above supports multi-point parallel computation due to continuous gradient update positions.

    - The derivative of $dist1_i$ to $y_{1_i}$ is $2*grad_dist1*(y_{1_i}-y_2)$.

      Where $y_{1_i}∈xyz1$, and $y_2$ denotes the y-coordinate of the nearest point in xyz2 indexed by the forward output id1. The single-point derivative formula above supports multi-point parallel computation due to continuous gradient update positions.

    - The derivative of $dist1_i$ with respect to $x_2$ is $-2*grad\_dist1*(x_1-x_{2_i})$.

      where $x_{2_i}∈xyz2$, and $x_1$ is the horizontal coordinate of the point with the minimum distance obtained from xyz2 based on the index value of id1 output in the forward direction. The preceding formula is used for calculating the gradient of a single point. Because the gradient of a single point needs to be updated based on the index value corresponding to the minimum distance, this operation can only be performed on a single point and cannot be parallelized.

    - The derivative of $dist1_i$ with respect to $y_2$ is $-2*grad\_dist1*(y_1-y_{2_i})$.

      where $y_{2_i}∈xyz2$, and $y_1$ is the vertical coordinate of the point with the minimum distance obtained from xyz2 based on the index value of id1 output in the forward direction. The preceding formula is used for calculating the gradient of a single point. Because the gradient of a single point needs to be updated based on the index value corresponding to the minimum distance, this operation can only be performed on a single point and cannot be parallelized.

  The derivatives of $dist2_i$ with respect to $x_{2_i}$, $x_1$, $y_{2_i}$, and $y_1$ are similar to the preceding process. Details are not described here.

  Final computation formulas (i∈[0,n)):

  $grad_xyz1[2*i] = 2*grad\_dist1*(x_{1_i}-x_2) - 2*grad\_dist1*(x_1-x_{2_i})$

  $grad_xyz1[2*i+1] = 2*grad\_dist1*(y_{1_i}-y_2) - 2*grad\_dist1*(y_1-y_{2_i})$

  $grad_xyz2[2*i] = 2*grad\_dist2*(x_{1_i}-x_2) - 2*grad\_dist2*(x_1-x_{2_i})$

  $grad_xyz2[2*i+1] = 2*grad\_dist2*(y_{1_i}-y_2) - 2*grad\_dist2*(y_1-y_{2_i})$

## Prototype

Each operator is divided into [two-phase API](../../../docs/en/context/two_phase_api.md). You must call aclnnChamferDistanceBackwardGetWorkspaceSize to obtain the input parameters and calculate the required workspace size based on the workflow, and then call aclnnChamferDistanceBackward to perform the computation.

```Cpp
aclnnStatus aclnnChamferDistanceBackwardGetWorkspaceSize(
    const aclTensor* xyz1, 
    const aclTensor* xyz2, 
    const aclTensor* idx1, 
    const aclTensor* idx2,
    const aclTensor* gradDist1, 
    const aclTensor* gradDist2, 
    aclTensor*       gradXyz1, 
    aclTensor*       gradXyz2,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnChamferDistanceBackward(
    void*            workspace, 
    uint64_t         workspaceSize, 
    aclOpExecutor*   executor, 
    aclrtStream      stream)
```

## aclnnChamferDistanceBackwardGetWorkspaceSize

- **Parameters**

  <table class="tg" style="undefined;table-layout: fixed; width: 1172px"><colgroup>
  <col style="width: 184px">
  <col style="width: 86px">
  <col style="width: 269px">
  <col style="width: 190px">
  <col style="width: 116px">
  <col style="width: 111px">
  <col style="width: 108px">
  <col style="width: 108px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Parameter Name</th>
      <th class="tg-0pky">Input/Output</th>
      <th class="tg-0pky">Description</th>
      <th class="tg-0pky">Usage Description</th>
      <th class="tg-0pky">Data Type</th>
      <th class="tg-0pky">Data Format</th>
      <th class="tg-0pky">Dimension (shape)</th>
      <th class="tg-0pky">Non-consecutive Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">xyz1 (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Coordinates of point set 1 input by the operator in the forward direction.</td>
      <td class="tg-0pky">The shape is (B, N, 2).</td>
      <td class="tg-0pky">FLOAT, FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">xyz2 (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Coordinates of point set 2 input by the operator in the forward direction.</td>
      <td class="tg-0pky">The shape is (B, M, 2).</td>
      <td class="tg-0pky">FLOAT, FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">idx1 (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Tensor of the index of the point in xyz2 that is the minimum distance from xyz1 in the forward output of the operator.</td>
      <td class="tg-0pky">The shape is (B, N).</td>
      <td class="tg-0pky">INT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">idx2 (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Index tensor of the points in xyz1 that are closest to xyz2 in the forward output of the operator.</td>
      <td class="tg-0pky">The shape is (B, N).</td>
      <td class="tg-0pky">INT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">gradDist1 (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Backward gradient of the forward output dist1, which is also the initial gradient of the backward operator.</td>
      <td class="tg-0pky">The shape is (B, N).</td>
      <td class="tg-0pky">FLOAT, FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">gradDist2 (aclTensor*) </td>
      <td class="tg-0pky">Input</td>
      <td class="tg-0pky">Forward output of the gradient descent of dist2, which is also the initial gradient of the backward operator.</td>
      <td class="tg-0pky">The shape is (B, N).</td>
      <td class="tg-0pky">FLOAT, FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">gradXyz1 (aclTensor*) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Gradient corresponding to the input xyz1 of the forward operator after gradient update.</td>
      <td class="tg-0pky">The shape is (B, N, 2).</td>
      <td class="tg-0pky">FLOAT, FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">gradXyz2 (aclTensor*) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Gradient of the input xyz2 corresponding to the forward operator after gradient update.</td>
      <td class="tg-0pky">The shape is (B, N, 2).</td>
      <td class="tg-0pky">FLOAT, FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">3</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize (uint64_t*) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the workspace size to be allocated on the device.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor (aclOpExecutor**) </td>
      <td class="tg-0pky">Output</td>
      <td class="tg-0pky">Returns the operator executor, including the operator computation process.</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **Returns**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.
  
  <table class="tg" style="undefined;table-layout: fixed; width: 951px"><colgroup>
  <col style="width: 258px">
  <col style="width: 86px">
  <col style="width: 607px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">Returns</th>
      <th class="tg-0pky">Error Code</th>
      <th class="tg-0pky">Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">The input xyz1, xyz2, idx1, idx2, gradDist1, gradDist2, or the output grad_xyz1 and grad_xyz2 is a null pointer. </td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="3">161002</td>
      <td class="tg-0pky">The input and output data types are not supported.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The data type of the input cannot be deduced.</td>
    </tr>
    <tr>
      <td class="tg-0pky">The deduced data type cannot be converted to the specified output type.</td>
    </tr>
  </tbody>
  </table>

## aclnnChamferDistanceBackward

- **Parameters**

   <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
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
          <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnChamferDistanceBackwardGetWorkspaceSize.</td>
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

- **Returns**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - **aclnnChamferDistanceBackward** is non-deterministic by default. Deterministic mode can be enabled via **aclrtCtxSetSysParamOpt**.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_chamfer_distance_backward.h"

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
    // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
    // Set the device ID (deviceId) based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Customize error handling based on your requirements.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct inputs and outputs based on API definitions.
    std::vector<int64_t> xyz1Shape = {2, 2, 2};
    std::vector<int64_t> xyz2Shape = {2, 2, 2};
    std::vector<int64_t> idx1Shape = {2, 2};
    std::vector<int64_t> idx2Shape = {2, 2};
    std::vector<int64_t> gradDist1Shape = {2, 2};
    std::vector<int64_t> gradDist2Shape = {2, 2};
    std::vector<int64_t> gradXyz1Shape = {2, 2, 2};
    std::vector<int64_t> gradXyz2Shape = {2, 2, 2};
    void* xyz1DeviceAddr = nullptr;
    void* xyz2DeviceAddr = nullptr;
    void* idx1DeviceAddr = nullptr;
    void* idx2DeviceAddr = nullptr;
    void* gradDist1DeviceAddr = nullptr;
    void* gradDist2DeviceAddr = nullptr;
    void* gradXyz1DeviceAddr = nullptr;
    void* gradXyz2DeviceAddr = nullptr;
    aclTensor* xyz1 = nullptr;
    aclTensor* xyz2 = nullptr;
    aclTensor* idx1 = nullptr;
    aclTensor* idx2 = nullptr;
    aclTensor* gradDist1 = nullptr;
    aclTensor* gradDist2 = nullptr;
    aclTensor* gradXyz1 = nullptr;
    aclTensor* gradXyz2 = nullptr;
    std::vector<float> xyz1HostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> xyz2HostData = {1, 1, 1, 2, 2, 2, 3, 3};
    std::vector<int32_t> idx1HostData = {0, 1, 2, 3};
    std::vector<int32_t> idx2HostData = {0, 1, 2, 3};
    std::vector<float> gradDist1HostData = {0, 1, 2, 3};
    std::vector<float> gradDist2HostData = {0, 1, 2, 3};
    std::vector<float> gradXyz1HostData = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> gradXyz2HostData = {0, 0, 0, 0, 0, 0, 0, 0};
    // Create an xyz1 aclTensor.
    ret = CreateAclTensor(xyz1HostData, xyz1Shape, &xyz1DeviceAddr, aclDataType::ACL_FLOAT, &xyz1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an xyz2 aclTensor.
    ret = CreateAclTensor(xyz2HostData, xyz2Shape, &xyz2DeviceAddr, aclDataType::ACL_FLOAT, &xyz2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an idx1 aclTensor.
    ret = CreateAclTensor(idx1HostData, idx1Shape, &idx1DeviceAddr, aclDataType::ACL_INT32, &idx1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an idx2 aclTensor.
    ret = CreateAclTensor(idx2HostData, idx2Shape, &idx2DeviceAddr, aclDataType::ACL_INT32, &idx2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a gradDist1 aclTensor.
    ret = CreateAclTensor(gradDist1HostData, gradDist1Shape, &gradDist1DeviceAddr, aclDataType::ACL_FLOAT, &gradDist1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a gradDist2 aclTensor.
    ret = CreateAclTensor(gradDist2HostData, gradDist2Shape, &gradDist2DeviceAddr, aclDataType::ACL_FLOAT, &gradDist2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a gradXyz1 aclTensor.
    ret = CreateAclTensor(gradXyz1HostData, gradXyz1Shape, &gradXyz1DeviceAddr, aclDataType::ACL_FLOAT, &gradXyz1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a gradXyz2 aclTensor.
    ret = CreateAclTensor(gradXyz2HostData, gradXyz2Shape, &gradXyz2DeviceAddr, aclDataType::ACL_FLOAT, &gradXyz2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Modify the API as required.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnChamferDistanceBackward.
    ret = aclnnChamferDistanceBackwardGetWorkspaceSize(xyz1, xyz2, idx1, idx2, gradDist1, gradDist2, gradXyz1, gradXyz2,
                                                       &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnChamferDistanceBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnChamferDistanceBackward.
    ret = aclnnChamferDistanceBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnChamferDistanceBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize the stream and wait for task completion.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(gradXyz1Shape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradXyz1DeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result1[%ld] is: %f\n", i, resultData[i]);
    }

    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradXyz2DeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result2[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
    aclDestroyTensor(xyz1);
    aclDestroyTensor(xyz2);
    aclDestroyTensor(idx1);
    aclDestroyTensor(idx2);
    aclDestroyTensor(gradDist1);
    aclDestroyTensor(gradDist2);
    aclDestroyTensor(gradXyz1);
    aclDestroyTensor(gradXyz2);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(xyz1DeviceAddr);
    aclrtFree(xyz2DeviceAddr);
    aclrtFree(idx1DeviceAddr);
    aclrtFree(idx2DeviceAddr);
    aclrtFree(gradDist1DeviceAddr);
    aclrtFree(gradDist2DeviceAddr);
    aclrtFree(gradXyz1DeviceAddr);
    aclrtFree(gradXyz2DeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
