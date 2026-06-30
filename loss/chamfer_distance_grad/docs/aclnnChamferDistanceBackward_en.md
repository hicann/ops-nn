# aclnnChamferDistanceBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/loss/chamfer_distance_grad)

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    ×     |

## Function

- Description: backward operator of ChamferDistance, which computes the gradients of the forward inputs based on their contribution to the forward output and the initial gradients.
- Formula:
  
  Assume there are two point sets: xyz1=[B,N,2], xyz2=[B,M,2]
  
  - Forward ChamferDistance formula:
    
    $dist1_i=Min((x_{1_i}-x_2)^2+(y_{1_i}-y_2)^2), x_2, y_2∈xyz2$
    $dist2_i=Min((x_{2_i}-x_1)^2+(y_{2_i}-y_1)^2), x_1,y_1∈xyz1$
  
  - Backward operator (derivative) formula:
    - Derivative of $dist1_i$ with respect to $x_{1_i}$$ = 2*grad\_dist1*(x_{1_i}-x_2)$

      Where $x_{1_i}∈xyz1$, and $x_2$ denotes the x-coordinate of the nearest point in xyz2 indexed by the forward output id1. The single-point derivative formula above supports multi-point parallel computation due to continuous gradient update positions.
      
    - Derivative of $dist1_i$ with respect to $y_{1_i}$$ = 2*grad\_dist1*(y_{1_i}-y_2)$
      
      Where $y_{1_i}∈xyz1$, and $y_2$ denotes the y-coordinate of the nearest point in xyz2 indexed by the forward output id1. The single-point derivative formula above supports multi-point parallel computation due to continuous gradient update positions.
      
    - Derivative of $dist1_i$ with respect to $x_2$$ = –2*grad\_dist1*(x_{1_i}-x_2)$
      
      Where $x_{1_i}∈xyz1$, and $x_2$ denotes the x-coordinate of the nearest point in xyz2 indexed by the forward output id1. Parallel computation is not supported; only single-point computation is allowed, as gradient updates rely on indices corresponding to minimum distances.
    
    - Derivative of $dist1_i$ with respect to $y_2$ = –2*grad\_dist1*(y_{1_i}-y_2)$
      
      Where $y_{1_i}∈xyz1$, and $y_2$ denotes the y-coordinate of the nearest point in xyz2 indexed by the forward output id1. Parallel computation is not supported; only single-point computation is allowed, as gradient updates rely on indices corresponding to minimum values.
  
  The derivatives of $dist2_i$ with respect to $x_{2_i}$, $x_1$, $y_{2_i}$, and $y_1$ follow a similar process and are omitted here.
  
  Final computation formulas (i∈[0,n)):

  $grad_xyz1[2*i] = 2*grad\_dist_1*(x_{1_i}-x_2) - 2*grad\_dist_1*(x_{1_i}-x_2)$
  
  $grad_xyz1[2*i+1] = 2*grad\_dist1*(y_{1_i}-y_2) - 2*grad\_dist1*(y_{1_i}-y_2)$
  
  $grad_xyz2[2*i] = 2*grad\_dist2*(x_{1_i}-x_2) - 2*grad\_dist2*(x_{1_i}-x_2)$
  
  $grad_xyz2[2*i+1] = 2*grad\_dist2*(y_{1_i}-y_2) - 2*grad\_dist2*(y_{1_i}-y_2)$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnChamferDistanceBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, **aclnnChamferDistanceBackward** is called to perform computation.

- `aclnnStatus aclnnChamferDistanceBackwardGetWorkspaceSize(const aclTensor* xyz1, const aclTensor* xyz2, const aclTensor* idx1, const aclTensor* idx2, const aclTensor* gradDist1, const aclTensor* gradDist2, aclTensor* gradXyz1, aclTensor* gradXyz2, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnChamferDistanceBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnChamferDistanceBackwardGetWorkspaceSize

- **Parameters**
  
  - **xyz1** (aclTensor\*, input): coordinates of point set 1 from the forward operator input, stored as a device-side aclTensor. Data type: FLOAT or FLOAT16; must match that of **xyz2**, **grad_dist1**, **grad_dist2**, **grad_xyz1**, and **grad_xyz2**. Shape: (B,N,2). [Data format](../../../docs/en/context/data_formats.md): ND.
  - **xyz2** (aclTensor\*, input): coordinates of point set 2 from the forward operator input, stored as a device-side aclTensor. Data type: FLOAT or FLOAT16; must match that of **xyz2**, **grad_dist1**, **grad_dist2**, **grad_xyz1**, and **grad_xyz2**. Shape: (B,N,2). [Data format](../../../docs/en/context/data_formats.md): ND.
  - **idx1** (aclTensor\*, input): indices of the nearest points in **xyz2** for each point in **xyz1** from the forward operator output, stored as a device-side aclTensor. Data type: INT32; must match that of **idx2**. Shape: (B,N). [Data format](../../../docs/en/context/data_formats.md): ND.
  - **idx2** (aclTensor\*, input): indices of the nearest points in **xyz1** for each point in **xyz2** from the forward operator output, stored as a device-side aclTensor. Data type: INT32; must match that of **idx1**. Shape: (B,N). [Data format](../../../docs/en/context/data_formats.md): ND.
  - **gradDist1** (aclTensor\*, input): backward gradient of the forward output **dist1**, serving as the initial gradient for the backward operator, stored as a device-side aclTensor. Data type: FLOAT or FLOAT16; must match that of **xyz2**, **xyz1**, **grad_dist2**, **grad_xyz1**, and **grad_xyz2**. Shape: (B,N). [Data format](../../../docs/en/context/data_formats.md): ND.
  - **gradDist2** (aclTensor\*, output): backward gradient of the forward output **dist2**, serving as the initial gradient for the backward operator, stored as a device-side aclTensor. Data type: FLOAT or FLOAT16; must match that of **xyz2**, **xyz1**, **grad_dist2**, **grad_xyz1**, and **grad_xyz2**. Shape: (B,N). [Data format](../../../docs/en/context/data_formats.md): ND.
  - **gradXyz1** (aclTensor\*, output): gradient of the forward input **xyz1** after gradient update, stored as a device-side aclTensor. Data type: FLOAT or FLOAT16; must match that of **xyz1**, **xyz2**, **grad_dist1**, **grad_dist2**, and **grad_xyz2**. Shape: (B,N,2). [Data format](../../../docs/en/context/data_formats.md): ND.
  - **gradXyz2** (aclTensor\*, output): backward gradient of the forward output **dist2**, serving as the initial gradient for the backward operator, stored as a device-side aclTensor. Data type: FLOAT or FLOAT16; must match that of **xyz1**, **xyz2**, **grad_dist1**, **grad_dist2**, and **grad_xyz1**. Shape: (B,N,2). [Data format](../../../docs/en/context/data_formats.md): ND.
  - **workspaceSize** (uint64_t\*, output): size of the workspace to be allocated on the device.
  - **executor** (aclOpExecutor\**, output): operator executor that contains the operator computation flow.
- **Returns**
  
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ```
  The first-phase API implements input parameter verification. The following errors may be thrown.
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. Any of the input tensors (such as xyz1 and xyz2) or output tensors (such as grad_xyz1 and grad_xyz2) is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The input and output data types are not supported.
                                        2. Data type deduction cannot be performed on the inputs.
                                        3. The deduced data type cannot be converted to the specified output data type.
  ```

## aclnnChamferDistanceBackward

- **Parameters**
  
  - **workspace** (void\*, input): address of the workspace to be allocated on the device.
  - **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained via the first-phase API **aclnnChamferDistanceBackwardGetWorkspaceSize**.
  - **executor** (aclOpExecutor\*, input): operator executor that contains the operator computation flow.
  - **stream** (aclrtStream, input): stream for executing the task.
- **Returns**
  
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic computation:
  - **aclnnChamferDistanceBackward** is non-deterministic by default. Deterministic mode can be enabled via **aclrtCtxSetSysParamOpt**.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

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
