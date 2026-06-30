# aclnnMultiScaleDeformableAttentionGrad

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/vfusion/multi_scale_deformable_attention_grad)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √    |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference series products</term>                            |    ×    |
| <term>Atlas training series products</term>                             |    ×    |

## Function

- Description:
  Traverses different sampling points of feature maps of different sizes by using parameters such as the sample location, attention weights, mapped value feature, start index location of a multi-scale feature, and spatial size of a multi-scale feature map (which facilitates changing a sampling location from a normalized value to an absolute location). A reverse operator calculates the gradient corresponding to the input based on the contribution of the forward input to the output and an initial gradient.
- Formula:

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnMultiScaleDeformableAttentionGrad** is called to perform computation.

```Cpp
aclnnStatus aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize(
    const aclTensor* value, 
    const aclTensor* spatialShape, 
    const aclTensor* levelStartIndex, 
    const aclTensor* location, 
    const aclTensor* attnWeight, 
    const aclTensor* gradOutput, 
    aclTensor*       gradValue, 
    aclTensor*       gradLocation, 
    aclTensor*       gradAttnWeight, 
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor)
```
```Cpp
aclnnStatus aclnnMultiScaleDeformableAttentionGrad(
    void*          workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor* executor, 
    aclrtStream    stream)
```

## aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 250px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 300px">
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
      <td>value</td>
      <td>Input</td>
      <td>Feature value of the feature map.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(bs, num_keys, num_heads, channels)<br>bs indicates the batch size, num_keys indicates the size of the feature map, num_heads indicates the number of heads, and channels indicates the dimensions of the feature map.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>spatialShape</td>
      <td>Input</td>
      <td>Height and width of each scale feature map.</td>
      <td>-</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>(num_levels, 2)<br>num_levels indicates the number of feature maps, and 2 represents H and W.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>levelStartIndex</td>
      <td>Input</td>
      <td>Start index of each feature map.</td>
      <td>-</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>(num_levels)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>location</td>
      <td>Input</td>
      <td>Sampling point tensor, for storing the coordinates of each sampling point.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(bs, num_queries, num_heads, num_levels, num_points, 2)<br>num_queries indicates the number of queries, num_points indicates the number of sampling points, and 2 represents y and x.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attnWeight</td>
      <td>Input</td>
      <td>Weight tensor of the sampling point.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(bs, num_queries, num_heads, num_levels, num_points)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradOutput</td>
      <td>Input</td>
      <td>Forward output gradient, which is also the initial gradient of the backward operator.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(bs, num_queries, num_heads, channels)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradValue</td>
      <td>Output</td>
      <td>Gradient corresponding to the input value.</td>
      <td>The shape must be the same as that of value.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradLocation</td>
      <td>Output</td>
      <td>Gradient corresponding to the input location.</td>
      <td>The shape must be the same as that of location.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradAttnWeight</td>
      <td>Output</td>
      <td>Gradient of the input attnWeight.</td>
      <td>The shape must be the same as that of attnWeight.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
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
  </tbody></table>
- **Returns:**
  
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown:
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <td>The passed input or output is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The input and output data types are not supported.</td>
    </tr>
    <tr>
      <td>The input and output data types are inconsistent.</td>
    </tr>
    <tr>
      <td>The API constraints are not met.</td>
    </tr>
  </tbody>
  </table>


## aclnnMultiScaleDeformableAttentionGrad

- **Parameters:**
  
  <table><thead>
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize.</td>
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

- Deterministic compute
  - **aclnnMultiScaleDeformableAttentionGrad** defaults to a non-deterministic implementation and does not support deterministic implementation currently.

- channels%8 = 0, and channels ≤ 256
- num_queries < 500000
- num_levels ≤ 16
- num_heads ≤ 16
- num_points ≤ 16

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_multi_scale_deformable_attention_grad.h"

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
    // 1. (Fixed writing) Initialize the device and stream. For details, see the list of external ACL APIs.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Use CHECK as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output based on the API.

    std::vector<int64_t> valueShape = {1, 1, 1, 8};
    std::vector<int64_t> spatialShapeShape = {1, 2};
    std::vector<int64_t> levelStartIndexShape = {1};
    std::vector<int64_t> locationShape = {1, 32, 1, 1, 1, 2};
    std::vector<int64_t> attnWeightShape = {1, 32, 1, 1, 1};
    std::vector<int64_t> gradOutputShape = {1, 32, 1, 8};
    std::vector<int64_t> gradValueShape = {1, 1, 1, 8};
    std::vector<int64_t> gradLocationShape = {1, 32, 1, 1, 1, 2};
    std::vector<int64_t> gradAttnWeightShape = {1, 32, 1, 1, 1};
    void* valueDeviceAddr = nullptr;
    void* spatialShapeDeviceAddr = nullptr;
    void* levelStartIndexDeviceAddr = nullptr;
    void* locationDeviceAddr = nullptr;
    void* attnWeightDeviceAddr = nullptr;
    void* gradOutputDeviceAddr = nullptr;
    void* gradValueDeviceAddr = nullptr;
    void* gradLocationDeviceAddr = nullptr;
    void* gradAttnWeightDeviceAddr = nullptr;
    aclTensor* value = nullptr;
    aclTensor* spatialShape = nullptr;
    aclTensor* levelStartIndex = nullptr;
    aclTensor* location = nullptr;
    aclTensor* attnWeight = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* gradValue = nullptr;
    aclTensor* gradLocation = nullptr;
    aclTensor* gradAttnWeight = nullptr;
    std::vector<float> valueHostData = {1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> spatialShapeHostData = {1, 1};
    std::vector<float> levelStartIndexHostData = {0};
    std::vector<float> gradValueHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> gradLocationHostData(GetShapeSize(gradLocationShape), 0);
    std::vector<float> gradAttnWeightHostData(GetShapeSize(gradAttnWeightShape), 0);
    std::vector<float> locationHostData(GetShapeSize(locationShape), 0);
    std::vector<float> attnWeightHostData(GetShapeSize(attnWeightShape), 1);
    std::vector<float> gradOutputHostData(GetShapeSize(gradOutputShape), 1);

    // value aclTensor
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a spatialShape aclTensor.
    ret = CreateAclTensor(spatialShapeHostData, spatialShapeShape, &spatialShapeDeviceAddr, aclDataType::ACL_INT32, &spatialShape);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a levelStartIndex aclTensor.
    ret = CreateAclTensor(levelStartIndexHostData, levelStartIndexShape, &levelStartIndexDeviceAddr, aclDataType::ACL_INT32, &levelStartIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a location aclTensor.
    ret = CreateAclTensor(locationHostData, locationShape, &locationDeviceAddr, aclDataType::ACL_FLOAT, &location);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an attnWeight aclTensor.
    ret = CreateAclTensor(attnWeightHostData, attnWeightShape, &attnWeightDeviceAddr, aclDataType::ACL_FLOAT, &attnWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // gradOutput aclTensor
    ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a gradValue aclTensor.
    ret = CreateAclTensor(gradValueHostData, gradValueShape, &gradValueDeviceAddr, aclDataType::ACL_FLOAT, &gradValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a gradLocation aclTensor.
    ret = CreateAclTensor(gradLocationHostData, gradLocationShape, &gradLocationDeviceAddr, aclDataType::ACL_FLOAT, &gradLocation);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a gradAttnWeight aclTensor.
    ret = CreateAclTensor(gradAttnWeightHostData, gradAttnWeightShape, &gradAttnWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradAttnWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMultiScaleDeformableAttentionGrad.
    ret = aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize(value, spatialShape, levelStartIndex, location, attnWeight, gradOutput, gradValue, gradLocation, gradAttnWeight, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnMultiScaleDeformableAttentionGrad.
    ret = aclnnMultiScaleDeformableAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultiScaleDeformableAttentionGrad failed. ERROR: %d\n", ret); return ret);
    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
    auto size = GetShapeSize(gradValueShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradValueDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
    aclDestroyTensor(value);
    aclDestroyTensor(spatialShape);
    aclDestroyTensor(levelStartIndex);
    aclDestroyTensor(location);
    aclDestroyTensor(attnWeight);
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(gradValue);
    aclDestroyTensor(gradLocation);
    aclDestroyTensor(gradAttnWeight);

    // 7. Release device resources. Modify the configuration based on the API definition.
    aclrtFree(valueDeviceAddr);
    aclrtFree(spatialShapeDeviceAddr);
    aclrtFree(levelStartIndexDeviceAddr);
    aclrtFree(locationDeviceAddr);
    aclrtFree(attnWeightDeviceAddr);
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(gradValueDeviceAddr);
    aclrtFree(gradLocationDeviceAddr);
    aclrtFree(gradAttnWeightDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
