# aclnnMultiScaleDeformableAttnFunction

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/vfusion/multi_scale_deformable_attn_function)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    √     |
| <term>Atlas training series products</term>                             |    ×     |


## Function

- Description: Traverses different sampling points of feature maps of different sizes by using parameters such as the sample location, attention weights, mapped value feature, start index location of a multi-scale feature, and spatial size of a multi-scale feature map (which facilitates changing a sampling location from a normalized value to an absolute location).

- Formula:

    Map the normalized coordinates $(u,v)\in[0,1]$ of the sampling point to the pixel coordinate system of the feature map at layer $\ell$:

    $$
    x = u \cdot W_\ell - 0.5, \qquad y = v \cdot H_\ell - 0.5
    $$


    Determine the four integer grid points between which the sampling point falls:

    $$
    x_0 = \lfloor x \rfloor,\quad x_1 = x_0 + 1,\qquad
    y_0 = \lfloor y \rfloor,\quad y_1 = y_0 + 1
    $$  

    Compute the offset of the sampling point relative to the upper-left grid point, which is used for interpolation weighting:

    $$
    \alpha_x = x - x_0, \qquad \alpha_y = y - y_0
    $$  

    Compute the bilinear interpolation weight. The sum of the four adjacent points is 1.

    $$
    \begin{aligned}
    w_{00} &= (1-\alpha_y)(1-\alpha_x), \\
    w_{10} &= (1-\alpha_y)\alpha_x, \\
    w_{01} &= \alpha_y(1-\alpha_x), \\
    w_{11} &= \alpha_y\alpha_x
    \end{aligned}
    $$

    Compute the feature vectors (length: $D$) corresponding to the sampling points.

    $$
    \operatorname{bilinear}(V;\,b,h,\ell,x,y) =
    w_{00} \, V_{b,\ell,y_0,x_0,h,:}
    + w_{10} \, V_{b,\ell,y_0,x_1,h,:}
    + w_{01} \, V_{b,\ell,y_1,x_0,h,:}
    + w_{11} \, V_{b,\ell,y_1,x_1,h,:}
    $$  

    Compute the weighted sum of the bilinear sampling results for all layers and all sampling points to obtain the final output:
    
    $$
    O_{b,q,h,:} =
    \sum_{\ell=0}^{L-1} \sum_{p=0}^{N_p-1}
    A_{b,q,h,\ell,p} \cdot
    \operatorname{bilinear}\!\left(V;\,b,h,\ell,
    x_{b,q,h,\ell,p}, y_{b,q,h,\ell,p}\right)
    $$  

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnMultiScaleDeformableAttnFunction** is called to perform computation.
```Cpp
aclnnStatus aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize(
    const aclTensor* value, 
    const aclTensor* spatialShape, 
    const aclTensor* levelStartIndex, 
    const aclTensor* location, 
    const aclTensor* attnWeight, 
    aclTensor*       output, 
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor)
```
```Cpp
aclnnStatus aclnnMultiScaleDeformableAttnFunction(
    void*          workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor* executor, 
    aclrtStream    stream)
```

## aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize

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
      <td>The data type must be the same as that of value.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(bs, num_queries, num_heads, num_levels, num_points, 2)<br>num_queries indicates the number of queries, num_points indicates the number of sampling points, and 2 represents y and x.</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attnWeight</td>
      <td>Input</td>
      <td>Weight tensor of the sampling point.</td>
      <td>The data type must be the same as that of value.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(bs, num_queries, num_heads, num_levels, num_points)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>Operator compute output.</td>
      <td>The data type must be the same as that of value.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(bs, num_queries, num_heads * channels)</td>
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
  
  - Atlas inference series products: BFLOAT16 is not supported.
- **Returns**:
  
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
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>The input and output data types are not supported.</td>
    </tr>
    <tr>
      <td>The input and output data types are inconsistent.</td>
    </tr>
    <tr>
      <td>The shape of value is not four-dimensional.</td>
    </tr>
    <tr>
      <td>The shape of spatialShape is not two-dimensional.</td>
    </tr>
    <tr>
      <td>The shape of levelStartIndex is not one-dimensional.</td>
    </tr>
    <tr>
      <td>The shape of location is not six-dimensional.</td>
    </tr>
    <tr>
      <td>The shape of attnWeight is not five-dimensional.</td>
    </tr>
    <tr>
      <td>The last axis of spatialShape is not 2.</td>
    </tr>
    <tr>
      <td>The last axis of location is not 2.</td>
    </tr>
    <tr>
      <td>The API constraints are not met.</td>
    </tr>
  </tbody>
  </table>

## aclnnMultiScaleDeformableAttnFunction

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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize.</td>
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

- **Returns**:
  
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic compute:
  - **aclnnMultiScaleDeformableAttnFunction** defaults to a deterministic implementation.

- <term>Atlas inference series products</term>:
  - channels%32 = 0, and channels ≤ 256
  - 32 ≤ num_queries < 500000
  - num_levels ≤ 16
  - num_heads = [2, 4, 8]
  - num_points = [4, 8]
- <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>:
  - channels%8 = 0, and channels ≤ 256
  - 32 ≤ num_queries < 500000
  - num_levels ≤ 16
  - num_heads ≤ 16
  - num_points ≤ 16

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_multi_scale_deformable_attn_function.h"

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
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
	// 2. Construct the input and output based on the API definition.
    std::vector<int64_t> valueShape = {1, 1, 2, 32};
    std::vector<int64_t> spatialShapeShape = {1, 2};
    std::vector<int64_t> levelStartIndexShape = {1};
    std::vector<int64_t> locationShape = {1, 32, 2, 1, 4, 2};
    std::vector<int64_t> attnWeightShape = {1, 32, 2, 1, 4};
    std::vector<int64_t> outputShape = {1, 32, 64};
    void* valueDeviceAddr = nullptr;
    void* spatialShapeDeviceAddr = nullptr;
    void* levelStartIndexDeviceAddr = nullptr;
    void* locationDeviceAddr = nullptr;
    void* attnWeightDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;
    aclTensor* value = nullptr;
    aclTensor* spatialShape = nullptr;
    aclTensor* levelStartIndex = nullptr;
    aclTensor* location = nullptr;
    aclTensor* attnWeight = nullptr;
    aclTensor* output = nullptr;
    std::vector<float> valueHostData = {static_cast<float>(GetShapeSize(locationShape)), 1};
    std::vector<float> spatialShapeHostData = {1, 1};
    std::vector<float> levelStartIndexHostData = {0};
    std::vector<float> locationHostData(static_cast<float>(GetShapeSize(locationShape)), 0);
    std::vector<float> attnWeightHostData = {static_cast<float>(GetShapeSize(attnWeightShape)), 1};
    std::vector<float> outputHostData = {static_cast<float>(GetShapeSize(outputShape)), 1};
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
    // Create an output aclTensor.
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMultiScaleDeformableAttnFunction.
    ret = aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize(value, spatialShape, levelStartIndex, location, attnWeight, output, 
                                                                &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnMultiScaleDeformableAttnFunction.
    ret = aclnnMultiScaleDeformableAttnFunction(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultiScaleDeformableAttnFunction failed. ERROR: %d\n", ret); return ret);
    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
    auto size = GetShapeSize(outputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputDeviceAddr, size * sizeof(float),
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
    aclDestroyTensor(output);

    // 7. Release device resources. Modify the configuration based on the API definition.
    aclrtFree(valueDeviceAddr);
    aclrtFree(spatialShapeDeviceAddr);
    aclrtFree(levelStartIndexDeviceAddr);
    aclrtFree(locationDeviceAddr);
    aclrtFree(attnWeightDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
