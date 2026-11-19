# aclnnMultiScaleDeformableAttnFunction

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:59:41.307Z -->

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

- Description:

  Different sampling points in feature maps of different sizes are traversed based on parameters such as the sampling location, attention weights, mapped value features, start index positions of the multi-scale features, and the spatial sizes of the multi-scale feature maps (to convert the sampling locations from normalized values to absolute positions).

- Formula:

    Let $b \in [0, bs)$, $q \in [0, num\_queries)$, $h \in [0, num\_heads)$, $\ell \in [0, num\_levels)$, $p \in [0, num\_points)$, and $c \in [0, channels)$ be the indices of the batch, query, head, feature map, sampling point, and channel, respectively. The height and width of the $\ell$-th layer feature map are $H_\ell$ and $W_\ell$, that is, $\mathrm{spatialShape}[\ell] = (H_\ell, W_\ell)$; the $num\_keys$ dimension of $\mathrm{value}$ flattens and concatenates the pixels of all layer feature maps layer by layer ($num\_keys = \sum_{\ell=0}^{num\_levels-1} H_\ell W_\ell$), and the start index of the $\ell$-th layer is $\mathrm{levelStartIndex}[\ell]$.

    1. Map the normalized coordinates of the sampling point $\mathrm{location}[b, q, h, \ell, p] = (u, v) \in [0,1]^2$ (where $u$ and $v$ are the 0th and 1st elements of the last dimension, corresponding to the $x$ and $y$ directions, respectively) to the pixel coordinate system of the $\ell$-th layer feature map:

       $$
       x = u \cdot W_\ell - 0.5, \qquad y = v \cdot H_\ell - 0.5
       $$

    2. Determine which four integer grid points the sampling point falls between:

       $$
       x_0 = \lfloor x \rfloor,\quad x_1 = x_0 + 1,\qquad
       y_0 = \lfloor y \rfloor,\quad y_1 = y_0 + 1
       $$

    3. Calculate the offset of the sampling point relative to the top-left grid point for interpolation weights:

       $$
       \alpha_x = x - x_0, \qquad \alpha_y = y - y_0
       $$

    4. Calculate the bilinear interpolation weights. The sum of the four neighboring points is 1:

       $$
       \begin{aligned}
       w_{00} &= (1-\alpha_y)(1-\alpha_x), \\
       w_{10} &= (1-\alpha_y)\alpha_x, \\
       w_{01} &= \alpha_y(1-\alpha_x), \\
       w_{11} &= \alpha_y\alpha_x
       \end{aligned}
       $$

    5. After flattening, the index of the pixel $(y, x)$ on the $\ell$ layer feature map in the $num\_keys$ dimension of $\mathrm{value}$ is:

       $$
       k_\ell(y, x) = \mathrm{levelStartIndex}[\ell] + y \cdot W_\ell + x
       $$

    6. Perform bilinear sampling on $\mathrm{value}$ to obtain the feature vector corresponding to the sampling point (length: $channels$):

       $$
       \begin{aligned}
       \operatorname{bilinear}(\mathrm{value};\,b,h,\ell,x,y) ={}&
       w_{00} \cdot \mathrm{value}[b,\; k_\ell(y_0, x_0),\; h,\; :] \\
       &+ w_{10} \cdot \mathrm{value}[b,\; k_\ell(y_0, x_1),\; h,\; :] \\
       &+ w_{01} \cdot \mathrm{value}[b,\; k_\ell(y_1, x_0),\; h,\; :] \\
       &+ w_{11} \cdot \mathrm{value}[b,\; k_\ell(y_1, x_1),\; h,\; :]
       \end{aligned}
       $$

    7. The bilinear sampling results of all layers and all sampling points are weighted and summed using $\mathrm{attnWeight}$ to obtain the final output $\mathrm{output}$ (shape: $(bs, num\_queries, num\_heads \times channels)$, with the last dimension arranged as $h \times channels + c$):

       $$
       \mathrm{output}[b,\; q,\; h \times channels + c] =
       \sum_{\ell=0}^{num\_levels-1} \sum_{p=0}^{num\_points-1}
       \mathrm{attnWeight}[b, q, h, \ell, p] \cdot
       \operatorname{bilinear}\!\left(\mathrm{value};\, b, h, \ell,\;
       x_{b,q,h,\ell,p},\, y_{b,q,h,\ell,p}\right)[c]
       $$

       Where $x_{b,q,h,\ell,p}$ and $y_{b,q,h,\ell,p}$ are the pixel coordinates obtained by mapping the sampling point $\mathrm{location}[b, q, h, \ell, p]$ through the preceding coordinate mapping.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the "aclnnMultiScaleDeformableAttnFunction" API to execute the computation.

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

- **Parameters**:

  <table style="undefined;table-layout: fixed; width: 100%"><colgroup>
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
      <td>value (aclTensor*)</td>
      <td>Input</td>
      <td>Feature values of the feature map. Corresponds to `value` in the formula.</td>
      <td>The shape is (bs, num_keys, num_heads, channels). bs is the batch size, num_keys is the size of the feature map, num_heads is the quantity of heads, and channels is the dimension of the feature map.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>spatialShape (aclTensor*)</td>
      <td>Input</td>
      <td>Stores the height and width of the feature map at each scale. Corresponds to `spatialShape` in the formula.</td>
      <td>The shape is (num_levels, 2). num_levels is the quantity of feature maps, and 2 indicates H and W, respectively.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>levelStartIndex (aclTensor*)</td>
      <td>Input</td>
      <td>Start index of each feature map, which is `levelStartIndex` in the corresponding formula.</td>
      <td>The shape is (num_levels).</td>
      <td>INT32, INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>location (aclTensor*)</td>
      <td>Input</td>
      <td>Sampling point position tensor, storing the coordinate position of each sampling point. Corresponds to `location` in the formula.</td>
      <td><ul><il>Data type: same as value.</il><il>Shape: (bs, num_queries, num_heads, num_levels, num_points, 2). num_queries is the quantity of queries, num_points is the quantity of sampling points, and 2 represents y and x respectively.</il></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>6</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attnWeight (aclTensor*)</td>
      <td>Input</td>
      <td>Sampling point weight tensor. Corresponds to `attnWeight` in the corresponding formula.</td>
      <td><ul><il>Data type must be the same as value.</il><il>Shape: (bs, num_queries, num_heads, num_levels, num_points).</il></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>output (aclTensor*)</td>
      <td>Output</td>
      <td>Operator computation output. Corresponds to `output` in the corresponding formula.</td>
      <td><ul><il>Data type is the same as value.</il><il>Shape is (bs, num_queries, num_heads * channels).</il></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="310p" id7 -->
  - <term>Atlas inference products</term>: BFLOAT16 data type is not supported.
  <!-- end id7 -->

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>Return value</th>
      <th>Error code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input or output is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>The data types of the input and output are not within the supported range.</td>
    </tr>
    <tr>
      <td>The input and output data types are inconsistent.</td>
    </tr>
    <tr>
      <td>The shape of value is not 4D.</td>
    </tr>
    <tr>
      <td>The shape of spatialShape is not 2D.</td>
    </tr>
    <tr>
      <td>The shape of levelStartIndex is not 1D.</td>
    </tr>
    <tr>
      <td>The shape of location is not 6D.</td>
    </tr>
    <tr>
      <td>The shape of attnWeight is not 5D.</td>
    </tr>
    <tr>
      <td>The last axis of spatialShape is not 2.</td>
    </tr>
    <tr>
      <td>The last axis of location is not 2.</td>
    </tr>
    <tr>
      <td>The constraints in the Constraints section are not met.</td>
    </tr>
  </tbody>
  </table>

## aclnnMultiScaleDeformableAttnFunction

- **Parameters**:

  <table><thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device side. Obtained by the first-phase API aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream on which tasks are executed.</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="A3,910b,310p" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Atlas inference products</term>: aclnnMultiScaleDeformableAttnFunction defaults to a deterministic implementation.
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnMultiScaleDeformableAttnFunction defaults to a non-deterministic implementation. You can enable deterministic computation through aclrtCtxSetSysParamOpt.

  <!-- end id9 -->

<!-- npu="310p" id10 -->
- <term>Atlas inference products</term>:
  - channels % 32 = 0, and channels <= 256.
  - Query count: 32 <= num_queries < 500000.
  - The number of feature maps: num_levels <= 16.
  - Head count: num_heads = [2, 4, 8].
  - Sampling point count: num_points = [4, 8].
<!-- end id10 -->
<!-- npu="950,A3,910b" id11 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:
  - Channel count: channels%8 = 0, and channels <= 256.
  - The number of queries: 32 <= num_queries < 500000.
  - The number of feature maps: num_levels <= 16.
  - The number of heads: num_heads <= 16.
  - The number of sampling points: num_points <= 16.

<!-- end id11 -->

## Example

The sample code is as follows and is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
    // Call aclrtMalloc to allocate memory on the device
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host-side data to the device-side memory
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of a contiguous tensor
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
       strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1.(Boilerplate) Initialize the device/stream, see the list of ACL external APIs
    // Set deviceId based on the actual device
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
   // 2. Construct the input and output, which must be custom-built based on the API.
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
    std::vector<float> valueHostData(GetShapeSize(valueShape), 0.0f);
    std::vector<int32_t> spatialShapeHostData = {1, 1};
    std::vector<float> levelStartIndexHostData = {0};
    std::vector<float> locationHostData(static_cast<float>(GetShapeSize(locationShape)), 0);
    std::vector<float> attnWeightHostData(GetShapeSize(attnWeightShape), 0.0f);
    std::vector<float> outputHostData(GetShapeSize(outputShape), 0.0f);
    // value aclTensor
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the spatialShape aclTensor.
    ret = CreateAclTensor(spatialShapeHostData, spatialShapeShape, &spatialShapeDeviceAddr, aclDataType::ACL_INT32, &spatialShape);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the levelStartIndex aclTensor.
    ret = CreateAclTensor(levelStartIndexHostData, levelStartIndexShape, &levelStartIndexDeviceAddr, aclDataType::ACL_INT32, &levelStartIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the location aclTensor.
    ret = CreateAclTensor(locationHostData, locationShape, &locationDeviceAddr, aclDataType::ACL_FLOAT, &location);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the attnWeight aclTensor.
    ret = CreateAclTensor(attnWeightHostData, attnWeightShape, &attnWeightDeviceAddr, aclDataType::ACL_FLOAT, &attnWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the output aclTensor.
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which must be changed to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnMultiScaleDeformableAttnFunction.
    ret = aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize(value, spatialShape, levelStartIndex, location, attnWeight, output,
                                                                &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnMultiScaleDeformableAttnFunction.
    ret = aclnnMultiScaleDeformableAttnFunction(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultiScaleDeformableAttnFunction failed. ERROR: %d\n", ret); return ret);
    // 4.(boilerplate)Synchronize and wait for task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5.Obtain the output value and copy the result from the device side to the host side. Modify the code according to the specific API definition.
    auto size = GetShapeSize(outputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6.Release aclTensor and aclScalar. Modify the code according to the specific API definition.
    aclDestroyTensor(value);
    aclDestroyTensor(spatialShape);
    aclDestroyTensor(levelStartIndex);
    aclDestroyTensor(location);
    aclDestroyTensor(attnWeight);
    aclDestroyTensor(output);

    // 7.Release device resources. Modify the code according to the specific API definition.
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
