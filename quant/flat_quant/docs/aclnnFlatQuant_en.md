# aclnnFlatQuant

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/quant/flat_quant)

## Product Support

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     ×    |
|  <term>Atlas training series products</term>   |     ×    |

## Function

- Description: Performs two small matrix multiplications on the input matrix **x**: it right-multiplies **x** by **kroneckerP2** and left-multiplies the result by **kroneckerP1**, then applies per-token quantization to the matrix multiplication result.

- Formula:
  
  1. Right-multiply **x** by **kroneckerP2**:
  
    $$
    x' = x @ kroneckerP2
    $$

  2. Left-multiply **x'** by **kroneckerP1**:

    $$
    x'' = kroneckerP1@x'
    $$
  
  3. Compute the maximum absolute value along dimension 0 of **x''** and divide by (7/clipRatio) to obtain the quantization factor for INT4 quantization:

    $$
    quantScale = [max(abs(x''[0,:,:])),max(abs(x''[1,:,:])),...,max(abs(x''[K,:,:]))]/(7 / clipRatio)
    $$
  
  4. Compute the output **out**:
  
    $$
    out = x'' / quantScale
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnFlatQuantGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, `aclnnFlatQuant` is called to perform computation.

```cpp
aclnnStatus aclnnFlatQuantGetWorkspaceSize(
  const aclTensor *x, 
  const aclTensor *kroneckerP1, 
  const aclTensor *kroneckerP2, 
  double           clipRatio, 
  aclTensor       *out, 
  aclTensor       *quantScale, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnFlatQuant(
    void          *workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```


## aclnnFlatQuantGetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
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
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input data, corresponding to `x` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape is [K, M, N], where K does not exceed 262144, and M and N do not exceed 256. </li><li>If the data type of out is INT32, N must be an integer multiple of 8. </li><li>If the data type of out is INT4, N must be an even number.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kroneckerP1</td>
      <td>Input</td>
      <td>First input computation matrix, corresponding to `kroneckerP1` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape is [M, M], where M is the same as the M dimension in x. </li><li>The data type is the same as that of x.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kroneckerP2</td>
      <td>Input</td>
      <td>Second input computation matrix, corresponding to `kroneckerP2` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape is [N, N], where N is the same as the N dimension in x. </li><li>The data type is the same as that of x.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>clipRatio</td>
      <td>Input</td>
      <td>Clipping ratio for quantization control, corresponding to `clipRatio` in the formula.</td>
      <td><ul><li>The input data range is (0, 1].</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor, corresponding to out in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>When the data type is INT32, the last dimension of the shape is 1/8 of the last dimension of x, while the remaining dimensions are the same as those of x. </li><li>When the data type is INT4, the shape is the same as that of x.</li></ul></td>
      <td>INT4, INT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantScale</td>
      <td>Output</td>
      <td>Output quantization factor, corresponding to quantScale in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape is [K], where K is the same as the K dimension in x.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
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
  </tbody>
  </table>


- **Returns**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.
  
  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td>The required input (x, kroneckerP1, or kroneckerP2) or output (out or quantScale) is passed as a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>The data type or format of x, kroneckerP1, kroneckerP2, out, or quantScale is not supported.</td>
    </tr>
    <tr>
      <td>The data type of kroneckerP1 or kroneckerP2 does not match that of x.</td>
    </tr>
     <tr>
      <td>x is not a 3D tensor.</td>
    </tr>
    <tr>
      <td>The first dimension of x is not in the range [1, 262144], the second dimension is not in the range [1, 256], or the third dimension is not in the range [1, 256].</td>
    </tr>
    <tr>
      <td>kroneckerP1 is not a 2D tensor, or the first and second dimensions are inconsistent with the second dimension of x.</td>
    </tr>
    <tr>
      <td>kroneckerP2 is not a 2D tensor, or the first and second dimensions are inconsistent with the third dimension of x.</td>
    </tr>
    <tr>
      <td>quantScale is not a 1D tensor, or the first dimension is inconsistent with the first dimension of x.</td>
    </tr>
    <tr>
      <td>The value of clipRatio is out of the range (0, 1].</td>
    </tr>
    <tr>
      <td>When the data type of out is INT4, the size of the last dimension of x is not an even number, or the shape of x is inconsistent with that of out.</td>
    </tr>
    <tr>
      <td>When the data type of out is INT32, the size of the last dimension of x is not 8 times the size of the last dimension of out, or the sizes of the non-last dimensions of the shapes of x and out are inconsistent.</td>
    </tr>
  </tbody></table>

## aclnnFlatQuant

- **Parameters**

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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnFlatQuantGetWorkspaceSize.</td>
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

- Deterministic computation:
  - **aclnnFlatQuant** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_flat_quant.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID (deviceId) based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Customize error handling based on your requirements.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs based on API definitions.
    std::vector<int64_t> xShape = {16, 16, 16};
    std::vector<int64_t> kroneckerP1Shape = {16, 16};
    std::vector<int64_t> kroneckerP2Shape = {16, 16};
    std::vector<int64_t> outShape = {16, 16, 2};
    std::vector<int64_t> quantScaleShape = {16};
    void* xDeviceAddr = nullptr;
    void* kroneckerP1DeviceAddr = nullptr;
    void* kroneckerP2DeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* quantScaleDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* kroneckerP1 = nullptr;
    aclTensor* kroneckerP2 = nullptr;
    aclTensor* out = nullptr;
    aclTensor* quantScale = nullptr;
    double clipRatio = 1.0;
    std::vector<aclFloat16> xHostData(16 * 16 * 16, aclFloatToFloat16(1));
    std::vector<aclFloat16> kroneckerP1HostData(16 * 16, aclFloatToFloat16(1));
    std::vector<aclFloat16> kroneckerP2HostData(16 * 16, aclFloatToFloat16(1));
    std::vector<int32_t> outHostData(16 * 16 * 2, 1);
    std::vector<float> quantScaleHostData(16, 0);
    // Create an x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create a kroneckerP1 aclTensor.
    ret = CreateAclTensor(
        kroneckerP1HostData, kroneckerP1Shape, &kroneckerP1DeviceAddr, aclDataType::ACL_FLOAT16, &kroneckerP1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a KroneckerP2 aclTensor.
    ret = CreateAclTensor(
        kroneckerP2HostData, kroneckerP2Shape, &kroneckerP2DeviceAddr, aclDataType::ACL_FLOAT16, &kroneckerP2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a quantScale aclTensor.
    ret = CreateAclTensor(
        quantScaleHostData, quantScaleShape, &quantScaleDeviceAddr, aclDataType::ACL_FLOAT, &quantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API. Modify the API as required.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnFlatQuant.
    ret = aclnnFlatQuantGetWorkspaceSize(
        x, kroneckerP1, kroneckerP2, clipRatio, out, quantScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlatQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnFlatQuant.
    ret = aclnnFlatQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFlatQuant failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize the stream and wait for task completion.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(outShape);
    std::vector<int32_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    auto quantScaleSize = GetShapeSize(quantScaleShape);
    std::vector<float> quantScaleResultData(quantScaleSize, 0);
    ret = aclrtMemcpy(
        quantScaleResultData.data(), quantScaleResultData.size() * sizeof(quantScaleResultData[0]),
        quantScaleDeviceAddr, quantScaleSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < quantScaleSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, quantScaleResultData[i]);
    }

    // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(kroneckerP1);
    aclDestroyTensor(kroneckerP2);
    aclDestroyTensor(out);
    aclDestroyTensor(quantScale);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(kroneckerP1DeviceAddr);
    aclrtFree(kroneckerP2DeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(quantScaleDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
