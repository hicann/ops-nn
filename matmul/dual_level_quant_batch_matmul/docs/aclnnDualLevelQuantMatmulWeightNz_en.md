# aclnnDualLevelQuantMatmulWeightNz

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:49:43.661Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Performs matrix multiplication computation for secondary quantization with mxfp4. x2 must be in NZ format. You can use [aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md) or [aclnnNpuFormatCast](https://gitcode.com/cann/ops-math/blob/master/conversion/npu_format_cast/docs/aclnnNpuFormatCast.md) to convert x2 in ND format to NZ format.
- Formula

  $$
  out =\sum_{i}^{level0GroupSize} x1Level0Scale @ x2Level0Scale \sum_{ij}^{level1GroupSize} ((x1Level1Scale @ x1_{ij})@ (x2Level1Scale @ x2_{ij})) + bias
  $$

  - x1 and x2 are the left and right matrices for matrix multiplication, with the data type FLOAT4_E2M1.
  - x1Level0Scale and x2Level0Scale: first-level quantization parameters, with the data type FLOAT32.
  - x1Level1Scale and x2Level1Scale: second-level quantization parameters, with the data type FLOAT8_E8M0.
  - bias: optional parameter, the bias accumulated after the matrix multiplication operation, with the data type FLOAT32.
  - level0GroupSize: the size of the first-level quantization groupsize, which only supports 512.
  - level1GroupSize specifies the groupsize of secondary quantization. Only 32 is supported.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize" first to obtain the workspace size required for computation and the executor containing the operator computation process. Then call "aclnnDualLevelQuantMatmulWeightNz" to execute the computation.

```c++
aclnnStatus aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize(
    const aclTensor* x1,
    const aclTensor* x2,
    const aclTensor* x1Level0Scale,
    const aclTensor* x2Level0Scale,
    const aclTensor* x1Level1Scale,
    const aclTensor* x2Level1Scale,
    const aclTensor* optionalBias,
    bool transposeX1,
    bool transposeX2,
    int64_t level0GroupSize,
    int64_t level1GroupSize,
    aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

```c++
aclnnStatus aclnnDualLevelQuantMatmulWeightNz(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1554px"><colgroup>
  <col style="width: 248px">
  <col style="width: 121px">
  <col style="width: 170px">
  <col style="width: 397px">
  <col style="width: 220px">
  <col style="width: 115px">
  <col style="width: 138px">
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
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x1(aclTensor*)</td>
        <td>Input</td>
        <td>Left matrix in the matrix multiplication operation.</td>
        <td>          <ul>
            <li>Empty tensors are not supported.</li>
            <li>Only non-transpose is supported.</li>
          </ul>
        </td>
        <td>FLOAT4_E2M1</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x2(aclTensor*)</td>
        <td>Input</td>
        <td>Right matrix in the matrix multiplication operation.</td>
        <td>          <ul>
            <li>Empty tensors are not supported.</li>
            <li>Only transpose is supported.</li>
          </ul>
        </td>
        <td>FLOAT4_E2M1</td>
        <td>FRACTAL_NZ</td>
        <td>4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x1Level0Scale(aclTensor*)</td>
        <td>Input</td>
        <td>Scaling factor of the primary quantization parameters of x1, corresponding to x1Level0Scale in the formula.</td>
        <td>        <ul>
            <li>Empty tensors are not supported.</li>
            <li>Only non-transpose is supported.</li>
          </ul>
          </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x1Level1Scale(aclTensor*)</td>
        <td>Input</td>
        <td>Scaling factor of the secondary quantization parameters of x1, corresponding to x1Level1Scale in the formula.</td>
        <td>          <ul>
            <li>Empty tensors are not supported.</li>
            <li>Only non-transpose is supported.</li>
          </ul>
        </td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>3</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x2Level0Scale(aclTensor*)</td>
        <td>Input</td>
        <td>Scaling factor of the first-level quantization parameters of x2, which is x2Level0Scale in the corresponding formula.</td>
        <td>          <ul>
            <li>Empty tensors are not supported.</li>
            <li>Only non-transpose is supported.</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x2Level1Scale(aclTensor*)</td>
        <td>Input</td>
        <td>Scaling factor of the secondary quantization parameters of x2, corresponding to x2Level1Scale in the formula.</td>
        <td>         <ul>
            <li>Empty tensors are not supported.</li>
            <li>Only transpose is supported.</li>
         </ul>
        </td>
        <td>FLOAT8_E8M0</td>
        <td>ND</td>
        <td>3</td>
        <td>-</td>
      </tr>
      <tr>
        <td>optionalBias(aclTensor*)</td>
        <td>Optional input</td>
        <td>Bias accumulated after the matrix multiplication operation, corresponding to bias in the formula.</td>
        <td>          <ul>
            <li>Empty tensors are not supported.</li>
          </ul>
        </td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeX1(bool)</td>
        <td>Input</td>
        <td>Specifies whether the input shape of x1 is transposed.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeX2(bool)</td>
        <td>Input</td>
        <td>Indicates whether the input shape of x2 is transposed.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>level0GroupSize(int64_t)</td>
        <td>Input</td>
        <td>In primary quantization, the groupSize input used to dequantize inputs x1 and x2. It describes the size, along the Reduce direction, of the data to be dequantized by a set of dequantization parameters.</td>
        <td>          <ul>
            <li>Primary quantization groupsize, only 512 is supported</li>
          </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>level1GroupSize(int64_t)</td>
        <td>Input</td>
        <td>In secondary quantization, the groupSize input for dequantization of inputs x1 and x2. It specifies the amount of data to be dequantized along the Reduce direction for a group of dequantization parameters.</td>
        <td>          <ul>
            <li>Secondary quantization groupSize. Only 32 is supported</li>
          </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out(aclTensor)</td>
        <td>Output</td>
        <td>out is the output of the matrix computation.</td>
        <td>          <ul>
            <li>Empty tensors are not supported.</li>
          </ul>
        </td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>workspaceSize(uint64_t)</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be applied for on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td style="white-space: nowrap">executor(aclOpExecutor)</td>
        <td>Output</td>
        <td>Returns the op executor containing the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 749px">
  </colgroup>
  <thead>
      <tr>
        <th>Return Value</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
      <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>x1, x2, x1Level0Scale, x1Level1Scale, x2Level0Scale, x2Level1Scale, or out is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="4">161002</td>
        <td>x1, x2, x1Level0Scale, x1Level1Scale, x2Level0Scale, x2Level1Scale, level0GroupSize, or level1GroupSize is an empty tensor.</td>
      </tr>
      <tr>
        <td>The data type and data format of x1, x2, x1Level0Scale, x1Level1Scale, x2Level0Scale, x2Level1Scale, level0GroupSize, level1GroupSize, or out are not within the supported range.</td>
      </tr>
      <tr>
        <td>The shape of x1, x2, x1Level0Scale, x1Level1Scale, x2Level0Scale, x2Level1Scale, level0GroupSize, level1GroupSize, or out does not meet the check requirements.</td>
      </tr>
      <tr>
        <td>level0GroupSize or level1GroupSize does not meet the check requirements.</td>
      </tr>
    </tbody></table>

## aclnnDualLevelQuantMatmulWeightNz

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
    <td>Workspace memory address requested on the device.</td>
  </tr>
  <tr>
    <td>workspaceSize</td>
    <td>Input</td>
    <td>Size of the workspace memory requested on the device, obtained by the first-phase API aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize.</td>
  </tr>
  <tr>
    <td>executor</td>
    <td>Input</td>
    <td>op executor, containing the operator computation process.</td>
  </tr>
  <tr>
    <td>stream</td>
    <td>Input</td>
    <td>Specifies the Stream for task execution.</td>
  </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  Deterministic computation: aclnnDualLevelQuantMatmulWeightNz defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_dual_level_quant_matmul_nz.h"
#include "aclnnop/aclnn_npu_format_cast.h"
#include "aclnnop/aclnn_cast.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T>
void PrintMat(std::vector<T> resultData, std::vector<int64_t> resultShape)
{
    int64_t m = resultShape[0];
    int64_t n = resultShape[1];
    for (size_t i = 0; i < m; i++) {
        printf(i == 0 ? "[[" : " [");
        for (size_t j = 0; j < n; j++) {
            std::cout << resultData[i * n + j] << (j == n - 1 ? "" : ", ");
            if (j == 2 && j + 3 < n) {
                printf("..., ");
                j = n - 4;
            }
        }
        printf(i < m - 1 ? "],\n" : "]]\n");
        if (i == 2 && i + 3 < m) {
            printf(" ... \n");
            i = m - 4;
        }
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate for resource initialization.
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
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, const int64_t* storageShape,
    int64_t storageShapeSize, void** deviceAddr, aclDataType dataType, aclTensor** tensor,
    aclFormat format = aclFormat::ACL_FORMAT_ND)
{
    auto size = hostData.size() * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, format, storageShape, storageShapeSize, *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int AclnnDualLevelQuantMatmulWeightNz(int32_t deviceId, aclrtStream stream)
{
    int ret = 0;
    // 2. Construct the inputs and outputs; custom construction is required based on the API.
    constexpr int64_t B4_IN_B8_NUMS = 2L;
    constexpr int64_t B8_IN_B16_NUMS = 2L;
    int64_t m = 256;
    int64_t k = 1024;
    int64_t n = 512;
    int64_t level0GroupSize = 512;
    int64_t level1GroupSize = 32;
    bool transposeX1 = false;
    bool transposeX2 = true;
    std::vector<int64_t> x1Shape = {m, k};
    std::vector<int64_t> x2Shape = {n, k};
    std::vector<int64_t> biasShape = {n};
    std::vector<int64_t> x1Level0ScaleShape = {m, k / level0GroupSize};
    std::vector<int64_t> x1Level1ScaleShape = {m, k / level1GroupSize / B8_IN_B16_NUMS, B8_IN_B16_NUMS};
    std::vector<int64_t> x2Level0ScaleShape = {k / level0GroupSize, n};
    std::vector<int64_t> x2Level1ScaleShape = {n, k / level1GroupSize / B8_IN_B16_NUMS, B8_IN_B16_NUMS};
    std::vector<int64_t> outShape = {m, n};

    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* x2NzDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* x1Level0ScaleDeviceAddr = nullptr;
    void* x1Level1ScaleDeviceAddr = nullptr;
    void* x2Level0ScaleDeviceAddr = nullptr;
    void* x2Level1ScaleDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* outFp32DeviceAddr = nullptr;

    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* x2Nz = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* x1Level0Scale = nullptr;
    aclTensor* x1Level1Scale = nullptr;
    aclTensor* x2Level0Scale = nullptr;
    aclTensor* x2Level1Scale = nullptr;
    aclTensor* out = nullptr;
    aclTensor* outFp32 = nullptr;

    // Binary representation of 1.0 in fp4: 0b0010. Each uint8_t represents two fp4 values, and fp4 is carried in uint8_t.
    std::vector<uint8_t> x1HostData(GetShapeSize(x1Shape) / 2, 0b0010'0010);
    std::vector<uint8_t> x2HostData(GetShapeSize(x2Shape), 0b0010'0010);
    std::vector<float> biasHostData(n, 1002.0f);
    std::vector<float> x1Level0ScaleHostData(GetShapeSize(x1Level0ScaleShape), 1.0f);
    std::vector<float> x2Level0ScaleHostData(GetShapeSize(x2Level0ScaleShape), 1.0f);
    // Binary representation of 1.0 in fp8e8m0: 0b0111'1111
    std::vector<uint8_t> x1Level1ScaleHostData(GetShapeSize(x1Level1ScaleShape), 0b0111'1111);
    std::vector<uint8_t> x2Level1ScaleHostData(GetShapeSize(x2Level1ScaleShape), 0b0111'1111);
    // Fill the output with 0.
    std::vector<uint16_t> outHostData(GetShapeSize(outShape), 0);
    std::vector<float> outFp32HostData(GetShapeSize(outShape), 0.0f);

    // Create the x1 aclTensor.
    ret = CreateAclTensor(
        x1HostData, x1Shape, x1Shape.data(), x1Shape.size(), &x1DeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &x1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the x2 aclTensor.
    // NpuFormatCast cannot convert the B4 type. B8 is used instead, so the inner axis is halved.
    x2Shape[1] /= 2;
    ret =
        CreateAclTensor(x2HostData, x2Shape, x2Shape.data(), x2Shape.size(), &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Use the NpuFormatCast API to convert to the actual tensor, or directly construct weightNZ data.
    int64_t* x2NzShape = nullptr;
    uint64_t x2NzShapeSize = 0;
    int x2NzFormat;
    // Calculate the shape and format of the target tensor.
    ret = aclnnNpuFormatCastCalculateSizeAndFormat(
        x2, static_cast<int>(aclFormat::ACL_FORMAT_FRACTAL_NZ), static_cast<int>(aclDataType::ACL_INT8), &x2NzShape,
        &x2NzShapeSize, &x2NzFormat);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret);
              return ret);
    ret = CreateAclTensor(
        x2HostData, x2Shape, x2NzShape, x2NzShapeSize, &x2NzDeviceAddr, aclDataType::ACL_INT8, &x2Nz,
        static_cast<aclFormat>(x2NzFormat));
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2NzTensorPtr(x2Nz, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2NzDeviceAddrPtr(x2NzDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

    // Create the x1Level0Scale aclTensor.
    ret = CreateAclTensor(
        x1Level0ScaleHostData, x1Level0ScaleShape, x1Level0ScaleShape.data(), x1Level0ScaleShape.size(),
        &x1Level0ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Level0Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1Level0ScaleTensorPtr(
        x1Level0Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1Level0ScaleDeviceAddrPtr(x1Level0ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the x1Level1Scale aclTensor.
    ret = CreateAclTensor(
        x1Level1ScaleHostData, x1Level1ScaleShape, x1Level1ScaleShape.data(), x1Level1ScaleShape.size(),
        &x1Level1ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x1Level1Scale, aclFormat::ACL_FORMAT_NCL);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1Level1ScaleTensorPtr(
        x1Level1Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1Level1ScaleDeviceAddrPtr(x1Level1ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the x2Level0Scale aclTensor.
    ret = CreateAclTensor(
        x2Level0ScaleHostData, x2Level0ScaleShape, x2Level0ScaleShape.data(), x2Level0ScaleShape.size(),
        &x2Level0ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Level0Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2Level0ScaleTensorPtr(
        x2Level0Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2Level0ScaleDeviceAddrPtr(x2Level0ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the x2Level1Scale aclTensor.
    ret = CreateAclTensor(
        x2Level1ScaleHostData, x2Level1ScaleShape, x2Level1ScaleShape.data(), x2Level1ScaleShape.size(),
        &x2Level1ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x2Level1Scale, aclFormat::ACL_FORMAT_NCL);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2Level1ScaleTensorPtr(
        x2Level1Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2Level1ScaleDeviceAddrPtr(x2Level1ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the bias aclTensor.
    ret = CreateAclTensor(
        biasHostData, biasShape, biasShape.data(), biasShape.size(), &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> biasTensorPtr(bias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the out aclTensor.
    ret = CreateAclTensor(
        outHostData, outShape, outShape.data(), outShape.size(), &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the outFp32 aclTensor.
    ret = CreateAclTensor(
        outFp32HostData, outShape, outShape.data(), outShape.size(), &outFp32DeviceAddr, aclDataType::ACL_FLOAT, &outFp32);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outFp32TensorPtr(outFp32, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outFp32DeviceAddrPtr(outFp32DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Replace it with the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);

    // Use NpuFormatCast to convert x2 to the NZ format first
    // Call the first-phase API of aclnnNpuFormatCastGetWorkspaceSize
    ret = aclnnNpuFormatCastGetWorkspaceSize(x2, x2Nz, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* workspaceNpuFormatCastAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceNpuFormatCastAddrPtr(nullptr, aclrtFree);
    // Allocate device memory based on the workspaceSize computed by the first-phase API
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceNpuFormatCastAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceNpuFormatCastAddrPtr.reset(workspaceNpuFormatCastAddr);
    }
    // Call the second-phase API of aclnnNpuFormatCastGetWorkspaceSize
    ret = aclnnNpuFormatCast(workspaceNpuFormatCastAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

    // Call the first-phase API of aclnnDualLevelQuantMatmulWeightNz
    ret = aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize(
        x1, x2Nz, x1Level0Scale, x2Level0Scale, x1Level1Scale, x2Level1Scale, bias, transposeX1, transposeX2,
        level0GroupSize, level1GroupSize, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second-phase API of aclnnDualLevelQuantMatmulWeightNz
    ret = aclnnDualLevelQuantMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDualLevelQuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronize and wait for the task to complete
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output, convert the result on the device to FP32, and copy it to the host. Modify this step based on the definition of the specific API
    ret = aclnnCastGetWorkspaceSize(out, aclDataType::ACL_FLOAT, outFp32, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on the workspaceSize computed by the first-phase API
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 6. Obtain the output value, copy the result from device memory to host memory, and modify it based on the definition of the specific API.
    auto size = GetShapeSize(outShape);
    ret = aclrtMemcpy(
        outFp32HostData.data(), size * sizeof(outFp32HostData[0]), outFp32DeviceAddr, size * sizeof(outFp32HostData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    PrintMat(outFp32HostData, outShape);
    return ret;
}

int main()
{
    // 1. (boilerplate) Initialize device/stream. For details, see the acl API manual.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // Run the test process and release memory.
    AclnnDualLevelQuantMatmulWeightNz(deviceId, stream);
    Finalize(deviceId, stream);
    return 0;
}
```
