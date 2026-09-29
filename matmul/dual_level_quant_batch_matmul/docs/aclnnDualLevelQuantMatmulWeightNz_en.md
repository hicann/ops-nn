# aclnnDualLevelQuantMatmulWeightNz

## Supported Products

| Product                                                    | Supported|
| :------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                  |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>|    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                 |    ×     |
| <term>Atlas inference products</term>                        |    ×     |
| <term>Atlas training products</term>                         |    ×     |

## Function

- This API is used to complete the matrix multiplication computation of level-2 quantization mxfp4. The parameter x2 must be in the NZ format. You can use [aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md) or [aclnnNpuFormatCast](https://gitcode.com/cann/ops-math/blob/master/conversion/npu_format_cast/docs/aclnnNpuFormatCast.md) to convert the format of x2 from ND to NZ.
- Formula

  $$
  out =\sum_{i}^{level0GroupSize} x1Level0Scale @ x2Level0Scale \sum_{ij}^{level1GroupSize} ((x1Level1Scale @ x1_{ij})@ (x2Level1Scale @ x2_{ij})) + bias
  $$

  - x1 and x2 are the left and right matrices for matrix computation, respectively. The data type is FLOAT4_E2M1.
  - x1Level0Scale and x2Level0Scale are level-1 quantization parameters. The data type is FLOAT32.
  - x1Level1Scale and x2Level1Scale are level-2 quantization parameters. The data type is FLOAT8_E8M0.
  - (Optional) bias is the bias added after the matrix multiplication operation. The data type is FLOAT32.
  - level0GroupSize indicates the group size for level-1 quantization. Only 512 is supported.
  - level1GroupSize indicates the group size for level-1 quantization. Only 32 is supported.

## Prototype

Each operator has <a href="../../../docs/en/context/two_phase_api.md">two-phase API calls</a>. You must call the aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize API to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the aclnnDualLevelQuantMatmulWeightNz API to perform the computation.

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
        <td>x1(aclTensor*)</td>
        <td>Input</td>
        <td>Left matrix in the matrix multiplication operation.</td>
        <td>
          <ul>
            <li>Empty tensors are not supported.</li>
            <li>Only non-transposition is supported.</li>
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
        <td>
          <ul>
            <li>Empty tensors are not supported.</li>
            <li>Only transposition is supported.</li>
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
        <td>Scale factor of the level-1 quantization parameters of x1, corresponding to x1Level0Scale in the formula.</td>
        <td>
        <ul>
            <li>Empty tensors are not supported.</li>
            <li>Only non-transposed tensors are supported.</li>
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
        <td>Scale factor of the level-2 quantization parameters of x1, corresponding to x1Level1Scale in the formula.</td>
        <td>
          <ul>
            <li>Empty tensors are not supported.</li>
            <li>Only non-transposed tensors are supported.</li>
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
        <td>Scale factor of the level-1 quantization parameters of x2, corresponding to x2Level0Scale in the formula.</td>
        <td>
          <ul>
            <li>Empty tensors are not supported.</li>
            <li>Only non-transposed tensors are supported.</li>
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
        <td>Scaling factor of the level-2 quantization parameter of x2, corresponding to x2Level1Scale in the formula.</td>
        <td>
         <ul>
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
        <td>Bias added after matrix multiplication, corresponding to bias in the formula.</td>
        <td>
          <ul>
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
        <td>Whether the input shape of x1 is transposed.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeX2(bool)</td>
        <td>Input</td>
        <td>Whether the input shape of x2 is transposed.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>level0GroupSize(int64_t)</td>
        <td>Input</td>
        <td>Group size input for dequantizing x1 and x2 in level-1 quantization, which describes the size of the data to be dequantized corresponding to a group of dequantization parameters in the Reduce direction.</td>
        <td>
          <ul>
            <li>Group size for level-1 quantization. Only 512 is supported.</li>
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
        <td>Group size input for dequantizing x1 and x2 in level-2 quantization, which describes the size of the data to be dequantized corresponding to a group of dequantization parameters in the Reduce direction.</td>
        <td>
          <ul>
            <li>Group size for level-2 quantization. Only 32 is supported.</li>
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
        <td>Output of the computation result.</td>
        <td>
          <ul>
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
        <td>Size of the workspace to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td style="white-space: nowrap">executor(aclOpExecutor)</td>
        <td>Output</td>
        <td>Operator executor, containing the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
  </tbody></table>

- **Return Value**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

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
        <td>The input x1, x2, x1Level0Scale, x1Level1Scale, x2Level0Scale, x2Level1Scale, or out is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="5">161002</td>
        <td>x1, x2, x1Level0Scale, x1Level1Scale, x2Level0Scale, x2Level1Scale, level0GroupSize and level1GroupSize are null tensors.</td>
      </tr>
      <tr>
        The data type and format of <td>x1, x2, x1Level0Scale, x1Level1Scale, x2Level0Scale, x2Level1Scale, level0GroupSize, level1GroupSize, or out are not supported.</td>
      </tr>
      <tr>
        The shape of <td>x1, x2, x1Level0Scale, x1Level1Scale, x2Level0Scale, x2Level1Scale, level0GroupSize, level1GroupSize, or out does not meet the verification conditions.</td>
      </tr>
      <tr>
        <td>The input level0GroupSize or level1GroupSize does not meet the verification conditions.</td>
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
    <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize.</td>
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

- **Return Value**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  Deterministic computing: The aclnnDualLevelQuantMatmulWeightNz function is implemented in deterministic mode by default.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, const int64_t* storageShape,
    int64_t storageShapeSize, void** deviceAddr, aclDataType dataType, aclTensor** tensor,
    aclFormat format = aclFormat::ACL_FORMAT_ND)
{
    auto size = hostData.size() * sizeof(T);
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
    // 2.Construct the inputs and outputs based on the API definition.
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

    // 1.0 binary representation of fp4, 0b0010. Each uint8_t represents two fp4s, and uint8_t is used to carry fp4.
    std::vector<uint8_t> x1HostData(GetShapeSize(x1Shape) / 2, 0b0010'0010);
    std::vector<uint8_t> x2HostData(GetShapeSize(x2Shape), 0b0010'0010);
    std::vector<float> biasHostData(n, 1002.0f);
    std::vector<float> x1Level0ScaleHostData(GetShapeSize(x1Level0ScaleShape), 1.0f);
    std::vector<float> x2Level0ScaleHostData(GetShapeSize(x2Level0ScaleShape), 1.0f);
    Binary representation of // fp8e8m0 1.0, 0b0111'1111
    std::vector<uint8_t> x1Level1ScaleHostData(GetShapeSize(x1Level1ScaleShape), 0b0111'1111);
    std::vector<uint8_t> x2Level1ScaleHostData(GetShapeSize(x2Level1ScaleShape), 0b0111'1111);
    //Use 0s for padding.
    std::vector<uint16_t> outHostData(GetShapeSize(outShape), 0);
    std::vector<float> outFp32HostData(GetShapeSize(outShape), 0.0f);

    // Create an x1 aclTensor.
    ret = CreateAclTensor(
        x1HostData, x1Shape, x1Shape.data(), x1Shape.size(), &x1DeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &x1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an x2 aclTensor.
    // NpuFormatCast cannot convert the B4 type. Therefore, B8 is used instead, and the inner axis is halved.
    x2Shape[1] /= 2;
    ret =
        CreateAclTensor(x2HostData, x2Shape, x2Shape.data(), x2Shape.size(), &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Use the NpuFormatCast API to convert the tensor to the actual tensor, or directly construct the weightNZ data.
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

    // Create a x1Level0Scale aclTensor.
    ret = CreateAclTensor(
        x1Level0ScaleHostData, x1Level0ScaleShape, x1Level0ScaleShape.data(), x1Level0ScaleShape.size(),
        &x1Level0ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Level0Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1Level0ScaleTensorPtr(
        x1Level0Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1Level0ScaleDeviceAddrPtr(x1Level0ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create a x1Level1Scale aclTensor.
    ret = CreateAclTensor(
        x1Level1ScaleHostData, x1Level1ScaleShape, x1Level1ScaleShape.data(), x1Level1ScaleShape.size(),
        &x1Level1ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x1Level1Scale, aclFormat::ACL_FORMAT_NCL);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1Level1ScaleTensorPtr(
        x1Level1Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1Level1ScaleDeviceAddrPtr(x1Level1ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    //Create a x2Level0Scale aclTensor.
    ret = CreateAclTensor(
        x2Level0ScaleHostData, x2Level0ScaleShape, x2Level0ScaleShape.data(), x2Level0ScaleShape.size(),
        &x2Level0ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Level0Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2Level0ScaleTensorPtr(
        x2Level0Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2Level0ScaleDeviceAddrPtr(x2Level0ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    //Create a x2Level1Scale aclTensor.
    ret = CreateAclTensor(
        x2Level1ScaleHostData, x2Level1ScaleShape, x2Level1ScaleShape.data(), x2Level1ScaleShape.size(),
        &x2Level1ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x2Level1Scale, aclFormat::ACL_FORMAT_NCL);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2Level1ScaleTensorPtr(
        x2Level1Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2Level1ScaleDeviceAddrPtr(x2Level1ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create a bias aclTensor.
    ret = CreateAclTensor(
        biasHostData, biasShape, biasShape.data(), biasShape.size(), &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> biasTensorPtr(bias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an out aclTensor.
    ret = CreateAclTensor(
        outHostData, outShape, outShape.data(), outShape.size(), &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create outFp32 aclTensor.
    ret = CreateAclTensor(
        outFp32HostData, outShape, outShape.data(), outShape.size(), &outFp32DeviceAddr, aclDataType::ACL_FLOAT, &outFp32);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outFp32TensorPtr(outFp32, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outFp32DeviceAddrPtr(outFp32DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);

    // First, use NpuFormatCast to convert x2 to the NZ format.
    // Call the first-phase API of aclnnNpuFormatCastGetWorkspaceSize.
    ret = aclnnNpuFormatCastGetWorkspaceSize(x2, x2Nz, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* workspaceNpuFormatCastAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceNpuFormatCastAddrPtr(nullptr, aclrtFree);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceNpuFormatCastAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceNpuFormatCastAddrPtr.reset(workspaceNpuFormatCastAddr);
    }
    // Call the second-phase API of aclnnNpuFormatCastGetWorkspaceSize.
    ret = aclnnNpuFormatCast(workspaceNpuFormatCastAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

    // Call the first part of the aclnnDualLevelQuantMatmulWeightNz API.
    ret = aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize(
        x1, x2Nz, x1Level0Scale, x2Level0Scale, x1Level1Scale, x2Level1Scale, bias, transposeX1, transposeX2,
        level0GroupSize, level1GroupSize, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second part of the aclnnDualLevelQuantMatmulWeightNz API.
    ret = aclnnDualLevelQuantMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDualLevelQuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value, convert the result in the device memory to FP32, and copy it to the host. The code needs to be modified based on the API definition.
    ret = aclnnCastGetWorkspaceSize(out, aclDataType::ACL_FLOAT, outFp32, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 6. Obtain the output value and copy the result from the device memory to the host memory. Modify the code based on the API definition.
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
    // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // Run the test and free the memory.
    AclnnDualLevelQuantMatmulWeightNz(deviceId, stream);
    Finalize(deviceId, stream);
    return 0;
}
```
