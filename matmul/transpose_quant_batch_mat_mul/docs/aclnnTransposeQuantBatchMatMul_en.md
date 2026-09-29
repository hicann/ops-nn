# aclnnTransposeQuantBatchMatMul

[📄 View Source Code](https://gitcode.com/cann/ops-nn/tree/master/matmul/transpose_quant_batch_mat_mul)

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- This API is used to perform the quantized matrix multiplication of tensor x1 and tensor x2. The K-C quantization mode is supported (../../../docs/en/context/quant_more_introduction.md). Only three-dimensional tensors are supported. Tensors can be transposed. The transposition sequence is changed based on the input sequence. **permX1** indicates the transposition sequence of tensor **x1**, and **permX2** indicates the transposition sequence of tensor **x2**. The sequence value **0** indicates the batch dimension, and the other two dimensions are used for matrix multiplication.

- Example:
  Assume that the shape of x1 is (M, B, K), the shape of x2 is (B, K, N), x1Scale and x2Scale are not None, and batchSplitFactor is 1. The shape of the output out is (M, B, N).

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnTransposeQuantBatchMatMulGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnTransposeQuantBatchMatMul` is called to perform computation.

```cpp
aclnnStatus aclnnTransposeQuantBatchMatMulGetWorkspaceSize(
    const aclTensor*   x1, 
    const aclTensor*   x2, 
    const aclTensor*   bias, 
    const aclTensor*   x1Scale, 
    const aclTensor*   x2Scale,
    const int32_t      dtype, 
    const int32_t      groupSize, 
    const aclIntArray* permX1, 
    const aclIntArray* permX2,
    const aclIntArray* permY, 
    const int32_t      batchSplitFactor, 
    aclTensor*         out, 
    uint64_t*          workspaceSize,
    aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnTransposeQuantBatchMatMul(
    void               *workspace, 
    uint64_t            workspaceSize,
    aclOpExecutor      *executor,
    const aclrtStream   stream)
```

## aclnnTransposeQuantBatchMatMulGetWorkSpaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed;width: 1545px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 350px">
    <col style="width: 210px">
    <col style="width: 120px">
    <col style="width: 130px">
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
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x1 (aclTensor*) </td>
        <td>Input</td>
        <td>First matrix for matrix multiplication.</td>
        <td>
          <ul>
            <li>Its data type and the data type of x2 must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>.</li>
            <li>The data type can only be FLOAT8_E5M2 or FLOAT8_E4M3FN.</li>
          </ul>
        </td>
        <td>FLOAT8_E5M2, FLOAT8_E4M3FN</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2 (aclTensor*) </td>
        <td>Input</td>
        <td>Second matrix for matrix multiplication.</td>
        <td>
        <ul>
            <li>Its data type and the data type of x1 must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>.</li>
            <li>The size of the k dimension of x2 must be the same as that of x1.</li>
            <li>The data type can only be FLOAT8_E5M2 or FLOAT8_E4M3FN.</li>
        </ul>
        </td>
        <td>FLOAT8_E5M2, FLOAT8_E4M3FN</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias (aclTensor*) </td>
        <td>Input</td>
        <td>Indicates the bias matrix for matrix multiplication.</td>
        <td>Reserved parameter, which is not supported currently.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
      <td>x1Scale (aclTensor*) </td>
        <td>Input</td>
        <td>Indicates the quantization coefficient of the left matrix.</td>
        <td>The shape must be a one-dimensional array and must be equal to [m].</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
      <td>x2Scale (aclTensor*) </td>
        <td>Input</td>
        <td>Indicates the quantization coefficient of the right matrix.</td>
        <td>The shape must be a one-dimensional array and must be equal to [n].</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dtype (int32_t) </td>
        <td>Input</td>
        <td>Data type of the output matrix. The supported values are 1 and 27.</td>
        <td>
        <ul>
          <li>The value 1 indicates that the output matrix type is FLOAT16.</li>
          <li>The value 27 indicates that the output matrix type is BFLOAT16.</li>
        </ul>
        </td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSize (int32_t) </td>
        <td>Input</td>
        <td>Quantization group size. This parameter is reserved. Currently, only the value 0 is supported.</td>
        <td>Currently, the value other than 0 does not take effect.</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>permX1 (aclIntArray*) </td>
        <td>Input</td>
        <td>Transposition sequence of the first matrix for matrix multiplication, which is an aclIntArray on the host.</td>
        <td>[1, 0, 2] is supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>permX2 (aclIntArray*) </td>
        <td>Input</td>
        <td>Transposition sequence of the second matrix for matrix multiplication, which is an aclIntArray on the host.</td>
        <td>[0, 1, 2] is supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>permY (aclIntArray*) </td>
        <td>Input</td>
        <td>Transposition sequence of the output matrix for matrix multiplication, which is an aclIntArray on the host.</td>
        <td>[1, 0, 2] is supported.</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>batchSplitFactor (int32_t) </td>
        <td>Input</td>
        <td>Split size of dimension B in the output matrix of matrix multiplication, which is an integer on the host. Currently, only the value 1 is supported.</td>
        <td>Currently, the value can only be 1.</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>Output matrix of matrix multiplication, which is out in the formula.</td>
        <td>
        <ul>
          <li>Its data type and the data type deduced from x1 and x2 must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</li>
          <li>Currently, only x1Scale and x2Scale are not empty, groupSize is 0, batchSplitFactor is 1, and the output shape is (M, B, N).</li>
        </ul>
        </td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
        <td>3</td>
        <td>-</td>
      </tr>
      </tbody>
      </table>
  
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.
  
  <table style="undefined;table-layout: fixed;width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <td>The input x1, x2, out, x1Scale, x2Scale, permX1, permX2 and permY are null pointers.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The data type of x1, x2, or out is not supported.</td>
    </tr>
    <tr>
      <td>The second dimension of x1 is not equal to the first dimension of x2.</td>
    </tr>
    <tr>
      <td>The dimensions of x1, x2, permX1, permX2 and permY are not equal to 3.</td>
    </tr>
    <tr>
      <td>The dimension size of x1Scale and x2Scale is not 1.</td>
    </tr>
    <tr>
      <td>batchSplitFactor is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data types of x1Scale and x2Scale are not supported.</td>
    </tr>
    <tr>
      <td>The values of permX1, permX2, and permY are not supported.</td>
    </tr>
    <tr>
      <td>The shape of x1Scale and x2Scale does not meet the requirements.</td>
    </tr>
  </tbody>
  </table>

## aclnnTransposeQuantBatchMatMul

- **Parameters**

  <div style="overflow-x: auto;">
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnTransposeQuantBatchMatMulGetWorkspaceSize.</td>
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
  </div>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic description: The aclnnTransposeQuantBatchMatMul is implemented in deterministic mode by default.

- Ascend 950PR/Ascend 950DT:
    - permX1 and permY support [1, 0, 2], and permX2 supports [0, 1, 2].
    - x1Scale and x2Scale are 1-dimensional, and x1Scale is (M,), and x2Scale is (N,).
    - out and dtype support float16 and bfloat16.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include <limits>
#include "acl/acl.h"
#include "aclnnop/aclnn_transpose_quant_batch_mat_mul.h"

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

// Function for converting BF16 to float.
float bf16_to_float(uint16_t bf16)
{
    uint16_t sign = (bf16 >> 15) & 0x1;
    uint16_t exp = (bf16 >> 7) & 0xFF; // 8-bit exponent
    uint16_t mant = bf16 & 0x7F;

    // Special value processing
    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        } else {
            // Non-normalized BF16 -> float
            return (sign ? -1.0f : 1.0f) * (float)mant * (1.0f / (1 << 7)) * (1.0f / (1, 127));
        }
    } else if (exp == 255) {
        // Infinity or NaN
        if (mant == 0) {
            return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        } else {
            return std::numeric_limits<float>::quiet_NaN();
        }
    } else {
        // Normalized number
        float f_exp = (float)(exp - 127); // Offset 127
        float f_mant = (float)mant / (1 << 7); // 7-bit fractional part
        float f = (sign ? -1.0f : 1.0f) * (1.0f + f_mant) * (1 << (int)f_exp);
        return f;
    }
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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int AclnnTransposeQuantBatchMatmulTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    int32_t M = 32;
    int32_t K = 512;
    int32_t N = 128;
    int32_t Batch = 16;
    std::vector<int64_t> x1Shape = {M, Batch, K};
    std::vector<int64_t> x2Shape = {Batch, K, N};
    std::vector<int64_t> x1ScaleShape = {M};
    std::vector<int64_t> x2ScaleShape = {N};
    std::vector<int64_t> outShape = {M, Batch, N};
    std::vector<int64_t> permX1Series = {1, 0, 2};
    std::vector<int64_t> permX2Series = {0, 1, 2};
    std::vector<int64_t> permYSeries = {1, 0, 2};
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* x1ScaleDeviceAddr = nullptr;
    void* x2ScaleDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* x1Scale = nullptr;
    aclTensor* x2Scale = nullptr;
    aclTensor* out = nullptr;
    std::vector<int8_t> x1HostData(GetShapeSize(x1Shape), 0x38);
    std::vector<int8_t> x2HostData(GetShapeSize(x2Shape), 0x38);
    std::vector<float> x1ScaleHostData(GetShapeSize(x1ScaleShape), 1);
    std::vector<float> x2ScaleHostData(GetShapeSize(x2ScaleShape), 1);
    std::vector<uint16_t> outHostData(GetShapeSize(outShape), 0); // bf16

    // Create an x1 aclTensor.
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1deviceAddrPtr(x1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an x2 aclTensor.
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2deviceAddrPtr(x2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an x1Scale aclTensor.
    ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1ScaledeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an x2Scale aclTensor.
    ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2ScaledeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BF16, &out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outdeviceAddrPtr(outDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclIntArray* permX1 = aclCreateIntArray(permX1Series.data(), permX1Series.size());
    aclIntArray* permX2 = aclCreateIntArray(permX2Series.data(), permX2Series.size());
    aclIntArray* permY = aclCreateIntArray(permYSeries.data(), permYSeries.size());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> executorAddrPtr(nullptr, aclrtFree);

    int32_t batchSplitFactor = 1;
    int32_t groupSize = 0;
    int32_t dtype = 27; // bf16

    // Example of calling the aclnnTransposeQuantBatchMatMul API
    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    // Call the first part of the aclnnTransposeQuantBatchMatMul API.
    ret = aclnnTransposeQuantBatchMatMulGetWorkspaceSize(
        x1, x2, (const aclTensor*)nullptr, x1Scale, x2Scale, dtype, groupSize, permX1, permX2, permY, batchSplitFactor,
        out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeQuantBatchMatMulGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        executorAddrPtr.reset(workspaceAddr);
    }
    // Call the second API of aclnnTransposeQuantBatchMatMul.
    ret = aclnnTransposeQuantBatchMatMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeQuantBatchMatMul failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0); // bf16
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    float resultDataBF16 = 0;
    for (int64_t i = 0; i < size; i++) {
        resultDataBF16 = bf16_to_float(resultData[i]);
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataBF16);
    }

    return ACL_SUCCESS;
}

int main()
{
    // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = AclnnTransposeQuantBatchMatmulTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeQuantBatchMatMulTest failed. ERROR: %d\n", ret);
                   return ret);
    Finalize(deviceId, stream);
    return 0;
}
```
