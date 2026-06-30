# aclnnTransposeBatchMatMul

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/matmul/transpose_batch_mat_mul)

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √    |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference series products</term>                            |    ×    |
| <term>Atlas training series products</term>                             |    ×    |


## Function

- Description: Performs matrix multiplication on tensors **x1** and **x2**. Only three-dimensional tensors can be passed. Tensors can be transposed. The transposition sequence is changed based on the input sequence. **permX1** indicates the transposition sequence of tensor **x1**, and **permX2** indicates the transposition sequence of tensor **x2**. The sequence value **0** indicates the batch dimension, and the other two dimensions are used for matrix multiplication.

- Example:
  - If the shape of **x1** is (B, M, K), the shape of **x2** is (B, K, N), **scale** is None, and **batchSplitFactor** is 1, the shape of the output **out** is (M, B, N).
  - If the shape of **x1** is (B, M, K), the shape of **x2** is (B, K, N), **scale** is not None, and **batchSplitFactor** is 1, the shape of the output **out** is (M, 1, B * N).
  - If the shape of **x1** is (B, M, K), the shape of **x2** is (B, K, N), **scale** is None, and **batchSplitFactor** is greater than 1, the shape of the output **out** is (batchSplitFactor, M, B * N / batchSplitFactor).

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnTransposeBatchMatMulGetWorkspaceSize** is called to obtain the input parameters and compute the required workspace size based on the process. Then, **aclnnTransposeBatchMatMul** is called to perform computation.
```cpp
aclnnStatus aclnnTransposeBatchMatMulGetWorkspaceSize(
    const aclTensor    *x1,
    const aclTensor    *x2,
    const aclTensor    *bias,
    const aclTensor    *scale,
    const aclIntArray  *permX1,
    const aclIntArray  *permX2,
    const aclIntArray  *permY,
    int8_t             cubeMathType,
    const int32_t      batchSplitFactor,
    aclTensor          *out,
    uint64_t           *workspaceSize,
    aclOpExecutor      **executor)
```
```cpp
aclnnStatus aclnnTransposeBatchMatMul(
    void               *workspace, 
    uint64_t           workspaceSize,
    aclOpExecutor      *executor,
    const aclrtStream  stream)
```
## aclnnTransposeBatchMatMulGetWorkSpaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed;width: 1567px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
    <col style="width: 145px">
    </colgroup>
    <thead>
      <tr>
        <th>Name</th>
        <th style="white-space: nowrap">Input/Output</th>
        <th>Description</th>
        <th>Usage Notes</th>
        <th>Data Type</th>
        <th><a href="../../../docs/en/context/data_formats.md" target="_blank"> Data Format</a></th>
        <th style="white-space: nowrap">Dimension</th>
        <th><a href="../../../docs/en/context/non_contiguous_tensors.md" target="_blank">Non-Contiguous Tensor</a></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x1</td>
        <td>Input</td>
        <td>First matrix for matrix multiplication, which is an aclTensor on the device.</td>
        <td>
          <ul>
            <li>Its data type and the data type of x2 must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>.</li>
            <li>The data type can be BFLOAT16, FLOAT16, or FLOAT32.</li>
            <li>If the data types of the input x1 and x2 are BFLOAT16 and FLOAT16, respectively, the data type cannot be deduced.</li>
            <li>If the data types of the input x1 and x2 are BFLOAT16 and FLOAT32, respectively, the data type cannot be deduced.</li>
          </ul>
        </td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2</td>
        <td>Input</td>
        <td>Second matrix for matrix multiplication, which is an aclTensor on the device.</td>
        <td>
        <ul>
            <li>Its data type and the data type of x1 must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>.</li>
            <li>The size of the Reduce dimension of x2 must be the same as the size of the Reduce dimension of x1.</li>
            <li>The data type can be BFLOAT16, FLOAT16, or FLOAT32.</li>
            <li>If the data types of the input x1 and x2 are BFLOAT16 and FLOAT16, respectively, the data type cannot be deduced.</li>
            <li>If the data types of the input x1 and x2 are BFLOAT16 and FLOAT32, respectively, the data type cannot be deduced.</li>
        </ul>
        </td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>Input</td>
        <td>Bias matrix for matrix multiplication, which is an aclTensor on the device.</td>
        <td>
        <ul>
            <li>This parameter is reserved and not supported currently.</li>
        </ul>
        </td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
      <td>scale</td>
        <td>Input (optional)</td>
        <td>Quantization coefficient of the output matrix. It can be enabled when the input is FLOAT16 and the output is INT8. It is an aclTensor on the device.</td>
        <td>
        <ul>
            <li>The shape must be one-dimensional and must be equal to [b*n].</li>
        </ul>
        </td>
        <td>INT64, UINT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>permX1</td>
        <td>Input</td>
        <td>Transposition sequence of the first matrix for matrix multiplication, which is an aclIntArray on the host.</td>
        <td>
        <ul>
          <li>[0, 1, 2] and [1, 0, 2] are supported.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>permX2</td>
        <td>Input</td>
        <td>Transposition sequence of the second matrix for matrix multiplication, which is an aclIntArray on the host.</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>permY</td>
        <td>Input</td>
        <td>Transposition sequence of the output matrix for matrix multiplication, which is an aclIntArray on the host.</td>
        <td>
        <ul>
            <li>[1, 0, 2] is supported.</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>cubeMathType</td>
        <td>Input</td>
        <td>Compute logic of the Cube unit, which is an integer on the host.</td>
        <td>If the input data types can be deduced from each other, this parameter processes the deduced data type by default. The enumerated values are as follows:<ul>
            <li>0: KEEP_DTYPE. The input data type is retained for computation.</li>
            <li>1: ALLOW_FP32_DOWN_PRECISION. The input data can be computed with reduced precision. If the input data type is FLOAT32, it is converted to HFLOAT32 for computation. If the input data type is not FLOAT32, the input type is retained for computation.</li>
            <li>2: USE_FP16. The input data can be downgraded to FLOAT16 for computation. If the input data type is BFLOAT16, this option is not supported.</li>
            <li>3: USE_HF32. The input data can be downgraded to HFLOAT32 for computation. If the input data type is FLOAT32, it is converted to HFLOAT32 for computation.</li></ul>
        </td>
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>batchSplitFactor</td>
        <td>Input</td>
        <td>Split size of dimension B in the output matrix for matrix multiplication, which is an integer on the host.</td>
        <td>
        <ul>
          <li>The value range is [1, B] and the value must be exactly divisible by B.</li>
          <li>When scale is not null, batchSplitFactor must be 1.</li>
        </ul>
        </td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Output matrix for matrix multiplication, which is out in the formula and is an aclTensor on the device.</td>
        <td>
        <ul>
          <li>Its data type and the data type deduced from x1 and x2 must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</li>
          <li>If scale has a value, the output shape is (M, 1, B * N).</li>
        </ul>
        <ul>
          If batchSplitFactor is greater than 1, the output shape of out is (batchSplitFactor, M, B * N / batchSplitFactor).
          <li>Example 1: If M, K, N, B = 32, 512, 128, 16 and batchSplitFactor = 2, the output shape of out is (2, 32, 1024).</li>
          <li>Example 2: If M, K, N, B = 32, 512, 128, 16 and batchSplitFactor = 4, the output shape of out is (4, 32, 512).</li>
        </ul>
        </td>
        <td>BFLOAT16, FLOAT16, FLOAT32, INT8</td>
        <td>ND</td>
        <td>3</td>
        <td>-</td>
      </tr>
      </tbody>
      </table>
  

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:

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
      <td>The passed x1, x2, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type of x1, x2, or out is not supported.</td>
    </tr>
    <tr>
      <td>The second dimension of x1 is not equal to the first dimension of x2.</td>
    </tr>
    <tr>
      <td>The dimension size of x1 or x2 is not 3.</td>
    </tr>
    <tr>
      <td>The second dimension or the third dimension of x2 cannot be exactly divided by 128.</td>
    </tr>
    <tr>
      <td>The data type of scale is not supported.</td>
    </tr>
    <tr>
      <td>The value of batchSplitFactor is not within the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnTransposeBatchMatMul

- **Parameters:**

  <div style="overflow-x: auto;">
  <table style="undefined;table-layout: fixed; width: 1030px" ><colgroup>
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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnTransposeBatchMatMulGetWorkSpaceSize.</td>
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

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic description:
  - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: **aclnnTransposeBatchMatMul** defaults to a deterministic implementation.

- <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>:
    - The value range of **B** is [1, 65536), and the value range of **N** is [1, 65536).
    - When the input shape of **x1** is (B, M, K), K ≤ 65535. When the input shape of **x1** is (M, B, K), B × K ≤ 65535.
    - **permX2** supports only the input [0, 1, 2].
    - When scale is not null, the product of **B** and **N** must be less than 65,536, and type deduction cannot be performed only when the input is FLOAT16 and the output is INT8.
    
## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_transpose_batch_mat_mul.h"

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

  // Call aclrtMemcpy to copy the data from the host to the device.
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
  // Handle the check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  int32_t M = 32;
  int32_t K = 512;
  int32_t N = 128;
  int32_t Batch = 16;
  std::vector<int64_t> x1Shape = {M, Batch, K};
  std::vector<int64_t> x2Shape = {Batch, K, N};
  std::vector<int64_t> outShape = {M, Batch, N};
  std::vector<int64_t> permX1Series = {1, 0, 2};
  std::vector<int64_t> permX2Series = {0, 1, 2};
  std::vector<int64_t> permYSeries = {1, 0, 2};
  void* x1DeviceAddr = nullptr;
  void* x2DeviceAddr = nullptr;
  void* scaleDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* x1 = nullptr;
  aclTensor* x2 = nullptr;
  aclTensor* scale = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> x1HostData(GetShapeSize(x1Shape));
  std::vector<float> x2HostData(GetShapeSize(x2Shape));
  std::vector<float> outHostData(GetShapeSize(outShape));
  int8_t cubeMathType = 1;
  int8_t batchSplitFactor = 1;

  // Create an x1 aclTensor.
  ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an x2 aclTensor.
  ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  aclIntArray *permX1 = aclCreateIntArray(permX1Series.data(), permX1Series.size());
  aclIntArray *permX2 = aclCreateIntArray(permX2Series.data(), permX2Series.size());
  aclIntArray *permY = aclCreateIntArray(permYSeries.data(), permYSeries.size());
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  // aclnnTransposeBatchMatMul API call example
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  // Call the first-phase API of aclnnTransposeBatchMatMul.
  ret = aclnnTransposeBatchMatMulGetWorkspaceSize(x1, x2, (const aclTensor*)nullptr, (const aclTensor*)nullptr,
                                                  permX1, permX2, permY, cubeMathType, batchSplitFactor, out,
                                                  &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeBatchMatMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnTransposeBatchMatMul.
  ret = aclnnTransposeBatchMatMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeBatchMatMul failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(x1);
  aclDestroyTensor(x2);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the configuration based on the API definition.
  aclrtFree(x1DeviceAddr);
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
