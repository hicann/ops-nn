# aclnnQuantBatchMatmulInplaceAdd

[📄 View Source Code](https://gitcode.com/cann/ops-nn/tree/master/matmul/quant_batch_matmul_inplace_add)

## Supported Products

| Product                                                                           | Supported|
| :------------------------------------------------------------------------------ | :------: |
| Ascend 950PR/Ascend 950DT                                         |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

- API usage: In the micro-batch training scenario, gradients need to be accumulated in the micro-batch. As a result, there are a large number of fusion scenarios where QuantBatchMatmul is followed by InplaceAdd. The QuantBatchMatmulInplaceAdd operator is used to fuse the preceding operators to improve the network performance. Performs quantized matrix multiplication and addition. The basic function is the combination of matrix multiplication and addition.

- Formula:

  - **mx quantization:**

  $$
  y[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice * x2Slice)) * (scale1[m, j] * scale2[j, n])) + y[m,n]
  $$

  $gsK$ indicates the quantized block size of the K axis, that is, 32. $x1Slice$ indicates the vector of length $gsK$ in row m of $x1$. $x2Slice$ indicates the vector of length $gsK$ in column n of $x2$. The K axis is sliced from $j*gsK$. The value range of j is [0, kLoops), and kLoops = ceil($K_i$/$gsK$). The length of the last slice can be less than $gsK$.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnQuantBatchMatmulInplaceAdd` is called to perform computation.

```cpp
aclnnStatus aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *x1Scale,
    const aclTensor *x2Scale,
    aclTensor       *yRef,
    bool            transposeX1,
    bool            transposeX2,
    int64_t         groupSize,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnQuantBatchMatmulInplaceAdd(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize

- **Parameters**
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
      <th>Usage</th>
      <th>Data Type</th>
      <th><a href="../../../docs/en/context/data_format.md" target="_blank">Data Format</a></th>
      <th style="white-space: nowrap">Dimension</th>
      <th><a href="../../../docs/en/context/non_contiguous_tensor.md" target="_blank">Non-Contiguous Tensor</a></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>aclTensor on the device, which is the input x1 in the formula.</td>
      <td>-</td>
      <td>FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>aclTensor on the device, which is the input x2 in the formula.</td>
      <td>-</td>
      <td>FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x1Scale</td>
      <td>Optional input</td>
      <td>Indicates the scaling factor introduced by x1 quantization in the quantization parameters, which is an ACLTensor on the device.</td>
      <td>
        <ul>
          <li>For the comprehensive constraints, see <a href="#constraints" target="_blank">Constraints</a>.</li>
        </ul>
      </td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2Scale</td>
      <td>Input</td>
      <td>Indicates the scaling factor introduced by x2 quantization in the quantization parameters, which is an ACLTensor on the device.</td>
      <td>
        <ul>
          <li>For the comprehensive constraints, see <a href="#constraints" target="_blank">Constraints</a>.</li>
        </ul>
      </td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>yRef</td>
      <td>Input and output</td>
      <td>Indicates the ACLTensor on the device, corresponding to the input/output y in the formula.</td>
      <td>
        <ul>
          <li>If x1 is an empty tensor with m = 0 or x2 is an empty tensor with n = 0, the output is an empty tensor.</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>transposeX1</td>
      <td>Input</td>
      <td>Whether the input shape of x1 is transposed.</td>
      <td>-</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX2</td>
      <td>Input</td>
      <td>Whether the input shape of x2 is transposed.</td>
      <td>-</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
    <tr>
      <td>groupSize</td>
      <td>Input</td>
      <td>Integer, specifying the size of the quantization group in the m, n, and k directions.</td>
      <td>
        <ul>
          <li>The value consists of three group sizes: groupSizeM, groupSizeN, and groupSizeK. Each value occupies 16 bits, and the total value occupies the lower 48 bits of the int64_t groupSize. The upper 16 bits of groupSize are invalid. For details about the calculation formula, see the following table.</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - Formula: <a name='f1'></a>

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

- **Returns**

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
      <td>The required input, output, or attribute is passed as a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types and formats of x1, x2, x1Scale, x2Scale, yRef, and groupSize are not supported.</td>
    </tr>
    <tr>
      <td>The shape of x1, x2, x1Scale, x2Scale, and yRef does not meet the verification conditions.</td>
    </tr>
    <tr>
      <td>x1, x2, x2Scale, and yRef are empty tensors.</td>
    </tr>
    <tr>
      <td>If the input groupSize does not meet the verification conditions or the input groupSize is 0, the shape relationship between x1, x2, x1Scale, and x2Scale cannot be used to infer the groupSize.</td>
    </tr>
  </tbody></table>

## aclnnQuantBatchMatmulInplaceAdd

- **Parameters**
  <table>
    <thead>
      <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
    </thead>
    <tbody>
      <tr><td>workspace</td><td>Input</td><td>Address of the workspace to be allocated on the device.</td></tr>
      <tr><td>workspaceSize</td><td>Input</td><td>Size of the workspace allocated on the device, which is obtained by the first API aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize.</td></tr>
      <tr><td>executor</td><td>Input</td><td>Operator executor, containing the operator computation process.</td></tr>
      <tr><td>stream</td><td>Input</td><td>AscendCL stream for executing the task.</td></tr>
    </tbody>
  </table>

- **Returns**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic description: The default deterministic implementation of aclnnQuantBatchMatmulInplaceAdd is used.
- Currently, only transposeX1 is true and transposeX2 is false.
- Restrictions on groupSize:
  - The input groupSize is decomposed into groupSizeM, groupSizeN, and groupSizeK according to the following formulas. If one or more of them are 0, groupSizeM, groupSizeN, and groupSizeK are reset based on the input shape of x1/x2/x1Scale/x2Scale for computation. Principle: If groupSizeM is 0, the quantization group value in the m direction is inferred by the API. The inference formula is groupSizeM = m/scaleM (m must be exactly divided by scaleM). m is the same as that in the shape of x1, and scaleM is the same as that in the shape of x1Scale.
    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$
- Restrictions on dynamic quantization (mx quantization):
  - The following table describes the supported input and output data type combinations.

    | x1 | x2 | x1Scale | x2Scale | outRef |
    |:-------:|:-------:| :------- | :------ | :------ |
    |FLOAT8_E5M2/FLOAT8_E4M3FN |FLOAT8_E5M2/FLOAT8_E4M3FN| FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT32 |

  - The value relationships between x1 data type, x2 data type, x1, x2, x1Scale, x2Scale, and groupSize are as follows:

      | x1 Data Type| x2 Data Type| x1 shape | x2 shape | x1Scale Shape | x2Scale Shape | yRef Shape | [gsM, gsN, gsK] | groupSize |
      |:-------:|:-------:| :------- | :------ | :------ | :------ | :------ | :------ | :------ |
      |FLOAT8_E5M2/FLOAT8_E4M3FN |FLOAT8_E5M2/FLOAT8_E4M3FN| (k, m) | (k, n) | (ceil(k / 64), m, 2) | (ceil(k / 64), n, 2) | (m, n) | [1, 1, 32] | 32 |

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <memory>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_batch_matmul_inplace_add.h"

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

template <typename T1, typename T2>
auto CeilDiv(T1 a, T2 b) -> T1
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int AclnnQuantBatchMatmulInplaceAddTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    int64_t M = 8;
    int64_t K = 16;
    int64_t N = 8;

    std::vector<int64_t> x1Shape = {K, M};
    std::vector<int64_t> x2Shape = {K, N};
    std::vector<int64_t> x2ScaleShape = {CeilDiv(K, 64), N, 2};
    std::vector<int64_t> yInputShape = {M, N};
    std::vector<int64_t> x1ScaleShape = {CeilDiv(K, 64), M, 2};
    std::vector<int64_t> yOutShape = {M, N};

    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* x2ScaleDeviceAddr = nullptr;
    void* yInputDeviceAddr = nullptr;
    void* x1ScaleDeviceAddr = nullptr;
    void* yOutputDeviceAddr = nullptr;

    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* x2Scale = nullptr;
    aclTensor* yInput = nullptr;
    aclTensor* x1Scale = nullptr;
    aclTensor* yOutput = nullptr;

    std::vector<uint8_t> x1HostData(M * K, 1); // 0b00111000 is 1.0 of fp8_e4m3fn.
    std::vector<uint8_t> x2HostData(N * K, 1); // 0b0010 is 1.0 of fp4_e2m1. Here, uint8 is used to represent two fp4s.
    std::vector<uint8_t> x2ScaleHostData(CeilDiv(K, 64) * N * 2, 1);
    1.0 of std::vector<float> yInputHostData(M * N, 1); // fp32
    std::vector<uint8_t> x1ScaleHostData(M * CeilDiv(K, 64) * 2, 1);
    std::vector<float> yOutputHostData(M * N, 1);

    // Create an x1 aclTensor.
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an x2 aclTensor.
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an x2Scale aclTensor.
    ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x2Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create yInput aclTensor.
    ret = CreateAclTensor(yInputHostData, yInputShape, &yInputDeviceAddr, aclDataType::ACL_FLOAT, &yInput);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yInputTensorPtr(yInput, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yInputDeviceAddrPtr(yInputDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an x1Scale aclTensor.
    ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x1Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    bool transposeX1 = true;
    bool transposeX2 = false;
    int64_t groupSize = 32;

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;

    // Call the first part of the aclnnQuantBatchMatmulInplaceAdd API.
    ret = aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize(x1, x2, x1Scale, x2Scale, yInput, transposeX1, transposeX2, groupSize, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnTransQuantParamV2.
    ret = aclnnQuantBatchMatmulInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantBatchMatmulInplaceAdd failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(yInputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), size * sizeof(uint32_t), yInputDeviceAddr,
                    size * sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t j = 0; j < size; j++) {
        LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = AclnnQuantBatchMatmulInplaceAddTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnQuantBatchMatmulInplaceAddTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
  ```
  