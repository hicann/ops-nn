# aclnnForeachMaximumScalar

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T23:49:32.240Z -->

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
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Performs element-wise comparison on a tensor list and a scalar, and computes the maximum value for each element.

- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = \max(x_i, \text{scalar}) (i=0,1,...n-1)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnForeachMaximumScalarGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the computation process. Then, call the "aclnnForeachMaximumScalar" API to execute the computation.

```cpp
aclnnStatus aclnnForeachMaximumScalarGetWorkspaceSize(
  const aclTensorList *x,
  const aclTensor     *scalar,
  aclTensorList       *out,
  uint64_t            *workspaceSize,
  aclOpExecutor      **executor)
```

```cpp
aclnnStatus aclnnForeachMaximumScalar(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnForeachMaximumScalarGetWorkspaceSize

- **Parameters**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
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
      <td>x (const aclTensorList*)</td>
      <td>Input</td>
      <td>Tensor list input for the maximum operation, corresponding to <code>x</code> in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>All tensors in this parameter must have the same data type.</li></ul></td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scalar (const aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor for the maximum operation, corresponding to <code>scalar</code> in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The number of elements is 1.</li><li>The data type has a specific correspondence with the data type of input <code>x</code>:<ul><li>If the data type of <code>x</code> is FLOAT32, FLOAT16, or INT32, the data type is the same as that of <code>x</code>.</li><li>If the data type of <code>x</code> is BFLOAT16, the supported data type is FLOAT32.</li></ul></li></ul></td>
      <td>FLOAT32, FLOAT16, INT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out (aclTensorList*)</td>
      <td>Output</td>
      <td>Output tensor of the maximum operation, corresponding to <code>y</code> in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The data types of all tensors in this parameter are the same.</li><li>The list has the same number of tensors as <code>x</code>. Each output has the same data type, data format and view shape as the corresponding input.</li></ul></td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>×</td>
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
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1170px">
    <colgroup>
      <col style="width: 268px">
      <col style="width: 140px">
      <col style="width: 762px">
    </colgroup>
    <thead>
      <tr>
        <th>Return Code</th>
        <th>Code Value</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>One of <code>x</code>, <code>scalar</code>, or <code>out</code> is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="9">161002</td>
        <td>The data type of <code>x</code>, <code>scalar</code>, or <code>out</code> is not supported.</td>
      </tr>
      <tr>
        <td>The data types of tensors in <code>x</code> or <code>out</code> are inconsistent, or the data type of <code>scalar</code> does not match the required mapping from <code>x</code>.</td>
      </tr>
      <tr>
        <td>The view shape of an output tensor differs from that of the corresponding input tensor.</td>
      </tr>
      <tr>
        <td>A tensor in <code>x</code> has more than 8 dimensions.</td>
      </tr>
      <tr>
        <td><code>x</code> and <code>out</code> contain different numbers of tensors.</td>
      </tr>
      <tr>
        <td>A tensor pointer in <code>x</code> or <code>out</code> is null.</td>
      </tr>
      <tr>
        <td>The number of elements in a tensor in <code>x</code> or <code>out</code> is outside the representable range.</td>
      </tr>
      <tr>
        <td>A tensor in <code>x</code> or <code>out</code> uses a private format.</td>
      </tr>
      <tr>
        <td>The view shape of <code>scalar</code> does not contain exactly one element.</td>
      </tr>
    </tbody>
  </table>

## aclnnForeachMaximumScalar

- **Parameters**

  <table style="table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnForeachMaximumScalarGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Non-contiguous input tensors in `x` and output tensors in `out` are supported.

- The input tensor list `x` contains at most 256 tensors.

- Deterministic computation:
  - aclnnForeachMaximumScalar defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about the compilation and execution process, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_foreach_maximum_scalar.h"

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
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Initialize the ACL resources.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        aclFinalize();
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host data to device memory.
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
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return 1);
    return 0;
}

int main()
{
    // 1. Initialize the device and stream. For details, see the ACL API reference.
    // Set deviceId to the actual device ID.
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs according to the API.
    std::vector<int64_t> selfShape1 = {2, 3};
    std::vector<int64_t> selfShape2 = {1, 3};
    std::vector<int64_t> outShape1 = {2, 3};
    std::vector<int64_t> outShape2 = {1, 3};
    std::vector<int64_t> alphaShape = {1};
    void* input1DeviceAddr = nullptr;
    void* input2DeviceAddr = nullptr;
    void* out1DeviceAddr = nullptr;
    void* out2DeviceAddr = nullptr;
    void* alphaDeviceAddr = nullptr;
    aclTensor* input1 = nullptr;
    aclTensor* input2 = nullptr;
    aclTensor* alpha = nullptr;
    aclTensor* out1 = nullptr;
    aclTensor* out2 = nullptr;
    std::vector<float> input1HostData = {1, 2, 3, 4, 5, 6};
    std::vector<float> input2HostData = {7, 8, 9};
    std::vector<float> out1HostData(6, 0);
    std::vector<float> out2HostData(3, 0);
    std::vector<float> alphaValueHostData = {1.2f};
    aclTensorList* tensorListInput = nullptr;
    aclTensorList* tensorListOutput = nullptr;
    void* workspaceAddr = nullptr;
    aclOpExecutor* executor = nullptr;
    auto run = [&]() -> int {
        // Create the input1 aclTensor.
        ret = CreateAclTensor(input1HostData, selfShape1, &input1DeviceAddr, aclDataType::ACL_FLOAT, &input1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create the input2 aclTensor.
        ret = CreateAclTensor(input2HostData, selfShape2, &input2DeviceAddr, aclDataType::ACL_FLOAT, &input2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create the alpha aclTensor.
        ret = CreateAclTensor(alphaValueHostData, alphaShape, &alphaDeviceAddr, aclDataType::ACL_FLOAT, &alpha);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create the out1 aclTensor.
        ret = CreateAclTensor(out1HostData, outShape1, &out1DeviceAddr, aclDataType::ACL_FLOAT, &out1);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        // Create the out2 aclTensor.
        ret = CreateAclTensor(out2HostData, outShape2, &out2DeviceAddr, aclDataType::ACL_FLOAT, &out2);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        std::vector<aclTensor*> tempInput{input1, input2};
        tensorListInput = aclCreateTensorList(tempInput.data(), tempInput.size());
        CHECK_RET(tensorListInput != nullptr, LOG_PRINT("aclCreateTensorList input failed.\n"); return 1);
        std::vector<aclTensor*> tempOutput{out1, out2};
        tensorListOutput = aclCreateTensorList(tempOutput.data(), tempOutput.size());
        CHECK_RET(tensorListOutput != nullptr, LOG_PRINT("aclCreateTensorList output failed.\n"); return 1);

        // 3. Call the CANN operator library API.
        uint64_t workspaceSize = 0;
        // Call the first-phase API aclnnForeachMaximumScalarGetWorkspaceSize.
        ret = aclnnForeachMaximumScalarGetWorkspaceSize(tensorListInput, alpha, tensorListOutput, &workspaceSize,
                                                        &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachMaximumScalarGetWorkspaceSize failed. ERROR: %d\n", ret);
                  return ret);
        // Allocate device memory using the workspaceSize returned by the first-phase API.
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        }
        // Call the second-phase API aclnnForeachMaximumScalar.
        ret = aclnnForeachMaximumScalar(workspaceAddr, workspaceSize, executor, stream);
        executor = nullptr; // The second-stage call consumes the executor.
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachMaximumScalar failed. ERROR: %d\n", ret); return ret);

        // 4. Wait for task execution to complete.
        ret = aclrtSynchronizeStream(stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

        // 5. Copy the outputs from device memory to host memory.
        auto size = GetShapeSize(outShape1);
        std::vector<float> out1Data(size, 0);
        ret = aclrtMemcpy(out1Data.data(), out1Data.size() * sizeof(out1Data[0]), out1DeviceAddr,
                          size * sizeof(out1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                  return ret);
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("out1 result[%ld] is: %f\n", i, out1Data[i]);
        }

        size = GetShapeSize(outShape2);
        std::vector<float> out2Data(size, 0);
        ret = aclrtMemcpy(out2Data.data(), out2Data.size() * sizeof(out2Data[0]), out2DeviceAddr,
                          size * sizeof(out2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                  return ret);
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("out2 result[%ld] is: %f\n", i, out2Data[i]);
        }

        return ACL_SUCCESS;
    };
    ret = run();
    // Keep the first failure while attempting every cleanup operation.
    auto cleanup = [&](int status, const char* operation) {
        if (status != ACL_SUCCESS) {
            LOG_PRINT("%s failed. ERROR: %d\n", operation, status);
            if (ret == ACL_SUCCESS) {
                ret = status;
            }
        }
    };
    cleanup(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    if (executor != nullptr) {
        cleanup(aclDestroyAclOpExecutor(executor), "aclDestroyAclOpExecutor");
    }
    // A successfully created TensorList owns its tensor descriptors.
    if (tensorListInput != nullptr) {
        cleanup(aclDestroyTensorList(tensorListInput), "aclDestroyTensorList input");
    } else {
        cleanup(aclDestroyTensor(input1), "aclDestroyTensor input1");
        cleanup(aclDestroyTensor(input2), "aclDestroyTensor input2");
    }
    if (tensorListOutput != nullptr) {
        cleanup(aclDestroyTensorList(tensorListOutput), "aclDestroyTensorList output");
    } else {
        cleanup(aclDestroyTensor(out1), "aclDestroyTensor out1");
        cleanup(aclDestroyTensor(out2), "aclDestroyTensor out2");
    }
    cleanup(aclDestroyTensor(alpha), "aclDestroyTensor scalar");
    for (auto address :
         {input1DeviceAddr, input2DeviceAddr, out1DeviceAddr, out2DeviceAddr, alphaDeviceAddr, workspaceAddr}) {
        if (address != nullptr) {
            cleanup(aclrtFree(address), "aclrtFree");
        }
    }
    cleanup(aclrtDestroyStream(stream), "aclrtDestroyStream");
    cleanup(aclrtResetDevice(deviceId), "aclrtResetDevice");
    cleanup(aclFinalize(), "aclFinalize");
    return ret;
}
```
