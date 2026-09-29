# aclnnDynamicMxQuant

[📄 View Source Code](https://gitcode.com/cann/ops-nn/tree/master/quant/dynamic_mx_quant)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- API description: performs MX quantization when the destination data type is FLOAT4 or FLOAT8. On the given axis, the quantization scale mxscale corresponding to the group of numbers is calculated based on the number of elements in each block. The calculated mxscale is used as the corresponding part of the output mxscaleOut. Then, each number in the group is divided by mxscale and converted to the corresponding dstType based on round_mode, and the obtained quantization result y is used as the corresponding part of the output yOut. When dstType is FLOAT8_E4M3FN or FLOAT8_E5M2, the algorithm for calculating mxscale is specified based on the value of scaleAlg.

- Formulas:
  - Scenario 1: When scaleAlg is 0:
    - The input x is grouped by k = blocksize in the axis dimension. A group of k numbers $\{\{V_i\}_{i=1}^{k}\}$ is dynamically quantized to $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, where k = blocksize.

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
    mxscale = 2^{shared\_exp}\\
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
    $$

    - ​The quantized $P_{i}$ forms the output yOut according to the positions of the corresponding $V_{i}$, and mxscale forms the output mxscaleOut according to the groups in the corresponding axis dimension.

    - emax: exponent bit of the maximum regular number of the corresponding data type.

        |   DataType    | emax |
        | :-----------: | :--: |
        |  FLOAT4_E2M1  |  2   |
        |  FLOAT4_E1M2  |  0   |
        | FLOAT8_E4M3FN |  8   |
        |  FLOAT8_E5M2  |  15  |

  - Scenario 2: When scaleAlg is 1, only the FP8 type is involved.
    - A long vector is divided into blocks, each with a length of k. A block scaling factor $S_{fp32}^b$ is calculated for each block, and then all elements in the block are mapped to the target low-precision type FP8 using the same $S_{fp32}^b$. If the last block contains less than k elements, the missing values are considered as 0 and processed as a complete block.
    - Find the maximum absolute value of the numbers in the block:
      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$
    - Map the FP32 to the range that can be represented by the target data type FP8. $Amax(DType)$ is the maximum value that can be represented by the target precision.
      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
      $$
    - Convert the block scaling factor $S_{fp32}^b$ to the scaling value $S_{ue8m0}^b$ that can be represented in the FP8 format.
    - Extract the unbiased exponent $E_{int}^b$ and mantissa $M_{fixp}^b$ from the floating-point scaling factor $S_{fp32}^b$ of the block.
    - To ensure that no overflow occurs during quantization, the exponent is rounded up and is within the range that can be represented by FP8.
      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{ if $S_{fp32}^b$ is a regular number, and $E_{int}^b < 254$ and } M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{ if $S_{fp32}^b$ is an unnormalized number, and } M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{ otherwise $\end{cases}$
      $$
    - Calculate the block scaling factor: $S_{ue8m0}^b=2^{E_{int}^b}$
    - Calculate the block conversion factor: $R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
    - The final step of quantization is to quantize each element in the block. The final output is $\left(S^b, [d^i]_{i=1}^k\right)$, where $d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$, $S^b$ is the scaling factor of the block ($S_{ue8m0}^b$), and $[d^i]_{i=1}^k$ is the quantized data in the block.

## Prototype

Each operator has <a href="../../../docs/en/context/two_phase_api.md">two-phase API calls</a>. You must call aclnnDynamicMxQuantGetWorkspaceSize to obtain the workspace size required for computation and the executor that contains the operator execution process, and then call aclnnDynamicMxQuant to perform the computation.

```cpp
aclnnStatus aclnnDynamicMxQuantGetWorkspaceSize(
  const aclTensor *x,
  int64_t          axis,
  char            *roundModeOptional,
  int64_t          dstType,
  int64_t          blocksize,
  int64_t          scaleAlg,
  const aclTensor *yOut,
  const aclTensor *mxscaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnDynamicMxQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDynamicMxQuantGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1547px"><colgroup>
  <col style="width: 200px">
  <col style="width: 120px">
  <col style="width: 250px">
  <col style="width: 330px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 190px">
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
      <td>x (aclTensor*) </td>
      <td>Input</td>
      <td>Input x, corresponding to Vi and di in the formula.</td>
      <td><ul><li>When the destination type is FLOAT4_E2M1 or FLOAT4_E1M2, the last dimension of x must be an even number.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis (int64_t) </td>
      <td>Input</td>
      <td>Axis where quantization occurs, corresponding to axis in the formula.</td>
      <td><ul><li>The value range is [–D, D – 1], where D is the number of dimensions in the shape of x.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundModeOptional (char*) </td>
      <td>Input</td>
      <td>Indicates the data conversion mode, corresponding to round_mode in the formula.</td>
      <td><ul><li>When dstType is set to 40 or 41, {"rint", "floor", "round"} is supported. </li><li>When dstType is set to 36 or 35, only {"rint"} is supported. </li><li>If a null pointer is passed, the rint mode is used.</li></ul></td>
      <td>STRING</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t) </td>
      <td>Input</td>
      <td>Indicates the type of yOut after data conversion, corresponding to DType in the formula.</td>
      <td><ul><li>The input value can be 35, 36, 40, or 41, which corresponds to the output data type of yOut being {35:FLOAT8_E5M2, 36:FLOAT8_E4M3FN, 40:FLOAT4_E2M1, 41:FLOAT4_E1M2}.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blocksize (int64_t) </td>
      <td>Input</td>
      <td>Number of elements to be quantized each time, corresponding to blocksize in the formula.</td>
      <td><ul><li>The value must be a multiple of 32, cannot be 0, and cannot exceed 1024.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg (int64_t) </td>
      <td>Input</td>
      <td>mxscaleOut calculation method, corresponding to scaleAlg in the formula.</td>
      <td><ul><li>The value can be 0 or 1. The value 0 indicates scenario 1, and the value 1 indicates scenario 2. </li><li>When dstType is set to FLOAT4_E2M1/FLOAT4_E1M2, the value can only be 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*) </td>
      <td>Output</td>
      <td>Quantized result of the input x, corresponding to Pi and di in the formula.</td>
      <td><ul><li>The shape is the same as that of the input x.</li></ul></td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mxscaleOut (aclTensor*) </td>
      <td>Output</td>
      <td>Quantization scale corresponding to each group, which corresponds to mxscale and Sb in the formula.</td>
      <td><ul><li>The value of shape on the axis is the value of the corresponding axis divided by blocksize and rounded up, and then even padding is performed. The padding value is 0. </li><li>When axis is not the last axis, the mxscaleOut output needs to interleave every two rows of data.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t) </td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**) </td>
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

- **Return Value**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>Null pointer error occurs in x.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types and formats of x, axis, roundModeOptional, dstType, blocksize, scaleAlg, yOut, and mxscaleOut are not supported.</td>
    </tr>
    <tr>
      <td>The shape of x, yOut, or mxscaleOut does not meet the verification conditions.</td>
    </tr>
    <tr>
      <td>The values of axis, roundModeOptional, dstType, blocksize, and scaleAlg are not supported.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>The current platform is not supported.</td>
    </tr>
  </tbody></table>

## aclnnDynamicMxQuant

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 173px">
  <col style="width: 124px">
  <col style="width: 852px">
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
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnDynamicMxQuantGetWorkspaceSize API.</td>
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

- **Return Value**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - Default deterministic implementation of aclnnDynamicMxQuant.
- The shape constraints on x and mxscaleOut are described as follows:
  - rank(mxscaleOut) = rank(x) + 1.
  - axis_change = axis if axis >= 0 else axis + rank(x).
  - mxscaleOut.shape[axis_change] = (ceil(x.shape[axis] / blocksize) + 2 - 1) / 2.
  - mxscaleOut.shape[-1] = 2.
  - Other dimensions are the same as those of the input x.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_dynamic_mx_quant.h"

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

    int64_t
    GetShapeSize(const std::vector<int64_t>& shape)
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnDynamicMxQuantTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    std::vector<int64_t> xShape = {1, 4};
    std::vector<int64_t> yOutShape = {1, 4};
    std::vector<int64_t> mxscaleOutShape = {1, 1, 2};
    void* xDeviceAddr = nullptr;
    void* yOutDeviceAddr = nullptr;
    void* mxscaleOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* yOut = nullptr;
    aclTensor* mxscaleOut = nullptr;
    // Value of BF16 (0, 8, 64, 512)
    std::vector<uint16_t> xHostData = {0, 16640, 17024, 17408};
    // Value of float8_e4m3 (0, 4, 32, 256)
    std::vector<uint8_t> yOutHostData = {0, 72, 96, 120};
    // Value of float8_e8m0 (2)
    std::vector<uint8_t> mxscaleOutHostData = {{128, 0}};
    int64_t axis = -1;
    char* roundModeOptional = const_cast<char*>("rint");
    int64_t dstType = 36;
    int64_t blocksize = 32;
    int64_t scaleAlg = 0;
    // Create an x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create yOut aclTensor.
    ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &yOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yOutTensorPtr(yOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yOutDeviceAddrPtr(yOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an mxscaleOut ACL tensor.
    ret = CreateAclTensor(mxscaleOutHostData, mxscaleOutShape, &mxscaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscaleOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscaleOutTensorPtr(mxscaleOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> mxscaleOutDeviceAddrPtr(mxscaleOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first API of aclnnDynamicMxQuant.
    ret = aclnnDynamicMxQuantGetWorkspaceSize(x, axis, roundModeOptional, dstType, blocksize, scaleAlg, yOut, mxscaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // Call the second API of aclnnDynamicMxQuant.
    ret = aclnnDynamicMxQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuant failed. ERROR: %d\n", ret); return ret);

    // (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(yOutShape);
    std::vector<uint8_t> yOutData(
        size, 0); // In C language, the fp4 data cannot be directly printed. You need to read the data using uint8 and convert it to fp4 in binary mode.
    ret = aclrtMemcpy(yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr,
                      size * sizeof(yOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("yOut[%ld] is: %d\n", i, yOutData[i]);
    }
    size = GetShapeSize(mxscaleOutShape);
    std::vector<uint8_t> mxscaleOutData(
        size, 0); // In C language, the fp8 data cannot be directly printed. You need to read the data using uint8 and convert it to fp8 in binary mode.
    ret = aclrtMemcpy(mxscaleOutData.data(), mxscaleOutData.size() * sizeof(mxscaleOutData[0]), mxscaleOutDeviceAddr,
                      size * sizeof(mxscaleOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy mxscaleOut from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("mxscaleOut[%ld] is: %d\n", i, mxscaleOutData[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnDynamicMxQuantTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
