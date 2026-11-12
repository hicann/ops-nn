# aclnnThnnFusedLstmCellBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:55:37.651Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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

- Description: Performs backward propagation for the remaining computation after matmul in the four gates of an LSTMCell, computing the gradients of the pre-activation values of the four gates in the forward output, the input cx, and the bias b.
- Formula:

**Variable Definitions**

* **Input gradients**: $\delta h_t$ (`gradHy`), $\delta c_t$ (`gradC`)
* **Forward cache**: $i, f, g, o$ (activation values of each gate `storage`), $c_{t-1}$ (`cx`), $c_t$ (`cy`)
* **Output gradients**: $\delta a_i, \delta a_f, \delta a_g, \delta a_o$ (stored in `gradGatesOut`), $\delta c_{t-1}$ (`gradCxOut`)

**Phase 1: Intermediate gradients and state backpropagation**

First, compute the contribution of the hidden state to the cell state, and aggregate to obtain the total gradient of the cell at the current time step $\text{grad\_}c_{total}$:

$$
\begin{aligned}
gcx &= \tanh(c_t) \\
\text{grad\_}c_{total} &= \delta h_t \cdot o \cdot (1 - gcx^2) + \delta c_t \\
\delta c_{t-1} &= \text{grad\_}c_{total} \cdot f
\end{aligned}
$$

**Second phase: gate component gradients (Pre-activation)**

Based on the code logic, the gradients $\delta a$ of each gate before entering the activation function are calculated as follows:

$$
\begin{aligned}
\delta a_o &= (\delta h_t \cdot gcx) \cdot o \cdot (1 - o) \\
\delta a_i &= (\text{grad\_}c_{total} \cdot g) \cdot i \cdot (1 - i) \\
\delta a_f &= (\text{grad\_}c_{total} \cdot c_{t-1}) \cdot f \cdot (1 - f) \\
\delta a_g &= (\text{grad\_}c_{total} \cdot i) \cdot (1 - g^2)
\end{aligned}
$$

**Stage 3: Parameter gradient (db)**

**1. Bias gradient (db):** Sum over the Batch dimension ($N$):

$$
\delta b = \sum_{n=1}^{N} \begin{bmatrix} \delta a_i \\ \delta a_f \\ \delta a_g \\ \delta a_o \end{bmatrix}_n
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnThnnFusedLstmCellBackwardGetWorkspaceSize" first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call "aclnnThnnFusedLstmCellBackward" to perform the computation.

```Cpp
aclnnStatus aclnnThnnFusedLstmCellBackwardGetWorkspaceSize(
  const aclTensor     *gradHyOptional,
  const aclTensor     *gradCOptional,
  const aclTensor     *cx,
  const aclTensor     *cy,
  const aclTensor     *storage,
  bool                hasBias,
  aclTensor           *gradGatesOut,
  aclTensor           *gradCxOut,
  aclTensor           *gradBiasOut,
  uint64_t            *workspaceSize,
  aclOpExecutor       **executor)
```

```Cpp
aclnnStatus aclnnThnnFusedLstmCellBackward(
  void            *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream     stream)
```

## aclnnFusedLstmCellBackwardGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1478px"><colgroup>
  <col style="width: 149px">
  <col style="width: 121px">
  <col style="width: 264px">
  <col style="width: 253px">
  <col style="width: 262px">
  <col style="width: 148px">
  <col style="width: 135px">
  <col style="width: 146px">
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
      <td>gradHyOptional</td>
      <td>Optional input</td>
      <td>Indicates the gradient of the hidden state in the forward output of LSTMCell.</td>
      <td>Optional input. Passing nullptr indicates that the gradient is absent, which is equivalent to an all-zero gradient of the same shape.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[batch, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradCOptional</td>
      <td>Optional input</td>
      <td>Indicates the gradient of the cell state in the forward output of LSTMCell.</td>
      <td>Optional input. Passing nullptr indicates that the gradient is absent, which is equivalent to an all-zero gradient of the same shape. When it is not null, the data type must be consistent with that of the other inputs.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[batch, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cx</td>
      <td>Input</td>
      <td>Indicates the forward input cell state of LSTMCell.</td>
      <td>The data type must be consistent with that of input.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[batch, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cy</td>
      <td>Input</td>
      <td>Indicates the cell state of the LSTMCell forward output.</td>
      <td>The data type is consistent with input.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[batch, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>storage</td>
      <td>Input</td>
      <td>Indicates the activation values of the four gates in the LSTMCell forward output.</td>
      <td>The data type must be consistent with that of input.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[batch, 4 * hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hasBias</td>
      <td>Input</td>
      <td>Whether the bias gradient needs to be computed.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradGatesOut</td>
      <td>Output</td>
      <td>Indicates the gradients of the four gate pre-activation values in the LSTMCell forward pass.</td>
      <td>The data type is consistent with input.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[batch, 4 * hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradCxOut</td>
      <td>Output</td>
      <td>Indicates the gradient of the input cell state in the forward propagation of LSTMCell.</td>
      <td>The data type is consistent with input.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[batch, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBiasOut</td>
      <td>Output</td>
      <td>Indicates the gradient of the input bias in LSTM forward propagation.</td>
      <td>Data type is consistent with input.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[4 * hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>Output parameter on the Host side.</td>
      <td>UINT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
      <td>Output parameter on the Host side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>
- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>If the input parameter is an aclTensor that is a non-optional input and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>If the input parameter is an aclTensor and the data type is out of the supported range.</td>
    </tr>
    <tr>
      <td>If the input parameter is an aclTensor and the data types are inconsistent.</td>
    </tr>
    <tr>
      <td>If the input parameter is an aclTensor and the shape does not meet the corresponding shape requirement.</td>
    </tr>
  </tbody>
  </table>

## aclnnThnnFusedLstmCellBackward

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnThnnFusedLstmCellBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>
- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnThnnFusedLstmCellBackward defaults to a deterministic implementation.
- Data types: all inputs and outputs that are actually passed must have the same data type. Optional inputs passed as nullptr are not involved in the data type consistency check.
- Optional inputs: gradHyOptional and gradCOptional can be nullptr, indicating that the corresponding gradient is absent and an all-zero gradient of the same shape is used instead. If both are nullptr, the output gradients are directly set to zero without running the backward computation (Inf/NaN propagation is not performed in this case).
- Boundary value scenario description:
  - When the input is Inf, the output is NAN.
  - When the input is NaN, the output is NaN.

## Example

The sample code is as follows and is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <cmath>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_thnn_fused_lstm_cell_backward.h"

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
  // Boilerplate: AscendCL initialization.
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
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to the device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (boilerplate) Device/stream initialization. For details, see the AscendCL external API list.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Construct them as required by the API.
  // Define variables.
  int64_t n = 1;
  int64_t hiddenSize = 8;

  // Shape definition.
  std::vector<int64_t> bShape = {hiddenSize * 4};
  std::vector<int64_t> dhShape = {n, hiddenSize};
  std::vector<int64_t> gatesShape = {n, 4 * hiddenSize};

  // Device address pointer.
  void* dhyDeviceAddr = nullptr;
  void* dcDeviceAddr = nullptr;
  void* cxDeviceAddr = nullptr;
  void* cyDeviceAddr = nullptr;
  void* storageDeviceAddr = nullptr;

  // Backward propagation output device address pointer
  void* dgatesDeviceAddr = nullptr;
  void* dcPrevDeviceAddr = nullptr;
  void* dbDeviceAddr = nullptr;

  // ACL Tensor pointer
  aclTensor* dhy = nullptr;
  aclTensor* dc = nullptr;
  aclTensor* cx = nullptr;
  aclTensor* cy = nullptr;
  aclTensor* storage = nullptr;

  // Backward propagation output ACL Tensor pointer
  aclTensor* dgates = nullptr;
  aclTensor* dcPrev = nullptr;
  aclTensor* db = nullptr;

  std::vector<float> dhyHostData(n * hiddenSize, 1.0f); // 1 * 1 * 8 = 8 ones
  std::vector<float> dcHostData(n * hiddenSize, 1.0f); // (8+8)*32 = 16 * 32 = 512 ones
  std::vector<float> cxHostData(n * hiddenSize, 1.0f); // (8+8)*32 = 16 * 32 = 512 ones
  std::vector<float> cyHostData(n * hiddenSize, 1.0f); // 32 ones
  std::vector<float> storageHostData(n * hiddenSize * 4, 1.0f); // 32 ones

  // Backward propagation output host data (initialized to 0)
  std::vector<float> dgatesHostData(n * hiddenSize * 4, 0.0f);
  std::vector<float> dcPrevHostData(n * hiddenSize, 0.0f);
  std::vector<float> dbHostData(hiddenSize * 4, 0.0f);

  // Create dhy aclTensor.
  ret = CreateAclTensor(dhyHostData, dhShape, &dhyDeviceAddr, aclDataType::ACL_FLOAT, &dhy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the params aclTensorList.
  ret = CreateAclTensor(dcHostData, dhShape, &dcDeviceAddr, aclDataType::ACL_FLOAT, &dc);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cxHostData, dhShape, &cxDeviceAddr, aclDataType::ACL_FLOAT, &cx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cyHostData, dhShape, &cyDeviceAddr, aclDataType::ACL_FLOAT, &cy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(storageHostData, gatesShape, &storageDeviceAddr, aclDataType::ACL_FLOAT, &storage);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the backward propagation output tensor.
  // Create the dgates aclTensor.
  ret = CreateAclTensor(dgatesHostData, gatesShape, &dgatesDeviceAddr, aclDataType::ACL_FLOAT, &dgates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the dcPrev aclTensor.
  ret = CreateAclTensor(dcPrevHostData, dhShape, &dcPrevDeviceAddr, aclDataType::ACL_FLOAT, &dcPrev);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the db aclTensor.
  ret = CreateAclTensor(dbHostData, bShape, &dbDeviceAddr, aclDataType::ACL_FLOAT, &db);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnThnnFusedLstmCellBackward.
  ret = aclnnThnnFusedLstmCellBackwardGetWorkspaceSize(dhy, dc, cx, cy, storage, true, dgates, dcPrev, db,
    &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThnnFusedLstmCellBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Request device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnThnnFusedLstmCellBackward.
  ret = aclnnThnnFusedLstmCellBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThnnFusedLstmCellBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait synchronously for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values, copy the results from the device-side memory to the host side, and modify the code based on the specific API definition.
  // Print the dparams result.
  auto dgatesSize = GetShapeSize(gatesShape);
  std::vector<float> resultDgatesData(dgatesSize, 0);
  ret = aclrtMemcpy(resultDgatesData.data(), resultDgatesData.size() * sizeof(resultDgatesData[0]), dgatesDeviceAddr,
                    dgatesSize * sizeof(resultDgatesData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dgates result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dgatesSize; i++) {
    LOG_PRINT("result dgates[%ld] is: %f\n", i, resultDgatesData[i]);
  }

  auto dbSize = GetShapeSize(bShape);
  std::vector<float> resultDwhData(dbSize, 0);
  ret = aclrtMemcpy(resultDwhData.data(), resultDwhData.size() * sizeof(resultDwhData[0]), dbDeviceAddr,
                    dbSize * sizeof(resultDwhData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy db result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dbSize; i++) {
    LOG_PRINT("result db[%ld] is: %f\n", i, resultDwhData[i]);
  }

  auto dcPrevSize = GetShapeSize(dhShape);
  std::vector<float> resultDcPrevData(dcPrevSize, 0);
  ret = aclrtMemcpy(resultDcPrevData.data(), resultDcPrevData.size() * sizeof(resultDcPrevData[0]), dcPrevDeviceAddr,
                    dcPrevSize * sizeof(resultDcPrevData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dcPrev result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dcPrevSize; i++) {
    LOG_PRINT("result dcPrev[%ld] is: %f\n", i, resultDcPrevData[i]);
  }
  // Release the aclTensor.
  aclDestroyTensor(dhy);
  aclDestroyTensor(dc);
  aclDestroyTensor(cx);
  aclDestroyTensor(cy);
  aclDestroyTensor(storage);
  aclDestroyTensor(dgates);
  aclDestroyTensor(dcPrev);
  aclDestroyTensor(db);

  // Release Device resources.
  aclrtFree(dhyDeviceAddr);
  aclrtFree(dcDeviceAddr);
  aclrtFree(cxDeviceAddr);
  aclrtFree(cyDeviceAddr);
  aclrtFree(storageDeviceAddr);
  aclrtFree(dgatesDeviceAddr);
  aclrtFree(dcPrevDeviceAddr);
  aclrtFree(dbDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
