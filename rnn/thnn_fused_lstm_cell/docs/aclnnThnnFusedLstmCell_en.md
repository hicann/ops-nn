# aclnnThnnFusedLstmCell

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:55:00.486Z -->

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

- Description: Performs the subsequent computation after matrix multiplication in a single-step forward computation of a long short-term memory (LSTM) cell. Outputs the hidden state and cell state at the current time step, as well as the current input gate, forget gate, candidate state, and output gate activations for backward computation.
- Formula:

  Compute the gate activation values:

  $$
  \begin{aligned}
  b &= b_{ih} + b_{hh} \\
  gates &= inputGates + hiddenGates + b \\
  i_{out} &= \sigma(gates_{i}) \\
  f_{out} &= \sigma(gates_{f}) \\
  g_{out} &= \tanh(gates_{g}) \\
  o_{out} &= \sigma(gates_{o})
  \end{aligned}
  $$

  Update the cell state:

  $$
  cy_{out} = f_{out} \odot cx + i_{out} \odot g_{out}
  $$

  Update the hidden state:

  $$
  \begin{aligned}
  tanhc &= \tanh(cy_{out}) \\
  hy_{out} &= o_{out} \odot tanhc
  \end{aligned}
  $$

  Description of related symbols:

  * Bias $b_{ih} = \text{inputBiasOptional}$, $b_{hh} = \text{hiddenBiasOptional}$; zero if no bias is provided.
  * Split $gates$ evenly into 4 components along the last dimension, i.e., $gates \xrightarrow{\text{split}} [gates_i, gates_f, gates_g, gates_o]$
  * Concatenate the four gate activation values along the last dimension into $\text{storageOut}$, i.e., $[i_{out}, f_{out}, g_{out}, o_{out}] \xrightarrow{\text{concat}} \text{storageOut}$.
  * $\sigma$ is the Sigmoid activation function, and $\odot$ is the element-wise product

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnThnnFusedLstmCellGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor containing the operator computation flow, and then call the "aclnnThnnFusedLstmCell" API to perform the computation.

```Cpp
aclnnStatus aclnnThnnFusedLstmCellGetWorkspaceSize(
  const aclTensor    *inputGates,
  const aclTensor    *hiddenGates,
  const aclTensor    *cx,
  const aclTensor    *inputBiasOptional,
  const aclTensor    *hiddenBiasOptional,
  aclTensor          *hyOut,
  aclTensor          *cyOut,
  aclTensor          *storageOut,
  uint64_t           *workspaceSize,
  aclOpExecutor      **executor);
```

```Cpp
aclnnStatus aclnnThnnFusedLstmCell(
  void              *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor     *executor,
  const aclrtStream  stream)
```

## aclnnThnnFusedLstmCellGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1599px"><colgroup>
  <col style="width: 210px">
  <col style="width: 125px">
  <col style="width: 344px">
  <col style="width: 244px">
  <col style="width: 179px">
  <col style="width: 122px">
  <col style="width: 230px">
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
        <td>inputGates (aclTensor*)</td>
        <td>Input</td>
        <td>Values of the 4 gates of the input layer, namely the input gate, forget gate, cell candidate, and output gate.</td>
        <td><ul>
          <li>Do not pass a null pointer. Empty tensors are supported: when inputGates is an empty tensor with numel 0 (for example, batch_size=0 or hidden_size=0), the API returns success directly after input parameter validation, without performing computation, and the output tensor contents remain unchanged.</li>
        </ul></td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(batch_size, 4*hidden_size)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>hiddenGates (aclTensor*)</td>
        <td>Input</td>
        <td>Values of the 4 gates of the hidden layer.</td>
        <td><ul>
          <li>Do not pass a null pointer. Empty tensors are supported.</li>
        </ul></td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(batch_size, 4*hidden_size)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>cx (aclTensor*)</td>
        <td>Input</td>
        <td>Cell state at the previous time step.</td>
        <td><ul>
          <li>Do not pass a null pointer. Empty tensors are supported.</li>
        </ul></td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(batch_size, hidden_size)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>inputBiasOptional (aclTensor*)</td>
        <td>Optional input</td>
        <td>Optional input bias. Passing nullptr indicates no bias.</td>
        <td><ul>
          <li>Optional input. Passing a null pointer indicates no bias, which is equivalent to an all-zero bias.</li>
        </ul></td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(4*hidden_size,)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>hiddenBiasOptional (aclTensor*)</td>
        <td>Optional input</td>
        <td>Optional hidden bias. Passing nullptr indicates no bias.</td>
        <td><ul>
          <li>Optional input. Passing a null pointer indicates no bias, which is equivalent to an all-zero bias.</li>
          <li>hiddenBiasOptional must be valid when the inputBiasOptional input is valid; otherwise, it must be nullptr.</li>
        </ul></td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(4*hidden_size,)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>hyOut (aclTensor*)</td>
        <td>Output</td>
        <td>Hidden state at the current time step, that is, the output of the current time step.</td>
        <td><ul>
          <li>Do not pass a null pointer. Empty tensors are supported (when inputGates is an empty tensor, the API does not perform computation, and the output contents remain unchanged).</li>
        </ul></td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(batch_size, hidden_size)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>cyOut (aclTensor*)</td>
        <td>Output</td>
        <td>Cell state at the current time step.</td>
        <td><ul>
          <li>Do not pass a null pointer. Empty tensors are supported (when inputGates is an empty tensor, the API does not perform computation, and the output contents remain unchanged).</li>
        </ul></td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(batch_size, hidden_size)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>storageOut (aclTensor*)</td>
        <td>Output</td>
        <td>Activation values of the four gates, provided for backward computation.</td>
        <td><ul>
          <li>Do not pass a null pointer. Empty tensors are supported (when inputGates is an empty tensor, the API does not perform computation, and the output contents remain unchanged).</li>
        </ul></td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>(batch_size, 4 * hidden_size)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size to be requested on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody> </table>

  <!-- npu="A3" id7 -->
  - <term>Atlas A3 products</term>: The data type BFLOAT16 is not supported.
  <!-- end id7 -->
  <!-- npu="910b" id8 -->
  - <term>Atlas A2 products</term>: The data type BFLOAT16 is not supported.
  <!-- end id8 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
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
        <td>The input tensor is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="2">161002</td>
        <td>The data type of the parameter is not within the supported range.</td>
      </tr>
      <tr>
        <td>The dimension of the parameter is not within the supported range, or the shape does not satisfy the quantity relationship between parameters.</td>
      </tr>
    </tbody></table>

## aclnnThnnFusedLstmCell

  - **Parameters:**

    <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
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
        <td>Workspace memory address allocated on the device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace allocated on the device, obtained by the first-phase API aclnnThnnFusedLstmCellGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Operator executor that contains the operator computation flow.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream for task execution.</td>
      </tr>
    </tbody>
    </table>

  - **Return value**

      aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation: aclnnThnnFusedLstmCell defaults to a deterministic implementation.
- The data types of all input and output parameters must be consistent.
- Dynamic shapes and dynamic ranks are supported: in graph mode, -1 in a shape denotes an unknown dimension and [-2] denotes an unknown rank. The actual shape and rank are derived at execution time and must still satisfy the shape constraints in the parameter description. Tensors passed in through direct calls of this API must have specific non-negative dimensions; literal negative values in a shape (such as -1 or -2) cause the API to return an error, and no computation is performed.
- The BFLOAT16 data type is supported only on Ascend 950PR&950DT products; Atlas A3 training/inference products and Atlas A2 training/inference products do not support it.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <cmath>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_thnn_fused_lstm_cell.h"

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
  // Call aclrtMemcpy to copy data from the host side to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the AscendCL external API list.
  // Fill in deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs, which must be custom-built based on the API.
  // Define variables.
  int64_t batchSize = 3;
  int64_t hiddenSize = 5;

  // Define shapes.
  std::vector<int64_t> biasShape = {hiddenSize * 4};
  std::vector<int64_t> commonShape = {batchSize, hiddenSize};
  std::vector<int64_t> gatesShape = {batchSize, 4 * hiddenSize};

  // Input device address pointers.
  void* inputGatesDeviceAddr = nullptr;
  void* hiddenGatesDeviceAddr = nullptr;
  void* cxDeviceAddr = nullptr;

  // Device address pointer of the output
  void* hyDeviceAddr = nullptr;
  void* cyDeviceAddr = nullptr;
  void* storageDeviceAddr = nullptr;

  // Pointer to the input ACL Tensor
  aclTensor* inputGates = nullptr;
  aclTensor* hiddenGates = nullptr;
  aclTensor* cx = nullptr;
  aclTensor* inputBias = nullptr;
  aclTensor* hiddenBias = nullptr;

  // Pointer to the output ACL Tensor
  aclTensor* hy = nullptr;
  aclTensor* cy = nullptr;
  aclTensor* storage = nullptr;

  std::vector<float> inputGatesHostData(batchSize * hiddenSize * 4, 1.0f);
  std::vector<float> hiddenGatesHostData(batchSize * hiddenSize * 4, 1.0f);
  std::vector<float> cxHostData(batchSize * hiddenSize, 1.0f);

  std::vector<float> hyHostData(batchSize * hiddenSize, 0.0f);
  std::vector<float> cyHostData(batchSize * hiddenSize, 0.0f);
  std::vector<float> storageHostData(batchSize * hiddenSize * 4, 0.0f);

  // Create the input aclTensor.
  ret = CreateAclTensor(inputGatesHostData, gatesShape, &inputGatesDeviceAddr, aclDataType::ACL_FLOAT, &inputGates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(hiddenGatesHostData, gatesShape, &hiddenGatesDeviceAddr, aclDataType::ACL_FLOAT, &hiddenGates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cxHostData, commonShape, &cxDeviceAddr, aclDataType::ACL_FLOAT, &cx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the output aclTensor.
  ret = CreateAclTensor(hyHostData, commonShape, &hyDeviceAddr, aclDataType::ACL_FLOAT, &hy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cyHostData, commonShape, &cyDeviceAddr, aclDataType::ACL_FLOAT, &cy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(storageHostData, gatesShape, &storageDeviceAddr, aclDataType::ACL_FLOAT, &storage);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the aclnn API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnThnnFusedLstmCell.
  ret = aclnnThnnFusedLstmCellGetWorkspaceSize(inputGates, hiddenGates, cx, inputBias, hiddenBias, hy, cy, storage,
    &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThnnFusedLstmCellGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnThnnFusedLstmCell.
  ret = aclnnThnnFusedLstmCell(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThnnFusedLstmCell failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait for the task execution to finish.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values, copy the results from device memory to host memory, and modify the code based on the specific API definition.
  // Print the hy result.
  auto commonSize = GetShapeSize(commonShape);
  std::vector<float> resultData(commonSize, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), hyDeviceAddr,
                    commonSize * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy hy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < commonSize and i < 10; i++) {
    LOG_PRINT("result hy[%ld] is: %f\n", i, resultData[i]);
  }

  // Release aclTensor.
  aclDestroyTensor(inputGates);
  aclDestroyTensor(hiddenGates);
  aclDestroyTensor(cx);
  aclDestroyTensor(hy);
  aclDestroyTensor(cy);
  aclDestroyTensor(storage);

  // Release Device resources.
  aclrtFree(inputGatesDeviceAddr);
  aclrtFree(hiddenGatesDeviceAddr);
  aclrtFree(cxDeviceAddr);
  aclrtFree(hyDeviceAddr);
  aclrtFree(cyDeviceAddr);
  aclrtFree(storageDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
