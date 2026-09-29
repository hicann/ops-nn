# aclnnThnnFusedLstmCell

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     ×    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- This API is used to perform subsequent calculations after matrix multiplication in the single-step forward calculation of the long short-term memory cell (LSTM cell). It outputs the hidden state and cell state at the current time step, and also outputs the current forget gate, input gate, output gate, and candidate state for backward calculation.
- Formula:

  Calculate the gating activation value:
  
  $$
  \begin{aligned}
  b &= b_{ih} + b_{hh} \\
  gates &= inputGates + hiddenGates + b \\
  i_{out} &= \sigma(gates_{i}) \\
  g_{out} &= \tanh(gates_{g}) \\
  f_{out} &= \sigma(gates_{f}) \\
  o_{out} &= \sigma(gates_{o})
  \end{aligned}
  $$
  
  Update the cell state.
  
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
  
  Relevant symbol description:
  
  * The bias $b_{ih} = \text{inputBiasOptional}$, $b_{hh} = \text{hiddenBiasOptional}$. If the bias is not specified, the value is 0.
  * Split $gates$ into four components along the last dimension, that is, $gates \xrightarrow{\text{split}} [gates_i, gates_g, gates_f, gates_o]$.
  * Concatenate the obtained four gating activation values along the last dimension to form $\text{storageOut}$, that is, $[i_{out}, g_{out}, f_{out}, o_{out}] \xrightarrow{\text{concat}} \text{storageOut}$.
  * $\sigma$ is the sigmoid activation function, and $\odot$ is the element-wise product.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnThnnFusedLstmCellGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnThnnFusedLstmCell` is called to perform computation.

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

- **Parameters**

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
        <td>inputGates (aclTensor*) </td>
        <td>Input</td>
        <td>Four gates of the input layer, that is, the values of the input gate, cell candidate, forget gate, and output gate.</td>
        <td>None.</td>
        <td>FLOAT, FLOAT16</td>
        <td>ND</td>
        <td>(batch_size, 4*hidden_size)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>hiddenGates (aclTensor*) </td>
        <td>Input</td>
        <td>Values of the four gates in the hidden layer.</td>
        <td>None.</td>
        <td>FLOAT, FLOAT16</td>
        <td>ND</td>
        <td>(batch_size, 4*hidden_size)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>cx (aclTensor*) </td>
        <td>Input</td>
        <td>Cell state of the previous time step.</td>
        <td>None.</td>
        <td>FLOAT, FLOAT16</td>
        <td>ND</td>
        <td>(batch_size, hidden_size)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>inputBiasOptional (aclTensor*) </td>
        <td>Optional input</td>
        <td>Optional input bias. If nullptr is passed, it indicates that there is no bias.</td>
        <td>None.</td>
        <td>FLOAT, FLOAT16</td>
        <td>ND</td>
        <td>(4*hidden_size,)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>hiddenBiasOptional (aclTensor*) </td>
        <td>Optional input</td>
        <td>Optional hidden layer bias. If nullptr is passed, it indicates that there is no bias.</td>
        <td>When inputBiasOptional is valid, hiddenBiasOptional must be valid. Otherwise, hiddenBiasOptional must be nullptr.</td>
        <td>FLOAT, FLOAT16</td>
        <td>ND</td>
        <td>(4*hidden_size,)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>hyOut (aclTensor*) </td>
        <td>Output</td>
        <td>Hidden state at the current time, that is, the output at the current time.</td>
        <td>None.</td>
        <td>FLOAT, FLOAT16</td>
        <td>ND</td>
        <td>(batch_size, hidden_size)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>cyOut (aclTensor*) </td>
        <td>Output</td>
        <td>Cell state at the current time.</td>
        <td>None.</td>
        <td>FLOAT, FLOAT16</td>
        <td>ND</td>
        <td>(batch_size, hidden_size)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>storageOut (aclTensor*) </td>
        <td>Output</td>
        <td>Activation values of the four gates, which are provided for backward propagation.</td>
        <td>None.</td>
        <td>FLOAT, FLOAT16</td>
        <td>ND</td>
        <td>(batch_size, 4 * hidden_size)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Size of the workspace to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Operator executor, containing the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

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
        <td>The data type or format of the parameter is not supported.</td>
      </tr>
      <tr>
        <td>The dimension of the parameter is not supported, or the shape does not meet the quantitative relationship between parameters.</td>
      </tr>
    </tbody></table>

## aclnnThnnFusedLstmCell

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
        <td>Memory address of the workspace to be allocated on the device.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnThnnFusedLstmCellGetWorkspaceSize.</td>
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

  - **Returns**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Restrictions

- Deterministic description: The aclnnThnnFusedLstmCell is implemented in deterministic mode by default.
- The data types of all input and output parameters must be the same.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

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
  // (Fixed writing) Initialize AscendCL.
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

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  // Define variables.
  int64_t batchSize = 3;
  int64_t hiddenSize = 5;

  // Shape definition
  std::vector<int64_t> biasShape = {hiddenSize * 4};
  std::vector<int64_t> commonShape = {batchSize, hiddenSize};
  std::vector<int64_t> gatesShape = {batchSize, 4 * hiddenSize};;

  // Pointer to the input device address
  void* inputGatesDeviceAddr = nullptr;
  void* hiddenGatesDeviceAddr = nullptr;
  void* cxDeviceAddr = nullptr;

  // Pointer to the output device address
  void* hyDeviceAddr = nullptr;
  void* cyDeviceAddr = nullptr;
  void* storageDeviceAddr = nullptr;

  // Pointer to the input ACL tensor
  aclTensor* inputGates = nullptr;
  aclTensor* hiddenGates = nullptr;
  aclTensor* cx = nullptr;
  aclTensor* inputBias = nullptr;
  aclTensor* hiddenBias = nullptr;

  // Pointer to the output ACL tensor
  aclTensor* hy = nullptr;
  aclTensor* cy = nullptr;
  aclTensor* storage = nullptr;

  std::vector<float> inputGatesHostData(batchSize * hiddenSize * 4, 1.0f);
  std::vector<float> hiddenGatesHostData(batchSize * hiddenSize * 4, 1.0f);
  std::vector<float> cxHostData(batchSize * hiddenSize, 1.0f);

  std::vector<float> hyHostData(batchSize * hiddenSize, 0.0f);
  std::vector<float> cyHostData(batchSize * hiddenSize, 0.0f);
  std::vector<float> storageHostData(batchSize * hiddenSize * 4, 0.0f);

  // Create an input aclTensor.
  ret = CreateAclTensor(inputGatesHostData, gatesShape, &inputGatesDeviceAddr, aclDataType::ACL_FLOAT, &inputGates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(hiddenGatesHostData, gatesShape, &hiddenGatesDeviceAddr, aclDataType::ACL_FLOAT, &hiddenGates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cxHostData, commonShape, &cxDeviceAddr, aclDataType::ACL_FLOAT, &cx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an output aclTensor.
  ret = CreateAclTensor(hyHostData, commonShape, &hyDeviceAddr, aclDataType::ACL_FLOAT, &hy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cyHostData, commonShape, &cyDeviceAddr, aclDataType::ACL_FLOAT, &cy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(storageHostData, gatesShape, &storageDeviceAddr, aclDataType::ACL_FLOAT, &storage);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the aclnn API. You need to change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first segment of the aclnnThnnFusedLstmCell API.
  ret = aclnnThnnFusedLstmCellGetWorkspaceSize(inputGates, hiddenGates, cx, inputBias, hiddenBias, hy, cy, storage,
    &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThnnFusedLstmCellGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second segment of the aclnnThnnFusedLstmCell API.
  ret = aclnnThnnFusedLstmCell(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThnnFusedLstmCell failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  // Print the hy result.
  auto commonSize = GetShapeSize(commonShape);
  std::vector<float> resultData(commonSize, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), hyDeviceAddr,
                    commonSize * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy hy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < commonSize and i < 10; i++) {
    LOG_PRINT("result hy[%ld] is: %f\n", i, resultData[i]);
  }

  // Destroy aclTensor.
  aclDestroyTensor(inputGates);
  aclDestroyTensor(hiddenGates);
  aclDestroyTensor(cx);
  aclDestroyTensor(hy);
  aclDestroyTensor(cy);
  aclDestroyTensor(storage);

  // Release device resources.
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
