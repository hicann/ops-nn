# aclnnBidirectionLSTM

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     ×    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     √    |
|  <term>Atlas training series products</term>   |     ×    |

## Function

- Description: Implements the long short-term memory (LSTM) network, which is a special recurrent neural network (RNN) model. Computes the LSTM network, receives the input sequence and initial state, and returns the output sequence and final state.
- Formula:
  
  $$
  f_t =sigm(W_f[h_{t-1}, x_t] + b_f)\\
  i_t =sigm(W_i[h_{t-1}, x_t] + b_i)\\
  o_t =sigm(W_o[h_{t-1}, x_t] + b_o)\\
  \tilde{c}_t =tanh(W_c[h_{t-1}, x_t] + b_c)\\
  c_t =f_t ⊙ c_{t-1} + i_t ⊙ \tilde{c}_t\\
  c_{o}^{t} =tanh(c_t)\\
  h_t =o_t ⊙ c_{o}^{t}\\
  $$

  - $x_t ∈ R^{d}$: input vector to the LSTM unit.
  - $f_t ∈ (0, 1)^{h}$: activation vector of the forget gate.
  - $i_t ∈ (0, 1)^{h}$: activation vector of the input/update gate.
  - $o_t ∈ (0, 1)^{h}$: activation vector of the output gate.
  - $h_i ∈ (-1, 1)^{h}$: hidden state vector, also known as output vector of the LSTM unit.
  - $\tilde{c}_t ∈ (-1, 1)^{h}$: cell input activation vector.
  - $c_t ∈ R^{h}$: cell state vector.
  - $W ∈ R^{h×d}, (U ∈ R^{h×h})∩(b ∈ R^{h})$: weight matrices and bias vector parameters which need to be learned during training.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnBidirectionLSTMGetWorkspaceSize** is called to obtain the input parameters and compute the required workspace size based on the process. Then, **aclnnBidirectionLSTM** is called to perform computation.

```Cpp
aclnnStatus aclnnBidirectionLSTMGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *initH,
  const aclTensor *initC,
  const aclTensor *wIh,
  const aclTensor *wHh,
  const aclTensor *bIhOptional,
  const aclTensor *bHhOptional,
  const aclTensor *wIhReverseOptional,
  const aclTensor *wHhReverseOptional,
  const aclTensor *bIhReverseOptional,
  const aclTensor *bHhReverseOptional,
  int64_t          numLayers,
  bool             isbias,
  bool             batchFirst,
  bool             bidirection,
  const aclTensor *yOut,
  const aclTensor *outputHOut,
  const aclTensor *outputCOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnBidirectionLSTM(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```


## aclnnBidirectionLSTMGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 201px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 240px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Precaution</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
   <tbody>
      <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input vector of the LSTM unit, corresponding to x in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape supports 3D (time_step, batch_size, input_size). `time_step` indicates the time dimension. `batch_size` indicates the number of batches to be processed at each moment. `input_size` indicates the number of input features.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>initH</td>
      <td>Input</td>
      <td>Initial hidden state, corresponding to h in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape supports 3D (num_layers, batch_size, hidden_size), or (2 * num_layers, batch_size, hidden_size) when **bidirection** is **True**. `num_layers` corresponds to the `numLayers` parameter, indicating the number of LSTM layers. `hidden_size` indicates the number of features in the hidden state.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>initC</td>
      <td>Input</td>
      <td>Initial cell state, corresponding to c in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape supports 3D (num_layers, batch_size, hidden_size), or (2 * num_layers, batch_size, hidden_size) when **bidirection** is **True**.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>wIh</td>
      <td>Input</td>
      <td>Input-hidden weight, corresponding to W in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape supports 2D (4 * hidden_size, input_size).</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>wHh</td>
      <td>Input</td>
      <td>Hidden-hidden weight, corresponding to W in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape supports 2D (4 * hidden_size, hidden_size).</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>bIhOptional</td>
      <td>Input</td>
      <td>Input-hidden weight, corresponding to b in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape supports 1D (4 * hidden_size).</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
       <tr>
      <td>bHhOptional</td>
      <td>Input</td>
      <td>Hidden-hidden offset, corresponding to b in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape supports 1D (4 * hidden_size).</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
       <tr>
      <td>wIhReverseOptional</td>
      <td>Input</td>
      <td>Reverse input-hidden weight, corresponding to W in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape supports 2D (4 * hidden_size, input_size).</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>wHhReverseOptional</td>
      <td>Input</td>
      <td>Reverse hidden-hidden weight, corresponding to W in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape supports 2D (4 * hidden_size, input_size).</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
       <tr>
      <td>bIhReverseOptional</td>
      <td>Input</td>
      <td>Reverse input-hidden weight, corresponding to b in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape supports 1D (4 * hidden_size).</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
       <tr>
      <td>bHhReverseOptional</td>
      <td>Input</td>
      <td>Reverse hidden-hidden weight, corresponding to b in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape supports 1D (4 * hidden_size).</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>numLayers</td>
      <td>Input</td>
      <td>Number of LSTM layers.</td>
      <td>Currently, only 1 is supported.</td>
      <td>INT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>isbias</td>
      <td>Input</td>
      <td>Whether bias exists.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>batchFirst</td>
      <td>Input</td>
      <td>Whether batch is the first dimension.</td>
      <td>Currently, only false is supported.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>bidirection</td>
      <td>Input</td>
      <td>Whether it is bidirectional.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>yOut</td>
      <td>Output</td>
      <td>Output vector of the LSTM unit.</td>
      <td>The shape supports 3D (time_step, batch_size, hidden_size), or (time_step, batch_size, 2 * hidden_size) when **bidirection** is **True**.</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
       <tr>
      <td>outputHOut</td>
      <td>Output</td>
      <td>Final hidden state, corresponding to h in the formula.</td>
      <td>The shape supports 3D (num_layers, batch_size, hidden_size), or (2 * num_layers, batch_size, hidden_size) when **bidirection** is **True**.</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
       <tr>
      <td>outputCOut</td>
      <td>Output</td>
      <td>Final cell state, corresponding to c in the formula.</td>
      <td>The shape supports 3D (num_layers, batch_size, hidden_size), or (2 * num_layers, batch_size, hidden_size) when **bidirection** is **True**.</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
       <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace required to be allocated on the device.</td>
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


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:

  <table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
  <col style="width: 319px">
  <col style="width: 108px">
  <col style="width: 621px">
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
      <td>The input parameter is a required input, output, or attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The input parameter type is aclTensor and its data type is not supported.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="3">561002</td>
      <td>The input parameter type is aclTensor and its shape does not comply with the preceding parameter description.</td>
    </tr>
  </tbody>
  </table>


## aclnnBidirectionLSTM

- **Parameters:**

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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API **aclnnBidirectionLSTMGetWorkspaceSize**.</td>
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


- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic compute:
  - **aclnnBidirectionLSTM** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_bidirection_lstm.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
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
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
  int time_step = 2;
  int batch_size = 32;
  int input_size = 32;
  int hidden_size = 32;

  int64_t numLayers = 1;
  bool isbias = true;
  bool batchFirst = false;
  bool bidirection = true;

  std::vector<int64_t> selfShape = {time_step, batch_size, input_size};
  std::vector<int64_t> weightHIShape = {4 * hidden_size, input_size};
  std::vector<int64_t> weightHHShape = {4 * hidden_size, hidden_size};
  std::vector<int64_t> initHShape = {2, batch_size, hidden_size};
  std::vector<int64_t> initCShape = {2, batch_size, hidden_size};
  std::vector<int64_t> biasHIShape = {4 * hidden_size};
  std::vector<int64_t> biasHHShape = {4 * hidden_size};  
  std::vector<int64_t> outShape = {time_step, batch_size, 2 * hidden_size};
  std::vector<int64_t> outHShape = {2, batch_size, hidden_size};
  std::vector<int64_t> outCShape = {2, batch_size, hidden_size};

  void* selfDeviceAddr = nullptr;
  void* weightHIDeviceAddr = nullptr;
  void* weightHHDeviceAddr = nullptr;
  void* weightHIReverseDeviceAddr = nullptr;
  void* weightHHReverseDeviceAddr = nullptr;
  void* initHDeviceAddr = nullptr;
  void* initCDeviceAddr = nullptr;
  void* biasHIDeviceAddr = nullptr;
  void* biasHHDeviceAddr = nullptr;
  void* biasHIReverseDeviceAddr = nullptr;
  void* biasHHReverseDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* outHDeviceAddr = nullptr;
  void* outCDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* weightHI = nullptr;
  aclTensor* weightHH = nullptr;
  aclTensor* weightHIReverse = nullptr;
  aclTensor* weightHHReverse = nullptr;
  aclTensor* biasHI = nullptr;
  aclTensor* biasHH = nullptr;
  aclTensor* biasHIReverse = nullptr;
  aclTensor* biasHHReverse = nullptr;
  aclTensor* initH = nullptr;
  aclTensor* initC = nullptr;
  aclTensor* out = nullptr;
  aclTensor* outH = nullptr;
  aclTensor* outC = nullptr;

  std::vector<uint16_t> selfHostData(GetShapeSize(selfShape));
  std::vector<uint16_t> weightHIHostData(GetShapeSize(weightHIShape));
  std::vector<uint16_t> weightHHHostData(GetShapeSize(weightHHShape));
  std::vector<uint16_t> biasHIHostData(GetShapeSize(biasHIShape));
  std::vector<uint16_t> biasHHHostData(GetShapeSize(biasHHShape));
  std::vector<uint16_t> initHHostData(GetShapeSize(initHShape));
  std::vector<uint16_t> initCHostData(GetShapeSize(initCShape));
  std::vector<uint16_t> outHostData(GetShapeSize(outShape));
  std::vector<uint16_t> outHHostData(GetShapeSize(outHShape));
  std::vector<uint16_t> outCHostData(GetShapeSize(outCShape));

  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHIHostData, weightHIShape, &weightHIDeviceAddr, aclDataType::ACL_FLOAT16, &weightHI);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHHHostData, weightHHShape, &weightHHDeviceAddr, aclDataType::ACL_FLOAT16, &weightHH);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(initHHostData, initHShape, &initHDeviceAddr, aclDataType::ACL_FLOAT16, &initH);
  CHECK_RET(ret == ACL_SUCCESS, return ret);  
  ret = CreateAclTensor(initCHostData, initCShape, &initCDeviceAddr, aclDataType::ACL_FLOAT16, &initC);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHIHostData, biasHIShape, &biasHIDeviceAddr, aclDataType::ACL_FLOAT16, &biasHI);
  CHECK_RET(ret == ACL_SUCCESS, return ret); 
  ret = CreateAclTensor(biasHHHostData, biasHHShape, &biasHHDeviceAddr, aclDataType::ACL_FLOAT16, &biasHH);
  CHECK_RET(ret == ACL_SUCCESS, return ret); 
  ret = CreateAclTensor(weightHIHostData, weightHIShape, &weightHIReverseDeviceAddr, aclDataType::ACL_FLOAT16, &weightHIReverse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHHHostData, weightHHShape, &weightHHReverseDeviceAddr, aclDataType::ACL_FLOAT16, &weightHHReverse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHIHostData, biasHIShape, &biasHIReverseDeviceAddr, aclDataType::ACL_FLOAT16, &biasHIReverse);
  CHECK_RET(ret == ACL_SUCCESS, return ret); 
  ret = CreateAclTensor(biasHHHostData, biasHHShape, &biasHHReverseDeviceAddr, aclDataType::ACL_FLOAT16, &biasHHReverse);
  CHECK_RET(ret == ACL_SUCCESS, return ret); 
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);  
  ret = CreateAclTensor(outHHostData, outHShape, &outHDeviceAddr, aclDataType::ACL_FLOAT16, &outH);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outCHostData, outCShape, &outCDeviceAddr, aclDataType::ACL_FLOAT16, &outC);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnBidirectionLSTM.
  ret = aclnnBidirectionLSTMGetWorkspaceSize(self, initH, initC, weightHI, weightHH,
                                            biasHI, biasHH, weightHIReverse, weightHHReverse, biasHIReverse, biasHHReverse,
                                            numLayers, isbias, batchFirst, bidirection,
                                            out, outH, outC,
                                            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBidirectionLSTMGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnBidirectionLSTM.
  ret = aclnnBidirectionLSTM(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBidirectionLSTM failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  PrintOutResult(outShape, &outDeviceAddr);
  PrintOutResult(outHShape, &outHDeviceAddr);
  PrintOutResult(outCShape, &outCDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(weightHI);
  aclDestroyTensor(weightHH);
  aclDestroyTensor(initH);
  aclDestroyTensor(initC);
  aclDestroyTensor(biasHI);
  aclDestroyTensor(biasHH);
  aclDestroyTensor(weightHIReverse);
  aclDestroyTensor(weightHHReverse);
  aclDestroyTensor(biasHIReverse);
  aclDestroyTensor(biasHHReverse);
  aclDestroyTensor(out);
  aclDestroyTensor(outH);
  aclDestroyTensor(outC);

  // 7. Release device resources.
  aclrtFree(selfDeviceAddr);
  aclrtFree(weightHIDeviceAddr);
  aclrtFree(weightHHDeviceAddr);
  aclrtFree(initHDeviceAddr);
  aclrtFree(initCDeviceAddr);
  aclrtFree(biasHIDeviceAddr);
  aclrtFree(biasHHDeviceAddr);
  aclrtFree(weightHIReverseDeviceAddr);
  aclrtFree(weightHHReverseDeviceAddr);
  aclrtFree(biasHIReverseDeviceAddr);
  aclrtFree(biasHHReverseDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(outHDeviceAddr);
  aclrtFree(outCDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
