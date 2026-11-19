# aclnnLSTM

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:50:35.013Z -->

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
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: The LSTM (Long Short-Term Memory) network is a special type of recurrent neural network (RNN) model. Performs LSTM network computation, receives an input sequence and initial states, and returns an output sequence and final states.
- Formula:

  $$
  \begin{aligned}
  (1)\qquad f_t &=\sigma(W_f[h_{t-1}, x_t] + b_f) \\
  (2)\qquad     i_t &=\sigma(W_i[h_{t-1}, x_t] + b_i) \\
  (3)\qquad     o_t &=\sigma(W_o[h_{t-1}, x_t] + b_o) \\
  (4)\qquad     \tilde{c}_t &=tanh(W_c[h_{t-1}, x_t] + b_c) \\
  (5)\qquad     c_t &=f_t ⊙ c_{t-1} + i_t ⊙ \tilde{c}_t \\
  (6)\qquad     c_{o}^{t} &=tanh(c_t) \\
  (7)\qquad     h_t &=o_t ⊙ c_{o}^{t} \\
  \end{aligned}
  $$

  - $x_t ∈ R^{d}$: Input vector of the LSTM unit.
  - $f_t ∈ (0, 1)^{h}$: forget gate activation vector.
  - $i_t ∈ (0, 1)^{h}$: input gate and update gate activation vector.
  - $o_t ∈ (0, 1)^{h}$: output gate activation vector.
  - $h_t ∈ (-1, 1)^{h}$: Hidden state vector, also known as the output vector of the LSTM unit.
  - $\tilde{c}_t ∈ (-1, 1)^{h}$: cell input activation vector.
  - $c_t ∈ R^{h}$: cell state vector.
  - $W ∈ R^{h×d}，(U ∈ R^{h×h})∩(b ∈ R^{h})$: weight matrices and bias vector parameters learned during training.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnLSTMGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the process, and then call the "aclnnLSTM" API to perform the computation.

```Cpp
aclnnStatus aclnnLSTMGetWorkspaceSize(
    const aclTensor     *input,
    const aclTensorList *params,
    const aclTensorList *hx,
    const aclTensor     *batchSizes,
    bool                 hasBias,
    int64_t              numLayers,
    double               dropout,
    bool                 train,
    bool                 bidirectional,
    bool                 batchFirst,
    aclTensor           *output,
    aclTensor           *hy,
    aclTensor           *cy,
    aclTensorList       *iOut,
    aclTensorList       *jOut,
    aclTensorList       *fOut,
    aclTensorList       *oOut,
    aclTensorList       *hOut,
    aclTensorList       *cOut,
    aclTensorList       *tanhCOut,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor);
```

```Cpp
aclnnStatus aclnnLSTM(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnLSTMGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1570px"><colgroup>
  <col style="width: 134px">
  <col style="width: 121px">
  <col style="width: 263px">
  <col style="width: 469px">
  <col style="width: 169px">
  <col style="width: 128px">
  <col style="width: 142px">
  <col style="width: 144px">
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
      <td>Input</td>
      <td>Input</td>
      <td>Input vector of the LSTM unit.</td>
      <td>      <ul>
          <li><strong>If batchSizes is a null pointer:</strong>
            <br>The shape format depends on the batchFirst parameter:
            <ul>
              <li>batchFirst=False: (time_step, batch_size, input_size)</li>
              <li>batchFirst=True: (batch_size, time_step, input_size)</li>
            </ul>
            Note: batchFirst indicates whether the batch dimension is the first dimension; time_step is the time dimension; batch_size is the number of samples processed at each time step; input_size is the number of input features.
          </li>
          <li><strong>If a valid batchSizes is passed in:</strong>
            <br>Shape format: (time_step * batch_size, input_size)
            <br>Note: The memory arrangement is the same as that of (time_step, batch_size, input_size).
          </li>
      </ul>
      </td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td> params</td>
      <td>Input</td>
      <td>Indicates the list of weight and bias tensors in the LSTM operation.</td>
  <td>  <ul>
    <p>List length calculation formula: <strong>2 * D * B * num_layers</strong></p>
    <ul>
    <li>num_layers: Corresponds to parameter numLayers, indicating the number of LSTM layers;</li>
    <li>D: D=2 when bidirectional=True; otherwise, D=1;</li>
    <li>B: B=2 when has_biases=True; otherwise, B=1.</li>
    </ul>

    <p><strong>Special use case (bidirectional=True and has_biases=True):</strong></p>
    <p style="padding-left: 20px;">
      Parameter layout: [weight_ih_0, weight_hh_0, bias_ih_0, bias_hh_0, weight_ih_reverse_0, weight_hh_reverse_0, bias_ih_reverse_0, bias_hh_reverse_0]
    </p>

    <p><strong>Description of key parameters (taking layer 0 as an example):</strong></p>
    <ul>
    <li>weight_ih_0: Input weight parameter of layer 0, shape=(4 * hidden_size, cur_input_size)
      <br>Note: cur_input_size is the number of input features for each layer (first layer=input_size; subsequent layers=hidden_size; bidirectional=2*hidden_size)
    </li>
    <li>weight_hh_0: Hidden layer weight parameter of layer 0, shape=(4 * hidden_size, hidden_size)</li>
    <li>bias_ih_0: Input weight bias of layer 0, shape=(4 * hidden_size)</li>
    <li>bias_hh_0: Hidden layer weight bias of layer 0, shape=(4 * hidden_size)</li>
    </ul>
    </ul>
  </td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
      <tr>
      <td>hx</td>
      <td>Optional input</td>
      <td>Indicates the list of initial hidden and cell states in the LSTM operation.</td>
      <td>The list length is 2. Each shape in the list supports three dimensions (D * num_layers, batch_size, hidden_size). If the input is empty, it indicates that the initial hidden and cell states are 0.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
      <tr>
      <td>batchSizes</td>
      <td>Optional input</td>
      <td>Indicates the number of valid batches that actually participate in the computation at each time step. If nullptr is passed in, the input is fixed-length mode data; otherwise, it is variable-length mode.</td>
      <td>The shape is (time_step,). Elements must be in descending order, be positive integers not greater than the total number of batches, and the first element must be equal to the total number of batches.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>hasBias</td>
      <td>Input</td>
      <td>Indicates whether biases exist.</td>
      <td>/</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
      <tr>
      <td>numLayers</td>
      <td>Input</td>
      <td>Indicates the number of LSTM layers.</td>
      <td>/</td>
      <td>INT64</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
       <tr>
      <td>dropout</td>
      <td>Input</td>
      <td>Indicates the probability of the random mask.</td>
      <td>This function is not supported currently.</td>
      <td>DOUBLE</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
       <tr>
      <td>train</td>
      <td>Input</td>
      <td>Indicates whether it is training mode.</td>
      <td>If train = True, intermediate results are saved during forward LSTM computation for backward propagation. If train = False, intermediate results are not saved during forward computation.</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
      <tr>
      <td>bidirectional</td>
      <td>Input</td>
      <td>Indicates whether it is bidirectional.</td>
      <td>/</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
       <tr>
      <td>batchFirst</td>
      <td>Input</td>
      <td>Indicates whether the input data format is batch on the first axis (B, T, H).</td>
      <td>/</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
       <tr>
      <td>Output</td>
      <td>Output</td>
      <td>Indicates the output of each time step of the last layer in the LSTM operation.</td>
      <td><ul><li>If a null pointer is passed in for batchSizes:<br>When batchFirst=False, the shape supports three dimensions (time_step, batch_size, D * hidden_size); otherwise, the shape supports three dimensions (batch_size, time_step, D * hidden_size).</li><li>If a valid batchSizes is passed in:<br>The shape should be (time_step, batch_size, D * hidden_size).</li></ul></td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hy</td>
      <td>Output</td>
      <td>Indicates the hidden layer at the last time step of each layer in the LSTM operation (output of formula (7)).</td>
      <td>shape supports three dimensions (D * num_layers, batch_size, hidden_size)</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cy</td>
      <td>Output</td>
      <td>Indicates the Cell state of the last time step in each layer of the LSTM operation (the output of formula (5)).</td>
      <td>shape supports three dimensions (D * num_layers, batch_size, hidden_size)</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hy</td>
      <td>Output</td>
      <td>Indicates the hidden layer at the last time step of each layer in the LSTM operation (the output of formula (7)).</td>
      <td>The shape supports three dimensions (D * num_layers, batch_size, hidden_size).</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cy</td>
      <td>Output</td>
      <td>Indicates the Cell state of the last time step of each layer in the LSTM operation (output of formula (5)).</td>
      <td>shape supports three dimensions (D * num_layers, batch_size, hidden_size).</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>iOut</td>
      <td>Output</td>
      <td>Indicates the activation value of the input gate of each layer in the LSTM operation (sigmoid output, output of formula (2)).</td>
      <td>The list length is D * num_layers. Each shape in the list supports three dimensions (time_step, batch_size, hidden_size). There is no output value when train=False.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>jOut</td>
      <td>Output</td>
      <td>Indicates the candidate cell state of each layer in the LSTM operation (tanh output, output of formula (4)).</td>
      <td>The list length is D * num_layers. Each shape in the list supports three dimensions (time_step, batch_size, hidden_size). There is no output value when train=False.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>fOut</td>
      <td>Output</td>
      <td>Indicates the activation value of the forget gate of each layer in the LSTM operation (sigmoid output, the output of formula (1)).</td>
      <td>The list length is D * num_layers. Each shape in the list supports three dimensions (time_step, batch_size, hidden_size). There is no output value when train=False.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>oOut</td>
      <td>Output</td>
      <td>Indicates the activation value of the output gate of each layer in performing the LSTM operation (sigmoid output, output of formula (3)).</td>
      <td>The list length is D * num_layers, and each shape in the list supports three dimensions (time_step, batch_size, hidden_size). When train=False, there is no output value.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>hOut</td>
      <td>Output</td>
      <td>Indicates the hidden layer of each layer in the LSTM operation (the output of formula (7)).</td>
      <td>The list length is D * num_layers, and each shape in the list supports three dimensions (time_step, batch_size, hidden_size). When train=False, there is no output value.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
      <tr>
      <td>cOut</td>
      <td>Output</td>
      <td>Indicates the final Cell state of each layer in performing the LSTM operation (output of formula (5)).</td>
      <td>The list length is D * num_layers. Each shape in the list supports three dimensions (time_step, batch_size, hidden_size). When train=False, there is no output value.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
      <tr>
      <td>tanhCOut</td>
      <td>Output</td>
      <td>Indicates the output after the final cell state of each layer in performing the LSTM operation passes through the tanh activation function (output of formula (6)).</td>
      <td>The list length is D * num_layers. Each shape in the list supports three dimensions (time_step, batch_size, hidden_size). No output value when train=False.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>

    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor</td>
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

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:
    <table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
    <col style="width: 319px">
    <col style="width: 108px">
    <col style="width: 621px">
    </colgroup>
    <thead>
      <tr>
        <th>Error Code</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>If a required input, output, or required attribute is passed in as a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="12">161002</td>
        <td>If the passed-in parameter is aclTensor or aclTensorList and the data type is not within the supported range.</td>
      </tr>
      <tr>
        <td>If the passed-in parameter is of type aclTensor or aclTensorList and the data types are different.</td>
      </tr>
      <tr>
        <td>If the passed-in parameter is of type aclTensor or aclTensorList and the shape does not meet the corresponding shape requirements.</td>
      </tr>
      <tr>
        <td>numLayers does not meet the requirement of >0.</td>
      </tr>
    </tbody>
    </table>

## aclnnLSTM

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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by calling the first-phase API aclnnLSTMGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- deterministic computation:
  - aclnnLSTM defaults to a deterministic implementation.
- All inputs and outputs that support FLOAT16 and FLOAT32 types must have the same data type.

## Example

The sample code is as follows and is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file main.cpp
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lstm.h"

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

void PrintOutResult(const std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // boilerplate, AscendCL initialization
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
    // Call aclrtMalloc to allocate memory on the device side.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorList(
    const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr, aclDataType dataType, aclTensorList** tensor,
    T initVal = 1)
{
    int size = shapes.size();
    aclTensor* tensors[size];
    for (int i = 0; i < size; i++) {
        std::vector<T> hostData(GetShapeSize(shapes[i]), initVal);
        int ret = CreateAclTensor<float>(hostData, shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

int main()
{
    // 1. (Boilerplate) Device/stream initialization. For details, see the list of AscendCL public APIs.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs. Custom construction is required based on the API.
    int time_step = 1;
    int batch_size = 1;
    int hidden_size = 1;
    int input_size = hidden_size;
    int64_t numLayers = 1;
    bool isbias = false;
    bool batchFirst = false;
    bool bidirectional = false;
    bool isTraining = true;
    int64_t d_scale = bidirectional == true ? 2 : 1;

    std::vector<int64_t> inputShape = {time_step, batch_size, input_size};
    std::vector<int64_t> outputShape = {time_step, batch_size, d_scale * hidden_size};
    std::vector<int64_t> hycyShape = {numLayers * d_scale, batch_size, hidden_size};
    std::vector<std::vector<int64_t>> paramsListShape = {};

    std::vector<std::vector<int64_t>> outIListShape = {};
    std::vector<std::vector<int64_t>> outJListShape = {};
    std::vector<std::vector<int64_t>> outFListShape = {};
    std::vector<std::vector<int64_t>> outOListShape = {};
    std::vector<std::vector<int64_t>> outHListShape = {};
    std::vector<std::vector<int64_t>> outCListShape = {};
    std::vector<std::vector<int64_t>> outTanhCListShape = {};

    auto cur_input_size = input_size;
    for (int i = 0; i < numLayers; i++) {
        paramsListShape.push_back({hidden_size * 4, cur_input_size});
        paramsListShape.push_back({hidden_size * 4, hidden_size});

        outIListShape.push_back({time_step, batch_size, hidden_size});
        outJListShape.push_back({time_step, batch_size, hidden_size});
        outFListShape.push_back({time_step, batch_size, hidden_size});
        outOListShape.push_back({time_step, batch_size, hidden_size});
        if (isTraining == true) {
            outHListShape.push_back({time_step, batch_size, hidden_size});
            outCListShape.push_back({time_step, batch_size, hidden_size});
        } else {
            outHListShape.push_back({batch_size, hidden_size});
            outCListShape.push_back({batch_size, hidden_size});
        }
        outTanhCListShape.push_back({time_step, batch_size, hidden_size});
        cur_input_size = hidden_size;
    }

    void* inputDeviceAddr = nullptr;
    void* paramsListDeviceAddr[2 * numLayers];

    void* outputDeviceAddr = nullptr;
    void* hyDeviceAddr = nullptr;
    void* cyDeviceAddr = nullptr;
    void* outIListDeviceAddr[numLayers];
    void* outJListDeviceAddr[numLayers];
    void* outFListDeviceAddr[numLayers];
    void* outOListDeviceAddr[numLayers];
    void* outHListDeviceAddr[numLayers];
    void* outCListDeviceAddr[numLayers];
    void* outTanhCListDeviceAddr[numLayers];

    aclTensor* input = nullptr;
    aclTensorList* params = nullptr;

    aclTensor* output = nullptr;
    aclTensor* hy = nullptr;
    aclTensor* cy = nullptr;
    aclTensorList* outIList = nullptr;
    aclTensorList* outJList = nullptr;
    aclTensorList* outFList = nullptr;
    aclTensorList* outOList = nullptr;
    aclTensorList* outHList = nullptr;
    aclTensorList* outCList = nullptr;
    aclTensorList* outTanhCList = nullptr;

    std::vector<float> inputHostData(GetShapeSize(inputShape), 1);

    ret = CreateAclTensor<float>(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(paramsListShape, paramsListDeviceAddr, aclDataType::ACL_FLOAT, &params, 1.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> outputHostData(GetShapeSize(outputShape), 1);
    ret = CreateAclTensor<float>(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> hycyHostData(GetShapeSize(hycyShape), 1);
    ret = CreateAclTensor<float>(hycyHostData, hycyShape, &hyDeviceAddr, aclDataType::ACL_FLOAT, &hy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(hycyHostData, hycyShape, &cyDeviceAddr, aclDataType::ACL_FLOAT, &cy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outIListShape, outIListDeviceAddr, aclDataType::ACL_FLOAT, &outIList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outJListShape, outJListDeviceAddr, aclDataType::ACL_FLOAT, &outJList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outFListShape, outFListDeviceAddr, aclDataType::ACL_FLOAT, &outFList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outOListShape, outOListDeviceAddr, aclDataType::ACL_FLOAT, &outOList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outHListShape, outHListDeviceAddr, aclDataType::ACL_FLOAT, &outHList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outCListShape, outCListDeviceAddr, aclDataType::ACL_FLOAT, &outCList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(
        outTanhCListShape, outTanhCListDeviceAddr, aclDataType::ACL_FLOAT, &outTanhCList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Modify it to the specific API name.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first-phase API of aclnnLSTM.
    ret = aclnnLSTMGetWorkspaceSize(
        input, params, nullptr, nullptr, isbias, numLayers, 0.0, isTraining, bidirectional, batchFirst, output, hy, cy,
        outIList, outJList, outFList, outOList, outHList, outCList, outTanhCList, &workspaceSize, &executor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLSTMGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnLSTM.
    ret = aclnnLSTM(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLSTM failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values, and copy the results from device memory to the host side. Modify based on the specific API definition.

    PrintOutResult(outputShape, &outputDeviceAddr);

    // 6. Release aclTensor and aclTensorList. Modify based on the specific API definition.
    aclDestroyTensor(input);
    aclDestroyTensorList(params);
    aclDestroyTensor(output);
    aclDestroyTensor(hy);
    aclDestroyTensor(cy);
    aclDestroyTensorList(outIList);
    aclDestroyTensorList(outJList);
    aclDestroyTensorList(outFList);
    aclDestroyTensorList(outOList);
    aclDestroyTensorList(outHList);
    aclDestroyTensorList(outCList);
    aclDestroyTensorList(outTanhCList);

    //   // 7. Release device resources.
    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclrtFree(hyDeviceAddr);
    aclrtFree(cyDeviceAddr);
    for (int i = 0; i < numLayers; i++) {
        aclrtFree(outIListDeviceAddr[i]);
        aclrtFree(outJListDeviceAddr[i]);
        aclrtFree(outFListDeviceAddr[i]);
        aclrtFree(outOListDeviceAddr[i]);
        aclrtFree(outHListDeviceAddr[i]);
        aclrtFree(outCListDeviceAddr[i]);
        aclrtFree(outTanhCListDeviceAddr[i]);
    }

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
