# aclnnGRUBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:53:29.922Z -->

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

* Description: backward propagation of GRU. Computes the gradients of the forward input input, weight params, and initial state hx.
* Forward formula:
  * Reset gate $r_t= \sigma(W_{ir}x_t + b_{ir} + W_{hr}h_{t-1} + b_{hr})$
  * Update gate $z_t = \sigma(W_{iz}x_t + b_{iz} + W_{hz}h_{t-1} + b_{hz})$
  * Candidate hidden state $n_t = \tanh(W_{in}x_t + b_{in} + r_t \odot (W_{hn}h_{t-1} + b_{hn}))$
  * Hidden state $h_t = (1-z) \odot n_t + z_t \odot h_{t-1}$
* Backward computation formula:
  * $\text{ Total upstream gradient:} \quad dh_t = dy_t + dh_{next} \quad $($dy_t$ is the gradient from the upper layer, and $dh_{next}$ is the gradient passed back at time step t+1)
  * $\text{ Update gate gradient:} \quad dz_t = dh_t * (h_{t-1} - \tilde{h}_t) * z_t * (1 - z_t)$
  * $\text{ Candidate state gradient:} \quad dh_{\tilde{h}t} = dh_t * (1 - z_t) * (1 - \tilde{h}_t^2)$
  * $\text{ Reset gate gradient:} \quad dr_t = dh_{\tilde{h}t} * lin_{hh}[2*hidden\_size:3*hidden\_size] * r_t * (1 - r_t)$
  * $\text{ Linear transformation gradient decomposition:}$
    $\quad dlin_{ih} = [dz_t; dr_t; dh_{\tilde{h}t}], \quad dlin_{hh} = [dz_t; dr_t; dh_{\tilde{h}t} * r_t]$
  * $\text{ Input gradient (passed to the lower layer):} \quad dx_t = W_{ih}^T @ dlin_{ih}$
  * $\text{ Previous hidden state gradient (passed to t-1):} \quad dh_{prev} = W_{hh}^T @ dlin_{hh} + dh_t * z_t$
  * $\text{ Weight/bias gradient accumulation:}$
    $\quad dW_{ih} += dlin_{ih} @ x_t^T, \quad dW_{hh} += dlin_{hh} @ h_{t-1}^T$
    $\quad db_{ih} += dlin_{ih}.sum(dim=1), \quad db_{hh} += dlin_{hh}.sum(dim=1)$

## Function Prototype

 Each operator is divided into a two-phase API. Call the "aclnnGRUBackwardGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnGRUBackward" API to perform the computation.

```Cpp
 aclnnStatus aclnnGRUBackwardGetWorkspaceSize(
   const aclTensor     *input,
   const aclTensorList *params,
   const aclTensor     *hx,
   const aclTensor     *dy,
   const aclTensor     *dh,
   const aclTensorList *r,
   const aclTensorList *z,
   const aclTensorList *n,
   const aclTensorList *h_n,
   const aclTensorList *h,
   const aclTensor     *batchSizesOptional,
   bool                hasBias,
   int64_t             numLayers,
   bool                bidirection,
   bool                batchFirst,
   aclTensor           *dxOut,
   aclTensor           *dhPrevOut,
   aclTensorList       *dparamsOut,
   uint64_t            *workspaceSize,
   aclOpExecutor       **executor)
```

```Cpp
 aclnnStatus aclnnGRUBackward(
   void            *workspace,
   uint64_t        workspaceSize,
   aclOpExecutor   *executor,
   aclrtStream     stream)
```

## aclnnGRUBackwardGetWorkspaceSize

* **Parameters:**

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
       <td>Input</td>
       <td>Input</td>
       <td>The input sequence of GRU, corresponding to x in the corresponding formula.</td>
       <td><ul><li>batch_size indicates the number of sequences; time_step indicates the time dimension; input_size indicates the number of input features.</li></ul></td>
       <td>FLOAT32, FLOAT16</td>
       <td>ND</td>
       <td><ul>
       <li>If a valid batchSizesOptional is passed in, the shape is [time_step * batch_size, input_size];</li>
       <li>If a null pointer batchSizesOptional is passed in, the shape is [time_step, batch_size, input_size] or [batch_size, time_step, input_size]</li>
       <li>Empty tensors are not supported</li></ul></td>
       <td>√</td>
     </tr>
     <tr>
       <td>params</td>
       <td>Input</td>
       <td>The weight and bias tensor list of each GRU layer, corresponding to w and b in the corresponding formula.</td>
       <td><ul><li>If bidirection is True, `D = 2`; otherwise, `D = 1`. If hasBiases is True, `B = 2`; otherwise, `B = 1`. The list length is D * B * num_layers * 2.</li><li>If both bidirection and hasBias are True, the layout is: [weight_ih_0, weight_hh_0, bias_ih_0, bias_hh_0, weight_ih_reverse_0, weight_hh_reverse_0, bias_ih_reverse_0, bias_hh_reverse_0].</li>
       <li>If hasBias is False, there is no bias item; if bidirection is False, there is no reverse item.</li><li>For multiple layers, tensors are laid out layer by layer.</li><li>The data type is consistent with input.</li></ul></td>
       <td>FLOAT32, FLOAT16</td>
       <td>ND</td>
       <td>weight_ih: [3*hidden_size, cur_input_size]<br>weight_hh: [3*hidden_size, hidden_size]<br>bias_ih: [3*hidden_size]<br>bias_hh: [3*hidden_size]<br>Empty tensors are not supported.</td>
       <td>√</td>
     </tr>
     <tr>
       <td>hx</td>
       <td>Input</td>
       <td>Initial hidden state of each GRU layer. Corresponds to h(t-1) at time 0.</td>
       <td><ul><li>For multi-layer bidirectional mode, each tensor is laid out along dim 0, bidirectional first and then layer by layer.</li><li>The data type is consistent with input.</li></ul></td>
       <td>FLOAT32, FLOAT16</td>
       <td>ND</td>
       <td>The shape of each tensor is [numLayers * D, batch_size, hidden_size]<br>Empty tensors are not supported</td>
       <td>√</td>
     </tr>
     <tr>
       <td>dy</td>
       <td>Input</td>
       <td>Gradient of the hidden output of the last layer in the GRU forward pass. Corresponds to ∂L/∂h^(l) in the formula.</td>
       <td><ul><li>In bidirectional mode, data is laid out along the last dimension in forward and backward direction.</li><li>The data type is consistent with input.</li></ul></td>
       <td>FLOAT32, FLOAT16</td>
       <td>ND</td>
       <td><ul>
       <li>If a valid batchSizesOptional is passed in, the shape is [time_step * batch_size, hidden_size * D];</li>
       <li>If batchSizesOptional is a null pointer, the shape is [time_step, batch_size, hidden_size * D] or [batch_size, time_step, hidden_size * D]</li>
       <li>Empty tensors are not supported</li></ul></td>
       <td>√</td>
     </tr>
     <tr>
       <td>dh</td>
       <td>Input</td>
       <td>Gradient passed from the next time step at time T for the hidden output of each layer in GRU forward propagation. Corresponds to δh_next.</td>
       <td><ul><li>In the multi-layer bidirectional case, data is laid out along dimension 0 in the order of bidirectional first and then layers.</li><li>The data type is consistent with that of input.</li></ul></td>
       <td>FLOAT32, FLOAT16</td>
       <td>ND</td>
       <td>[numLayers * D, batch_size, hidden_size]<br>Empty tensor is not supported.</td>
       <td>√</td>
     </tr>
     <tr>
       <td>r</td>
       <td>Input</td>
       <td>Activation value of the reset gate of each layer at each time step in GRU forward. Corresponds to r in the corresponding formula.</td>
       <td><ul><li>List length is D * num_layers.</li><li>For multi-layer bidirectional, the tensor layout is bidirection first and then multi-layer.</li><li>The data type is consistent with input.</li></ul></td>
       <td>FLOAT32, FLOAT16</td>
       <td>ND</td>
       <td><ul>
       <li>If a valid batchSizesOptional is passed in, the shape is [time_step * batch_size, hidden_size];</li>
       <li>If a null pointer batchSizesOptional is passed in, the shape is [time_step, batch_size, hidden_size]</li>
       <li>Empty tensors are not supported</li></ul></td>
       <td>√</td>
     </tr>
     <tr>
       <td>z</td>
       <td>Input</td>
       <td>Activation value of the update gate of each layer at each time step in GRU forward. Corresponds to z in the corresponding formula.</td>
       <td><ul><li>The list length is D * num_layers.</li><li>For multi-layer bidirectional, the layout of tensors is bidirectional first and then layer by layer.</li><li>The data type is consistent with that of input.</li></ul></td>
       <td>FLOAT32, FLOAT16</td>
       <td>ND</td>
       <td><ul>
       <li>If a valid batchSizesOptional is passed in, the shape is [time_step * batch_size, hidden_size];</li>
       <li>If an empty pointer batchSizesOptional is passed in, the shape is [time_step, batch_size, hidden_size]</li>
       <li>Empty tensors are not supported</li></ul></td>
       <td>√</td>
     </tr>
     <tr>
       <td>n</td>
       <td>Input</td>
       <td>Activation value of the candidate hidden state for each layer at each time step in GRU forward propagation. It corresponds to n in the corresponding formula.</td>
       <td><ul><li>The list length is D * num_layers.</li><li>For multi-layer bidirectional, the layout of tensors is bidirectional first and then layer by layer.</li><li>The data type is consistent with that of input.</li></ul></td>
       <td>FLOAT32, FLOAT16</td>
       <td>ND</td>
       <td><ul>
       <li>If a valid batchSizesOptional is passed in, the shape is [time_step * batch_size, hidden_size];</li>
       <li>If an empty pointer batchSizesOptional is passed in, the shape is [time_step, batch_size, hidden_size]</li>
       <li>Empty tensors are not supported</li></ul></td>
       <td>√</td>
     </tr>
     <tr>
       <td>h_n</td>
       <td>Input</td>
       <td>Intermediate value of the candidate hidden state at each time step of each layer in GRU forward. It corresponds to $W_{hn}h_{t-1} + b_{hn}$ in the corresponding formula.</td>
       <td><ul><li>The list length is D * num_layers.</li><li>For multi-layer bidirectional, the layout of tensors is bidirectional first and then layer by layer.</li><li>The data type is consistent with input.</li></ul></td>
       <td>FLOAT32, FLOAT16</td>
       <td>ND</td>
       <td><ul>
       <li>If a valid batchSizesOptional is passed in, the shape is [time_step * batch_size, hidden_size];</li>
       <li>If an empty pointer batchSizesOptional is passed in, the shape is [time_step, batch_size, hidden_size]</li>
       <li>Empty tensors are not supported</li></ul></td>
       <td>√</td>
     </tr>
     <tr>
       <td>h</td>
       <td>Input</td>
       <td>Hidden state of each layer at each time step in GRU forward. It corresponds to h in the formula.</td>
       <td><ul><li>List length is D * num_layers.</li><li>In multi-layer bidirectional mode, tensors are laid out bidirectional first and then by layer.</li><li>The data type is consistent with input.</li></ul></td>
       <td>FLOAT32, FLOAT16</td>
       <td>ND</td>
       <td><ul>
       <li>If a valid batchSizesOptional is passed in, the shape is [time_step * batch_size, hidden_size];</li>
       <li>If a null pointer batchSizesOptional is passed in, the shape is [time_step, batch_size, hidden_size]</li>
       <li>Empty tensors are not supported</li></ul></td>
       <td>√</td>
     </tr>
     <tr>
       <td>batchSizesOptional</td>
       <td>Input</td>
       <td>Number of valid sequence batches at each time step of the variable-length GRU input sequence.</td>
       <td><ul><li>Supported for variable-length sequences.</li></ul></td>
       <td>INT64</td>
       <td>ND</td>
       <td>[time_step]</td>
       <td>√</td>
     </tr>
     <tr>
       <td>hasBias</td>
       <td>Input</td>
       <td>Indicates whether the bias b exists.</td>
       <td>-</td>
       <td>BOOL</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
     </tr>
     <tr>
       <td>numLayers</td>
       <td>Input</td>
       <td>Represents the number of GRU layers.</td>
       <td><ul><li>Value greater than 0.</li></ul></td>
       <td>INT64</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
     </tr>
     <tr>
       <td>bidirection</td>
       <td>Input</td>
       <td>Specifies whether it is bidirectional.</td>
       <td>-</td>
       <td>BOOL</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
     </tr>
     <tr>
       <td>batchFirst</td>
       <td>Input</td>
       <td>Indicates whether the data format of inputs input, y, and dy has batch as the first dimension.</td>
       <td>-</td>
       <td>BOOL</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
     </tr>
     <tr>
       <td>dxOut(grad_x_t)</td>
       <td>Output</td>
       <td>Gradient of input, corresponding to δx in the formula.</td>
       <td><ul><li>The shape is consistent with input.</li><li>The data type is consistent with input.</li></ul></td>
       <td>FLOAT32, FLOAT16</td>
       <td>ND</td>
       <td><ul>
        <li>If a valid batchSizesOptional is passed, the shape is [time_step * batch_size, input_size];</li>
        <li>If a null pointer batchSizesOptional is passed, the shape is [time_step, batch_size, input_size] or [batch_size, time_step, input_size]</li>
        <li>Empty tensors are not supported</li></ul></td>
       <td>√</td>
     </tr>
     <tr>
       <td>dhPrevOut(grad_h_prev)</td>
       <td>Output</td>
       <td>Gradient of the initial hidden state of each GRU layer, corresponding to δh_prev at t=0.</td>
       <td><ul><li>In multi-layer bidirectional mode, data is laid out along dimension 0 with bidirection first, then layer by layer.</li><li>The data type is consistent with input.</li></ul></td>
       <td>FLOAT32, FLOAT16</td>
       <td>ND</td>
       <td>[D * num_layers, batch_size, hidden_size]<br>Empty tensors are not supported.</td>
       <td>√</td>
     </tr>
     <tr>
       <td>dparamsOut(total_grad_w_ih, total_grad_w_hh)</td>
       <td>Output</td>
       <td>List of gradient tensors for weights and biases. They correspond to δw and δb in the corresponding formula.</td>
       <td><ul><li>The list length is D * B * num_layers * 2.</li><li>The layout is consistent with input params.</li><li>The data type is consistent with input.</li></ul></td>
       <td>FLOAT32, FLOAT16</td>
       <td>ND</td>
       <td>dweight_ih: [3*hidden_size, cur_input_size]<br>dweight_hh: [3*hidden_size, hidden_size]<br>dbias: [3*hidden_size]<br>Empty tensors are not supported.</td>
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
       <td>Output parameter on the Host.</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
     </tr>
   </tbody></table>

* **Return value**

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
       <th>Error Code</th>
       <th>Description</th>
     </tr></thead>
   <tbody>
     <tr>
       <td>ACLNN_ERR_PARAM_NULLPTR</td>
       <td>161001</td>
       <td>If the input parameter is aclTensor or aclTensorList and is not batchSizesOptional, it is a null pointer.</td>
     </tr>
     <tr>
       <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
       <td rowspan="4">161002</td>
       <td>If the input parameter is aclTensor or aclTensorList, the data type is not in the supported range.</td>
     </tr>
     <tr>
       <td>If the input parameter is aclTensor or aclTensorList, the data types are inconsistent.</td>
     </tr>
     <tr>
       <td>The input parameter is of type aclTensor or aclTensorList, and the shape does not meet the corresponding shape requirements.</td>
     </tr>
     <tr>
       <td>numLayers is not greater than 0.</td>
     </tr>
   </tbody>
   </table>

## aclnnGRUBackward

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
      <td>Workspace memory address requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device, obtained by calling the first-phase API aclnnGRUBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>
- **Return value**

  aclnnStatus: return status code. For details, see [aclnn_return_code].

## Constraints

  - aclnnGRUBackward defaults to a deterministic implementation.
  - FP16/FP32 is supported. The data types of all inputs must be consistent.
  - When W = numLayers・D・T・B・H・(I+H) ≥ 9,622,080, a timeout may occur.

## Example

 The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gru_backward.h"

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

void PrintOutResult(const std::string& name, const std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s from device to host failed. ERROR: %d\n", name.c_str(), ret);
              return );
    LOG_PRINT("=== %s shape=[", name.c_str());
    for (size_t i = 0; i < shape.size(); i++) {
        LOG_PRINT("%ld%s", shape[i], (i + 1 < shape.size()) ? "," : "");
    }
    LOG_PRINT("] ===\n");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("  [%ld] = %f\n", i, resultData[i]);
    }
    LOG_PRINT("\n");
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to device memory.
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

template <typename T>
int CreateAclTensorList(const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr, aclDataType dataType,
                        aclTensorList** tensor, T initVal = 1)
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
    // 1. (Boilerplate) Initialize the device and stream. For details, see the AscendCL API list.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs as required by the API.
    int64_t timeStep = 2;
    int64_t batchSize = 3;
    int64_t inputSize = 4;
    int64_t hiddenSize = 5;
    int64_t gateNum = 3;
    int64_t numLayers = 1;
    bool hasBias = false;
    bool batchFirst = false;
    bool bidirection = false;
    int64_t dScale = bidirection ? 2 : 1;
    int64_t ldScale = numLayers * dScale;

    std::vector<int64_t> inputShape = {timeStep, batchSize, inputSize};
    std::vector<int64_t> dyShape = {timeStep, batchSize, dScale * hiddenSize};
    std::vector<int64_t> dhShape = {ldScale, batchSize, hiddenSize};
    std::vector<int64_t> hxShape = {ldScale, batchSize, hiddenSize};
    std::vector<std::vector<int64_t>> paramsListShape = {};

    auto curLayerInputSize = inputSize;
    for (int i = 0; i < numLayers; i++) {
        for (int64_t j = 0; j < dScale; j++) {
            paramsListShape.push_back({hiddenSize * gateNum, curLayerInputSize});
            paramsListShape.push_back({hiddenSize * gateNum, hiddenSize});
            if (hasBias) {
                paramsListShape.push_back({hiddenSize * gateNum});
                paramsListShape.push_back({hiddenSize * gateNum});
            }
        }
        curLayerInputSize = dScale * hiddenSize;
    }

    // gate lists: r, z, n, hn, h each has ldScale tensors of [T, B, H]
    std::vector<std::vector<int64_t>> gateListShape;
    for (int64_t i = 0; i < ldScale; i++) {
        gateListShape.push_back({timeStep, batchSize, hiddenSize});
    }

    void* inputDeviceAddr = nullptr;
    std::vector<void*> paramsListDeviceAddr(paramsListShape.size(), nullptr);
    void* dyDeviceAddr = nullptr;
    void* dhDeviceAddr = nullptr;
    void* hxDeviceAddr = nullptr;

    std::vector<void*> rDeviceAddr;
    std::vector<void*> zDeviceAddr;
    std::vector<void*> nDeviceAddr;
    std::vector<void*> hnDeviceAddr;
    std::vector<void*> hDeviceAddr;

    // output
    void* dxDeviceAddr = nullptr;
    std::vector<void*> dparamsListDeviceAddr(paramsListShape.size(), nullptr);
    void* dhPrevDeviceAddr = nullptr;

    aclTensor* input = nullptr;
    aclTensorList* params = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* dh = nullptr;
    aclTensor* hx = nullptr;

    aclTensorList* r = nullptr;
    aclTensorList* z = nullptr;
    aclTensorList* n = nullptr;
    aclTensorList* hn = nullptr;
    aclTensorList* h = nullptr;

    aclTensor* dxOut = nullptr;
    aclTensor* dhPrevOut = nullptr;
    aclTensorList* dparamsOut = nullptr;

    // Construct the inputs.
    std::vector<float> inputHostData(GetShapeSize(inputShape), 1.0);
    ret = CreateAclTensor<float>(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(paramsListShape, paramsListDeviceAddr.data(), aclDataType::ACL_FLOAT, &params,
                                     1.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> dyHostData(GetShapeSize(dyShape), 0.5);
    ret = CreateAclTensor<float>(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> dhHostData(GetShapeSize(dhShape), 0.1);
    ret = CreateAclTensor<float>(dhHostData, dhShape, &dhDeviceAddr, aclDataType::ACL_FLOAT, &dh);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> hxHostData(GetShapeSize(hxShape), 0.0);
    ret = CreateAclTensor<float>(hxHostData, hxShape, &hxDeviceAddr, aclDataType::ACL_FLOAT, &hx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Construct gate lists (r, z, n, hn, h) - intermediate values of forward computation.
    rDeviceAddr.resize(ldScale, nullptr);
    zDeviceAddr.resize(ldScale, nullptr);
    nDeviceAddr.resize(ldScale, nullptr);
    hnDeviceAddr.resize(ldScale, nullptr);
    hDeviceAddr.resize(ldScale, nullptr);

    ret = CreateAclTensorList<float>(gateListShape, rDeviceAddr.data(), aclDataType::ACL_FLOAT, &r, 0.5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorList<float>(gateListShape, zDeviceAddr.data(), aclDataType::ACL_FLOAT, &z, 0.5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorList<float>(gateListShape, nDeviceAddr.data(), aclDataType::ACL_FLOAT, &n, 0.5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorList<float>(gateListShape, hnDeviceAddr.data(), aclDataType::ACL_FLOAT, &hn, 0.5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorList<float>(gateListShape, hDeviceAddr.data(), aclDataType::ACL_FLOAT, &h, 0.5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Construct the output.
    std::vector<float> dxHostData(GetShapeSize(inputShape), 0.0);
    ret = CreateAclTensor<float>(dxHostData, inputShape, &dxDeviceAddr, aclDataType::ACL_FLOAT, &dxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> dhPrevHostData(GetShapeSize(hxShape), 0.0);
    ret = CreateAclTensor<float>(dhPrevHostData, hxShape, &dhPrevDeviceAddr, aclDataType::ACL_FLOAT, &dhPrevOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(paramsListShape, dparamsListDeviceAddr.data(), aclDataType::ACL_FLOAT, &dparamsOut,
                                     0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // Call the first-phase API of aclnnGRUBackward.
    ret = aclnnGRUBackwardGetWorkspaceSize(
        input, params, hx, dy, dh, r, z, n, hn, h, nullptr, hasBias, numLayers, bidirection, batchFirst,
        dxOut, dhPrevOut, dparamsOut, &workspaceSize, &executor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGRUBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnGRUBackward.
    ret = aclnnGRUBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGRUBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Wait for task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output values, and copy the results from device memory to the host side.
    PrintOutResult("dxOut", inputShape, &dxDeviceAddr);
    PrintOutResult("dhPrevOut", hxShape, &dhPrevDeviceAddr);
    for (size_t i = 0; i < paramsListShape.size(); i++) {
        PrintOutResult("dparamsOut[" + std::to_string(i) + "]", paramsListShape[i], &dparamsListDeviceAddr[i]);
    }

    // 6. Release aclTensor and aclTensorList.
    aclDestroyTensor(input);
    aclDestroyTensorList(params);
    aclDestroyTensor(dy);
    aclDestroyTensor(dh);
    aclDestroyTensor(hx);

    aclDestroyTensorList(r);
    aclDestroyTensorList(z);
    aclDestroyTensorList(n);
    aclDestroyTensorList(hn);
    aclDestroyTensorList(h);

    aclDestroyTensor(dxOut);
    aclDestroyTensor(dhPrevOut);
    aclDestroyTensorList(dparamsOut);

    // 7. Release device resources.
    aclrtFree(inputDeviceAddr);
    for (size_t i = 0; i < paramsListShape.size(); i++) {
        aclrtFree(paramsListDeviceAddr[i]);
    }
    aclrtFree(dyDeviceAddr);
    aclrtFree(dhDeviceAddr);
    aclrtFree(hxDeviceAddr);

    for (size_t i = 0; i < rDeviceAddr.size(); i++) {
        aclrtFree(rDeviceAddr[i]);
        aclrtFree(zDeviceAddr[i]);
        aclrtFree(nDeviceAddr[i]);
        aclrtFree(hnDeviceAddr[i]);
        aclrtFree(hDeviceAddr[i]);
    }

    aclrtFree(dxDeviceAddr);
    aclrtFree(dhPrevDeviceAddr);
    for (size_t i = 0; i < dparamsListDeviceAddr.size(); i++) {
        aclrtFree(dparamsListDeviceAddr[i]);
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
