# aclnnSwigluGroupGrad

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T15:22:59.404Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
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

- Description: The SwigluGroupGrad operator implements the backward propagation computation of the SwiGLU activation function. It computes the input gradient `grad_x` and the weight gradient `grad_weight`.

-  Formula

-  Basic computation flow

```text
Step 0: GroupIndex processing (optional) → compute trunc
Step 1: Input splitting (split x into x0 and x1)
Step 2: Clamp processing (optional)
Step 3: SwiGLU backward propagation computation
Step 4: Weight gradient computation (optional)
Step 5: Gradient concatenation output
```

Step 0: GroupIndex processing (optional)

When `groupIndex` is provided, it is used to dynamically compute the number of tokens actually processed:

$$
\text{trunc} = \sum_{g=0}^{G-1} \text{groupIndex}[g]
$$

Where:
- $G$ is the number of MoE expert groups
- All subsequent steps process only the first $\text{trunc}$ rows of data

**MoE use case description**: In MoE inference backward propagation, different experts may process different numbers of tokens. groupIndex allows dynamic adjustment of the processing range.

Step 1: Input Splitting

The input tensor $\mathbf{x} \in \mathbb{R}^{T \times 2H}$ or $\mathbb{R}^{B \times S \times 2H}$ is split into two parts along the last dimension:

$$
\mathbf{x}_0[t, h] = \mathbf{x}[t, h], \quad h \in [0, H)

$$
\mathbf{x}_1[t, h] = \mathbf{x}[t, h + H], \quad h \in [0, H)

Step 2: Clamp Processing (Optional)

When `clamp_limit > 0`, the input is clamped:

$$
\mathbf{x}_0'[t, h] = \min(\mathbf{x}_0[t, h], c)

$$
\mathbf{x}_1'[t, h] = \min(\max(\mathbf{x}_1[t, h], -c), c)

where $c$ is `clamp_limit`.

Step 3: SwiGLU backward propagation computation

**Forward propagation review**:
$$
\text{Swish}(\mathbf{x}_0) = \mathbf{x}_0 \cdot \sigma(\mathbf{x}_0)
$$

$$
\mathbf{y}_{\text{swiglu}} = \text{Swish}(\mathbf{x}_0) \cdot \mathbf{x}_1
$$

**Backward Propagation Gradient Calculation**:

First, calculate the Sigmoid:

$$
\sigma(\mathbf{x}_0') = \frac{1}{1 + e^{-\mathbf{x}_0'}}
$$

Then, calculate the SiLU:

$$
\text{SiLU}(\mathbf{x}_0') = \mathbf{x}_0' \cdot \sigma(\mathbf{x}_0')
$$

**SiLU Gradient Formula**:

$$
\frac{d\text{SiLU}}{d\mathbf{x}_0'} = \sigma(\mathbf{x}_0') \cdot \left(1 + \mathbf{x}_0' \cdot (1 - \sigma(\mathbf{x}_0'))\right)
$$

**Input gradient computation**:

$$
\mathbf{grad}_{x_0}[t, h] = \mathbf{grad}_{y_0}[t, h] \cdot \mathbf{x}_1'[t, h] \cdot \frac{d\text{SiLU}}{d\mathbf{x}_0'}[t, h]
$$

$$
\mathbf{grad}_{x_1}[t, h] = \mathbf{grad}_{y_0}[t, h] \cdot \text{SiLU}(\mathbf{x}_0'[t, h])
$$

Where:
- $\mathbf{grad}_{y_0}$ is the processed gradient input (see Step 4)
- If weight is provided, $\mathbf{grad}_{y_0} = \mathbf{grad}_{\text{output}} \cdot \mathbf{weight}$
- If weight is not provided, $\mathbf{grad}_{y_0} = \mathbf{grad}_{\text{output}}$

Step 4: Weight Gradient Computation (Optional)

When `weight` is provided, compute the weight gradient:

$$
\mathbf{grad}_{\text{weight}}[t] = \sum_{h=0}^{H-1} \mathbf{grad}_{\text{output}}[t, h] \cdot \mathbf{y}_{\text{origin}}[t, h]
$$

Where:
- $\mathbf{y}_{\text{origin}}$ is the original activation output of SwiGLU forward propagation
- Summed along the last dimension (H dimension)

**GroupIndex handling**:

When `groupIndex` is provided, only the weight gradients of the first $\text{trunc}$ rows are valid:

$$
\mathbf{grad}_{\text{weight}}[t] = \mathbf{grad}_{\text{weight}}[t] \cdot \mathbb{I}(t < \text{trunc})
$$

where $\mathbb{I}$ is the indicator function.

Step 5: Clamp backward propagation (optional)

When `clamp_limit > 0`, apply the gradient mask for Clamp:

**Clamp mask of x0**:

$$
\mathbf{m}_{x_0}[t, h] = \mathbb{I}(\mathbf{x}_0[t, h] < c)
$$

**Clamp mask of x1**:

$$
\mathbf{m}_{x_1}[t, h] = \mathbb{I}(-c < \mathbf{x}_1[t, h] < c)
$$

**Gradient mask application**:

$$
\mathbf{grad}_{x_0}[t, h] = \mathbf{grad}_{x_0}[t, h] \cdot \mathbf{m}_{x_0}[t, h]
$$

$$
\mathbf{grad}_{x_1}[t, h] = \mathbf{grad}_{x_1}[t, h] \cdot \mathbf{m}_{x_1}[t, h]
$$

Step 6: Gradient Concatenation and GroupIndex Processing

**Gradient concatenation**:

$$
\mathbf{grad}_x[t, h] = \begin{cases}
\mathbf{grad}_{x_0}[t, h] & h \in [0, H) \\
\mathbf{grad}_{x_1}[t, h-H] & h \in [H, 2H)
\end{cases}
$$

**GroupIndex processing**:

When `groupIndex` is provided, only gradients in the first $\text{trunc}$ rows are valid:

$$
\mathbf{grad}_x[t, :] = \mathbf{grad}_x[t, :] \cdot \mathbb{I}(t < \text{trunc})
$$



## Function Prototype

Each operator consists of a two-phase API. You must first call the "aclnnSwigluGroupGradGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnSwigluGroupGrad" API to perform the computation.

```cpp
aclnnStatus aclnnSwigluGroupGradGetWorkspaceSize(
  const aclTensor *gradY,
  const aclTensor *x,
  const aclTensor *weightOptional,
  const aclTensor *yOriginOptional,
  const aclTensor *groupIndexOptional,
  float            clampLimit,
  aclTensor       *gradXOut,
  aclTensor       *gradWeightOutOptional,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnSwigluGroupGrad(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnSwigluGroupGradGetWorkspaceSize

- **Parameters**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 350px">
  <col style="width: 220px">
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 100px">
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
      <td>gradY (aclTensor*)</td>
      <td>Input</td>
      <td>Gradient output tensor, which is the gradient from the downstream layer.</td>
      <td><ul><li>shape=[..., H]. Common shapes are [T, H] or [B, S, H].</li><li>Empty tensors are supported. Returns directly when the tensor is empty.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Input tensor for forward propagation.</td>
      <td><ul><li>shape=[..., 2H]. Common shapes are [T, 2H] or [B, S, 2H]. The last dimension must be an even number.</li><li>Empty tensors are supported. The API returns directly when the tensor is empty.</li></ul></td>      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightOptional (aclTensor*)</td>
      <td>Input (optional)</td>
      <td>MoE weight tensor.</td>
      <td><ul><li>The number of elements equals the number of elements in gradY excluding the last dimension.</li><li>Must be provided together with yOriginOptional.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>yOriginOptional (aclTensor*)</td>
      <td>Input (optional)</td>
      <td>SwiGLU forward output. If weight exists, this output is multiplied by weight.</td>
      <td><ul><li>shape=[..., H]. Common shapes are [T, H] or [B, S, H].</li><li>Must be provided together with weightOptional.</li><li>The number of non-trailing axis elements equals that of gradY.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndexOptional (aclTensor*)</td>
      <td>Input (optional)</td>
      <td>GroupIndex tensor for dynamic kernel allocation.</td>
      <td><ul><li>shape=[G], G > 0.</li><li>Data type: INT64.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1 ([G])</td>
      <td>√</td>
    </tr>
    <tr>
      <td>clampLimit (float)</td>
      <td>Input</td>
      <td>Clamp threshold.</td>
      <td><ul><li>Value range: -1.0 or >0.0.</li><li>clampLimit=-1.0 indicates that the Clamp backward propagation mask is disabled. If enabled, clampLimit must be >0.0.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradXOut (aclTensor*)</td>
      <td>Output</td>
      <td>Input gradient tensor.</td>
      <td><ul><li>shape=[..., 2H]. Common shapes are [T, 2H] or [B, S, 2H], consistent with x.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeightOutOptional (aclTensor*)</td>
      <td>Output (optional)</td>
      <td>Weight gradient tensor.</td>
      <td><ul><li>Output when both weightOptional and yOriginOptional are provided.</li><li>shape is consistent with weightOptional.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>Same as weightOptional</td>
      <td>√</td>
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
      <td>Returns an op executor containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

aclnnStatus: Returns a status code. For details, see [aclnn Return Codes](../../../docs/zh/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>Required parameter gradY, x, or gradXOut is nullptr.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>clampLimit is not -1.0 or greater than 0.0 (including NaN and 0.0).</td>
    </tr>
    <tr>
      <td>The last dimension of x is not twice the last dimension of gradY.</td>
    </tr>
    <tr>
      <td>The total number of non-trailing axis elements in gradY is not equal to that in x.</td>
    </tr>
    <tr>
      <td>The parameter shape is invalid.</td>
    </tr>
    <tr>
      <td>The parameter data types do not match.</td>
    </tr>
  </tbody></table>

## aclnnSwigluGroupGrad

- **Parameters**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr><th>Parameter</th><th>Input/Output</th><th>Description</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>Input</td><td>Workspace memory address requested on the Device.</td></tr>
    <tr><td>workspaceSize</td><td>Input</td><td>Workspace size requested on the Device, obtained by calling the first-phase API aclnnSwigluGroupGradGetWorkspaceSize.</td></tr>
    <tr><td>executor</td><td>Input</td><td>Op executor that contains the operator computation flow.</td></tr>
    <tr><td>stream</td><td>Input</td><td>Specifies the Stream for executing tasks.</td></tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Determinism: aclnnSwigluGroupGrad defaults to a deterministic implementation.
- H must be greater than 0.
- The last dimension of x must be an even number (2H). The last dimension of gradY is H, which is half of the last dimension of x.
- The trailing axes of gradY and x satisfy x.shape[-1] == 2 * gradY.shape[-1], and the total numbers of non-trailing axis elements are equal.
- weightOptional and yOriginOptional must be provided together or left empty together; the number of elements in weightOptional is equal to the number of non-trailing axis elements in gradY; the last dimension of yOriginOptional is the same as that of gradY, and the number of non-trailing axis elements is equal to that of gradY.
- When groupIndexOptional is non-empty, it must be a 1-D non-empty tensor.
- gradY, x, yOriginOptional, and gradXOut must have the same data type (FLOAT, FLOAT16, or BFLOAT16).
- weightOptional and gradWeightOutOptional must be of the FLOAT type.
- groupIndexOptional must be of the INT64 type.
- The value of clampLimit is -1.0 or greater than 0.0. A value of -1.0 indicates that the Clamp backward propagation mask is disabled. When the mask is enabled, clampLimit must be greater than 0.0.

## Example

The sample code is for reference only. For details about how to compile and run the sample, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<term>Ascend 950PR&950DT products</term>:

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_group_grad.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor, aclFormat format)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(),
                              shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. Initialize Device and Stream.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Create the input tensor.
    int64_t T = 2;
    int64_t H = 256;
    int64_t twoH = 2 * H;

    std::vector<int64_t> gradYShape = {T, H};
    std::vector<int64_t> xShape = {T, twoH};
    std::vector<int64_t> weightShape = {T, 1};
    std::vector<int64_t> yOriginShape = {T, H};

    void *gradYDeviceAddr = nullptr;
    void *xDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *yOriginDeviceAddr = nullptr;
    aclTensor *gradY = nullptr;
    aclTensor *x = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *yOrigin = nullptr;

    std::vector<uint16_t> hostGradY(GetShapeSize(gradYShape), 1);
    std::vector<uint16_t> hostX(GetShapeSize(xShape), 1);
    std::vector<float> hostWeight(GetShapeSize(weightShape), 1.0f);
    std::vector<uint16_t> hostYOrigin(GetShapeSize(yOriginShape), 1);

    ret = CreateAclTensor(hostGradY, gradYShape, &gradYDeviceAddr, aclDataType::ACL_BF16, &gradY,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostX, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostWeight, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostYOrigin, yOriginShape, &yOriginDeviceAddr, aclDataType::ACL_BF16, &yOrigin,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Create the output tensor.
    std::vector<int64_t> gradXOutShape = {T, twoH};
    std::vector<int64_t> gradWeightOutShape = {T, 1};

    void *gradXOutDeviceAddr = nullptr;
    void *gradWeightOutDeviceAddr = nullptr;
    aclTensor *gradXOut = nullptr;
    aclTensor *gradWeightOut = nullptr;

    std::vector<uint16_t> hostGradXOut(GetShapeSize(gradXOutShape), 0);
    std::vector<float> hostGradWeightOut(GetShapeSize(gradWeightOutShape), 0.0f);

    ret = CreateAclTensor(hostGradXOut, gradXOutShape, &gradXOutDeviceAddr, aclDataType::ACL_BF16, &gradXOut,
                          aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(hostGradWeightOut, gradWeightOutShape, &gradWeightOutDeviceAddr,
                          aclDataType::ACL_FLOAT, &gradWeightOut, aclFormat::ACL_FORMAT_ND);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 4. Obtain the workspace size.
    float clampLimit = -1.0f;
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    ret = aclnnSwigluGroupGradGetWorkspaceSize(gradY, x, weight, yOrigin, nullptr, clampLimit, gradXOut,
                                               gradWeightOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 5. Request the workspace and perform computation.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnSwigluGroupGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupGrad failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 6. Release resources.
    aclDestroyTensor(gradY);
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(yOrigin);
    aclDestroyTensor(gradXOut);
    aclDestroyTensor(gradWeightOut);
    aclrtFree(gradYDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(yOriginDeviceAddr);
    aclrtFree(gradXOutDeviceAddr);
    aclrtFree(gradWeightOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
