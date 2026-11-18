# aclnnApplyAdamV2

## 产品支持情况

以下支持情况适用于本仓提供的`aclnnApplyAdamV2`接口。ApplyAdamV2算子的其他调用方式请参见[算子说明](../README.md)。

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id4 -->
<!-- npu="310b" id5 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id5 -->
<!-- npu="310p" id6 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id6 -->
<!-- npu="910" id7 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id7 -->

## 功能说明

- **接口功能：** 实现Adam（Adaptive Moment Estimation，自适应矩估计）优化器的参数更新，支持`adam`和`mbart_adam`两种模式。`varRef`、`mRef`、`vRef`均原地更新，分别保存更新后的模型参数、一阶矩和二阶矩估计。
- **计算公式：** 记输入模型参数、一阶矩估计、二阶矩估计和梯度分别为$\mathrm{var}$、$m$、$v$、$g$，学习率为$\mathrm{lr}$，权重衰减系数为$\mathrm{wd}$。$\beta_1$、$\beta_2$为矩估计的衰减率，$\epsilon$为数值稳定性常数，$s$为更新步长。

  `adam`模式下，仅当提供`maxGradNormOptional`、其值大于0，且`globalGradNorm`为有限值并大于1时，使用$\tilde{g} = g / \mathrm{globalGradNorm}$；其他情况使用$\tilde{g} = g$。`globalGradNorm`为非数值（NaN）、无穷大（Inf）或小于`1e-10`时不缩放梯度。`maxGradNormOptional`用作裁剪开关，其数值不作为缩放分子。

  `mbart_adam`模式下，使用$\tilde{g} = g$，不进行上述梯度缩放。

  以下公式均按元素计算。两种模式均先计算：

  $$
  m_{\mathrm{new}} = \beta_1 \cdot m + (1 - \beta_1) \cdot \tilde{g}
  $$

  $$
  v_{\mathrm{new}} = \beta_2 \cdot v + (1 - \beta_2) \cdot \tilde{g}^2
  $$

  $$
  u = \frac{m_{\mathrm{new}}}{\sqrt{v_{\mathrm{new}}} + \epsilon}
  $$

  `adam`模式更新为：

  $$
  \mathrm{var}_{\mathrm{new}} = \mathrm{var} - \mathrm{lr} \cdot \left(u + \mathrm{wd} \cdot \mathrm{var}\right)
  $$

  `mbart_adam`模式更新为：

  $$
  \mathrm{var}_{\mathrm{new}} = \mathrm{var} - \left(s \cdot u + \mathrm{wd} \cdot (\mathrm{lr} \cdot \mathrm{var})\right)
  $$

  `weightDecay`为0时跳过衰减项。更新步长$s$由`stepSizeOptional`指定，仅在`mbart_adam`模式使用，缺省时取`lr`；在`adam`模式不参与计算。接口内部不计算矩估计的偏差修正，需要时由调用方计算相应的更新系数。

## 函数原型

本算子采用[两段式接口](../../../docs/zh/context/two_phase_api.md)：先调用`aclnnApplyAdamV2GetWorkspaceSize`获取workspace大小和执行器，再调用`aclnnApplyAdamV2`提交计算。

头文件为`aclnnop/aclnn_apply_adam_v2.h`，也可使用算子仓总头文件`aclnnop/aclnn_ops_nn.h`；链接库为`libopapi_nn.so`。

```cpp
aclnnStatus aclnnApplyAdamV2GetWorkspaceSize(
    aclTensor *varRef,
    aclTensor *mRef,
    aclTensor *vRef,
    const aclTensor *lr,
    const aclTensor *beta1,
    const aclTensor *beta2,
    const aclTensor *epsilon,
    const aclTensor *grad,
    const aclTensor *maxGradNormOptional,
    const aclTensor *globalGradNorm,
    const aclTensor *weightDecay,
    const aclTensor *stepSizeOptional,
    char *adamModeOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnApplyAdamV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## aclnnApplyAdamV2GetWorkspaceSize

### 参数说明

下列Tensor（张量）均位于Device（设备）侧，数据格式为[ND](../../../docs/zh/context/data_format.md)。所有已提供的Tensor数据类型须与`varRef`一致，支持FLOAT16（16位浮点数）和FLOAT32（32位浮点数）。shape表示形状，dtype表示数据类型；标量参数以Tensor传入，shape为`[]`或`[1]`。

| 参数名 | 输入/输出 | 描述与约束 |
|:-------|:----------|:-----------|
| varRef（aclTensor*） | 输入/输出 | 待更新的模型参数。维度数支持0至8，计算后原地保存$\mathrm{var}_{\mathrm{new}}$。 |
| mRef（aclTensor*） | 输入/输出 | 一阶矩估计，与`varRef`的shape、dtype一致，计算后原地保存$m_{\mathrm{new}}$。 |
| vRef（aclTensor*） | 输入/输出 | 二阶矩估计，与`varRef`的shape、dtype一致，计算后原地保存$v_{\mathrm{new}}$。 |
| lr（const aclTensor*） | 输入 | 学习率标量。 |
| beta1（const aclTensor*） | 输入 | 一阶矩衰减率标量。 |
| beta2（const aclTensor*） | 输入 | 二阶矩衰减率标量。 |
| epsilon（const aclTensor*） | 输入 | 数值稳定性常数标量。 |
| grad（const aclTensor*） | 输入 | 梯度，与`varRef`的shape、dtype一致。 |
| maxGradNormOptional（const aclTensor*） | 输入 | 可选的梯度裁剪控制标量。传`nullptr`时不进行梯度缩放，仅在`adam`模式使用。 |
| globalGradNorm（const aclTensor*） | 输入 | 全局梯度范数标量。必选输入，仅在`adam`模式启用裁剪时参与计算。 |
| weightDecay（const aclTensor*） | 输入 | 权重衰减系数标量。 |
| stepSizeOptional（const aclTensor*） | 输入 | 可选的更新步长标量。`mbart_adam`模式下传`nullptr`时使用`lr`；`adam`模式忽略其数值。 |
| adamModeOptional（char*） | 输入 | 以空字符结尾的模式字符串，仅支持`"adam"`、`"mbart_adam"`；传`nullptr`时默认`"adam"`。 |
| workspaceSize（uint64_t*） | 输出 | 返回计算所需的Device侧workspace字节数，不能为`nullptr`。 |
| executor（aclOpExecutor**） | 输出 | 返回包含算子计算流程的执行器，不能为`nullptr`。 |

### 返回值

返回`aclnnStatus`，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

- `ACLNN_SUCCESS`（0）：第一段接口执行成功。
- `ACLNN_ERR_PARAM_NULLPTR`（161001）：必选计算输入或执行器输出指针为空。
- 其他非零返回值：参数校验、算子匹配或tiling等阶段执行失败，按返回码和错误日志定位。第一段接口成功不代表Device计算已完成。

## aclnnApplyAdamV2

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|:-------|:----------|:-----|
| workspace（void*） | 输入 | Device侧workspace地址。第一段返回的大小为0时可传`nullptr`；否则须分配不少于`workspaceSize`字节的内存。 |
| workspaceSize（uint64_t） | 输入 | 第一段接口返回的workspace大小。 |
| executor（aclOpExecutor*） | 输入 | 第一段接口返回的执行器。 |
| stream（aclrtStream） | 输入 | 执行计算的Stream（流）。 |

### 返回值

返回`aclnnStatus`，0表示任务提交成功。接口采用异步执行方式，读取更新后的`varRef`、`mRef`、`vRef`之前须等待对应Stream完成；计算完成前须保持Tensor数据和workspace有效。

## 约束说明

- `varRef`、`mRef`、`vRef`和`grad`的shape、dtype必须完全一致，不支持这四个Tensor之间的广播。
- 支持FLOAT16和FLOAT32，不支持BFLOAT16。所有已提供的标量Tensor均使用与`varRef`相同的dtype。
- 所有输入位于同一Device。调用本接口时须提供已确定的非负shape，不能将图引擎（Graph Engine，GE）图中的未知维度`-1`或未知维度数`-2`直接传入本接口。
- 支持合法元数据的空Tensor，三个状态Tensor的shape保持不变，不进行数值计算。
- `maxGradNormOptional`和`stepSizeOptional`均可缺省，但`globalGradNorm`和`weightDecay`仍为必选Tensor。
- `varRef`、`mRef`、`vRef`是三个独立的状态量，调用方应提供独立的存储，避免与其他计算输入发生存储重叠。
- 数值范围遵循[ApplyAdamV2约束说明](../README.md#数值范围)。其中`vRef`表示非负二阶矩，`mRef`和`grad`允许为负值；调用方应保证参数和中间结果符合Adam更新的数值定义域。
- 本接口默认采用确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_apply_adam_v2.h"

namespace {
struct Resources {
    int32_t deviceId = 0;
    bool initialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    std::vector<void*> allocations;
    std::vector<aclTensor*> tensors;

    ~Resources()
    {
        if (stream != nullptr) {
            (void)aclrtSynchronizeStream(stream);
        }
        for (auto* tensor : tensors) {
            (void)aclDestroyTensor(tensor);
        }
        for (auto* allocation : allocations) {
            (void)aclrtFree(allocation);
        }
        if (stream != nullptr) {
            (void)aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            (void)aclrtResetDevice(deviceId);
        }
        if (initialized) {
            (void)aclFinalize();
        }
    }
};

bool CheckStatus(int status, const char* operation)
{
    if (status != 0) {
        std::fprintf(stderr, "%s failed: %d\n", operation, status);
        return false;
    }
    return true;
}

bool CreateTensor(Resources& resources, const std::vector<float>& values, const std::vector<int64_t>& shape,
                  aclTensor** tensor)
{
    const size_t bytes = values.size() * sizeof(float);
    void* data = nullptr;
    if (!CheckStatus(aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc")) {
        return false;
    }
    resources.allocations.push_back(data);
    if (!CheckStatus(aclrtMemcpy(data, bytes, values.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy")) {
        return false;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i) {
        strides[i - 2] = shape[i - 1] * strides[i - 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), data);
    if (*tensor == nullptr) {
        std::fprintf(stderr, "aclCreateTensor failed\n");
        return false;
    }
    resources.tensors.push_back(*tensor);
    return true;
}
} // namespace

int main()
{
    // Initialize the runtime, select a device and create a stream.
    Resources resources;
    if (!CheckStatus(aclInit(nullptr), "aclInit")) {
        return 1;
    }
    resources.initialized = true;
    if (!CheckStatus(aclrtSetDevice(resources.deviceId), "aclrtSetDevice")) {
        return 1;
    }
    resources.deviceSet = true;
    if (!CheckStatus(aclrtCreateStream(&resources.stream), "aclrtCreateStream")) {
        return 1;
    }

    // Create the input tensors in device memory, in API parameter order:
    // var, m, v, lr, beta1, beta2, epsilon, grad, maxGradNorm, globalGradNorm, weightDecay, stepSize.
    const std::array<std::vector<float>, 12> values = {{{2.0f, -4.0f},
                                                        {0.0f, 0.0f},
                                                        {4.0f, 16.0f},
                                                        {0.25f},
                                                        {0.5f},
                                                        {0.5f},
                                                        {0.5f},
                                                        {2.0f, -4.0f},
                                                        {-1.0f},
                                                        {0.0f},
                                                        {0.0f},
                                                        {0.125f}}};
    std::array<aclTensor*, 12> inputs = {};
    for (size_t i = 0; i < inputs.size(); ++i) {
        const std::vector<int64_t> shape = (i < 3 || i == 7) ? std::vector<int64_t>{2} : std::vector<int64_t>{1};
        if (!CreateTensor(resources, values[i], shape, &inputs[i])) {
            return 1;
        }
    }

    // Query the workspace size and execution plan, then allocate workspace if needed.
    char adamMode[] = "adam";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnApplyAdamV2GetWorkspaceSize(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
                                                   inputs[6], inputs[7], inputs[8], inputs[9], inputs[10], inputs[11],
                                                   adamMode, &workspaceSize, &executor);
    if (!CheckStatus(status, "aclnnApplyAdamV2GetWorkspaceSize")) {
        return 1;
    }
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        if (!CheckStatus(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace")) {
            return 1;
        }
        resources.allocations.push_back(workspace);
    }
    // Submit the update and wait for completion before reading the in-place results.
    if (!CheckStatus(aclnnApplyAdamV2(workspace, workspaceSize, executor, resources.stream), "aclnnApplyAdamV2") ||
        !CheckStatus(aclrtSynchronizeStream(resources.stream), "aclrtSynchronizeStream")) {
        return 1;
    }

    const char* names[] = {"var", "m", "v"};
    for (size_t i = 0; i < 3; ++i) {
        std::array<float, 2> result = {};
        if (!CheckStatus(aclrtMemcpy(result.data(), sizeof(result), resources.allocations[i], sizeof(result),
                                     ACL_MEMCPY_DEVICE_TO_HOST),
                         "aclrtMemcpy result")) {
            return 1;
        }
        std::printf("%s = [%.6f, %.6f]\n", names[i], result[0], result[1]);
    }
    return 0;
}
```
