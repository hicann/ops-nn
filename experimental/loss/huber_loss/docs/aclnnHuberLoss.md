# aclnnHuberLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品</term>     |     √    |

## 功能说明

计算 `input` 与 `target` 之间的 Huber 损失，对齐 PyTorch `aten::huber_loss` 前向语义，支持 `none`/`mean`/`sum` 三种规约模式。

设 $e = input - target$，逐元素损失为：

$$
l =
\begin{cases}
0.5e^2, & |e| \leq delta \\
delta(|e| - 0.5delta), & |e| > delta
\end{cases}
$$

再按 `reduction` 规约：

$$
loss =
\begin{cases}
l, & reduction = 0\ (none) \\
\frac{1}{N}\sum l, & reduction = 1\ (mean) \\
\sum l, & reduction = 2\ (sum)
\end{cases}
$$

`FLOAT16` 与 `BFLOAT16` 在 Kernel 内部提升为 `FLOAT` 计算，末端一次舍入回输出数据类型；`mean`/`sum` 的累加同样在 `FLOAT` 域完成。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnHuberLossGetWorkspaceSize”接口获取计算所需 workspace 大小以及包含算子计算流程的执行器，再调用“aclnnHuberLoss”接口执行计算。

```Cpp
aclnnStatus aclnnHuberLossGetWorkspaceSize(
    const aclTensor* input,
    const aclTensor* target,
    int64_t          reduction,
    double           delta,
    const aclTensor* out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnHuberLoss(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnHuberLossGetWorkspaceSize

- **参数说明**：

  <table class="tg" style="undefined;table-layout: fixed; width: 1450px"><colgroup>
  <col style="width: 220px">
  <col style="width: 110px">
  <col style="width: 250px">
  <col style="width: 410px">
  <col style="width: 180px">
  <col style="width: 100px">
  <col style="width: 180px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input（aclTensor*）</td>
      <td>输入</td>
      <td>预测值输入。</td>
      <td>shape 与数据类型需与 `target` 完全一致。支持非连续张量，由框架完成连续化。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>任意维，含 0 维与空张量</td>
    </tr>
    <tr>
      <td>target（aclTensor*）</td>
      <td>输入</td>
      <td>目标值输入。</td>
      <td>shape 与数据类型需与 `input` 完全一致。支持非连续张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>与 `input` 一致</td>
    </tr>
    <tr>
      <td>reduction（int64_t）</td>
      <td>输入</td>
      <td>规约模式。</td>
      <td>取值仅支持 `0`（none，不规约）、`1`（mean，求均值）、`2`（sum，求和）。算子属性默认值为 `1`；ACLNN 接口调用时需显式传入。<b>注意：同目录 SmoothL1LossV2 使用 1=sum、2=mean 的相反约定。</b></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>delta（double）</td>
      <td>输入</td>
      <td>Huber 损失的分段阈值。</td>
      <td>必须大于 `0`，允许 `+∞`（此时公式退化为 `0.5e²`）。内部窄化为 `FLOAT`，因此正但窄化后为 `0` 的取值（如 `1e-300`）会在窄化之后被拒绝。算子属性默认值为 `1.0`。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>Huber 损失。</td>
      <td>数据类型需与 `input`、`target` 一致。`reduction=0` 时 shape 与 `input` 一致；`reduction=1/2` 时为 0 维标量（rank 为 `0`），也兼容 shape 为 `{1}` 的 1 维张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>见使用说明</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回 op 执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值**：

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，报错场景包括：

  - `input`、`target` 或 `out` 是空指针。
  - `input`、`target` 或 `out` 的数据类型不在支持范围内，或三者数据类型不一致。
  - `input` 与 `target` 的 shape 不一致。
  - `out` 的 shape 与 `reduction` 不匹配：`reduction=0` 时 `out` 必须与 `input` 同 shape，`reduction=1/2` 时 `out` 必须是 1 个元素且 rank ≤ 1。
  - `reduction` 不是 `0`、`1`、`2` 之一。
  - `delta` 小于或等于 `0`，或为 `NaN`，或窄化到 `FLOAT` 后为 `0`。

## aclnnHuberLoss

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 200px">
    <col style="width: 150px">
    <col style="width: 800px">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在 Device 侧申请的 workspace 内存地址。第一段接口返回的大小 = 算子自身所需 + 框架保留的系统 workspace，后者无条件叠加，因此三种 `reduction` 下返回值均不为 `0`。算子自身的那部分仅 `reduction=1/2` 非零，用于承载跨核归约的中间结果。<b>一律以第一段接口返回值为准，不要按 `reduction` 自行假设。</b></td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnHuberLossGetWorkspaceSize 获取。</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op 执行器，包含了算子计算流程。</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的 Stream。</td>
      </tr>
    </tbody>
  </table>

- **返回值**：

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 输入张量 `input`、`target` 的 shape 必须完全一致，不支持广播。
- `input`、`target` 与 `out` 的数据类型必须一致，仅支持 `FLOAT`、`FLOAT16`、`BFLOAT16`，数据格式仅支持 `ND`。
- `reduction` 仅支持 `0`/`1`/`2`，算子属性默认值为 `1`（mean）。
- `delta` 必须大于 `0`，算子属性默认值为 `1.0`。
- 支持动态 rank 和动态 shape，支持非连续输入。
- `reduction=1/2` 需要算子自身的 workspace，并启用 `BATCH_MODE` 调度以保证参与跨核归约的各核共驻；`reduction=0` 不需要，但第一段接口返回的大小仍包含框架保留的系统 workspace。
- 空张量：`sum` 返回 `0`，`mean` 返回 `NaN`（`0/0`），与 PyTorch 一致。
- 半精度累加在 `FLOAT` 域完成，仅在写出时舍入一次。因此 `FLOAT16`/`BFLOAT16` 的结果可能与 PyTorch CPU 实现不逐位相同——后者在原生数据类型下逐步舍入。
- 仅支持 Atlas A2 训练系列产品。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。仓库中可运行的完整样例参见 [test_aclnn_huber_loss.cpp](../examples/test_aclnn_huber_loss.cpp)。

```Cpp
#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnn_huber_loss.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
  int64_t size = 1;
  for (int64_t dim : shape) {
    size *= dim;
  }
  return size;
}

int CreateAclTensor(
  const std::vector<float>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclTensor** tensor)
{
  const size_t bytes = static_cast<size_t>(GetShapeSize(shape)) * sizeof(float);
  auto ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtMemcpy(*deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(
    shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtCreateStream(&stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  const std::vector<int64_t> shape = {7};
  const std::vector<float> inputHost = {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0};
  const std::vector<float> targetHost(inputHost.size(), 0.0);
  constexpr int64_t reduction = 1; // mean
  constexpr double delta = 1.0;

  // reduction=1/2 输出为 0 维标量：shape 长度为 0，元素个数为 1。
  const std::vector<int64_t> lossShape = {};
  std::vector<float> lossHost(1, 0.0);

  void* inputDevice = nullptr;
  void* targetDevice = nullptr;
  void* lossDevice = nullptr;
  aclTensor* input = nullptr;
  aclTensor* target = nullptr;
  aclTensor* loss = nullptr;
  ret = CreateAclTensor(inputHost, shape, &inputDevice, &input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(targetHost, shape, &targetDevice, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(lossHost, lossShape, &lossDevice, &loss);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnHuberLossGetWorkspaceSize(input, target, reduction, delta, loss, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  void* workspace = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  ret = aclnnHuberLoss(workspace, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtMemcpy(
    lossHost.data(), sizeof(float), lossDevice, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::printf("NPU loss (mean): %.6f\n", lossHost[0]);

  aclDestroyTensor(input);
  aclDestroyTensor(target);
  aclDestroyTensor(loss);
  aclrtFree(inputDevice);
  aclrtFree(targetDevice);
  aclrtFree(lossDevice);
  if (workspace != nullptr) {
    aclrtFree(workspace);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
