# aclnnSleep

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/control/sleep)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：让NPU设备休眠指定的时钟周期数。该算子通过SIMT`clock()`忙等待实现精确的延时控制，语义与CUDA的`spin_kernel`/`torch.cuda._sleep`一致。

- 计算公式：

  对于给定的时钟周期数cycles，aclnnSleep执行以下计算：

  1. 获取当前clock计数器值作为起始时间：

     $$
     start = clock()
     $$

  2. 忙等待直到经过的cycle数达到指定值：

     $$
     while(clock() - start < cycles): spin
     $$

  3. 实际休眠时间（秒）与clock频率相关：

     $$
     t_{sleep} = \frac{cycles}{f_{clock}}
     $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnSleepGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSleep”接口执行计算。

> **说明**：`cycles` 参数为 `aclIntArray*` 类型，直接传值，数组包含1个元素即休眠周期数。

```Cpp
aclnnStatus aclnnSleepGetWorkspaceSize(
  const aclIntArray* cycles,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnSleep(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnSleepGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
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
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>cycles（aclIntArray*）</td>
      <td>输入</td>
      <td>休眠的时钟周期数。因声明值依赖，aclnn接口层转换为aclIntArray*传值。</td>
      <td><ul><li>必须为正整数（cycles &gt; 0）。</li><li>不支持空数组。</li><li>受AICore超时限制，cycles最大值约为1.782e12（Ascend 950PR/Ascend 950DT主频1.65GHz，约18分钟），详见约束说明。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>[1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>当前实现返回0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入数据类型不在支持范围之内（非INT64）。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="2">561002</td>
      <td>cycles为空数组，tiling阶段获取数据失败。</td>
    </tr>
    <tr>
      <td>cycles参数非正数（cycles &lt;= 0）。</td>
    </tr>
  </tbody></table>

## aclnnSleep

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
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
      <td>在Device侧申请的workspace内存地址。当前实现无需workspace，可传入nullptr。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSleepGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：aclnnSleep默认确定性实现。
- 输入限制：cycles必须为正整数（cycles &gt; 0），否则返回ACLNN_ERR_INNER_TILING_ERROR（561002）。
- cycles以aclIntArray*传入，数组包含1个元素即休眠周期数。
- 不支持空数组：cycles为空数组时，返回ACLNN_ERR_INNER_TILING_ERROR（561002）。
- 输入不支持包含±inf或nan：cycles为INT64类型，本身不存在inf/nan。若上游通过浮点类型转换传入（如PyTorch的`.to(torch.int64)`），`+inf`会转换为INT64_MAX，cycles校验通过但运行时会触发AICore超时；`-inf`和`nan`转换为非正数，tiling阶段拒绝并返回ACLNN_ERR_INNER_TILING_ERROR（561002）。
- AICore超时限制：AICore默认执行超时时间为18分钟。在Ascend 950PR/Ascend 950DT（主频1.65GHz）下，cycles的最大值约为1.782e12（= 1080s × 1.65GHz）。超出此值可能导致算子被强制终止，请根据实际场景合理设置cycles。如需更长的休眠时间，可通过`aclrtSetOpExecuteTimeOut`接口修改AICore超时配置来调整此限制。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

<term>Ascend 950PR/Ascend 950DT</term>：

```Cpp

#include <iostream>
#include "acl/acl.h"
#include "aclnnop/aclnn_sleep.h"

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

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  int64_t cyclesValue = 1000000;
  LOG_PRINT("Testing aclnnSleep with cycles = %ld\n", cyclesValue);

  aclIntArray* cycles = aclCreateIntArray(&cyclesValue, 1);
  CHECK_RET(cycles != nullptr, LOG_PRINT("aclCreateIntArray failed.\n"); return ACL_ERROR_BAD_ALLOC);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnSleepGetWorkspaceSize(cycles, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSleepGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnSleep(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSleep failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("aclnnSleep test passed.\n");

  aclDestroyIntArray(cycles);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
