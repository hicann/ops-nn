# aclnnAssertAsync

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：断言输入张量 `inputCondition`（要求 `numel == 1`）的数值非 0。若其数值为 0（含 bool 的 `False`），
  设备侧打印 `assertMsg` 指定的消息并触发算子失败；数值非 0 直接通过。本接口**无输出张量**，
  语义对齐 PyTorch 的 `torch._assert_async`（含 `.msg` 重载语义），可用于在设备侧异步校验条件值。

- 判定规则（bit-exact，无浮点计算）：

  ```text
  inputCondition 的单一元素值为 0（含 bool False）  ->  断言失败：打印 assertMsg + 触发算子失败
  inputCondition 的单一元素值非 0（含 bool True）   ->  断言通过：无任何输出 / 无副作用
  ```

  - 浮点类型（FLOAT16、FLOAT、DOUBLE、BFLOAT16）按符号位掩码判定：FLOAT 取 `& 0x7FFFFFFF`，FLOAT16/BFLOAT16 取 `& 0x7FFF`，
    DOUBLE 取 `& 0x7FFFFFFFFFFFFFFF`；`+0.0` 与 `-0.0` 均判失败，`NaN`、`+inf`、`-inf` 判通过。
  - 复数类型（COMPLEX64、COMPLEX128）实部与虚部精确同时为 0（含 `-0.0` 分量）判失败；`NaN`/`±inf` 分量通过。
  - 整型与布尔类型判全字节零：`0`（含 bool 的 `False`）判失败，非 0（含 `True`）判通过。

- 失败通路：kernel 内以 `AscendC::printf` 打印 `"<assertMsg>\n"` 后调用 `AscendC::Trap()`
  中断 AI Core，使算子执行失败；失败状态经 stream 同步回传给上层。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnAssertAsyncGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAssertAsync”接口执行计算。

```Cpp
aclnnStatus aclnnAssertAsyncGetWorkspaceSize(
  const aclTensor* inputCondition,
  const char*      assertMsg,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnAssertAsync(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnAssertAsyncGetWorkspaceSize

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
      <td>inputCondition（aclTensor*）</td>
      <td>输入</td>
      <td>待断言的条件张量，对应算子输入 input_condition。数值为 0（含 bool 的 False）判失败。</td>
      <td><ul><li>不支持空Tensor（numel 必须为 1）。</li><li>numel 必须为 1（rank-0 或任意收缩为单元素的 shape），否则返回 ACLNN_ERR_PARAM_INVALID（161002）。</li><li>维度不超过 8 维，否则返回 ACLNN_ERR_PARAM_INVALID（161002）。</li><li>数据类型须在支持范围内，否则返回 ACLNN_ERR_PARAM_INVALID（161002）。</li></ul></td>
      <td>BOOL、INT8、INT16、INT32、INT64、UINT8、FLOAT16、FLOAT、DOUBLE、BFLOAT16、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
      <td>0-8 维（numel == 1）</td>
      <td>√</td>
    </tr>
    <tr>
      <td>assertMsg（char*）</td>
      <td>输入</td>
      <td>断言失败时打印的消息字符串。</td>
      <td><ul><li>可为 nullptr，此时按空字符串 "" 处理。</li><li>为 C 字符串，需以 '\0' 结尾。</li><li>可用长度上限 254 字节（长度须小于 255；缓冲 256 字节，含结尾 '\0'），超出时返回 ACLNN_ERR_PARAM_INVALID（161002），不截断。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>workspaceSize 或 executor 为空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>inputCondition 为空指针。</td>
    </tr>
    <tr>
      <td>inputCondition 的维度超过 8 维。</td>
    </tr>
    <tr>
      <td>inputCondition 的 numel 不为 1（含空Tensor）。</td>
    </tr>
    <tr>
      <td>inputCondition 的数据类型不在支持范围之内。</td>
    </tr>
    <tr>
      <td>assertMsg 的长度超出上限（长度须小于 255 字节，可用长度上限为 254 字节）。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td>561101</td>
      <td>API内部创建 aclOpExecutor 失败。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>API内部转连续（l0op::Contiguous）返回空指针，或内部 L0 启动器添加算子（ADD_TO_LAUNCHER_LIST_AICORE）失败。</td>
    </tr>
  </tbody></table>

## aclnnAssertAsync

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAssertAsyncGetWorkspaceSize获取。</td>
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

  若 `inputCondition` 的数值为 0（含 bool 的 `False`），则断言失败：设备侧打印 `assertMsg` 并触发算子失败，
  需通过 stream 同步（如 `aclrtSynchronizeStream`）感知失败返回。

## 约束说明

- 确定性说明：aclnnAssertAsync默认确定性实现。
- 本接口无输出张量、无浮点计算，判定按行为 bit-exact 进行，不适用输出张量的浮点容差（atol/rtol）精度标准。
- 断言失败发生在设备侧 kernel 内，需通过 stream 同步感知算子失败。
- 本算子仅支持 aclnn 接口直调通路，不支持动态图（动态 shape）场景。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

<term>Ascend 950PR&950DT系列产品</term>：

```Cpp
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_assert_async.h"

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

int Init(int32_t deviceId, aclrtStream* stream)
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = static_cast<int64_t>(hostData.size()) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

static int RunAssertAsyncCase(bool value, const char* assertMsg)
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    void* conditionDeviceAddr = nullptr;
    aclTensor* condition = nullptr;
    ret = CreateAclTensor<uint8_t>({static_cast<uint8_t>(value ? 1 : 0)}, {1}, &conditionDeviceAddr, ACL_BOOL,
                                   &condition);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor failed. ERROR: %d\n", ret); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAssertAsyncGetWorkspaceSize(condition, assertMsg, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAssertAsyncGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnAssertAsync(workspaceAddr, workspaceSize, executor, stream);
    LOG_PRINT("aclnnAssertAsync ret = %d (case value=%d)\n", ret, static_cast<int>(value));

    ret = aclrtSynchronizeStream(stream);
    LOG_PRINT("aclrtSynchronizeStream ret = %d (case value=%d)\n", ret, static_cast<int>(value));

    aclDestroyTensor(condition);
    aclrtFree(conditionDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

int main()
{
    LOG_PRINT("Testing aclnnAssertAsync with condition == true (expect pass).\n");
    RunAssertAsyncCase(true, "condition should be true");

    LOG_PRINT("Testing aclnnAssertAsync with condition == false (expect failure + printed message).\n");
    RunAssertAsyncCase(false, "condition is false");

    return 0;
}
```
