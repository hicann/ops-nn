# aclnnL1LossBackward

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id6 -->

## 功能说明

  计算[aclnnL1Loss](../../lp_loss/docs/aclnnL1Loss.md)的反向传播。reduction指定损失函数的计算方式，支持'none'、'mean'、'sum'。'none'表示不应用缩减，'mean'表示输出的总和将除以输出中的元素数，'sum'表示输出将被求和。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnL1LossBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnL1LossBackward”接口执行计算。

```Cpp
aclnnStatus aclnnL1LossBackwardGetWorkspaceSize(
    const aclTensor* gradOutput,
    const aclTensor* self,
    const aclTensor* target,
    int64_t          reduction,
    aclTensor*       gradInput,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnL1LossBackward(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnL1LossBackwardGetWorkspaceSize

- **参数说明**

  <table class="tg" style="undefined;table-layout: fixed; width: 1435px"><colgroup>
  <col style="width: 205px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 130px">
  <col style="width: 115px">
  <col style="width: 120px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">参数名</th>
      <th class="tg-0pky">输入/输出</th>
      <th class="tg-0pky">描述</th>
      <th class="tg-0pky">使用说明</th>
      <th class="tg-0pky">数据类型</th>
      <th class="tg-0pky">数据格式</th>
      <th class="tg-0pky">维度(shape)</th>
      <th class="tg-0pky">非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">gradOutput（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">梯度反向输入。</td>
      <td class="tg-0pky">shape需要与self满足<a href="../../../docs/zh/context/broadcast_relationship.md" target="_blank">broadcast关系</a>。<br>数据类型与self、target的数据类型满足数据类型推导规则（参见<a href="../../../docs/zh/context/conversion_relationship.md" target="_blank">互转换关系</a>）。</td>
      <td class="tg-0pky">FLOAT、FLOAT16、BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">self（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">输入张量</td>
      <td class="tg-0pky">shape需要与gradOutput、target满足<a href="../../../docs/zh/context/broadcast_relationship.md" target="_blank">broadcast关系</a>。<br>数据类型与gradOutput、target满足数据类型推导规则（参见<a href="../../../docs/zh/context/conversion_relationship.md" target="_blank">互转换关系</a>）。</td>
      <td class="tg-0pky">FLOAT、FLOAT16、BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">target（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">真实的标签</td>
      <td class="tg-0pky">shape需要与gradOutput、self满足<a href="../../../docs/zh/context/broadcast_relationship.md" target="_blank">broadcast关系</a>。<br>数据类型要与self、gradOutput满足数据类型推导规则（参见<a href="../../../docs/zh/context/conversion_relationship.md" target="_blank">互转换关系</a>）。</td>
      <td class="tg-0pky">FLOAT、FLOAT16、BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">reduction（int64_t）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">指定要应用到输出的缩减。</td>
      <td class="tg-0pky">支持0('none')|1('mean')|2('sum')。<br>-'none'表示不应用缩减<br>-'mean'表示输出的总和将除以输出中的元素数<br>-'sum'表示输出将被求和</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">gradInput（aclTensor*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">计算输出</td>
      <td class="tg-0pky">shape为gradOutput、self、target三个输入做<a href="../../../docs/zh/context/broadcast_relationship.md" target="_blank">broadcast</a>后的shape。<br>数据类型要与self满足数据类型推导规则（参见<a href="../../../docs/zh/context/conversion_relationship.md" target="_blank">互转换关系</a>）</td>
      <td class="tg-0pky">FLOAT、FLOAT16、BFLOAT16、DOUBLE</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize（uint64_t*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回需要在Device侧申请的workspace大小。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor（aclOpExecutor**）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回op执行器，包含了算子计算流程。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

  上表“数据类型”列为各产品支持的数据类型全集，按产品的差异如下：

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：gradOutput、self、target、gradInput的数据类型支持FLOAT、FLOAT16、BFLOAT16。
  <!-- end id7 -->
  <!-- npu="A3" id8 -->
  - <term>Atlas A3系列产品</term>：gradOutput、self、target、gradInput的数据类型支持FLOAT、FLOAT16、BFLOAT16。
  <!-- end id8 -->
  <!-- npu="910b" id9 -->
  - <term>Atlas A2系列产品</term>：gradOutput、self、target、gradInput的数据类型支持FLOAT、FLOAT16、BFLOAT16。
  <!-- end id9 -->
  <!-- npu="310p" id10 -->
  - <term>Atlas推理系列产品</term>：gradOutput、self、target、gradInput的数据类型支持FLOAT、FLOAT16，不支持BFLOAT16。数据类型推导结果为BFLOAT16时返回161002。
  <!-- end id10 -->
  <!-- npu="910" id11 -->
  - <term>Atlas训练系列产品</term>：gradOutput、self、target、gradInput的数据类型支持FLOAT、FLOAT16，不支持BFLOAT16。数据类型推导结果为BFLOAT16时返回161002。
  <!-- end id11 -->

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

    <table class="tg" style="undefined;table-layout: fixed; width: 991px"><colgroup>
    <col style="width: 269px">
    <col style="width: 90px">
    <col style="width: 632px">
    </colgroup>
    <thead>
      <tr>
        <th class="tg-0pky">返回值</th>
        <th class="tg-0pky">错误码</th>
        <th class="tg-0pky">描述</th>
      </tr></thead>
    <tbody>
      <tr>
        <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
        <td class="tg-0pky">161001</td>
        <td class="tg-0pky">传入的gradOutput、self、target和gradInput是空指针。</td>
      </tr>
      <tr>
        <td class="tg-0pky" rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
        <td class="tg-0pky" rowspan="7">161002</td>
        <td class="tg-0pky">reduction值不在0~2范围之内。</td>
      </tr>
      <tr>
        <td class="tg-0pky">gradOutput、self、target三者的数据类型两两之间无法完成数据类型推导（推导结果为未定义），例如其中之一为COMPLEX类型而其余为浮点类型。</td>
      </tr>
      <tr>
        <td class="tg-0pky">数据类型推导成功，但推导结果无法cast到gradInput的数据类型。例如推导结果为FLOAT而gradInput为INT32。</td>
      </tr>
      <tr>
        <td class="tg-0pky">数据类型推导成功且可cast到gradInput，但推导结果不在当前产品支持的数据类型范围之内。例如在<term>Atlas训练系列产品</term>或<term>Atlas推理系列产品</term>上推导结果为BFLOAT16，或在任意产品上推导结果为UINT16、INT32等非上表所列类型。</td>
      </tr>
      <tr>
        <td class="tg-0pky">gradOutput、self、target或gradInput的维度超过8维。</td>
      </tr>
      <tr>
        <td class="tg-0lax">gradOutput、self和target的shape无法做<a href="../../../docs/zh/context/broadcast_relationship.md" target="_blank">broadcast</a>。</td>
      </tr>
      <tr>
        <td class="tg-0lax">gradOutput、self和target做<a href="../../../docs/zh/context/broadcast_relationship.md" target="_blank">broadcast</a>后的shape与gradInput不一致。</td>
      </tr>
    </tbody>
    </table>

  上表各条件按接口内部的实际校验顺序排列：reduction取值 → 数据类型推导 → 推导结果到gradInput的cast可行性 → 推导结果是否在产品支持列表内 → 维度上限 → broadcast关系 → 输出shape一致性。当一次调用同时命中多个条件时，以排序在前的条件为实际触发原因；所有条件均返回同一错误码161002，无法通过错误码区分具体原因，需结合日志判断。

## aclnnL1LossBackward

- **参数说明：**
    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
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
          <td>在Device侧申请的workspace内存地址。</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>输入</td>
          <td>在Device侧申请的workspace大小，由第一段接口aclnnL1LossBackwardGetWorkspaceSize获取。</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 数据类型与维度：
    - gradOutput、self、target三者的数据类型需满足[互推导关系](../../../docs/zh/context/deduction_relationship.md)，推导结果需在当前产品支持的数据类型范围之内（见“参数说明”中的产品差异说明），且需可cast到gradInput的数据类型。
    - gradOutput、self、target、gradInput的shape支持0~8维，其中0维表示标量输入。

- 空Tensor：
    - 当gradOutput、self、target中任一输入为空Tensor（包含长度为0的轴）时，aclnnL1LossBackwardGetWorkspaceSize返回ACLNN_SUCCESS且workspaceSize为0，不对gradInput写入任何数据，gradInput保持调用前的内存内容。调用者需自行保证该场景下不使用gradInput的内容。

- inf/nan输入：
    - 输入包含inf或nan时的输出行为未定义，本接口不做保证。

- 动态shape与动态rank：
    - 算子注册了动态shape与动态rank能力标记，但动态维标记-1、动态rank标记-2在本接口上的公开语义未定义，不建议依赖。

- 数据格式：
    - 数据格式仅声明支持ND。传入非ND的存储格式（如FRACTAL_NZ、NCHW）时，本接口不做拦截，仅在FRACTAL_NZ场景打印告警日志，计算结果的精度不做保证。

- 非连续Tensor：
    - 本接口支持非连续的Tensor：内部对gradOutput、self、target各做一次连续化转换，输出经ViewCopy写回，故gradInput也可为非连续Tensor。

- 确定性计算：
    - aclnnL1LossBackward默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_l1_loss_backward.h"

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
  // 固定写法，资源初始化
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
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1.（固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造gradOutput
  std::vector<int64_t> gradOutputShape = {2, 2};
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> targetShape = {2, 2};
  std::vector<int64_t> gradInputShape = {2, 2};
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* gradInput = nullptr;
  std::vector<float> gradOutputHostData = {0, 1, 2, 3};
  std::vector<float> selfHostData = {0, 1, 2, 3};
  std::vector<float> targetHostData = {1, 1, 1, 1};
  std::vector<float> gradInputHostData(4, 0);
  // 创建gradOutput aclTensor
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr,
                        aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建target aclTensor
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gradInput aclTensor
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建reduction
  int64_t reduction = 1;

  // 3. 调用CANN算子库API，需要修改为具体的API名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnL1LossBackward第一段接口
  ret = aclnnL1LossBackwardGetWorkspaceSize(gradOutput, self, target, reduction, gradInput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnL1LossBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnL1LossBackward第二段接口
  ret = aclnnL1LossBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnL1LossBackward failed. ERROR: %d\n", ret); return ret);

  // 4.（固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(gradInputShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(gradInput);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(gradInputDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
