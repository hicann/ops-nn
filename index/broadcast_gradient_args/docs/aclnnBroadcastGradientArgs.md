# aclnnBroadcastGradientArgs

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/index/broadcast_gradient_args)

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

- 接口功能：BroadcastGradientArgs是TensorFlow中用于计算梯度传播所需的广播维度索引的算子。在反向传播过程中，根据两个张量在正向传播时的原始形状，自动识别出它们因广播机制而扩展的维度，并输出需要在哪些维度上对梯度进行约简，以便将梯度从广播后的形状还原为每个原始张量的形状。
- 计算规则：
  1. 若两个输入的shape长度相同且各维度值逐一相等（即两个输入完全elementwise一致），则y1和y2均为空，表示无需约简。
  2. 若两个输入的shape长度不同，将较短的一侧在左侧补1至与较长一侧长度一致。补齐后，较短一侧多出的前导维度视为该张量在该维度上值为1，需加入对应的约简列表。
  3. 除场景1外，对齐后逐维比较x1[i]与x2[i]，若其中任意一方值为1，则该维度需加入值为1的一方对应的约简列表（x1[i]==1时索引i加入y1，x2[i]==1时索引i加入y2）。
- 示例：

  ```text
  例1（两输入长度相等，部分维度存在1）：
    原始张量a的shape为[2, 1, 4, 1, 6]
    原始张量b的shape为[2, 3, 1, 5, 1]
    x1_data: [2, 1, 4, 1, 6]   # x1_shape=[5]
    x2_data: [2, 3, 1, 5, 1]   # x2_shape=[5]

    逐维比较：
      dim0: x1=2, x2=2   → 相等，不输出
      dim1: x1=1, x2=3   → x1[i]==1，y1 收集索引 1
      dim2: x1=4, x2=1   → x2[i]==1，y2 收集索引 2
      dim3: x1=1, x2=5   → x1[i]==1，y1 收集索引 3
      dim4: x1=6, x2=1   → x2[i]==1，y2 收集索引 4

    y1_data: [1, 3]            # y1_shape=[2]
    y2_data: [2, 4]            # y2_shape=[2]

  例2（两输入长度不等，左侧补1对齐）：
    原始张量a的shape为[4, 1, 6]
    原始张量b的shape为[2, 3, 1, 5, 1]
    x1_data: [4, 1, 6]         # x1_shape=[3]，较短
    x2_data: [2, 3, 1, 5, 1]   # x2_shape=[5]，较长

    对齐：将 x1 左侧补 1 至长度 5，得到 [1, 1, 4, 1, 6]
          x1(补齐后): [1, 1, 4, 1, 6]
          x2:         [2, 3, 1, 5, 1]

    逐维比较（使用补齐后的 x1）：
      dim0: x1=1(补), x2=2 → x1[i]==1，y1 收集索引 0
      dim1: x1=1(补), x2=3 → x1[i]==1，y1 收集索引 1
      dim2: x1=4,     x2=1 → x2[i]==1，y2 收集索引 2
      dim3: x1=1,     x2=5 → x1[i]==1，y1 收集索引 3
      dim4: x1=6,     x2=1 → x2[i]==1，y2 收集索引 4

    y1_data: [0, 1, 3]         # y1_shape=[3]
    y2_data: [2, 4]            # y2_shape=[2]

  例3（两输入完全一致，无需约简）：
    原始张量a的shape为[2, 1, 4, 1, 6]
    原始张量b的shape为[2, 1, 4, 1, 6]
    x1_data: [2, 1, 4, 1, 6]   # x1_shape=[5]
    x2_data: [2, 1, 4, 1, 6]   # x2_shape=[5]

    两输入长度相同且各维度值逐一相等，属于场景1，无需约简。

    y1_data: []                # y1_shape=[0]
    y2_data: []                # y2_shape=[0]
  ```

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnBroadcastGradientArgsGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnBroadcastGradientArgs”接口执行计算。

```Cpp
aclnnStatus aclnnBroadcastGradientArgsGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  aclTensor       *y1,
  aclTensor       *y2,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnBroadcastGradientArgs(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnBroadcastGradientArgsGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 330px">
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 145px">
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
      <td>x1</td>
      <td>输入</td>
      <td>表示原始张量a的shape。</td>
      <td>必须为1维。<br>数据类型需与x2一致。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示原始张量b的shape。</td>
      <td>必须为1维。<br>数据类型需与x1一致。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>输出</td>
      <td>表示x1对应的张量shape中需要广播的索引。</td>
      <td>数据类型需与x1一致。<br>输出size需按max(x1_len, x2_len)申请，实际输出可能小于此值，执行完成后通过aclGetViewShape获取实际shape。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>输出</td>
      <td>表示x2对应的张量shape中需要广播的索引。</td>
      <td>数据类型需与x1一致。<br>输出size需按max(x1_len, x2_len)申请，实际输出可能小于此值，执行完成后通过aclGetViewShape获取实际shape。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1071px"><colgroup>
  <col style="width: 265px">
  <col style="width: 149px">
  <col style="width: 657px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的x1、x2、y1或y2是空指针时。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>x1或x2的数据类型不在支持的范围之内（仅支持INT32、INT64）。</td>
    </tr>
    <tr>
      <td>x1和x2的数据类型不一致，或y1/y2的数据类型与x1/x2不一致。</td>
    </tr>
    <tr>
      <td>x1、x2、y1或y2的shape不是1维。</td>
    </tr>
    <tr>
      <td>y1或y2的容量小于max(x1长度, x2长度)，可能导致kernel写入越界。</td>
    </tr>
    <tr>
      <td>当前设备不是Ascend 950PR/Ascend 950DT，算子不支持该架构。</td>
    </tr>
  </tbody>
  </table>

## aclnnBroadcastGradientArgs

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1042px"><colgroup>
  <col style="width: 141px">
  <col style="width: 110px">
  <col style="width: 791px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnBroadcastGradientArgsGetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnBroadcastGradientArgs默认确定性实现。

- 仅支持<term>Ascend 950PR/Ascend 950DT</term>，不支持其他架构。
- 输入x1和x2需满足广播规则：对应维度要么相等，要么至少一个为1。
- y1和y2为动态shape输出，输出内存需按max(x1长度, x2长度)预分配，实际输出元素数可能小于此值，执行完成后需通过aclGetViewShape获取实际输出shape。
- y1和y2不支持非连续Tensor。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_broadcast_gradient_args.h"

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
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            aclrtResetDevice(deviceId); aclFinalize(); return ret);
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
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出
  //    x1 = {2, 1, 4, 1, 6}  (原始张量a的shape)
  //    x2 = {2, 3, 1, 5, 1}  (原始张量b的shape)
  //    输出按最大可能size申请，max(5, 5) = 5
  std::vector<int32_t> x1HostData = {2, 1, 4, 1, 6};
  std::vector<int32_t> x2HostData = {2, 3, 1, 5, 1};
  std::vector<int64_t> y1Shape = {5};
  std::vector<int64_t> y2Shape = {5};
  void* x1DeviceAddr = nullptr;
  void* x2DeviceAddr = nullptr;
  void* y1DeviceAddr = nullptr;
  void* y2DeviceAddr = nullptr;
  aclTensor* x1 = nullptr;
  aclTensor* x2 = nullptr;
  aclTensor* y1 = nullptr;
  aclTensor* y2 = nullptr;
  std::vector<int32_t> y1HostData(5, 0);
  std::vector<int32_t> y2HostData(5, 0);
  ret = CreateAclTensor(x1HostData, {5}, &x1DeviceAddr, aclDataType::ACL_INT32, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(x2HostData, {5}, &x2DeviceAddr, aclDataType::ACL_INT32, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(y1HostData, y1Shape, &y1DeviceAddr, aclDataType::ACL_INT32, &y1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(y2HostData, y2Shape, &y2DeviceAddr, aclDataType::ACL_INT32, &y2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnBroadcastGradientArgs第一段接口
  ret = aclnnBroadcastGradientArgsGetWorkspaceSize(x1, x2, y1, y2, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBroadcastGradientArgsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnBroadcastGradientArgs第二段接口
  ret = aclnnBroadcastGradientArgs(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBroadcastGradientArgs failed. ERROR: %d\n", ret); return ret);

  // 4.（固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的实际shape（动态shape关键步骤），并将device侧结果拷贝至host侧
  int64_t* y1ViewDims = nullptr;
  uint64_t y1ViewDimsNum = 0;
  ret = aclGetViewShape(y1, &y1ViewDims, &y1ViewDimsNum);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclGetViewShape y1 failed. ERROR: %d\n", ret); return ret);
  int64_t y1Size = 1;
  for (uint64_t i = 0; i < y1ViewDimsNum; i++) {
    y1Size *= y1ViewDims[i];
  }
  if (y1Size > 0) {
    std::vector<int32_t> y1ResultData(y1Size, 0);
    ret = aclrtMemcpy(y1ResultData.data(), y1ResultData.size() * sizeof(int32_t), y1DeviceAddr,
                      y1Size * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y1 result failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < y1Size; i++) {
      LOG_PRINT("y1[%ld] is: %d\n", i, y1ResultData[i]);
    }
  } else {
    LOG_PRINT("y1 is empty (no broadcast axis)\n");
  }
  delete[] y1ViewDims;

  int64_t* y2ViewDims = nullptr;
  uint64_t y2ViewDimsNum = 0;
  ret = aclGetViewShape(y2, &y2ViewDims, &y2ViewDimsNum);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclGetViewShape y2 failed. ERROR: %d\n", ret); return ret);
  int64_t y2Size = 1;
  for (uint64_t i = 0; i < y2ViewDimsNum; i++) {
    y2Size *= y2ViewDims[i];
  }
  if (y2Size > 0) {
    std::vector<int32_t> y2ResultData(y2Size, 0);
    ret = aclrtMemcpy(y2ResultData.data(), y2ResultData.size() * sizeof(int32_t), y2DeviceAddr,
                      y2Size * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y2 result failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < y2Size; i++) {
      LOG_PRINT("y2[%ld] is: %d\n", i, y2ResultData[i]);
    }
  } else {
    LOG_PRINT("y2 is empty (no broadcast axis)\n");
  }
  delete[] y2ViewDims;

  // 6. 释放aclTensor
  aclDestroyTensor(x1);
  aclDestroyTensor(x2);
  aclDestroyTensor(y1);
  aclDestroyTensor(y2);

  // 7. 释放device资源
  aclrtFree(x1DeviceAddr);
  aclrtFree(x2DeviceAddr);
  aclrtFree(y1DeviceAddr);
  aclrtFree(y2DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
