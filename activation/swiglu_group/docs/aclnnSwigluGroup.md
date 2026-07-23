# aclnnSwigluGroup

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/activation/swiglu_group)

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

- 接口功能：执行SwiGLU激活，输出与输入x相同数据类型的激活结果。

- 计算公式：

  令输入x的最后一维大小为D，左半部分为A，右半部分为B（即`h = D // 2`，`A = x[..., :h]`，`B = x[..., h:]`）。

  当`clampLimit`大于0时，先对A、B进行截断：

  $$
  A = min(A, clampLimit)
  $$

  $$
  B = min(max(B, -clampLimit), clampLimit)
  $$

  计算SwiGLU：

  $$
  y = silu(A) \times B = \frac{A}{1 + e^{-A}} \times B
  $$

  当`weightOptional`非空时，按token乘以权重：

  $$
  y = y \times weightOptional
  $$

  当`groupIndexOptional`非空时，按count模式控制实际处理的token数：实际处理的token数为`groupIndexOptional`所有元素之和与`bs`两者中的较小值，其中`bs`为x除最后一维外的维度乘积。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnSwigluGroupGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSwigluGroup”接口执行计算。

```Cpp
aclnnStatus aclnnSwigluGroupGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weightOptional,
  const aclTensor *groupIndexOptional,
  double           clampLimit,
  const aclTensor *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor);
```

```Cpp
aclnnStatus aclnnSwigluGroup(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream);
```

## aclnnSwigluGroupGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 300px">
  <col style="width: 360px">
  <col style="width: 230px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 120px">
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
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的输入x，最后一维被均分为A、B两部分。</td>
      <td>不支持空指针，不支持空Tensor。维度必须大于0，最后一维必须为偶数。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>weightOptional（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的输入weightOptional，逐token权重。</td>
      <td>支持空指针，不支持空Tensor。不为空指针时，元素个数需等于x除最后一维外的维度乘积（bs）。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>groupIndexOptional（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的输入groupIndexOptional，count模式下的分组token数量。</td>
      <td>支持空指针，不支持空Tensor。不为空指针时，维度要求为1维，元素需大于等于0。第i个元素代表第i组需要处理x的token数量，所有元素之和经截断后为实际处理的token数。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>clampLimit（double）</td>
      <td>输入</td>
      <td>公式中的输入clampLimit，表示激活前对A、B进行截断的上界。</td>
      <td>传入-1.0表示不进行截断；传入大于0的值时，在激活前对A、B进行截断。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>不涉及</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>公式中的输出y，SwiGLU计算结果。</td>
      <td>数据类型与x一致，最后一维为x最后一维的一半，其余维度与x一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>不涉及</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>不涉及</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的x、out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>x、out的数据类型不在支持的范围内，或out与x的数据类型不一致。</td>
    </tr>
    <tr>
      <td>x的最后一维不是偶数，或维度不在支持的范围内。</td>
    </tr>
    <tr>
      <td>weightOptional的元素个数与x除最后一维外的维度乘积不一致。</td>
    </tr>
  </tbody>
  </table>

## aclnnSwigluGroup

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSwigluGroupGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 不支持空Tensor：输入x、weightOptional、groupIndexOptional均不支持空Tensor（其中weightOptional、groupIndexOptional可传入空指针，但传入时不可为空Tensor）。
- 不支持非连续Tensor：输入x、weightOptional、groupIndexOptional及输出out均需为连续Tensor。
- 输入x的最后一维必须为偶数（被均分为A、B两部分，输出最后一维为输入的一半）。
- 输入x的数据类型仅支持FLOAT16、BFLOAT16、FLOAT32，且out的数据类型需与x一致。
- 确定性计算：aclnnSwigluGroup默认为确定性实现，暂不支持非确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_group.h"

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

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，acl初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl对外接口列表
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {2, 256};
    std::vector<int64_t> yShape = {2, 128};

    std::vector<uint16_t> xHostData(GetShapeSize(xShape), 0);
    for (size_t i = 0; i < xHostData.size(); ++i) {
        xHostData[i] = static_cast<uint16_t>(i % 23);
    }
    std::vector<uint16_t> yHostData(GetShapeSize(yShape), 0);

    void* xDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y aclTensor
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // weightOptional与groupIndexOptional为可选输入，此处传入nullptr表示不使用
    double clampLimit = -1.0;

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // 调用aclnnSwigluGroup第一段接口
    ret = aclnnSwigluGroupGetWorkspaceSize(x, nullptr, nullptr, clampLimit, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请Device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnSwigluGroup第二段接口
    ret = aclnnSwigluGroup(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroup failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧
    std::vector<uint16_t> resultData(GetShapeSize(yShape), 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              return ret);
    LOG_PRINT("result[0] is: %u\n", resultData[0]);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(y);

    // 7. 释放Device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
