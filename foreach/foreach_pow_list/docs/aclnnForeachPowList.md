# aclnnForeachPowList

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
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：对输入张量列表的每个张量进行幂运算（底数为x1，指数为x2）。
- 计算公式：

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}]\\
  x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = x1_i^{x2_i}  (i=0,1,...n-1)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnForeachPowListGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnForeachPowList”接口执行计算。

```Cpp
aclnnStatus aclnnForeachPowListGetWorkspaceSize(
  const aclTensorList *x1,
  const aclTensorList *x2,
  aclTensorList       *out,
  uint64_t            *workspaceSize,
  aclOpExecutor      **executor)
```

```Cpp
aclnnStatus aclnnForeachPowList(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnForeachPowListGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 271px">
    <col style="width: 330px">
    <col style="width: 223px">
    <col style="width: 101px">
    <col style="width: 190px">
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
      <td>x1（aclTensorList*）</td>
      <td>输入</td>
      <td>表示进行幂运算的底数张量列表，对应公式中的`x1`。</td>
      <td><ul><li>支持空Tensor。</li><li>该参数中所有Tensor的数据类型保持一致。</li><li>x1、x2、out三个张量列表的长度必须一致。</li><li>shape与入参`x2`的shape完全一致，即各维度逐维相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2（aclTensorList*）</td>
      <td>输入</td>
      <td>表示进行幂运算的指数张量列表，对应公式中的`x2`。</td>
      <td><ul><li>支持空Tensor。</li><li>该参数中所有Tensor的数据类型保持一致。</li><li>x1、x2、out三个张量列表的长度必须一致。</li><li>数据类型、数据格式与入参`x1`的数据类型、数据格式一致，shape与入参`x1`的shape完全一致，即各维度逐维相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out（aclTensorList*）</td>
      <td>输出</td>
      <td>表示进行幂运算的输出张量列表，对应公式中的`y`。</td>
      <td><ul><li>支持空Tensor。</li><li>该参数中所有Tensor的数据类型保持一致。</li><li>x1、x2、out三个张量列表的长度必须一致。</li><li>数据类型和数据格式与入参`x1`的数据类型和数据格式一致，shape与入参`x1`的shape完全一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
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

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td>传入的x1、x2、out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>x1、x2、out的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>x1、x2、out的数据类型不一致。</td>
    </tr>
    <tr>
      <td>x1、x2、out的数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>x1、x2、out的shape不满足约束。</td>
    </tr>
    <tr>
      <td>x1、x2或out中的Tensor维度超过8维。</td>
    </tr>
  </tbody></table>

## aclnnForeachPowList

- **参数说明**

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnForeachPowListGetWorkspaceSize获取。</td>
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

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT系列产品</term>：单次调用支持1至256组张量；超过256组时需分批调用。
<!-- end id7 -->
- 确定性计算：
  - aclnnForeachPowList默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <utility>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_foreach_pow_list.h"

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

template <typename F>
class ScopeGuard {
public:
    explicit ScopeGuard(const F& cleanup) : cleanup_(cleanup), active_(true) {}
    ScopeGuard(ScopeGuard&& other) noexcept : cleanup_(std::move(other.cleanup_)), active_(other.active_)
    {
        other.active_ = false;
    }
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
    ~ScopeGuard()
    {
        if (active_) {
            cleanup_();
        }
    }

private:
    F cleanup_;
    bool active_;
};

template <typename F>
ScopeGuard<F> MakeScopeGuard(F cleanup)
{
    return ScopeGuard<F>(cleanup);
}

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
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        aclFinalize();
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    if (*tensor == nullptr) {
        LOG_PRINT("aclCreateTensor failed.\n");
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ACL_ERROR_FAILURE;
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape1 = {2, 3};
    std::vector<int64_t> selfShape2 = {1, 3};
    std::vector<int64_t> otherShape1 = {2, 3};
    std::vector<int64_t> otherShape2 = {1, 3};
    std::vector<int64_t> outShape1 = {2, 3};
    std::vector<int64_t> outShape2 = {1, 3};
    void* input1DeviceAddr = nullptr;
    void* input2DeviceAddr = nullptr;
    void* other1DeviceAddr = nullptr;
    void* other2DeviceAddr = nullptr;
    void* out1DeviceAddr = nullptr;
    void* out2DeviceAddr = nullptr;
    aclTensor* input1 = nullptr;
    aclTensor* input2 = nullptr;
    aclTensor* other1 = nullptr;
    aclTensor* other2 = nullptr;
    aclTensor* out1 = nullptr;
    aclTensor* out2 = nullptr;
    aclTensorList* tensorListInput1 = nullptr;
    aclTensorList* tensorListInput2 = nullptr;
    aclTensorList* tensorListOutput = nullptr;
    void* workspaceAddr = nullptr;

    auto cleanup = MakeScopeGuard([&]() {
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        if (tensorListInput1 != nullptr) {
            aclDestroyTensorList(tensorListInput1);
        } else {
            if (input1 != nullptr)
                aclDestroyTensor(input1);
            if (input2 != nullptr)
                aclDestroyTensor(input2);
        }
        if (tensorListInput2 != nullptr) {
            aclDestroyTensorList(tensorListInput2);
        } else {
            if (other1 != nullptr)
                aclDestroyTensor(other1);
            if (other2 != nullptr)
                aclDestroyTensor(other2);
        }
        if (tensorListOutput != nullptr) {
            aclDestroyTensorList(tensorListOutput);
        } else {
            if (out1 != nullptr)
                aclDestroyTensor(out1);
            if (out2 != nullptr)
                aclDestroyTensor(out2);
        }
        if (input1DeviceAddr != nullptr)
            aclrtFree(input1DeviceAddr);
        if (input2DeviceAddr != nullptr)
            aclrtFree(input2DeviceAddr);
        if (other1DeviceAddr != nullptr)
            aclrtFree(other1DeviceAddr);
        if (other2DeviceAddr != nullptr)
            aclrtFree(other2DeviceAddr);
        if (out1DeviceAddr != nullptr)
            aclrtFree(out1DeviceAddr);
        if (out2DeviceAddr != nullptr)
            aclrtFree(out2DeviceAddr);
        if (stream != nullptr)
            aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
    });
    std::vector<float> input1HostData = {1, 2, 3, 4, 5, 6};
    std::vector<float> input2HostData = {7, 8, 9};
    std::vector<float> other1HostData = {6, 5, 4, 3, 2, 1};
    std::vector<float> other2HostData = {9, 8, 7};
    std::vector<float> out1HostData(6, 0);
    std::vector<float> out2HostData(3, 0);
    // 创建input1 aclTensor
    ret = CreateAclTensor(input1HostData, selfShape1, &input1DeviceAddr, aclDataType::ACL_FLOAT, &input1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建input2 aclTensor
    ret = CreateAclTensor(input2HostData, selfShape2, &input2DeviceAddr, aclDataType::ACL_FLOAT, &input2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建other1 aclTensor
    ret = CreateAclTensor(other1HostData, otherShape1, &other1DeviceAddr, aclDataType::ACL_FLOAT, &other1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建other2 aclTensor
    ret = CreateAclTensor(other2HostData, otherShape2, &other2DeviceAddr, aclDataType::ACL_FLOAT, &other2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out1 aclTensor
    ret = CreateAclTensor(out1HostData, outShape1, &out1DeviceAddr, aclDataType::ACL_FLOAT, &out1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out2 aclTensor
    ret = CreateAclTensor(out2HostData, outShape2, &out2DeviceAddr, aclDataType::ACL_FLOAT, &out2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<aclTensor*> tempInput1{input1, input2};
    tensorListInput1 = aclCreateTensorList(tempInput1.data(), tempInput1.size());
    std::vector<aclTensor*> tempInput2{other1, other2};
    tensorListInput2 = aclCreateTensorList(tempInput2.data(), tempInput2.size());
    std::vector<aclTensor*> tempOutput{out1, out2};
    tensorListOutput = aclCreateTensorList(tempOutput.data(), tempOutput.size());

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnForeachPowList第一段接口
    ret = aclnnForeachPowListGetWorkspaceSize(tensorListInput1, tensorListInput2, tensorListOutput, &workspaceSize,
                                              &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachPowListGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnForeachPowList第二段接口
    ret = aclnnForeachPowList(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachPowList failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape1);
    std::vector<float> out1Data(size, 0);
    ret = aclrtMemcpy(out1Data.data(), out1Data.size() * sizeof(out1Data[0]), out1DeviceAddr,
                      size * sizeof(out1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out1 result[%ld] is: %f\n", i, out1Data[i]);
    }

    size = GetShapeSize(outShape2);
    std::vector<float> out2Data(size, 0);
    ret = aclrtMemcpy(out2Data.data(), out2Data.size() * sizeof(out2Data[0]), out2DeviceAddr,
                      size * sizeof(out2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out2 result[%ld] is: %f\n", i, out2Data[i]);
    }

    return 0;
}
```
