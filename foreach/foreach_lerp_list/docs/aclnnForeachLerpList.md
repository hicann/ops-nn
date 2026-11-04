# aclnnForeachLerpList

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

- 接口功能：对两个张量列表对应位置元素执行插值计算，其中张量列表weight是插值系数。
- 计算公式：

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
  weight = [{weight_0}, {weight_1}, ... {weight_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = x1_i + {\rm weight}_i × (x2_i - x1_i) (i=0,1,...n-1)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnForeachLerpListGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnForeachLerpList”接口执行计算。

```Cpp
aclnnStatus aclnnForeachLerpListGetWorkspaceSize(
  const aclTensorList *x1,
  const aclTensorList *x2,
  const aclTensorList *weight,
  aclTensorList       *out,
  uint64_t            *workspaceSize,
  aclOpExecutor      **executor)
```

```Cpp
aclnnStatus aclnnForeachLerpList(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnForeachLerpListGetWorkspaceSize

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
      <td>表示进行线性插值计算的第一个输入张量列表。对应公式中的<code>x1</code>。</td>
      <td><ul><li>支持空Tensor。</li><li>该参数中所有Tensor的数据类型保持一致。</li><li>shape与入参<code>x2</code>、<code>weight</code>的shape一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2（aclTensorList*）</td>
      <td>输入</td>
      <td>表示进行线性插值计算的第二个输入张量列表。对应公式中的<code>x2</code>。</td>
      <td><ul><li>支持空Tensor。</li><li>该参数中所有Tensor的数据类型保持一致。</li><li>数据类型、数据格式和shape与入参<code>x1</code>一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight（aclTensorList*）</td>
      <td>输入</td>
      <td>表示进行线性插值计算的插值系数张量列表。对应公式中的<code>weight</code>。</td>
      <td><ul><li>支持空Tensor。</li><li>该参数中所有Tensor的数据类型保持一致。</li><li>数据类型、数据格式和shape与入参<code>x1</code>的数据类型、数据格式和shape一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out（aclTensorList*）</td>
      <td>输出</td>
      <td>表示进行线性插值计算的输出张量列表。对应公式中的<code>y</code>。</td>
      <td><ul><li>支持空Tensor。</li><li>该参数中所有Tensor的数据类型保持一致。</li><li>数据类型和数据格式与入参<code>x1</code>的数据类型和数据格式一致，shape与入参<code>x1</code>一致，storage shape size不小于入参<code>x1</code>的storage shape size。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
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
      <td>传入的x1、x2、weight、out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>x1、x2、weight、out的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>x1、x2、weight、out的数据类型不一致（包括各TensorList内部Tensor之间的数据类型不一致）。</td>
    </tr>
    <tr>
      <td>x1、x2、weight、out的TensorList长度不一致。</td>
    </tr>
    <tr>
      <td>x1、x2、weight的shape不一致，或out的shape与x1不一致、out的storage shape size小于x1的storage shape size。</td>
    </tr>
    <tr>
      <td>x1、x2、weight、out中存在维度超过8维的Tensor。</td>
    </tr>
    <tr>
      <td>x1、x2、weight、out的TensorList中存在空指针Tensor。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>内部tiling阶段发生异常，例如x1、x2、weight、out的TensorList长度超过上限（256）。</td>
    </tr>
  </tbody></table>

## aclnnForeachLerpList

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnForeachLerpListGetWorkspaceSize获取。</td>
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

- 数据格式：仅支持 ND 格式，不支持FRACTAL_Z、FRACTAL_NZ等私有格式。传入私有格式时，aclnnForeachLerpListGetWorkspaceSize接口返回161002（ACLNN_ERR_PARAM_INVALID）。
- 维度与列表长度：每个Tensor的维度范围为0-8，超过8维时接口返回161002（ACLNN_ERR_PARAM_INVALID）；TensorList长度上限为256，超过上限时在内部tiling阶段返回561002（ACLNN_ERR_INNER_TILING_ERROR）。
- out 的 shape 需与 x1 一致，且 storage shape size 不小于 x1。
- 确定性计算：
  - aclnnForeachLerpList默认确定性实现。

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
#include <memory>
#include <type_traits>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_foreach_lerp_list.h"

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

using StreamPtr = std::unique_ptr<std::remove_pointer<aclrtStream>::type, decltype(&aclrtDestroyStream)>;
using DeviceMemPtr = std::unique_ptr<void, decltype(&aclrtFree)>;
using TensorPtr = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;
using TensorListPtr = std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)>;

void PrintOutResult(const std::vector<int64_t>& shape, const DeviceMemPtr& deviceAddr, const char* name)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceAddr.get(),
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("%s result[%ld] is: %f\n", name, i, resultData[i]);
    }
}

int Init(int32_t deviceId, StreamPtr& stream, bool& initialized, bool& deviceSet)
{
    // 固定写法，初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    initialized = true;
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    deviceSet = true;
    aclrtStream rawStream = nullptr;
    ret = aclrtCreateStream(&rawStream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    stream.reset(rawStream);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
                    DeviceMemPtr& deviceAddr, TensorPtr& tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    void* rawDeviceAddr = nullptr;
    auto ret = aclrtMalloc(&rawDeviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    deviceAddr.reset(rawDeviceAddr);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(deviceAddr.get(), size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    aclTensor* rawTensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), deviceAddr.get());
    CHECK_RET(rawTensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    tensor.reset(rawTensor);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    bool initialized = false;
    bool deviceSet = false;
    std::shared_ptr<void> aclGuard(nullptr, [&](void*) {
        if (deviceSet) {
            aclrtResetDevice(deviceId);
        }
        if (initialized) {
            aclFinalize();
        }
    });
    StreamPtr stream(nullptr, &aclrtDestroyStream);
    auto ret = Init(deviceId, stream, initialized, deviceSet);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape1 = {2, 3};
    std::vector<int64_t> selfShape2 = {1, 3};
    std::vector<int64_t> otherShape1 = {2, 3};
    std::vector<int64_t> otherShape2 = {1, 3};
    std::vector<int64_t> weightShape1 = {2, 3};
    std::vector<int64_t> weightShape2 = {1, 3};
    std::vector<int64_t> outShape1 = {2, 3};
    std::vector<int64_t> outShape2 = {1, 3};
    std::vector<float> input1HostData = {1, 2, 3, 4, 5, 6};
    std::vector<float> input2HostData = {7, 8, 9};
    std::vector<float> other1HostData = {1, 2, 3, 4, 5, 6};
    std::vector<float> other2HostData = {7, 8, 9};
    std::vector<float> weight1HostData = {1, 2, 3, 4, 5, 6};
    std::vector<float> weight2HostData = {7, 8, 9};
    std::vector<float> out1HostData(6, 0);
    std::vector<float> out2HostData(3, 0);

    DeviceMemPtr input1DeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr input2DeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr other1DeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr other2DeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr weight1DeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr weight2DeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr out1DeviceAddr(nullptr, &aclrtFree);
    DeviceMemPtr out2DeviceAddr(nullptr, &aclrtFree);

    TensorPtr input1(nullptr, &aclDestroyTensor);
    TensorPtr input2(nullptr, &aclDestroyTensor);
    TensorPtr other1(nullptr, &aclDestroyTensor);
    TensorPtr other2(nullptr, &aclDestroyTensor);
    TensorPtr weight1(nullptr, &aclDestroyTensor);
    TensorPtr weight2(nullptr, &aclDestroyTensor);
    TensorPtr out1(nullptr, &aclDestroyTensor);
    TensorPtr out2(nullptr, &aclDestroyTensor);

    TensorListPtr tensorListInput1(nullptr, &aclDestroyTensorList);
    TensorListPtr tensorListInput2(nullptr, &aclDestroyTensorList);
    TensorListPtr tensorListWeight(nullptr, &aclDestroyTensorList);
    TensorListPtr tensorListOutput(nullptr, &aclDestroyTensorList);

    // 创建input1 aclTensor
    ret = CreateAclTensor(input1HostData, selfShape1, aclDataType::ACL_FLOAT, input1DeviceAddr, input1);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create input1 failed. ERROR: %d\n", ret); return ret);
    // 创建input2 aclTensor
    ret = CreateAclTensor(input2HostData, selfShape2, aclDataType::ACL_FLOAT, input2DeviceAddr, input2);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create input2 failed. ERROR: %d\n", ret); return ret);
    // 创建other1 aclTensor
    ret = CreateAclTensor(other1HostData, otherShape1, aclDataType::ACL_FLOAT, other1DeviceAddr, other1);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create other1 failed. ERROR: %d\n", ret); return ret);
    // 创建other2 aclTensor
    ret = CreateAclTensor(other2HostData, otherShape2, aclDataType::ACL_FLOAT, other2DeviceAddr, other2);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create other2 failed. ERROR: %d\n", ret); return ret);
    // 创建weight1 aclTensor
    ret = CreateAclTensor(weight1HostData, weightShape1, aclDataType::ACL_FLOAT, weight1DeviceAddr, weight1);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create weight1 failed. ERROR: %d\n", ret); return ret);
    // 创建weight2 aclTensor
    ret = CreateAclTensor(weight2HostData, weightShape2, aclDataType::ACL_FLOAT, weight2DeviceAddr, weight2);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create weight2 failed. ERROR: %d\n", ret); return ret);
    // 创建out1 aclTensor
    ret = CreateAclTensor(out1HostData, outShape1, aclDataType::ACL_FLOAT, out1DeviceAddr, out1);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create out1 failed. ERROR: %d\n", ret); return ret);
    // 创建out2 aclTensor
    ret = CreateAclTensor(out2HostData, outShape2, aclDataType::ACL_FLOAT, out2DeviceAddr, out2);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("create out2 failed. ERROR: %d\n", ret); return ret);

    // 创建TensorList
    std::vector<aclTensor*> tempInput1{input1.get(), input2.get()};
    tensorListInput1.reset(aclCreateTensorList(tempInput1.data(), tempInput1.size()));
    CHECK_RET(tensorListInput1 != nullptr, LOG_PRINT("aclCreateTensorList failed.\n"); return ACL_ERROR_FAILURE);
    // aclCreateTensorList接管aclTensor的所有权，释放本地智能指针避免重复释放
    input1.release();
    input2.release();

    std::vector<aclTensor*> tempInput2{other1.get(), other2.get()};
    tensorListInput2.reset(aclCreateTensorList(tempInput2.data(), tempInput2.size()));
    CHECK_RET(tensorListInput2 != nullptr, LOG_PRINT("aclCreateTensorList failed.\n"); return ACL_ERROR_FAILURE);
    other1.release();
    other2.release();

    std::vector<aclTensor*> tempWeight{weight1.get(), weight2.get()};
    tensorListWeight.reset(aclCreateTensorList(tempWeight.data(), tempWeight.size()));
    CHECK_RET(tensorListWeight != nullptr, LOG_PRINT("aclCreateTensorList failed.\n"); return ACL_ERROR_FAILURE);
    weight1.release();
    weight2.release();

    std::vector<aclTensor*> tempOutput{out1.get(), out2.get()};
    tensorListOutput.reset(aclCreateTensorList(tempOutput.data(), tempOutput.size()));
    CHECK_RET(tensorListOutput != nullptr, LOG_PRINT("aclCreateTensorList failed.\n"); return ACL_ERROR_FAILURE);
    out1.release();
    out2.release();

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 调用aclnnForeachLerpList第一段接口
    ret = aclnnForeachLerpListGetWorkspaceSize(tensorListInput1.get(), tensorListInput2.get(), tensorListWeight.get(),
                                               tensorListOutput.get(), &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachLerpListGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    DeviceMemPtr workspaceAddr(nullptr, &aclrtFree);
    if (workspaceSize > 0) {
        void* rawWorkspaceAddr = nullptr;
        ret = aclrtMalloc(&rawWorkspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddr.reset(rawWorkspaceAddr);
    }
    // 调用aclnnForeachLerpList第二段接口
    ret = aclnnForeachLerpList(workspaceAddr.get(), workspaceSize, executor, stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachLerpList failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream.get());
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    PrintOutResult(outShape1, out1DeviceAddr, "out1");
    PrintOutResult(outShape2, out2DeviceAddr, "out2");

    return 0;
}
```
