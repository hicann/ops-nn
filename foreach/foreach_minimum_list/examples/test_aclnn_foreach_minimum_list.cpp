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
#include "aclnnop/aclnn_foreach_minimum_list.h"

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
    // 调用aclnnForeachMinimumList第一段接口
    ret = aclnnForeachMinimumListGetWorkspaceSize(tensorListInput1, tensorListInput2, tensorListOutput, &workspaceSize,
                                                  &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachMinimumListGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnForeachMinimumList第二段接口
    ret = aclnnForeachMinimumList(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachMinimumList failed. ERROR: %d\n", ret); return ret);

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
