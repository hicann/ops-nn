/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_max.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnQuantMaxTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("Init acl success.\n");

    std::vector<int64_t> xShape = {4, 4};
    std::vector<int64_t> scaleShape = {1};
    std::vector<int64_t> amaxShape = {1};

    // x 的值选择可以覆盖多种情况
    std::vector<float> xHostData = {1.0f, -2.0f, 3.0f, -4.0f,  5.0f,    -6.0f,  7.0f,    -8.0f,
                                    0.5f, -0.5f, 0.0f, 448.0f, -448.0f, 0.125f, -0.125f, 2.5f};

    // scale = 1.0
    std::vector<float> scaleHostData = {1.0f};

    // 输出 y 初始化为 0
    std::vector<uint8_t> yHostData(GetShapeSize(xShape), 0);

    // amax 初始化为 0
    std::vector<float> amaxHostData = {0.0f};

    void* xDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* amaxDeviceAddr = nullptr;
    aclTensor* xTensor = nullptr;
    aclTensor* scaleTensor = nullptr;
    aclTensor* yTensor = nullptr;
    aclTensor* amaxTensor = nullptr;

    // Create input x tensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &xTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(xTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create scale tensor (float, shape [1])
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scaleTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scaleTensorPtr(scaleTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create output y tensor (FLOAT8_E4M3FN, same shape as x)
    ret = CreateAclTensor(yHostData, xShape, &yDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &yTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(yTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create output amax tensor (float, shape [1])
    ret = CreateAclTensor(amaxHostData, amaxShape, &amaxDeviceAddr, aclDataType::ACL_FLOAT, &amaxTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> amaxTensorPtr(amaxTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> amaxDeviceAddrPtr(amaxDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call quant_max API
    int64_t dstType = 36; // FLOAT8_E4M3FN
    char* roundMode = const_cast<char*>("rint");
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // First stage: get workspace size
    ret = aclnnQuantMaxGetWorkspaceSize(
        xTensor, scaleTensor, roundMode, dstType, yTensor, amaxTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMaxGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("aclnnQuantMaxGetWorkspaceSize success, workspaceSize: %lu\n", workspaceSize);

    // Allocate workspace
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // Second stage: execute
    ret = aclnnQuantMax(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMax failed. ERROR: %d\n", ret); return ret);

    // 4. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("aclnnQuantMax execution success.\n");

    // 5. Get results
    // Copy y (FLOAT8_E4M3FN) back to host
    auto ySize = GetShapeSize(xShape);
    std::vector<uint8_t> yOutData(ySize, 0);
    ret = aclrtMemcpy(
        yOutData.data(), ySize * sizeof(uint8_t), yDeviceAddr, ySize * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y from device to host failed. ERROR: %d\n", ret); return ret);

    // Copy amax back to host
    std::vector<float> amaxOutData(1, 0);
    ret = aclrtMemcpy(amaxOutData.data(), sizeof(float), amaxDeviceAddr, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy amax from device to host failed. ERROR: %d\n", ret); return ret);

    // Print results
    auto size = GetShapeSize(xShape);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("yOut[%ld] is: %d\n", i, yOutData[i]);
    }
    LOG_PRINT("amaxOut is: %f\n", amaxOutData[0]);

    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnQuantMaxTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMaxTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    LOG_PRINT("All test cases passed!\n");
    return 0;
}