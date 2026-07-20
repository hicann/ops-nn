/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "aclnn_masked_scatter.h"

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
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. Initialize device and stream.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Create input and output tensors.
    constexpr int64_t dataSize = 1024;
    std::vector<int64_t> selfShape = {dataSize};
    std::vector<int64_t> maskShape = {dataSize};
    void* selfDeviceAddr = nullptr;
    void* maskDeviceAddr = nullptr;
    void* sourceDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* mask = nullptr;
    aclTensor* source = nullptr;
    std::vector<float> selfHostData(dataSize, 0.0f);
    std::vector<uint8_t> maskHostData(dataSize, 0);
    int64_t sourceSize = 0;
    for (int64_t i = 0; i < dataSize; ++i) {
        selfHostData[i] = static_cast<float>(i);
        if (i % 3 == 0 || i % 7 == 0) {
            maskHostData[i] = 1;
            ++sourceSize;
        }
    }
    std::vector<int64_t> sourceShape = {sourceSize};
    std::vector<float> sourceHostData(sourceSize, 0.0f);
    for (int64_t i = 0; i < sourceSize; ++i) {
        sourceHostData[i] = 10000.0f + static_cast<float>(i);
    }
    std::vector<float> expectedHostData = selfHostData;
    int64_t sourceIdx = 0;
    for (int64_t i = 0; i < dataSize; ++i) {
        if (maskHostData[i] != 0) {
            expectedHostData[i] = sourceHostData[sourceIdx++];
        }
    }

    // Create self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create mask aclTensor.
    ret = CreateAclTensor(maskHostData, maskShape, &maskDeviceAddr, aclDataType::ACL_BOOL, &mask);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create source aclTensor.
    ret = CreateAclTensor(sourceHostData, sourceShape, &sourceDeviceAddr, aclDataType::ACL_FLOAT, &source);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call CANN operator API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnInplaceMaskedScatterGetWorkspaceSize(self, mask, source, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceMaskedScatterGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnInplaceMaskedScatter(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceMaskedScatter failed. ERROR: %d\n", ret); return ret);

    // 4. Wait for the task to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Copy result from device to host.
    auto size = GetShapeSize(selfShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    int64_t mismatch = 0;
    int64_t firstMismatch = -1;
    for (int64_t i = 0; i < size; i++) {
        if (resultData[i] != expectedHostData[i]) {
            if (firstMismatch < 0) {
                firstMismatch = i;
            }
            ++mismatch;
        }
    }
    LOG_PRINT("dataSize: %ld, sourceSize: %ld, mismatch: %ld\n", dataSize, sourceSize, mismatch);
    if (firstMismatch >= 0) {
        LOG_PRINT("first mismatch index: %ld, result: %f, expected: %f\n", firstMismatch, resultData[firstMismatch],
                  expectedHostData[firstMismatch]);
        return 1;
    }

    // 6. Release aclTensor.
    aclDestroyTensor(self);
    aclDestroyTensor(mask);
    aclDestroyTensor(source);

    // 7. Release device resources.
    aclrtFree(selfDeviceAddr);
    aclrtFree(maskDeviceAddr);
    aclrtFree(sourceDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
