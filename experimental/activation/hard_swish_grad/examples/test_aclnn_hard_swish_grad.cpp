/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnn_hardswish_backward.h"

namespace {

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

int CreateTensor(const std::vector<float>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                 aclTensor** tensor)
{
    const size_t byteSize = static_cast<size_t>(GetShapeSize(shape)) * sizeof(float);
    aclError ret = aclrtMalloc(deviceAddr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtMalloc failed, ret=%d\n", ret);
        return ret;
    }

    ret = aclrtMemcpy(*deviceAddr, byteSize, hostData.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtMemcpy host-to-device failed, ret=%d\n", ret);
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    if (*tensor == nullptr) {
        std::printf("aclCreateTensor failed\n");
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ACL_ERROR_FAILURE;
    }
    return ACL_SUCCESS;
}

float Golden(float grad, float x)
{
    if (x <= -3.0f) {
        return 0.0f;
    }
    if (x >= 3.0f) {
        return grad;
    }
    return grad * (x * 0.333333343f + 0.5f);
}

bool IsClose(float actual, float expected)
{
    constexpr float absTolerance = 1e-5f;
    constexpr float relTolerance = 1e-5f;
    return std::fabs(actual - expected) <= absTolerance + relTolerance * std::fabs(expected);
}

} // namespace

int main()
{
    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclTensor* grad = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclOpExecutor* executor = nullptr;
    void* gradDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    int result = ACL_SUCCESS;

    const std::vector<int64_t> shape = {9};
    const std::vector<float> gradData = {1.0f, 2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, 8.0f, -9.0f};
    const std::vector<float> xData = {-4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> outputData(shape[0], 0.0f);

    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::printf("aclInit failed, ret=%d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtSetDevice failed, ret=%d\n", ret);
        aclFinalize();
        return ret;
    }
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtCreateStream failed, ret=%d\n", ret);
        result = ret;
        goto cleanup;
    }

    ret = CreateTensor(gradData, shape, &gradDeviceAddr, &grad);
    if (ret != ACL_SUCCESS) {
        result = ret;
        goto cleanup;
    }
    ret = CreateTensor(xData, shape, &xDeviceAddr, &x);
    if (ret != ACL_SUCCESS) {
        result = ret;
        goto cleanup;
    }
    ret = CreateTensor(outputData, shape, &yDeviceAddr, &y);
    if (ret != ACL_SUCCESS) {
        result = ret;
        goto cleanup;
    }

    ret = aclnnHardswishBackwardGetWorkspaceSize(grad, x, y, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::printf("aclnnHardswishBackwardGetWorkspaceSize failed, ret=%d\n", ret);
        result = ret;
        goto cleanup;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::printf("workspace allocation failed, ret=%d\n", ret);
            result = ret;
            goto cleanup;
        }
    }

    ret = aclnnHardswishBackward(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::printf("aclnnHardswishBackward failed, ret=%d\n", ret);
        result = ret;
        goto cleanup;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtSynchronizeStream failed, ret=%d\n", ret);
        result = ret;
        goto cleanup;
    }

    {
        std::vector<float> actual(outputData.size(), 0.0f);
        const size_t byteSize = actual.size() * sizeof(float);
        ret = aclrtMemcpy(actual.data(), byteSize, yDeviceAddr, byteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            std::printf("aclrtMemcpy device-to-host failed, ret=%d\n", ret);
            result = ret;
            goto cleanup;
        }

        for (size_t i = 0; i < actual.size(); ++i) {
            const float expected = Golden(gradData[i], xData[i]);
            std::printf("[%zu] grad=%f, x=%f, actual=%f, expected=%f\n", i, gradData[i], xData[i], actual[i], expected);
            if (!IsClose(actual[i], expected)) {
                std::printf("HardSwishGrad result mismatch at index %zu\n", i);
                result = ACL_ERROR_FAILURE;
            }
        }
    }

    if (result == ACL_SUCCESS) {
        std::printf("HardSwishGrad eager example passed.\n");
    }

cleanup:
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    if (grad != nullptr) {
        aclDestroyTensor(grad);
    }
    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (y != nullptr) {
        aclDestroyTensor(y);
    }
    if (gradDeviceAddr != nullptr) {
        aclrtFree(gradDeviceAddr);
    }
    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (yDeviceAddr != nullptr) {
        aclrtFree(yDeviceAddr);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return result;
}
