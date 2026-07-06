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
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_mish_backward.h"

#define CHECK_RET(cond, expr) \
    do {                      \
        if (!(cond)) {        \
            expr;             \
        }                     \
    } while (0)

namespace {

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclInit failed: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtSetDevice failed: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtCreateStream failed: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtMalloc failed: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtMemcpy failed: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return 0;
}

float StableSoftplus(float x) { return std::max(x, 0.0f) + std::log1p(std::exp(-std::fabs(x))); }

float MishGradRef(float grad, float x, float tanhx)
{
    return grad * (tanhx + x * (1.0f - tanhx * tanhx) / (1.0f + std::exp(-x)));
}

int RunCase(aclrtStream stream)
{
    std::vector<int64_t> shape = {8};
    std::vector<float> gradHost = {1.0f, -0.5f, 0.25f, 1.5f, -1.0f, 0.75f, 0.5f, -0.25f};
    std::vector<float> xHost = {-3.0f, -1.0f, -0.1f, 0.0f, 0.2f, 1.5f, 3.0f, 5.0f};
    std::vector<float> outHost(shape[0], 0.0f);
    std::vector<float> refHost(shape[0], 0.0f);
    int ret = 0;

    for (size_t i = 0; i < xHost.size(); ++i) {
        auto tanhx = std::tanh(StableSoftplus(xHost[i]));
        refHost[i] = MishGradRef(gradHost[i], xHost[i], tanhx);
    }

    void* gradDevice = nullptr;
    void* xDevice = nullptr;
    void* outDevice = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* gradTensor = nullptr;
    aclTensor* xTensor = nullptr;
    aclTensor* outTensor = nullptr;

    ret = CreateAclTensor(gradHost, shape, &gradDevice, ACL_FLOAT, &gradTensor);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    ret = CreateAclTensor(xHost, shape, &xDevice, ACL_FLOAT, &xTensor);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    ret = CreateAclTensor(outHost, shape, &outDevice, ACL_FLOAT, &outTensor);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnMishBackwardGetWorkspaceSize(gradTensor, xTensor, outTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("GetWorkspaceSize failed: %d\n", ret); goto cleanup);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, std::printf("workspace malloc failed: %d\n", ret); goto cleanup);
    }

    ret = aclnnMishBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclnnMishBackward failed: %d\n", ret); goto cleanup);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtSynchronizeStream failed: %d\n", ret); goto cleanup);

    ret = aclrtMemcpy(outHost.data(), outHost.size() * sizeof(float), outDevice, outHost.size() * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("result memcpy failed: %d\n", ret); goto cleanup);

    for (size_t i = 0; i < outHost.size(); ++i) {
        if (std::fabs(outHost[i] - refHost[i]) > 1e-5f) {
            std::printf("compat case mismatch at %zu: got=%f ref=%f\n", i, outHost[i], refHost[i]);
            ret = 1;
            goto cleanup;
        }
    }

cleanup:
    if (gradTensor != nullptr) {
        aclDestroyTensor(gradTensor);
    }
    if (xTensor != nullptr) {
        aclDestroyTensor(xTensor);
    }
    if (outTensor != nullptr) {
        aclDestroyTensor(outTensor);
    }
    if (gradDevice != nullptr) {
        aclrtFree(gradDevice);
    }
    if (xDevice != nullptr) {
        aclrtFree(xDevice);
    }
    if (outDevice != nullptr) {
        aclrtFree(outDevice);
    }
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    CHECK_RET(ret == 0, return ret);
    std::printf("compat example passed\n");
    return 0;
}

} // namespace

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, return ret);
    ret = RunCase(stream);
    CHECK_RET(ret == 0, return ret);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
