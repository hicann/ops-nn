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
#include <cstdlib>
#include <numeric>
#include <vector>
#include "acl/acl.h"
#include "aclnn_gaussian_nll_loss.h"

namespace {
aclError CreateFloatTensor(const std::vector<float>& hostData, const std::vector<int64_t>& shape, void** device,
                           aclTensor** tensor)
{
    const size_t bytes = hostData.size() * sizeof(float);
    aclError ret = aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(*device, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *device);
    return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

std::vector<float> Golden(const std::vector<float>& input, const std::vector<float>& target,
                          const std::vector<float>& var, size_t lastDimension, bool full, float eps)
{
    constexpr float halfLogTwoPi = 0.91893853320467274178f;
    std::vector<float> loss(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const size_t row = i / lastDimension;
        const float difference = input[i] - target[row];
        const float variance = std::max(var[row], eps);
        loss[i] = 0.5f * (std::log(variance) + difference * difference / variance);
        if (full) {
            loss[i] += halfLogTwoPi;
        }
    }
    return loss;
}

void Print(const char* label, const std::vector<float>& values)
{
    std::printf("%s: [", label);
    for (size_t i = 0; i < values.size(); ++i) {
        std::printf(i + 1 == values.size() ? "%.6f" : "%.6f, ", values[i]);
    }
    std::printf("]\n");
}
} // namespace

int main()
{
    constexpr bool full = true;
    constexpr double eps = 1e-6;
    constexpr float tolerance = 1e-5f;
    char reduction[] = "mean";
    const std::vector<int64_t> inputShape = {2, 3};
    const std::vector<int64_t> targetShape = {2, 1};
    const std::vector<int64_t> varShape = {2};
    const std::vector<int64_t> lossShape = {1};
    const std::vector<float> inputHost = {0.0f, 0.5f, 1.0f, -1.0f, 0.0f, 1.0f};
    const std::vector<float> targetHost = {0.25f, -0.5f};
    const std::vector<float> varHost = {0.5f, 2.0f};
    const std::vector<float> elementwiseGolden = Golden(inputHost, targetHost, varHost, 3, full,
                                                        static_cast<float>(eps));
    const std::vector<float> golden = {std::accumulate(elementwiseGolden.begin(), elementwiseGolden.end(), 0.0f) /
                                       elementwiseGolden.size()};
    std::vector<float> lossHost(1, 0.0f);

    int32_t deviceId = 0;
    if (const char* value = std::getenv("ASCEND_DEVICE_ID")) {
        deviceId = std::atoi(value);
    }
    aclrtStream stream = nullptr;
    aclTensor* input = nullptr;
    aclTensor* target = nullptr;
    aclTensor* var = nullptr;
    aclTensor* loss = nullptr;
    void* inputDevice = nullptr;
    void* targetDevice = nullptr;
    void* varDevice = nullptr;
    void* lossDevice = nullptr;
    void* workspace = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclError finalRet = ACL_SUCCESS;
    bool initialized = false;
    bool deviceSet = false;
    auto cleanup = [&]() {
        if (input != nullptr) {
            aclDestroyTensor(input);
        }
        if (target != nullptr) {
            aclDestroyTensor(target);
        }
        if (var != nullptr) {
            aclDestroyTensor(var);
        }
        if (loss != nullptr) {
            aclDestroyTensor(loss);
        }
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        if (inputDevice != nullptr) {
            aclrtFree(inputDevice);
        }
        if (targetDevice != nullptr) {
            aclrtFree(targetDevice);
        }
        if (varDevice != nullptr) {
            aclrtFree(varDevice);
        }
        if (lossDevice != nullptr) {
            aclrtFree(lossDevice);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(deviceId);
        }
        if (initialized) {
            aclFinalize();
        }
        return finalRet;
    };

    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        finalRet = ret;
        return cleanup();
    }
    initialized = true;
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        finalRet = ret;
        return cleanup();
    }
    deviceSet = true;
    ret = aclrtCreateStream(&stream);
    if (ret == ACL_SUCCESS) {
        ret = CreateFloatTensor(inputHost, inputShape, &inputDevice, &input);
    }
    if (ret == ACL_SUCCESS) {
        ret = CreateFloatTensor(targetHost, targetShape, &targetDevice, &target);
    }
    if (ret == ACL_SUCCESS) {
        ret = CreateFloatTensor(varHost, varShape, &varDevice, &var);
    }
    if (ret == ACL_SUCCESS) {
        ret = CreateFloatTensor(lossHost, lossShape, &lossDevice, &loss);
    }
    if (ret != ACL_SUCCESS) {
        finalRet = ret;
        return cleanup();
    }

    ret = aclnnGaussianNllLossGetWorkspaceSize(input, target, var, full, eps, reduction, loss, &workspaceSize,
                                               &executor);
    if (ret == ACL_SUCCESS && workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (ret == ACL_SUCCESS) {
        ret = aclnnGaussianNllLoss(workspace, workspaceSize, executor, stream);
    }
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream);
    }
    if (ret == ACL_SUCCESS) {
        ret = aclrtMemcpy(lossHost.data(), lossHost.size() * sizeof(float), lossDevice, lossHost.size() * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (ret != ACL_SUCCESS) {
        finalRet = ret;
        return cleanup();
    }

    float maxError = 0.0f;
    for (size_t i = 0; i < lossHost.size(); ++i) {
        maxError = std::max(maxError, std::fabs(lossHost[i] - golden[i]));
    }
    Print("输入 input", inputHost);
    Print("输入 target", targetHost);
    Print("输入 var", varHost);
    std::printf("full: %s\n", full ? "true" : "false");
    std::printf("eps: %.8f\n", eps);
    std::printf("reduction: %s\n", reduction);
    Print("Golden loss", golden);
    Print("NPU loss", lossHost);
    std::printf("最大误差（loss）: %.6f\n", maxError);
    const bool passed = maxError <= tolerance;
    std::printf("验证结果: %s\n", passed ? "PASS" : "FAIL");
    std::printf("------------------------------------------------------------\n");
    if (!passed) {
        finalRet = ACL_ERROR_FAILURE;
    }
    return cleanup();
}
