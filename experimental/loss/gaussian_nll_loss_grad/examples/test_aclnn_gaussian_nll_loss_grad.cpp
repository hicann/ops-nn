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
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_gaussian_nll_loss_grad.h"

#define CHECK_RET(cond, action) \
    do {                        \
        if (!(cond)) {          \
            action;             \
        }                       \
    } while (0)

static int64_t ShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

static int CreateTensor(const std::vector<float>& host, const std::vector<int64_t>& shape, void** device,
                        aclTensor** tensor)
{
    const size_t bytes = static_cast<size_t>(ShapeSize(shape)) * sizeof(float);
    aclError ret = aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(*device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *device);
    return *tensor == nullptr ? 1 : 0;
}

static void PrintVector(const char* name, const std::vector<float>& values)
{
    std::printf("%s: [", name);
    for (size_t i = 0; i < values.size(); ++i) {
        std::printf(i + 1 == values.size() ? "%.7f" : "%.7f, ", values[i]);
    }
    std::printf("]\n");
}

int main()
{
    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclInit failed: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtSetDevice failed: %d\n", ret); return ret);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtCreateStream failed: %d\n", ret); return ret);

    const std::vector<int64_t> inputShape = {2, 3};
    const std::vector<int64_t> targetShape = {2, 1};
    const std::vector<int64_t> varShape = {2, 1};
    const std::vector<float> gradOutputHost = {1.0f, 0.5f, -1.0f, 2.0f, -0.5f, 1.5f};
    const std::vector<float> inputHost = {0.2f, -0.1f, 1.0f, 1.5f, 2.0f, 2.5f};
    const std::vector<float> targetHost = {0.0f, 2.0f};
    const std::vector<float> varHost = {0.5f, 2.0f};
    std::vector<float> gradInputHost(inputHost.size(), 0.0f);
    std::vector<float> gradVarHost(varHost.size(), 0.0f);

    void* gradOutputDevice = nullptr;
    void* inputDevice = nullptr;
    void* targetDevice = nullptr;
    void* varDevice = nullptr;
    void* gradInputDevice = nullptr;
    void* gradVarDevice = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* input = nullptr;
    aclTensor* target = nullptr;
    aclTensor* var = nullptr;
    aclTensor* gradInput = nullptr;
    aclTensor* gradVar = nullptr;
    CHECK_RET(CreateTensor(gradOutputHost, inputShape, &gradOutputDevice, &gradOutput) == 0, return 1);
    CHECK_RET(CreateTensor(inputHost, inputShape, &inputDevice, &input) == 0, return 1);
    CHECK_RET(CreateTensor(targetHost, targetShape, &targetDevice, &target) == 0, return 1);
    CHECK_RET(CreateTensor(varHost, varShape, &varDevice, &var) == 0, return 1);
    CHECK_RET(CreateTensor(gradInputHost, inputShape, &gradInputDevice, &gradInput) == 0, return 1);
    CHECK_RET(CreateTensor(gradVarHost, varShape, &gradVarDevice, &gradVar) == 0, return 1);

    constexpr bool full = true;
    constexpr float eps = 1e-6f;
    char reduction[] = "none";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnGaussianNllLossGradGetWorkspaceSize(gradOutput, input, target, var, full, eps, reduction, gradInput,
                                                   gradVar, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclnnGaussianNllLossGradGetWorkspaceSize failed: %d\n", ret);
              return ret);
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    ret = aclnnGaussianNllLossGrad(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclnnGaussianNllLossGrad failed: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(gradInputHost.data(), gradInputHost.size() * sizeof(float), gradInputDevice,
                      gradInputHost.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(gradVarHost.data(), gradVarHost.size() * sizeof(float), gradVarDevice,
                      gradVarHost.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> goldenInput(inputHost.size());
    std::vector<float> goldenVar(varHost.size(), 0.0f);
    for (size_t i = 0; i < inputHost.size(); ++i) {
        const size_t row = i / 3;
        const float d = inputHost[i] - targetHost[row];
        const float v = std::max(varHost[row], eps);
        goldenInput[i] = gradOutputHost[i] * d / v;
        goldenVar[row] += gradOutputHost[i] * 0.5f * (1.0f / v - d * d / (v * v));
    }
    float maxInputError = 0.0f;
    float maxVarError = 0.0f;
    for (size_t i = 0; i < goldenInput.size(); ++i) {
        maxInputError = std::max(maxInputError, std::fabs(goldenInput[i] - gradInputHost[i]));
    }
    for (size_t i = 0; i < goldenVar.size(); ++i) {
        maxVarError = std::max(maxVarError, std::fabs(goldenVar[i] - gradVarHost[i]));
    }
    PrintVector("输入 gradOutput", gradOutputHost);
    PrintVector("输入 input", inputHost);
    PrintVector("输入 target", targetHost);
    PrintVector("输入 var", varHost);
    std::printf("属性 full=true, eps=%.7g, reduction=%s\n", eps, reduction);
    PrintVector("Golden gradInput", goldenInput);
    PrintVector("NPU gradInput", gradInputHost);
    PrintVector("Golden gradVar", goldenVar);
    PrintVector("NPU gradVar", gradVarHost);
    std::printf("最大误差（gradInput）: %.9f\n", maxInputError);
    std::printf("最大误差（gradVar）: %.9f\n", maxVarError);
    const bool passed = maxInputError <= 1e-5f && maxVarError <= 1e-5f;
    std::printf("验证结果: %s\n", passed ? "PASS" : "FAIL");
    std::printf("------------------------------------------------------------\n");

    aclDestroyTensor(gradOutput);
    aclDestroyTensor(input);
    aclDestroyTensor(target);
    aclDestroyTensor(var);
    aclDestroyTensor(gradInput);
    aclDestroyTensor(gradVar);
    aclrtFree(gradOutputDevice);
    aclrtFree(inputDevice);
    aclrtFree(targetDevice);
    aclrtFree(varDevice);
    aclrtFree(gradInputDevice);
    aclrtFree(gradVarDevice);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return passed ? 0 : 1;
}
