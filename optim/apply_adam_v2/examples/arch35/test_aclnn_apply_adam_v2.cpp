/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_apply_adam_v2.h"

namespace {
struct Resources {
    int32_t deviceId = 0;
    bool initialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    std::vector<void*> allocations;
    std::vector<aclTensor*> tensors;

    ~Resources()
    {
        if (stream != nullptr) {
            (void)aclrtSynchronizeStream(stream);
        }
        for (auto* tensor : tensors) {
            (void)aclDestroyTensor(tensor);
        }
        for (auto* allocation : allocations) {
            (void)aclrtFree(allocation);
        }
        if (stream != nullptr) {
            (void)aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            (void)aclrtResetDevice(deviceId);
        }
        if (initialized) {
            (void)aclFinalize();
        }
    }
};

bool CheckStatus(int status, const char* operation)
{
    if (status != 0) {
        std::fprintf(stderr, "%s failed: %d\n", operation, status);
        return false;
    }
    return true;
}

bool CreateTensor(Resources& resources, const std::vector<float>& values, const std::vector<int64_t>& shape,
                  aclTensor** tensor)
{
    const size_t bytes = values.size() * sizeof(float);
    void* data = nullptr;
    if (!CheckStatus(aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc")) {
        return false;
    }
    resources.allocations.push_back(data);
    if (!CheckStatus(aclrtMemcpy(data, bytes, values.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy")) {
        return false;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i) {
        strides[i - 2] = shape[i - 1] * strides[i - 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), data);
    if (*tensor == nullptr) {
        std::fprintf(stderr, "aclCreateTensor failed\n");
        return false;
    }
    resources.tensors.push_back(*tensor);
    return true;
}
} // namespace

int main()
{
    // Initialize the runtime, select a device and create a stream.
    Resources resources;
    if (!CheckStatus(aclInit(nullptr), "aclInit")) {
        return 1;
    }
    resources.initialized = true;
    if (!CheckStatus(aclrtSetDevice(resources.deviceId), "aclrtSetDevice")) {
        return 1;
    }
    resources.deviceSet = true;
    if (!CheckStatus(aclrtCreateStream(&resources.stream), "aclrtCreateStream")) {
        return 1;
    }

    // Create the input tensors in device memory, in API parameter order:
    // var, m, v, lr, beta1, beta2, epsilon, grad, maxGradNorm, globalGradNorm, weightDecay, stepSize.
    const std::array<std::vector<float>, 12> values = {{{2.0f, -4.0f},
                                                        {0.0f, 0.0f},
                                                        {4.0f, 16.0f},
                                                        {0.25f},
                                                        {0.5f},
                                                        {0.5f},
                                                        {0.5f},
                                                        {2.0f, -4.0f},
                                                        {-1.0f},
                                                        {0.0f},
                                                        {0.0f},
                                                        {0.125f}}};
    std::array<aclTensor*, 12> inputs = {};
    for (size_t i = 0; i < inputs.size(); ++i) {
        const std::vector<int64_t> shape = (i < 3 || i == 7) ? std::vector<int64_t>{2} : std::vector<int64_t>{1};
        if (!CreateTensor(resources, values[i], shape, &inputs[i])) {
            return 1;
        }
    }

    // Query the workspace size and execution plan, then allocate workspace if needed.
    char adamMode[] = "adam";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnApplyAdamV2GetWorkspaceSize(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
                                                   inputs[6], inputs[7], inputs[8], inputs[9], inputs[10], inputs[11],
                                                   adamMode, &workspaceSize, &executor);
    if (!CheckStatus(status, "aclnnApplyAdamV2GetWorkspaceSize")) {
        return 1;
    }
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        if (!CheckStatus(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace")) {
            return 1;
        }
        resources.allocations.push_back(workspace);
    }
    // Submit the update and wait for completion before reading the in-place results.
    if (!CheckStatus(aclnnApplyAdamV2(workspace, workspaceSize, executor, resources.stream), "aclnnApplyAdamV2") ||
        !CheckStatus(aclrtSynchronizeStream(resources.stream), "aclrtSynchronizeStream")) {
        return 1;
    }

    const char* names[] = {"var", "m", "v"};
    for (size_t i = 0; i < 3; ++i) {
        std::array<float, 2> result = {};
        if (!CheckStatus(aclrtMemcpy(result.data(), sizeof(result), resources.allocations[i], sizeof(result),
                                     ACL_MEMCPY_DEVICE_TO_HOST),
                         "aclrtMemcpy result")) {
            return 1;
        }
        std::printf("%s = [%.6f, %.6f]\n", names[i], result[0], result[1]);
    }
    return 0;
}
