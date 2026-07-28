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
#include <cstdlib>
#include <vector>
#include "acl/acl.h"
#include "aclnn_huber_loss.h"

namespace {
int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

aclError CreateFloatTensor(const std::vector<float>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                           aclTensor** tensor)
{
    const int64_t elementCount = GetShapeSize(shape);
    if (elementCount < 0 || static_cast<size_t>(elementCount) != hostData.size()) {
        return ACL_ERROR_INVALID_PARAM;
    }
    const size_t bytes = hostData.size() * sizeof(float);
    aclError ret = aclrtMalloc(deviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddr, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ret;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    if (*tensor == nullptr) {
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ACL_ERROR_FAILURE;
    }
    return ACL_SUCCESS;
}

std::vector<float> ComputeGolden(const std::vector<float>& predictions, const std::vector<float>& targets, float delta)
{
    std::vector<float> golden(predictions.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
        const float error = predictions[i] - targets[i];
        const float absError = std::fabs(error);
        golden[i] = absError <= delta ? 0.5f * error * error : delta * (absError - 0.5f * delta);
    }
    return golden;
}

void PrintVector(const char* label, const std::vector<float>& values)
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
    constexpr float delta = 1.0f;
    constexpr float tolerance = 1e-5f;
    const std::vector<int64_t> shape = {7};
    const std::vector<float> predictionsHost = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
    const std::vector<float> targetsHost(predictionsHost.size(), 0.0f);
    const std::vector<float> golden = ComputeGolden(predictionsHost, targetsHost, delta);
    std::vector<float> lossHost(predictionsHost.size(), 0.0f);

    int32_t deviceId = 0;
    if (const char* value = std::getenv("ASCEND_DEVICE_ID")) {
        deviceId = std::atoi(value);
    }
    aclrtStream stream = nullptr;
    aclTensor* predictions = nullptr;
    aclTensor* targets = nullptr;
    aclTensor* loss = nullptr;
    void* predictionsDevice = nullptr;
    void* targetsDevice = nullptr;
    void* lossDevice = nullptr;
    void* workspace = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclError finalRet = ACL_SUCCESS;
    bool initialized = false;
    bool deviceSet = false;

    auto cleanup = [&]() {
        if (predictions != nullptr)
            aclDestroyTensor(predictions);
        if (targets != nullptr)
            aclDestroyTensor(targets);
        if (loss != nullptr)
            aclDestroyTensor(loss);
        if (workspace != nullptr)
            aclrtFree(workspace);
        if (predictionsDevice != nullptr)
            aclrtFree(predictionsDevice);
        if (targetsDevice != nullptr)
            aclrtFree(targetsDevice);
        if (lossDevice != nullptr)
            aclrtFree(lossDevice);
        if (stream != nullptr)
            aclrtDestroyStream(stream);
        if (deviceSet)
            aclrtResetDevice(deviceId);
        if (initialized)
            aclFinalize();
        return finalRet;
    };

    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::printf("aclInit failed. ERROR: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }
    initialized = true;
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtSetDevice failed. ERROR: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }
    deviceSet = true;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtCreateStream failed. ERROR: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }
    ret = CreateFloatTensor(predictionsHost, shape, &predictionsDevice, &predictions);
    if (ret == ACL_SUCCESS)
        ret = CreateFloatTensor(targetsHost, shape, &targetsDevice, &targets);
    if (ret == ACL_SUCCESS)
        ret = CreateFloatTensor(lossHost, shape, &lossDevice, &loss);
    if (ret != ACL_SUCCESS) {
        std::printf("CreateFloatTensor failed. ERROR: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }

    ret = aclnnHuberLossGetWorkspaceSize(predictions, targets, delta, loss, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::printf("aclnnHuberLossGetWorkspaceSize failed. ERROR: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::printf("aclrtMalloc workspace failed. ERROR: %d\n", ret);
            finalRet = ret;
            return cleanup();
        }
    }
    ret = aclnnHuberLoss(workspace, workspaceSize, executor, stream);
    if (ret == ACL_SUCCESS)
        ret = aclrtSynchronizeStream(stream);
    if (ret == ACL_SUCCESS) {
        ret = aclrtMemcpy(lossHost.data(), lossHost.size() * sizeof(float), lossDevice, lossHost.size() * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (ret != ACL_SUCCESS) {
        std::printf("HuberLoss execution failed. ERROR: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }

    float maxError = 0.0f;
    for (size_t i = 0; i < lossHost.size(); ++i) {
        maxError = std::max(maxError, std::fabs(lossHost[i] - golden[i]));
    }
    PrintVector("输入 predictions", predictionsHost);
    PrintVector("输入 targets", targetsHost);
    std::printf("delta: %.4f\n", delta);
    PrintVector("Golden loss", golden);
    PrintVector("NPU loss", lossHost);
    std::printf("最大误差（loss）: %.6f\n", maxError);
    const bool passed = maxError <= tolerance;
    std::printf("验证结果: %s\n", passed ? "PASS" : "FAIL");
    std::printf("------------------------------------------------------------\n");
    if (!passed)
        finalRet = ACL_ERROR_FAILURE;
    return cleanup();
}
