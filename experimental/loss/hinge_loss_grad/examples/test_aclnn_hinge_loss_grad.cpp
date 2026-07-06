/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_hinge_loss_grad.h"

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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr,
                    const std::vector<float>& predictHostData, const std::vector<float>& targetHostData,
                    const std::vector<float>& gradOutputHostData)
{
    auto size = std::min(GetShapeSize(shape), static_cast<int64_t>(10));
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    LOG_PRINT("Notice: Only printing the first 10 elements.\n");
    for (int64_t i = 0; i < size; i++) {
        float predictVal = predictHostData[i];
        float targetVal = targetHostData[i];
        float gradOutVal = gradOutputHostData[i];
        float margin = 1.0f - targetVal * predictVal;
        float expected = (margin > 0.0f) ? (-targetVal * gradOutVal) : 0.0f;
        LOG_PRINT("hinge_loss_grad predict[%ld]=%f, target[%ld]=%f, grad_output[%ld]=%f, result[%ld]=%f, expected=%f\n",
            i, predictVal, i, targetVal, i, gradOutVal, i, resultData[i], expected);
    }
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
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

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Input: predict
    aclTensor* predict = nullptr;
    void* predictDeviceAddr = nullptr;
    std::vector<int64_t> predictShape = {32, 4, 4, 4};
    std::vector<float> predictHostData = {0.5f, -0.3f, 2.0f, -1.5f, 0.8f, -0.7f, 1.2f, 0.1f,
                                          1.5f, -2.0f, 0.3f, -0.9f, 0.0f, 1.0f, -1.0f, 0.6f};
    predictHostData.resize(2048, 0.5f);
    ret = CreateAclTensor(predictHostData, predictShape, &predictDeviceAddr, aclDataType::ACL_FLOAT, &predict);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Input: target
    aclTensor* targetT = nullptr;
    void* targetDeviceAddr = nullptr;
    std::vector<int64_t> targetShape = {32, 4, 4, 4};
    std::vector<float> targetHostData = {1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
                                          1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f};
    targetHostData.resize(2048, 1.0f);
    ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &targetT);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Input: grad_output
    aclTensor* gradOutput = nullptr;
    void* gradOutputDeviceAddr = nullptr;
    std::vector<int64_t> gradOutputShape = {32, 4, 4, 4};
    std::vector<float> gradOutputHostData(2048, 0.25f);
    ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Output: grad_input
    aclTensor* gradInput = nullptr;
    void* gradInputDeviceAddr = nullptr;
    std::vector<int64_t> gradInputShape = {32, 4, 4, 4};
    std::vector<float> gradInputHostData(2048, 0.0f);
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Call API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnHingeLossGradGetWorkspaceSize(predict, targetT, gradOutput, gradInput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHingeLossGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnHingeLossGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHingeLossGrad failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    PrintOutResult(gradInputShape, &gradInputDeviceAddr, predictHostData, targetHostData, gradOutputHostData);

    aclDestroyTensor(predict);
    aclDestroyTensor(targetT);
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(gradInput);

    aclrtFree(predictDeviceAddr);
    aclrtFree(targetDeviceAddr);
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);

    aclFinalize();

    return 0;
}
