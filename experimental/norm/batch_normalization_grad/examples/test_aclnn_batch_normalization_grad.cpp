/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
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
#include "aclnn_batch_normalization_grad.h"

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

void PrintOutResult(const std::vector<int64_t>& gradInputShape, void** gradInputDeviceAddr,
                    const std::vector<int64_t>& gradWeightShape, void** gradWeightDeviceAddr,
                    const std::vector<int64_t>& gradBiasShape, void** gradBiasDeviceAddr)
{
    auto size = std::min(GetShapeSize(gradInputShape), static_cast<int64_t>(10));
    std::vector<float> giData(size, 0);
    auto ret = aclrtMemcpy(
        giData.data(), giData.size() * sizeof(giData[0]), *gradInputDeviceAddr, size * sizeof(giData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy grad_input from device to host failed. ERROR: %d\n", ret); return);
    LOG_PRINT("Notice: Only printing the first 10 elements.\n");
    LOG_PRINT("grad_input:\n");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("  [%ld] %f\n", i, giData[i]);
    }

    auto gwSize = std::min(GetShapeSize(gradWeightShape), static_cast<int64_t>(10));
    std::vector<float> gwData(gwSize, 0);
    ret = aclrtMemcpy(
        gwData.data(), gwData.size() * sizeof(gwData[0]), *gradWeightDeviceAddr, gwSize * sizeof(gwData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy grad_weight from device to host failed. ERROR: %d\n", ret); return);
    LOG_PRINT("grad_weight:\n");
    for (int64_t i = 0; i < gwSize; i++) {
        LOG_PRINT("  [%ld] %f\n", i, gwData[i]);
    }

    auto gbSize = std::min(GetShapeSize(gradBiasShape), static_cast<int64_t>(10));
    std::vector<float> gbData(gbSize, 0);
    ret = aclrtMemcpy(
        gbData.data(), gbData.size() * sizeof(gbData[0]), *gradBiasDeviceAddr, gbSize * sizeof(gbData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy grad_bias from device to host failed. ERROR: %d\n", ret); return);
    LOG_PRINT("grad_bias:\n");
    for (int64_t i = 0; i < gbSize; i++) {
        LOG_PRINT("  [%ld] %f\n", i, gbData[i]);
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
    // 1. 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入
    std::vector<int64_t> dataShape = {2, 4, 8};
    std::vector<int64_t> paramShape = {4};
    int64_t dataTotal = GetShapeSize(dataShape);
    int64_t paramTotal = GetShapeSize(paramShape);

    // grad_output: 全1
    aclTensor* gradOutput = nullptr;
    void* gradOutputDeviceAddr = nullptr;
    std::vector<float> gradOutputHostData(dataTotal, 1.0f);
    ret = CreateAclTensor(gradOutputHostData, dataShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // input: 全2
    aclTensor* input = nullptr;
    void* inputDeviceAddr = nullptr;
    std::vector<float> inputHostData(dataTotal, 2.0f);
    ret = CreateAclTensor(inputHostData, dataShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // weight: 全1
    aclTensor* weight = nullptr;
    void* weightDeviceAddr = nullptr;
    std::vector<float> weightHostData(paramTotal, 1.0f);
    ret = CreateAclTensor(weightHostData, paramShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // bias: 全0
    aclTensor* bias = nullptr;
    void* biasDeviceAddr = nullptr;
    std::vector<float> biasHostData(paramTotal, 0.0f);
    ret = CreateAclTensor(biasHostData, paramShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // save_mean: 全0
    aclTensor* saveMean = nullptr;
    void* saveMeanDeviceAddr = nullptr;
    std::vector<float> saveMeanHostData(paramTotal, 0.0f);
    ret = CreateAclTensor(saveMeanHostData, paramShape, &saveMeanDeviceAddr, aclDataType::ACL_FLOAT, &saveMean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // save_invstd: 全1
    aclTensor* saveInvstd = nullptr;
    void* saveInvstdDeviceAddr = nullptr;
    std::vector<float> saveInvstdHostData(paramTotal, 1.0f);
    ret = CreateAclTensor(saveInvstdHostData, paramShape, &saveInvstdDeviceAddr, aclDataType::ACL_FLOAT, &saveInvstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 输出: grad_input
    aclTensor* gradInput = nullptr;
    void* gradInputDeviceAddr = nullptr;
    std::vector<float> gradInputHostData(dataTotal, 0);
    ret = CreateAclTensor(gradInputHostData, dataShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 输出: grad_weight
    aclTensor* gradWeight = nullptr;
    void* gradWeightDeviceAddr = nullptr;
    std::vector<float> gradWeightHostData(paramTotal, 0);
    ret = CreateAclTensor(gradWeightHostData, paramShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 输出: grad_bias
    aclTensor* gradBias = nullptr;
    void* gradBiasDeviceAddr = nullptr;
    std::vector<float> gradBiasHostData(paramTotal, 0);
    ret = CreateAclTensor(gradBiasHostData, paramShape, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // BatchNormalizationGrad epsilon 属性
    float epsilon = 1e-5f;

    // 3. 调用第一段接口获取workspace大小
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnBatchNormalizationGradGetWorkspaceSize(
        gradOutput, input, weight, bias, saveMean, saveInvstd, epsilon,
        gradInput, gradWeight, gradBias, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormalizationGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 4. 申请workspace内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. 调用第二段接口执行算子
    ret = aclnnBatchNormalizationGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormalizationGrad failed. ERROR: %d\n", ret); return ret);

    // 6. 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. 打印结果
    PrintOutResult(dataShape, &gradInputDeviceAddr, paramShape, &gradWeightDeviceAddr, paramShape, &gradBiasDeviceAddr);

    // 8. 释放资源
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(input);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(saveMean);
    aclDestroyTensor(saveInvstd);
    aclDestroyTensor(gradInput);
    aclDestroyTensor(gradWeight);
    aclDestroyTensor(gradBias);

    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(saveMeanDeviceAddr);
    aclrtFree(saveInvstdDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    aclrtFree(gradWeightDeviceAddr);
    aclrtFree(gradBiasDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
