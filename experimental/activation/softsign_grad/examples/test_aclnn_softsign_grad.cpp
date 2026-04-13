/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file test_aclnn_softsign_grad.cpp
 * @brief aclnn 两段式调用示例 - SoftsignGrad
 *
 * 数学公式: output = gradients / (1 + |features|)^2
 *
 * 使用 aclnnSoftsignBackward 接口
 */

#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnn_softsign_backward.h"

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
                    const std::vector<float>& gradHost, const std::vector<float>& featHost)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]),
        *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);

    for (int64_t i = 0; i < size && i < 10; i++) {
        float expected = gradHost[i] / std::pow(1.0f + std::fabs(featHost[i]), 2.0f);
        LOG_PRINT("softsign_grad gradients[%ld]=%.6f, features[%ld]=%.6f, "
                  "result[%ld]=%.6f (expected=%.6f)\n",
                  i, gradHost[i], i, featHost[i], i, resultData[i], expected);
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
    const std::vector<T>& hostData, const std::vector<int64_t>& shape,
    void** deviceAddr, aclDataType dataType, aclTensor** tensor)
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
        shape.data(), shape.size(), dataType,
        strides.data(), 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 构造测试数据: shape = {4, 4, 4}, dtype = float32
    std::vector<int64_t> shape = {4, 4, 4};
    int64_t numElements = GetShapeSize(shape);

    // 输入 gradients
    std::vector<float> gradHost(numElements);
    for (int64_t i = 0; i < numElements; i++) {
        gradHost[i] = 1.0f;  // 均匀梯度
    }
    aclTensor* gradients = nullptr;
    void* gradDeviceAddr = nullptr;
    ret = CreateAclTensor(gradHost, shape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &gradients);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 输入 features
    std::vector<float> featHost(numElements);
    for (int64_t i = 0; i < numElements; i++) {
        featHost[i] = static_cast<float>(i) * 0.1f - 3.0f;  // [-3.0, 3.3]
    }
    aclTensor* features = nullptr;
    void* featDeviceAddr = nullptr;
    ret = CreateAclTensor(featHost, shape, &featDeviceAddr, aclDataType::ACL_FLOAT, &features);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 输出
    std::vector<float> outHost(numElements, 0.0f);
    aclTensor* out = nullptr;
    void* outDeviceAddr = nullptr;
    ret = CreateAclTensor(outHost, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用 aclnnSoftsignBackward 第一段
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnSoftsignBackwardGetWorkspaceSize(gradients, features, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnSoftsignBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用 aclnnSoftsignBackward 第二段
    ret = aclnnSoftsignBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnSoftsignBackward failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 打印结果
    PrintOutResult(shape, &outDeviceAddr, gradHost, featHost);

    // 释放资源
    aclDestroyTensor(gradients);
    aclDestroyTensor(features);
    aclDestroyTensor(out);
    aclrtFree(gradDeviceAddr);
    aclrtFree(featDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
