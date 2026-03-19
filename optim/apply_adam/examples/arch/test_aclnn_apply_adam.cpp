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
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_adam.h"

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

    std::vector<int64_t> varShape = {2, 2};
    std::vector<int64_t> mShape = {2, 2};
    std::vector<int64_t> vShape = {2, 2};
    std::vector<int64_t> beta1PowerShape = {1};
    std::vector<int64_t> beta2PowerShape = {1};
    std::vector<int64_t> lrShape = {1};
    std::vector<int64_t> beta1Shape = {1};
    std::vector<int64_t> beta2Shape = {1};
    std::vector<int64_t> epsilonShape = {1};
    std::vector<int64_t> gradShape = {2, 2};
    void* varDeviceAddr = nullptr;
    void* mDeviceAddr = nullptr;
    void* vDeviceAddr = nullptr;
    void* beta1PowerDeviceAddr = nullptr;
    void* beta2PowerDeviceAddr = nullptr;
    void* lrDeviceAddr = nullptr;
    void* beta1DeviceAddr = nullptr;
    void* beta2DeviceAddr = nullptr;
    void* epsilonDeviceAddr = nullptr;
    void* gradDeviceAddr = nullptr;
    aclTensor* var = nullptr;
    aclTensor* m = nullptr;
    aclTensor* v = nullptr;
    aclTensor* beta1Power = nullptr;
    aclTensor* beta2Power = nullptr;
    aclTensor* lr = nullptr;
    aclTensor* beta1 = nullptr;
    aclTensor* beta2 = nullptr;
    aclTensor* epsilon = nullptr;
    aclTensor* grad = nullptr;
    std::vector<float> varHostData = {0, 1, 2, 3};
    std::vector<float> mHostData = {0, 1, 2, 3};
    std::vector<float> vHostData = {0, 1, 2, 3};
    std::vector<float> beta1PowerHostData = {0.431};
    std::vector<float> beta2PowerHostData = {0.992};
    std::vector<float> lrHostData = {0.001};
    std::vector<float> beta1HostData = {0.9};
    std::vector<float> beta2HostData = {0.999};
    std::vector<float> epsilonHostData = {1e-8};
    std::vector<float> gradHostData = {0, 1, 2, 3};
    bool useLocking = false;
    bool useNesterov = false;

    ret = CreateAclTensor(varHostData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mHostData, mShape, &mDeviceAddr, aclDataType::ACL_FLOAT, &m);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        beta1PowerHostData, beta1PowerShape, &beta1PowerDeviceAddr, aclDataType::ACL_FLOAT, &beta1Power);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        beta2PowerHostData, beta2PowerShape, &beta2PowerDeviceAddr, aclDataType::ACL_FLOAT, &beta2Power);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(lrHostData, lrShape, &lrDeviceAddr, aclDataType::ACL_FLOAT, &lr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(beta1HostData, beta1Shape, &beta1DeviceAddr, aclDataType::ACL_FLOAT, &beta1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(beta2HostData, beta2Shape, &beta2DeviceAddr, aclDataType::ACL_FLOAT, &beta2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(epsilonHostData, epsilonShape, &epsilonDeviceAddr, aclDataType::ACL_FLOAT, &epsilon);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnApplyAdamGetWorkspaceSize(
        var, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad, useLocking, useNesterov, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnApplyAdam(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdam failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(varShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), varDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    aclDestroyTensor(var);
    aclDestroyTensor(m);
    aclDestroyTensor(v);
    aclDestroyTensor(beta1Power);
    aclDestroyTensor(beta2Power);
    aclDestroyTensor(lr);
    aclDestroyTensor(beta1);
    aclDestroyTensor(beta2);
    aclDestroyTensor(epsilon);
    aclDestroyTensor(grad);

    aclrtFree(varDeviceAddr);
    aclrtFree(mDeviceAddr);
    aclrtFree(vDeviceAddr);
    aclrtFree(beta1PowerDeviceAddr);
    aclrtFree(beta2PowerDeviceAddr);
    aclrtFree(lrDeviceAddr);
    aclrtFree(beta1DeviceAddr);
    aclrtFree(beta2DeviceAddr);
    aclrtFree(epsilonDeviceAddr);
    aclrtFree(gradDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
