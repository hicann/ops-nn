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
 * @file test_aclnn_apply_adam_d.cpp
 * @brief aclnnApplyAdamD two-stage calling example.
 */
#include <cstdio>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_apply_adam_d.h"

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
    for (auto dim : shape) {
        shapeSize *= dim;
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> shape = {4, 2};
    std::vector<int64_t> scalarShape = {1};
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
    void* workspaceAddr = nullptr;

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
    aclTensor* varOut = nullptr;
    aclTensor* mOut = nullptr;
    aclTensor* vOut = nullptr;

    std::vector<float> varHostData = {1.0f, 2.0f, -1.0f, 0.5f, 0.0f, -0.5f, 1.5f, -2.0f};
    std::vector<float> mHostData = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.1f, 0.2f, 0.3f};
    std::vector<float> vHostData = {0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.01f, 0.02f, 0.03f};
    std::vector<float> gradHostData = {0.5f, -0.3f, 0.1f, -0.2f, 0.4f, -0.1f, 0.3f, -0.4f};
    std::vector<float> beta1PowerHostData = {0.9f};
    std::vector<float> beta2PowerHostData = {0.999f};
    std::vector<float> lrHostData = {0.001f};
    std::vector<float> beta1HostData = {0.9f};
    std::vector<float> beta2HostData = {0.999f};
    std::vector<float> epsilonHostData = {1e-8f};

    ret = CreateAclTensor(varHostData, shape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mHostData, shape, &mDeviceAddr, aclDataType::ACL_FLOAT, &m);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(vHostData, shape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(beta1PowerHostData, scalarShape, &beta1PowerDeviceAddr, aclDataType::ACL_FLOAT, &beta1Power);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(beta2PowerHostData, scalarShape, &beta2PowerDeviceAddr, aclDataType::ACL_FLOAT, &beta2Power);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(lrHostData, scalarShape, &lrDeviceAddr, aclDataType::ACL_FLOAT, &lr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(beta1HostData, scalarShape, &beta1DeviceAddr, aclDataType::ACL_FLOAT, &beta1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(beta2HostData, scalarShape, &beta2DeviceAddr, aclDataType::ACL_FLOAT, &beta2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(epsilonHostData, scalarShape, &epsilonDeviceAddr, aclDataType::ACL_FLOAT, &epsilon);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradHostData, shape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    varOut = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, strides.data(), 0,
                             aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), varDeviceAddr);
    mOut = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, strides.data(), 0,
                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), mDeviceAddr);
    vOut = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, strides.data(), 0,
                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), vDeviceAddr);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    bool useLocking = false;
    bool useNesterov = false;
    ret = aclnnApplyAdamDGetWorkspaceSize(var, m, v, beta1Power, beta2Power, lr, beta1, beta2, epsilon, grad,
                                          useLocking, useNesterov, varOut, mOut, vOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamDGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnApplyAdamD(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamD failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(shape);
    std::vector<float> varResult(size, 0.0f);
    std::vector<float> mResult(size, 0.0f);
    std::vector<float> vResult(size, 0.0f);
    ret = aclrtMemcpy(varResult.data(), varResult.size() * sizeof(float), varDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy var result failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(mResult.data(), mResult.size() * sizeof(float), mDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy m result failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(vResult.data(), vResult.size() * sizeof(float), vDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy v result failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; ++i) {
        LOG_PRINT("var_out[%ld]=%.6f  m_out[%ld]=%.6f  v_out[%ld]=%.6f\n", i, varResult[i], i, mResult[i], i,
                  vResult[i]);
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
    aclDestroyTensor(varOut);
    aclDestroyTensor(mOut);
    aclDestroyTensor(vOut);
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
