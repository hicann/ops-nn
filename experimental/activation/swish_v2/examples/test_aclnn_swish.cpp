/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_swish.h"

#define CHECK_RET(cond, expr) \
    do {                      \
        if (!(cond)) {        \
            expr;             \
        }                     \
    } while (0)

#define LOG_PRINT(message, ...)              \
    do {                                     \
        std::printf(message, ##__VA_ARGS__); \
    } while (0)

namespace {

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (int64_t dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
}

aclError CreateAclTensor(const std::vector<int64_t>& shape, aclDataType dtype, void* deviceAddr, aclTensor** tensor)
{
    std::vector<int64_t> strides = MakeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), deviceAddr);
    return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

} // namespace

int main()
{
    const std::vector<int64_t> shape = {4, 2};
    const std::vector<float> selfHostData = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
    std::vector<float> outHostData(selfHostData.size(), 0.f);
    float betaValue = 1.1f;
    const size_t bytes = static_cast<size_t>(GetShapeSize(shape)) * sizeof(float);

    aclrtStream stream = nullptr;
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    aclScalar* betaOptional = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    int finalRet = 0;
    bool aclInitialized = false;
    bool deviceSet = false;

    int32_t deviceId = 0;
    if (const char* deviceIdEnv = std::getenv("ASCEND_DEVICE_ID")) {
        deviceId = std::atoi(deviceIdEnv);
    }

    auto cleanup = [&]() -> int {
        if (betaOptional != nullptr) {
            aclDestroyScalar(betaOptional);
        }
        if (self != nullptr) {
            aclDestroyTensor(self);
        }
        if (out != nullptr) {
            aclDestroyTensor(out);
        }
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        if (selfDeviceAddr != nullptr) {
            aclrtFree(selfDeviceAddr);
        }
        if (outDeviceAddr != nullptr) {
            aclrtFree(outDeviceAddr);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(deviceId);
        }
        if (aclInitialized) {
            aclFinalize();
        }
        return finalRet;
    };

    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclInit failed: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }
    aclInitialized = true;
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice(%d) failed: %d\n", deviceId, ret);
        finalRet = ret;
        return cleanup();
    }
    deviceSet = true;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }

    ret = aclrtMalloc(&selfDeviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc(self) failed: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }
    ret = aclrtMalloc(&outDeviceAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMalloc(out) failed: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }

    ret = aclrtMemcpy(selfDeviceAddr, bytes, selfHostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy(self) failed: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }
    ret = aclrtMemcpy(outDeviceAddr, bytes, outHostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy(out) failed: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }

    ret = CreateAclTensor(shape, ACL_FLOAT, selfDeviceAddr, &self);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("CreateAclTensor(self) failed: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }
    ret = CreateAclTensor(shape, ACL_FLOAT, outDeviceAddr, &out);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("CreateAclTensor(out) failed: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }
    betaOptional = aclCreateScalar(&betaValue, ACL_FLOAT);
    if (betaOptional == nullptr) {
        LOG_PRINT("aclCreateScalar failed\n");
        finalRet = ACL_ERROR_FAILURE;
        return cleanup();
    }

    ret = aclnnSwishGetWorkspaceSize(self, betaOptional, out, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnSwishGetWorkspaceSize failed: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("aclrtMalloc(workspace) failed: %d\n", ret);
            finalRet = ret;
            return cleanup();
        }
    }

    ret = aclnnSwish(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnSwish failed: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSynchronizeStream failed: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }

    ret = aclrtMemcpy(outHostData.data(), bytes, outDeviceAddr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy(result) failed: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }

    for (size_t i = 0; i < outHostData.size(); ++i) {
        std::printf("result[%zu] = %f\n", i, outHostData[i]);
    }

    finalRet = 0;
    return cleanup();
}
