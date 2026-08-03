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
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_fused_patch_mlp.h"

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
    for (const auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

int PrintOutResult(const std::vector<int64_t>& shape, const void* deviceAddr)
{
    const auto size = GetShapeSize(shape);
    std::vector<aclFloat16> resultData(size, 0);
    const auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceAddr,
                                 size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; ++i) {
        LOG_PRINT("result[%ld] is: %f\n", i, aclFloat16ToFloat(resultData[i]));
    }
    return ACL_SUCCESS;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        aclFinalize();
        return ret;
    }

    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    if (deviceAddr == nullptr || tensor == nullptr) {
        LOG_PRINT("CreateAclTensor received a null output pointer.\n");
        return -1;
    }
    *deviceAddr = nullptr;
    *tensor = nullptr;

    const auto elementCount = GetShapeSize(shape);
    if (elementCount <= 0 || static_cast<uint64_t>(elementCount) != hostData.size()) {
        LOG_PRINT("CreateAclTensor received invalid shape or host data.\n");
        return -1;
    }
    const auto size = static_cast<size_t>(elementCount) * sizeof(T);

    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        const auto freeRet = aclrtFree(*deviceAddr);
        if (freeRet != ACL_SUCCESS) {
            LOG_PRINT("aclrtFree after aclrtMemcpy failure failed. ERROR: %d\n", freeRet);
        }
        *deviceAddr = nullptr;
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    if (*tensor == nullptr) {
        LOG_PRINT("aclCreateTensor failed.\n");
        const auto freeRet = aclrtFree(*deviceAddr);
        if (freeRet != ACL_SUCCESS) {
            LOG_PRINT("aclrtFree after aclCreateTensor failure failed. ERROR: %d\n", freeRet);
        }
        *deviceAddr = nullptr;
        return -1;
    }
    return ACL_SUCCESS;
}

int main()
{
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    const int64_t numLayers = 3;
    const int64_t patchRows = 4;
    const int64_t patchSize = 16;
    const int64_t hidden = 32;
    const std::vector<int64_t> xShape = {patchRows, patchSize};
    const int64_t totalWeights = patchSize * hidden + (numLayers - 1) * hidden * hidden;
    const std::vector<int64_t> weightsShape = {totalWeights};
    const std::vector<int64_t> biasesShape = {numLayers * hidden};
    const std::vector<int64_t> outputShape = {patchRows, hidden};

    std::vector<aclFloat16> xHostData(GetShapeSize(xShape));
    std::vector<aclFloat16> weightsHostData(GetShapeSize(weightsShape));
    std::vector<aclFloat16> biasesHostData(GetShapeSize(biasesShape));
    std::vector<aclFloat16> outputHostData(GetShapeSize(outputShape));
    for (size_t i = 0; i < xHostData.size(); ++i) {
        xHostData[i] = aclFloatToFloat16(0.1f * static_cast<float>(i + 1));
    }
    for (auto& value : weightsHostData) {
        value = aclFloatToFloat16(0.02f);
    }
    for (auto& value : biasesHostData) {
        value = aclFloatToFloat16(0.0f);
    }

    void* xDeviceAddr = nullptr;
    void* weightsDeviceAddr = nullptr;
    void* biasesDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weights = nullptr;
    aclTensor* biases = nullptr;
    aclTensor* output = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;

    auto cleanup = [&]() {
        for (aclTensor** tensor : {&x, &weights, &biases, &output}) {
            if (*tensor != nullptr) {
                aclDestroyTensor(*tensor);
                *tensor = nullptr;
            }
        }
        for (void** address :
             {&xDeviceAddr, &weightsDeviceAddr, &biasesDeviceAddr, &outputDeviceAddr, &workspaceAddr}) {
            if (*address != nullptr) {
                aclrtFree(*address);
                *address = nullptr;
            }
        }
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
    };

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);
    ret = CreateAclTensor(weightsHostData, weightsShape, &weightsDeviceAddr, ACL_FLOAT16, &weights);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);
    ret = CreateAclTensor(biasesHostData, biasesShape, &biasesDeviceAddr, ACL_FLOAT16, &biases);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, ACL_FLOAT16, &output);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);

    ret = aclnnFusedPatchMlpGetWorkspaceSize(x, weights, biases, numLayers, output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedPatchMlpGetWorkspaceSize failed. ERROR: %d\n", ret); cleanup();
              return ret);

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); cleanup(); return ret);
    }

    ret = aclnnFusedPatchMlp(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedPatchMlp failed. ERROR: %d\n", ret); cleanup(); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); cleanup(); return ret);

    ret = PrintOutResult(outputShape, outputDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, cleanup(); return ret);

    cleanup();
    return ACL_SUCCESS;
}
