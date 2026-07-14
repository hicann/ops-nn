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
#include "aclnn_relu_grad_v3.h"

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

int64_t GetBroadcastIndex(int64_t outIndex, const std::vector<int64_t>& outShape, const std::vector<int64_t>& inShape)
{
    int64_t inIndex = 0;
    int64_t inStride = 1;
    for (int64_t outDim = static_cast<int64_t>(outShape.size()) - 1, inDim = static_cast<int64_t>(inShape.size()) - 1;
         outDim >= 0; --outDim, --inDim) {
        int64_t coord = outIndex % outShape[outDim];
        outIndex /= outShape[outDim];
        int64_t curInDim = (inDim >= 0) ? inShape[inDim] : 1;
        if (curInDim != 1) {
            inIndex += coord * inStride;
        }
        inStride *= curInDim;
    }
    return inIndex;
}

bool PrintOutResult(const std::vector<int64_t>& outShape, void** deviceAddr, const std::vector<float>& xHostData,
                    const std::vector<int64_t>& xShape, const std::vector<float>& yHostData,
                    const std::vector<int64_t>& yShape)
{
    auto totalSize = GetShapeSize(outShape);
    std::vector<float> resultData(totalSize, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return false);

    int64_t mismatchCount = 0;
    for (int64_t i = 0; i < totalSize; i++) {
        int64_t xIndex = GetBroadcastIndex(i, outShape, xShape);
        int64_t yIndex = GetBroadcastIndex(i, outShape, yShape);
        float expected = (xHostData[xIndex] > 0.0f) ? yHostData[yIndex] : 0.0f;
        if (resultData[i] != expected) {
            if (mismatchCount < 10) {
                LOG_PRINT("Mismatch out[%ld], x[%ld]=%f, y[%ld]=%f, result=%f, expected=%f\n", i, xIndex,
                          xHostData[xIndex], yIndex, yHostData[yIndex], resultData[i], expected);
            }
            mismatchCount++;
        }
    }
    LOG_PRINT("Broadcast check: total=%ld, mismatch=%ld\n", totalSize, mismatchCount);

    auto printSize = std::min(totalSize, static_cast<int64_t>(10));
    LOG_PRINT("Notice: Only printing the first 10 elements.\n");
    for (int64_t i = 0; i < printSize; i++) {
        int64_t xIndex = GetBroadcastIndex(i, outShape, xShape);
        int64_t yIndex = GetBroadcastIndex(i, outShape, yShape);
        float xVal = xHostData[xIndex];
        float yVal = yHostData[yIndex];
        float expected = (xVal > 0.0f) ? yVal : 0.0f;
        LOG_PRINT("relu_grad_v3 out[%ld], x[%ld]=%f, y[%ld]=%f, result=%f, expected=%f\n", i, xIndex, xVal, yIndex,
                  yVal, resultData[i], expected);
    }
    return mismatchCount == 0;
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
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // Input: x (ReLU forward input)
    aclTensor* x = nullptr;
    void* xDeviceAddr = nullptr;
    std::vector<int64_t> xShape = {8, 32, 1, 64};
    std::vector<float> xHostData(GetShapeSize(xShape));
    for (size_t i = 0; i < xHostData.size(); ++i) {
        xHostData[i] = (i % 5 == 0) ? -1.0f : static_cast<float>((i % 17) + 1) / 17.0f;
    }
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Input: y (grad_output)
    aclTensor* y = nullptr;
    void* yDeviceAddr = nullptr;
    std::vector<int64_t> yShape = {1, 32, 128, 64};
    std::vector<float> yHostData(GetShapeSize(yShape));
    for (size_t i = 0; i < yHostData.size(); ++i) {
        yHostData[i] = static_cast<float>((i % 23) + 1) / 23.0f;
    }
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Output: z (grad_input)
    aclTensor* z = nullptr;
    void* zDeviceAddr = nullptr;
    std::vector<int64_t> zShape = {8, 32, 128, 64};
    std::vector<float> zHostData(GetShapeSize(zShape), 0.0f);
    ret = CreateAclTensor(zHostData, zShape, &zDeviceAddr, aclDataType::ACL_FLOAT, &z);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Call API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnReluGradV3GetWorkspaceSize(x, y, z, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReluGradV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnReluGradV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReluGradV3 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    bool checkOk = PrintOutResult(zShape, &zDeviceAddr, xHostData, xShape, yHostData, yShape);
    CHECK_RET(checkOk, return 1);

    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclDestroyTensor(z);

    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(zDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);

    aclFinalize();

    return 0;
}
