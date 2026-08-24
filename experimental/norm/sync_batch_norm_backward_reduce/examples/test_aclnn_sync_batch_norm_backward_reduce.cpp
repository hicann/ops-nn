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
#include <cstring>
#include <cstdint>
#include <algorithm>
#include "acl/acl.h"
#include "aclnn_sync_batch_norm_backward_reduce.h"

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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
              aclFinalize(); return ret);
    return 0;
}

static uint16_t FloatToHalf(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0)
        return sign;
    if (exp >= 31)
        return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

static uint16_t FloatToBFloat16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto elemCount = GetShapeSize(shape);
    int64_t elemSize = sizeof(T);
    switch (dataType) {
        case aclDataType::ACL_FLOAT16:
        case aclDataType::ACL_BF16:
        case aclDataType::ACL_INT16:
        case aclDataType::ACL_UINT16:
            elemSize = 2;
            break;
        case aclDataType::ACL_INT8:
        case aclDataType::ACL_UINT8:
        case aclDataType::ACL_BOOL:
            elemSize = 1;
            break;
        case aclDataType::ACL_INT64:
        case aclDataType::ACL_UINT64:
        case aclDataType::ACL_DOUBLE:
            elemSize = 8;
            break;
        default:
            break;
    }
    auto size = elemCount * elemSize;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    std::vector<uint8_t> convBuf(size);
    if (dataType == aclDataType::ACL_FLOAT16) {
        for (int64_t i = 0; i < elemCount; i++) {
            uint16_t h = FloatToHalf(static_cast<float>(hostData[i]));
            memcpy(convBuf.data() + i * 2, &h, 2);
        }
    } else if (dataType == aclDataType::ACL_BF16) {
        for (int64_t i = 0; i < elemCount; i++) {
            uint16_t b = FloatToBFloat16(static_cast<float>(hostData[i]));
            memcpy(convBuf.data() + i * 2, &b, 2);
        }
    } else if (dataType == aclDataType::ACL_DOUBLE) {
        for (int64_t i = 0; i < elemCount; i++) {
            double d = static_cast<double>(hostData[i]);
            memcpy(convBuf.data() + i * 8, &d, 8);
        }
    } else {
        auto copySize = std::min((int64_t)(elemCount * sizeof(T)), size);
        memcpy(convBuf.data(), hostData.data(), copySize);
    }
    ret = aclrtMemcpy(*deviceAddr, size, convBuf.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
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

    // 构造输入 tensor
    aclTensor* sum_dy = nullptr;
    void* sum_dyDeviceAddr = nullptr;
    aclTensor* sum_dy_dx_pad = nullptr;
    void* sum_dy_dx_padDeviceAddr = nullptr;
    aclTensor* mean = nullptr;
    void* meanDeviceAddr = nullptr;
    aclTensor* invert_std = nullptr;
    void* invert_stdDeviceAddr = nullptr;
    std::vector<int64_t> sum_dyShape = {5};
    std::vector<float> sum_dyHostData(5, 1);
    ret = CreateAclTensor(sum_dyHostData, sum_dyShape, &sum_dyDeviceAddr, aclDataType::ACL_FLOAT16, &sum_dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> sum_dy_dx_padShape = {5};
    std::vector<float> sum_dy_dx_padHostData(5, 1);
    ret = CreateAclTensor(sum_dy_dx_padHostData, sum_dy_dx_padShape, &sum_dy_dx_padDeviceAddr, aclDataType::ACL_FLOAT16,
                          &sum_dy_dx_pad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> meanShape = {5};
    std::vector<float> meanHostData(5, 1);
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT16, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> invert_stdShape = {5};
    std::vector<float> invert_stdHostData(5, 1);
    ret = CreateAclTensor(invert_stdHostData, invert_stdShape, &invert_stdDeviceAddr, aclDataType::ACL_FLOAT16,
                          &invert_std);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 构造输出 tensor
    aclTensor* sum_dy_xmu = nullptr;
    void* sum_dy_xmuDeviceAddr = nullptr;
    aclTensor* y = nullptr;
    void* yDeviceAddr = nullptr;
    std::vector<int64_t> sum_dy_xmuShape = {5};
    std::vector<float> sum_dy_xmuHostData(5, 0);
    ret = CreateAclTensor(sum_dy_xmuHostData, sum_dy_xmuShape, &sum_dy_xmuDeviceAddr, aclDataType::ACL_FLOAT16,
                          &sum_dy_xmu);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<int64_t> yShape = {5};
    std::vector<float> yHostData(5, 0);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用 aclnnSyncBatchNormBackwardReduceGetWorkspaceSize 第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnSyncBatchNormBackwardReduceGetWorkspaceSize(sum_dy, sum_dy_dx_pad, mean, invert_std, sum_dy_xmu, y,
                                                           &workspaceSize, &executor);
    CHECK_RET(ret == ACLNN_SUCCESS,
              LOG_PRINT("aclnnSyncBatchNormBackwardReduceGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 申请 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用 aclnnSyncBatchNormBackwardReduce 第二段接口
    ret = aclnnSyncBatchNormBackwardReduce(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACLNN_SUCCESS, LOG_PRINT("aclnnSyncBatchNormBackwardReduce failed. ERROR: %d\n", ret); return ret);

    // 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 释放资源
    aclDestroyTensor(sum_dy);
    aclrtFree(sum_dyDeviceAddr);
    aclDestroyTensor(sum_dy_dx_pad);
    aclrtFree(sum_dy_dx_padDeviceAddr);
    aclDestroyTensor(mean);
    aclrtFree(meanDeviceAddr);
    aclDestroyTensor(invert_std);
    aclrtFree(invert_stdDeviceAddr);

    aclDestroyTensor(sum_dy_xmu);
    aclrtFree(sum_dy_xmuDeviceAddr);
    aclDestroyTensor(y);
    aclrtFree(yDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    if (executor != nullptr) {
        aclDestroyOpExecutor(executor);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
