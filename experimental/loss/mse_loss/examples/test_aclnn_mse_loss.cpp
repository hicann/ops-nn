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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "aclnn_mse_loss.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)              \
    do {                                     \
        std::printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

static uint16_t FloatToHalf(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) {
        return sign;
    }
    if (exp >= 31) {
        return sign | 0x7c00;
    }
    return sign | (exp << 10) | mant;
}

static uint16_t FloatToBFloat16(float f)
{
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return static_cast<uint16_t>(bits >> 16);
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    const int64_t elemCount = GetShapeSize(shape);
    CHECK_RET(elemCount >= 0 && static_cast<uint64_t>(elemCount) <= hostData.size(),
              LOG_PRINT("invalid tensor shape or insufficient host data.\n");
              return ACL_ERROR_FAILURE);
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
    const size_t size = static_cast<size_t>(elemCount * elemSize);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    std::vector<uint8_t> convBuf(size);
    if (dataType == aclDataType::ACL_FLOAT16) {
        for (int64_t i = 0; i < elemCount; ++i) {
            uint16_t h = FloatToHalf(static_cast<float>(hostData[i]));
            memcpy(convBuf.data() + i * 2, &h, 2);
        }
    } else if (dataType == aclDataType::ACL_BF16) {
        for (int64_t i = 0; i < elemCount; ++i) {
            uint16_t b = FloatToBFloat16(static_cast<float>(hostData[i]));
            memcpy(convBuf.data() + i * 2, &b, 2);
        }
    } else if (dataType == aclDataType::ACL_DOUBLE) {
        for (int64_t i = 0; i < elemCount; ++i) {
            double d = static_cast<double>(hostData[i]);
            memcpy(convBuf.data() + i * 8, &d, 8);
        }
    } else {
        const size_t copySize = std::min(static_cast<size_t>(elemCount) * sizeof(T), size);
        memcpy(convBuf.data(), hostData.data(), copySize);
    }
    ret = aclrtMemcpy(*deviceAddr, size, convBuf.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    const int64_t* shapeData = shape.empty() ? nullptr : shape.data();
    const int64_t* strideData = strides.empty() ? nullptr : strides.data();
    *tensor = aclCreateTensor(shapeData, shape.size(), dataType, strideData, 0, aclFormat::ACL_FORMAT_ND, shapeData,
                              shape.size(), *deviceAddr);
    if (*tensor == nullptr) {
        LOG_PRINT("aclCreateTensor failed.\n");
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ACL_ERROR_FAILURE;
    }
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    if (const char* deviceIdEnv = std::getenv("ASCEND_DEVICE_ID")) {
        deviceId = std::atoi(deviceIdEnv);
    }
    aclrtStream stream = nullptr;
    aclTensor* predict = nullptr;
    void* predictDeviceAddr = nullptr;
    aclTensor* label = nullptr;
    void* labelDeviceAddr = nullptr;
    aclTensor* y = nullptr;
    void* yDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int finalRet = ACL_SUCCESS;
    bool aclInitialized = false;
    bool deviceSet = false;

    auto cleanup = [&]() -> int {
        if (predict != nullptr) {
            aclDestroyTensor(predict);
        }
        if (label != nullptr) {
            aclDestroyTensor(label);
        }
        if (y != nullptr) {
            aclDestroyTensor(y);
        }
        if (workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }
        if (predictDeviceAddr != nullptr) {
            aclrtFree(predictDeviceAddr);
        }
        if (labelDeviceAddr != nullptr) {
            aclrtFree(labelDeviceAddr);
        }
        if (yDeviceAddr != nullptr) {
            aclrtFree(yDeviceAddr);
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
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }
    aclInitialized = true;
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }
    deviceSet = true;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }

    const std::vector<int64_t> predictShape = {3, 5};
    const std::vector<float> predictHostData(15, 1);
    ret = CreateAclTensor(predictHostData, predictShape, &predictDeviceAddr, aclDataType::ACL_FLOAT16, &predict);
    if (ret != ACL_SUCCESS) {
        finalRet = ret;
        return cleanup();
    }
    const std::vector<int64_t> labelShape = {3, 5};
    const std::vector<float> labelHostData(15, 1);
    ret = CreateAclTensor(labelHostData, labelShape, &labelDeviceAddr, aclDataType::ACL_FLOAT16, &label);
    if (ret != ACL_SUCCESS) {
        finalRet = ret;
        return cleanup();
    }
    const std::vector<int64_t> yShape = {};
    const std::vector<float> yHostData(1, 0);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    if (ret != ACL_SUCCESS) {
        finalRet = ret;
        return cleanup();
    }

    constexpr int64_t reduction = 1; // 0: none, 1: mean, 2: sum
    ret = aclnnMseLossGetWorkspaceSize(predict, label, reduction, y, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnMseLossGetWorkspaceSize failed. ERROR: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }

    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            finalRet = ret;
            return cleanup();
        }
    }

    ret = aclnnMseLoss(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnMseLoss failed. ERROR: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        finalRet = ret;
        return cleanup();
    }

    return cleanup();
}
