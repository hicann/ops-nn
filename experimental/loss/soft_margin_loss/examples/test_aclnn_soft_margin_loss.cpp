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

#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnn_soft_margin_loss.h"

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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr, aclDataType dtype)
{
    auto size = GetShapeSize(shape);
    if (dtype == ACL_FLOAT) {
        std::vector<float> resultData(size, 0);
        aclrtMemcpy(resultData.data(), size * sizeof(float), *deviceAddr, size * sizeof(float),
                     ACL_MEMCPY_DEVICE_TO_HOST);
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
        }
    } else if (dtype == ACL_FLOAT16) {
        // fp16: 拷回 uint16_t 再转 float 显示
        std::vector<uint16_t> resultData(size, 0);
        aclrtMemcpy(resultData.data(), size * sizeof(uint16_t), *deviceAddr, size * sizeof(uint16_t),
                     ACL_MEMCPY_DEVICE_TO_HOST);
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("result[%ld] is: 0x%04x\n", i, resultData[i]);
        }
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
    auto elemCount = hostData.size();
    auto size = elemCount * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
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

    // 2. 构造输入数据
    // SoftMarginLoss: loss = log(1 + exp(-target * self))
    // reduction: 0=none, 1=mean, 2=sum
    std::vector<int64_t> selfShape = {4, 8};
    int64_t totalNum = GetShapeSize(selfShape);

    // self: 输入预测值
    std::vector<float> selfHostData = {
         0.5f,  1.0f, -0.3f,  2.0f,  0.1f, -1.5f,  0.8f, -0.2f,
        -1.0f,  0.7f,  1.5f, -0.5f,  0.3f,  0.9f, -0.8f,  1.2f,
         0.4f, -0.6f,  1.1f, -1.3f,  0.6f, -0.4f,  1.4f,  0.2f,
        -0.9f,  1.3f, -0.1f,  0.0f,  1.6f, -1.1f,  0.5f, -0.7f
    };

    // target: 标签值（+1 或 -1）
    std::vector<float> targetHostData = {
         1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
         1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f
    };

    // 3. 计算 CPU golden (reduction=mean)
    int64_t reduction = 1;  // mean
    {
        double totalLoss = 0.0;
        LOG_PRINT("=== CPU Golden (reduction=mean) ===\n");
        for (int64_t i = 0; i < totalNum; i++) {
            double loss = std::log(1.0 + std::exp(-targetHostData[i] * selfHostData[i]));
            totalLoss += loss;
        }
        LOG_PRINT("expected mean loss: %f\n\n", totalLoss / totalNum);
    }

    // 4. 创建 aclTensor
    aclTensor* selfTensor = nullptr;
    void* selfDeviceAddr = nullptr;
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, ACL_FLOAT, &selfTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclTensor* targetTensor = nullptr;
    void* targetDeviceAddr = nullptr;
    ret = CreateAclTensor(targetHostData, selfShape, &targetDeviceAddr, ACL_FLOAT, &targetTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // output shape: reduction=none -> same as input; reduction=mean/sum -> scalar (0-dim)
    std::vector<int64_t> outShape = (reduction == 0) ? selfShape : std::vector<int64_t>{};
    int64_t outNum = (reduction == 0) ? totalNum : 1;
    std::vector<float> outHostData(outNum, 0.0f);

    aclTensor* outTensor = nullptr;
    void* outDeviceAddr = nullptr;
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, ACL_FLOAT, &outTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 5. 调用 aclnnSoftMarginLoss
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnSoftMarginLossGetWorkspaceSize(selfTensor, targetTensor, reduction, outTensor,
                                              &workspaceSize, &executor);
    LOG_PRINT("aclnnSoftMarginLossGetWorkspaceSize returned %d, workspaceSize=%llu\n",
              ret, (unsigned long long)workspaceSize);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnSoftMarginLossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnSoftMarginLoss(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSoftMarginLoss failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 6. 打印结果
    LOG_PRINT("=== NPU Output ===\n");
    PrintOutResult(outShape, &outDeviceAddr, ACL_FLOAT);

    // 7. 释放资源
    aclDestroyTensor(selfTensor);
    aclDestroyTensor(targetTensor);
    aclDestroyTensor(outTensor);

    aclrtFree(selfDeviceAddr);
    aclrtFree(targetDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
