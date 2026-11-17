/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_chunk_loss_arch35.cpp
 * \brief aclnnChunkLoss arch35 调用样例
 *
 * 演示 Ascend950 上 aclnnChunkLossGetWorkspaceSize / aclnnChunkLoss 的基本用法。
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_chunk_loss.h"

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

uint16_t FloatToFp16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000U;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffU) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffU;
    if (exponent <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
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
    auto size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    if (shape.empty()) {
        size = sizeof(T);
    }
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor（0维标量时shape为空）
    *tensor = aclCreateTensor(shape.empty() ? nullptr : shape.data(), shape.size(), dataType,
                              shape.empty() ? nullptr : strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.empty() ? nullptr : shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出：M=4，H=8，V=16。第3个label等于ignoreIndex
    std::vector<int64_t> hiddenShape = {4, 8};
    std::vector<int64_t> weightShape = {16, 8};
    std::vector<int64_t> labelsShape = {4};
    std::vector<int64_t> lossShape; // 0维标量
    std::vector<int64_t> maxOutShape = {4};
    std::vector<int64_t> sumOutShape = {4};
    std::vector<int64_t> dHShape = {4, 8};

    std::vector<uint16_t> hiddenHostData(32, FloatToFp16(0.1f));
    std::vector<uint16_t> weightHostData(128, FloatToFp16(0.01f));
    std::vector<int64_t> labelsHostData = {1, 2, -100, 0};
    std::vector<float> lossHostData = {0.0f};
    std::vector<float> maxOutHostData(4, 0.0f);
    std::vector<float> sumOutHostData(4, 0.0f);
    std::vector<uint16_t> dHHostData(32, 0);
    double alpha = 4.0;
    int64_t ignoreIndex = -100;

    void* hiddenDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* labelsDeviceAddr = nullptr;
    void* lossDeviceAddr = nullptr;
    void* maxOutDeviceAddr = nullptr;
    void* sumOutDeviceAddr = nullptr;
    void* dHDeviceAddr = nullptr;
    aclTensor* hidden = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* labels = nullptr;
    aclTensor* loss = nullptr;
    aclTensor* maxOut = nullptr;
    aclTensor* sumOut = nullptr;
    aclTensor* dH = nullptr;

    ret = CreateAclTensor(hiddenHostData, hiddenShape, &hiddenDeviceAddr, aclDataType::ACL_FLOAT16, &hidden);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(labelsHostData, labelsShape, &labelsDeviceAddr, aclDataType::ACL_INT64, &labels);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(lossHostData, lossShape, &lossDeviceAddr, aclDataType::ACL_FLOAT, &loss);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maxOutHostData, maxOutShape, &maxOutDeviceAddr, aclDataType::ACL_FLOAT, &maxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sumOutHostData, sumOutShape, &sumOutDeviceAddr, aclDataType::ACL_FLOAT, &sumOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dHHostData, dHShape, &dHDeviceAddr, aclDataType::ACL_FLOAT16, &dH);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // 调用aclnnChunkLoss第一段接口
    ret = aclnnChunkLossGetWorkspaceSize(hidden, weight, labels, alpha, ignoreIndex, loss, maxOut, sumOut, dH,
                                         &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnChunkLossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnChunkLoss第二段接口
    ret = aclnnChunkLoss(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnChunkLoss failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧结果拷贝至host侧，需要根据具体API的接口定义修改
    float lossResult = 0.0f;
    ret = aclrtMemcpy(&lossResult, sizeof(lossResult), lossDeviceAddr, sizeof(lossResult), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy loss from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("loss is: %f\n", lossResult);

    auto maxSize = GetShapeSize(maxOutShape);
    std::vector<float> maxResult(maxSize, 0.0f);
    ret = aclrtMemcpy(maxResult.data(), maxResult.size() * sizeof(maxResult[0]), maxOutDeviceAddr,
                      maxSize * sizeof(maxResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy maxOut from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < maxSize; ++i) {
        LOG_PRINT("maxOut[%ld] is: %f\n", i, maxResult[i]);
    }

    auto sumSize = GetShapeSize(sumOutShape);
    std::vector<float> sumResult(sumSize, 0.0f);
    ret = aclrtMemcpy(sumResult.data(), sumResult.size() * sizeof(sumResult[0]), sumOutDeviceAddr,
                      sumSize * sizeof(sumResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy sumOut from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < sumSize; ++i) {
        LOG_PRINT("sumOut[%ld] is: %f\n", i, sumResult[i]);
    }

    auto dHSize = GetShapeSize(dHShape);
    std::vector<uint16_t> dHResult(dHSize, 0);
    ret = aclrtMemcpy(dHResult.data(), dHResult.size() * sizeof(dHResult[0]), dHDeviceAddr,
                      dHSize * sizeof(dHResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dH from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < dHSize; ++i) {
        LOG_PRINT("dH[%ld] raw_fp16 is: %u\n", i, static_cast<unsigned>(dHResult[i]));
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(hidden);
    aclDestroyTensor(weight);
    aclDestroyTensor(labels);
    aclDestroyTensor(loss);
    aclDestroyTensor(maxOut);
    aclDestroyTensor(sumOut);
    aclDestroyTensor(dH);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(hiddenDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(labelsDeviceAddr);
    aclrtFree(lossDeviceAddr);
    aclrtFree(maxOutDeviceAddr);
    aclrtFree(sumOutDeviceAddr);
    aclrtFree(dHDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
