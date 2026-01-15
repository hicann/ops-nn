/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "aclnn_fused_linear_cross_entropy_loss_grad.h"

#define CHECK_RET(cond, return_expr) \
    do                               \
    {                                \
        if (!(cond))                 \
        {                            \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do                                  \
    {                                   \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
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
std::vector<T> GenZeroVector(const std::vector<int64_t>& shape) {
    // 1. 计算总元素数量
    size_t total = 1;
    for (auto dim : shape) {
        total *= dim;
    }

    // 2. 填充0
    std::vector<T> vec(total);
    for (auto& elem : vec) {
        elem = 0;
    }
    return vec;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateEmptyAclTensor(const std::vector<int64_t> &shape, void **deviceAddr,
                         aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t BT = 1024;
    int64_t V = 1024;
    int64_t H = 1024;
    std::vector<int64_t> gradShape = {BT};
    std::vector<int64_t> inputShape = {BT, H};
    std::vector<int64_t> weightShape = {V, H};
    std::vector<int64_t> targetMaskShape = {BT};
    std::vector<int64_t> maskedTargetShape = {BT};
    std::vector<int64_t> softmaxOptionalShape = {BT, V};
    std::vector<int64_t> inputGradOutShape = {BT, H};
    std::vector<int64_t> weightGradOutShape = {V, H};
    void *gradDeviceAddr = nullptr;
    void *inputDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *targetMaskDeviceAddr = nullptr;
    void *maskedTargetDeviceAddr = nullptr;
    void *softmaxOptionalDeviceAddr = nullptr;
    void *inputGradOutDeviceAddr = nullptr;
    void *weightGradOutDeviceAddr = nullptr;
    aclTensor *grad = nullptr;
    aclTensor *input = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *targetMask = nullptr;
    aclTensor *maskedTarget = nullptr;
    float labelSmoothing = 0.0;
    aclTensor *logitsMaxOptional = nullptr;
    aclTensor *sumExpLogitsOptional = nullptr;
    aclTensor *softmaxOptional = nullptr;
    aclTensor *inputGradOut = nullptr;
    aclTensor *weightGradOut = nullptr;
    // 创建aclTensor
    auto gradData = GenZeroVector<int32_t>(gradShape);
    ret = CreateAclTensor<int32_t>(gradData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto inputData = GenZeroVector<int16_t>(inputShape);
    ret = CreateAclTensor<int16_t>(inputData, inputShape, &inputDeviceAddr, aclDataType::ACL_BF16, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto weightData = GenZeroVector<int16_t>(weightShape);
    ret = CreateAclTensor<int16_t>(weightData, weightShape, &weightDeviceAddr, aclDataType::ACL_BF16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto targetMaskData = GenZeroVector<int8_t>(targetMaskShape);
    ret = CreateAclTensor<int8_t>(targetMaskData, targetMaskShape, &targetMaskDeviceAddr, aclDataType::ACL_UINT8, &targetMask);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto maskedTargetData = GenZeroVector<int32_t>(maskedTargetShape);
    ret = CreateAclTensor<int32_t>(maskedTargetData, maskedTargetShape, &maskedTargetDeviceAddr, aclDataType::ACL_INT32, &maskedTarget);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto softmaxOptionalData = GenZeroVector<int32_t>(softmaxOptionalShape);
    ret = CreateAclTensor<int32_t>(softmaxOptionalData, softmaxOptionalShape, &softmaxOptionalDeviceAddr, aclDataType::ACL_FLOAT, &softmaxOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto inputGradOutData = GenZeroVector<int16_t>(inputGradOutShape);
    ret = CreateAclTensor<int16_t>(inputGradOutData, inputGradOutShape, &inputGradOutDeviceAddr, aclDataType::ACL_BF16, &inputGradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto weightGradOutData = GenZeroVector<int16_t>(weightGradOutShape);
    ret = CreateAclTensor<int16_t>(weightGradOutData, weightGradOutShape, &weightGradOutDeviceAddr, aclDataType::ACL_BF16, &weightGradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 调用aclnnFusedLinearCrossEntropyLossGrad第一段接口
    ret = aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize(grad, input, weight, targetMask, maskedTarget, labelSmoothing, logitsMaxOptional, sumExpLogitsOptional, softmaxOptional, inputGradOut, weightGradOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnFusedLinearCrossEntropyLossGrad第二段接口
    ret = aclnnFusedLinearCrossEntropyLossGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedLinearCrossEntropyLossGrad failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    // inputGradOut
    auto size = GetShapeSize(inputGradOutShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), inputGradOutDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < 16; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(grad);
    aclDestroyTensor(input);
    aclDestroyTensor(weight);
    aclDestroyTensor(targetMask);
    aclDestroyTensor(maskedTarget);
    aclDestroyTensor(logitsMaxOptional);
    aclDestroyTensor(sumExpLogitsOptional);
    aclDestroyTensor(softmaxOptional);
    aclDestroyTensor(inputGradOut);
    aclDestroyTensor(weightGradOut);

    // 7. 释放Device资源，需要根据具体API的接口定义修改
    aclrtFree(gradDeviceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(targetMaskDeviceAddr);
    aclrtFree(maskedTargetDeviceAddr);
    aclrtFree(softmaxOptionalDeviceAddr);
    aclrtFree(inputGradOutDeviceAddr);
    aclrtFree(weightGradOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}