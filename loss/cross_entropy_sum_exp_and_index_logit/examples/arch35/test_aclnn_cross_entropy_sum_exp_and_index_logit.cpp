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
 * \file test_aclnn_cross_entropy_sum_exp_and_index_logit.cpp
 * \brief aclnnCrossEntropySumExpAndIndexLogit 调用示例
 */
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cross_entropy_sum_exp_and_index_logit.h"

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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

void PrintOutIntResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<int32_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }
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

int main()
{
    // 1. 固定写法，device/stream初始化, 参考acl API
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口定义构造
    // 示例使用 2 维 shape：N=4, V_local=16（FLOAT32 时 V_local 需为 8 的倍数）
    int64_t N = 4;
    int64_t vLocal = 16;
    int64_t vocabStart = 0;
    int64_t vocabEnd = vLocal; // target 均落在 [0, 16) 内，target_mask 全 0

    std::vector<int64_t> logitsShape = {N, vLocal};
    std::vector<int64_t> targetShape = {N};

    void* logitsDeviceAddr = nullptr;
    void* targetDeviceAddr = nullptr;
    void* maxDeviceAddr = nullptr;
    void* predictedDeviceAddr = nullptr;
    void* sumExpDeviceAddr = nullptr;
    void* expLogitsDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* maskDeviceAddr = nullptr;

    aclTensor* logitsTensor = nullptr;
    aclTensor* targetTensor = nullptr;
    aclTensor* maxTensor = nullptr;
    aclTensor* predictedTensor = nullptr;
    aclTensor* sumExpTensor = nullptr;
    aclTensor* expLogitsTensor = nullptr;
    aclTensor* offsetTensor = nullptr;
    aclTensor* maskTensor = nullptr;

    // 构造示例输入数据
    std::vector<float> logitsHostData(N * vLocal, 0);
    for (int64_t i = 0; i < N; i++) {
        for (int64_t j = 0; j < vLocal; j++) {
            logitsHostData[i * vLocal + j] = static_cast<float>(i) * 1.0f + static_cast<float>(j) * 0.1f;
        }
    }
    std::vector<int32_t> targetHostData = {2, 5, 10, 15};
    std::vector<float> maxHostData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> predictedHostData(N, 0);
    std::vector<float> sumExpHostData(N, 0);
    std::vector<float> expLogitsHostData(N * vLocal, 0);
    std::vector<int32_t> offsetHostData(N, 0);
    std::vector<int32_t> maskHostData(N, 0);

    ret = CreateAclTensor(logitsHostData, logitsShape, &logitsDeviceAddr, aclDataType::ACL_FLOAT, &logitsTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT32, &targetTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maxHostData, targetShape, &maxDeviceAddr, aclDataType::ACL_FLOAT, &maxTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(predictedHostData, targetShape, &predictedDeviceAddr, aclDataType::ACL_FLOAT,
                          &predictedTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sumExpHostData, targetShape, &sumExpDeviceAddr, aclDataType::ACL_FLOAT, &sumExpTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expLogitsHostData, logitsShape, &expLogitsDeviceAddr, aclDataType::ACL_FLOAT,
                          &expLogitsTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(offsetHostData, targetShape, &offsetDeviceAddr, aclDataType::ACL_INT32, &offsetTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maskHostData, targetShape, &maskDeviceAddr, aclDataType::ACL_INT32, &maskTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用第一段接口
    ret = aclnnCrossEntropySumExpAndIndexLogitGetWorkspaceSize(logitsTensor, targetTensor, maxTensor, vocabStart,
                                                               vocabEnd, predictedTensor, sumExpTensor, expLogitsTensor,
                                                               offsetTensor, maskTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnCrossEntropySumExpAndIndexLogitGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用第二段接口
    ret = aclnnCrossEntropySumExpAndIndexLogit(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCrossEntropySumExpAndIndexLogit failed. ERROR: %d\n", ret);
              return ret);

    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值
    LOG_PRINT("predicted_logits:\n");
    PrintOutResult(targetShape, &predictedDeviceAddr);
    LOG_PRINT("sum_exp_logits:\n");
    PrintOutResult(targetShape, &sumExpDeviceAddr);
    LOG_PRINT("exp_logits:\n");
    PrintOutResult(logitsShape, &expLogitsDeviceAddr);
    LOG_PRINT("target_offset:\n");
    PrintOutIntResult(targetShape, &offsetDeviceAddr);
    LOG_PRINT("target_mask:\n");
    PrintOutIntResult(targetShape, &maskDeviceAddr);

    // 6. 释放aclTensor
    aclDestroyTensor(logitsTensor);
    aclDestroyTensor(targetTensor);
    aclDestroyTensor(maxTensor);
    aclDestroyTensor(predictedTensor);
    aclDestroyTensor(sumExpTensor);
    aclDestroyTensor(expLogitsTensor);
    aclDestroyTensor(offsetTensor);
    aclDestroyTensor(maskTensor);

    // 7. 释放device资源
    aclrtFree(logitsDeviceAddr);
    aclrtFree(targetDeviceAddr);
    aclrtFree(maxDeviceAddr);
    aclrtFree(predictedDeviceAddr);
    aclrtFree(sumExpDeviceAddr);
    aclrtFree(expLogitsDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(maskDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
