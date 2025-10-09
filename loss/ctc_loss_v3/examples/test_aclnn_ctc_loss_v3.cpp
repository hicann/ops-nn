/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <unistd.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_ctc_loss.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化, 参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> logProbsShape = {12, 4, 5};
    std::vector<int64_t> targetsShape = {4, 7};
    std::vector<int64_t> negLoglikelihoodOutShape = {4};
    std::vector<int64_t> logAlphaOutShape = {4, 12, 16};
    void* logProbsDeviceAddr = nullptr;
    void* targetsDeviceAddr = nullptr;
    void* negLoglikelihoodOutDeviceAddr = nullptr;
    void* logAlphaOutDeviceAddr = nullptr;
    aclTensor* logProbs = nullptr;
    aclTensor* targets = nullptr;
    aclIntArray* inputLengths = nullptr;
    aclIntArray* targetLengths = nullptr;
    aclTensor* negLoglikelihoodOut = nullptr;
    aclTensor* logAlphaOut = nullptr;
    std::vector<float> logProbsHostData = {
        -1.0894, -2.7162, -0.9764, -1.9126, -2.6162, -2.0684, -2.4871, -2.0866, -1.7205, -0.7187,
        -2.4423, -1.2017, -1.4653, -1.1821, -2.5942, -2.4670, -2.7257, -1.4135, -2.1042, -0.7248,

        -3.7759, -1.3742, -1.2549, -1.5807, -1.4562, -1.3826, -1.8995, -1.8527, -0.9493, -2.8895,
        -1.6316, -2.6603, -2.5014, -0.6992, -1.8609, -1.9269, -2.2350, -0.8073, -1.8906, -1.8947,

        -0.3468, -2.5855, -2.0723, -2.7147, -3.6668, -0.9541, -1.7258, -2.0693, -1.6378, -2.1531,
        -3.5386, -3.4830, -0.2532, -2.0557, -3.3261, -1.1480, -1.8080, -0.8244, -3.2414, -3.1909,

        -0.8866, -0.7540, -4.4312, -3.4634, -2.6000, -1.2785, -1.8347, -3.3122, -0.7620, -2.8349,
        -1.4975, -1.3865, -0.9645, -3.8171, -2.0939, -2.3536, -2.0773, -1.4981, -0.8372, -2.0938,

        -1.2186, -0.8285, -2.9399, -2.1159, -2.3620, -2.3139, -0.6503, -2.7249, -1.2340, -3.7927,
        -0.7143, -2.5084, -3.2826, -2.6651, -1.1334, -1.6965, -1.9728, -2.3849, -1.6052, -0.9554,

        -1.6384, -1.2596, -2.1680, -1.8476, -1.3866, -3.0455, -0.5737, -2.5339, -2.1118, -1.6681,
        -2.4675, -2.8842, -0.4329, -3.6266, -1.6925, -3.1023, -2.7696, -1.2755, -0.6470, -2.4143,

        -2.0107, -2.0912, -1.3053, -0.8557, -3.0683, -1.2872, -3.6523, -1.6703, -2.7596, -0.8063,
        -2.4633, -1.2959, -1.6153, -2.3072, -1.0705, -3.0543, -0.6473, -1.1650, -2.9025, -2.7710,

        -3.5519, -2.0400, -1.8667, -1.4289, -0.8050, -1.4602, -0.7452, -1.5754, -3.1624, -3.1247,
        -1.4677, -1.2725, -2.9575, -1.8883, -1.2513, -1.2164, -1.5894, -2.2217, -2.3714, -1.2110,

        -2.0843, -0.6515, -1.4252, -2.9402, -2.7964, -1.5261, -2.5471, -1.7167, -1.9846, -0.9488,
        -1.4847, -1.7093, -1.4095, -1.7293, -1.7675, -0.9203, -4.2299, -1.8740, -1.4076, -1.6671,

        -1.9052, -0.8330, -2.1839, -2.2459, -1.6193, -2.9108, -1.2114, -1.4616, -1.7297, -1.4330,
        -2.2656, -0.7878, -1.8533, -1.8711, -2.0349, -2.2457, -2.1395, -1.4509, -0.7538, -2.6381,

        -0.8078, -2.1054, -2.6703, -1.1108, -3.3867, -1.7774, -1.8426, -1.9473, -1.3293, -1.3273,
        -1.3490, -1.9842, -2.5357, -2.2161, -0.8800, -1.5412, -1.8003, -2.7603, -0.8606, -2.0066,

        -1.8342, -2.2741, -1.8348, -1.5833, -0.9877, -3.5196, -2.3361, -0.9124, -0.9307, -2.5531,
        -1.4862, -1.2153, -1.4453, -3.4462, -1.5625, -2.6455, -1.4153, -1.3079, -1.1568, -2.2897};
    std::vector<int64_t> targetsHostData = {1, 2, 1, 1, 2, 4, 1, 2, 2, 2, 2, 2, 2, 3,
                                            4, 2, 1, 4, 3, 1, 4, 4, 1, 4, 2, 2, 2, 3};

    std::vector<float> negLoglikelihoodOutHostData = {0, 0, 0, 0};
    std::vector<float> logAlphaOutHostData = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // 创建logProbs aclTensor
    ret = CreateAclTensor(logProbsHostData, logProbsShape, &logProbsDeviceAddr, aclDataType::ACL_FLOAT, &logProbs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建targets aclTensor
    ret = CreateAclTensor(targetsHostData, targetsShape, &targetsDeviceAddr, aclDataType::ACL_INT64, &targets);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> inputLengthsSizeData = {10, 10, 10, 10};
    inputLengths = aclCreateIntArray(inputLengthsSizeData.data(), 4);
    CHECK_RET(inputLengths != nullptr, return ACL_ERROR_BAD_ALLOC);
    std::vector<int64_t> targetLengthsSizeData = {2, 3, 1, 5};
    targetLengths = aclCreateIntArray(targetLengthsSizeData.data(), 4);
    CHECK_RET(targetLengths != nullptr, return ACL_ERROR_BAD_ALLOC);

    // 创建negLoglikelihoodOut aclTensor
    ret = CreateAclTensor(
        negLoglikelihoodOutHostData, negLoglikelihoodOutShape, &negLoglikelihoodOutDeviceAddr, aclDataType::ACL_FLOAT,
        &negLoglikelihoodOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建logAlphaOut aclTensor
    ret = CreateAclTensor(
        logAlphaOutHostData, logAlphaOutShape, &logAlphaOutDeviceAddr, aclDataType::ACL_FLOAT, &logAlphaOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnCtcLoss第一段接口
    ret = aclnnCtcLossGetWorkspaceSize(
        logProbs, targets, inputLengths, targetLengths, 0, false, negLoglikelihoodOut, logAlphaOut, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCtcLossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnCtcLoss第二段接口
    ret = aclnnCtcLoss(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCtcLoss failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的negLoglikelihoodOut值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(negLoglikelihoodOutShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), negLoglikelihoodOutDeviceAddr,
        size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("negLoglikelihoodOut result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 获取输出的logAlphaOut值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size1 = GetShapeSize(logAlphaOutShape);
    std::vector<float> resultData1(size1, 0);
    ret = aclrtMemcpy(
        resultData1.data(), resultData1.size() * sizeof(resultData1[0]), logAlphaOutDeviceAddr, size1 * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size1; i++) {
        LOG_PRINT("logAlphaOut result[%ld] is: %f\n", i, resultData1[i]);
    }

    // 7. 释放aclTensor和IntArray，需要根据具体API的接口定义修改
    aclDestroyTensor(logProbs);
    aclDestroyTensor(targets);
    aclDestroyIntArray(inputLengths);
    aclDestroyIntArray(targetLengths);
    aclDestroyTensor(negLoglikelihoodOut);
    aclDestroyTensor(logAlphaOut);

    // 8. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(logProbsDeviceAddr);
    aclrtFree(targetsDeviceAddr);
    aclrtFree(negLoglikelihoodOutDeviceAddr);
    aclrtFree(logAlphaOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}