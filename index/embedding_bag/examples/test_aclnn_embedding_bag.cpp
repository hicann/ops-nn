/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file test_aclnn_embedding_bag.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_embedding_bag.h"

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
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> weightShape = {3, 3};
    std::vector<int64_t> indicesShape = {6};
    std::vector<int64_t> offsetsShape = {4};
    std::vector<int64_t> perSampleWeightsShape = {6};
    std::vector<int64_t> outputShape = {4, 3};
    std::vector<int64_t> offset2bagShape = {6};
    std::vector<int64_t> bagSizeShape = {4};
    std::vector<int64_t> maxIndicesShape = {4};

    void* weightDeviceAddr = nullptr;
    void* indicesDeviceAddr = nullptr;
    void* offsetsDeviceAddr = nullptr;
    void* perSampleWeightsDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;
    void* offset2bagDeviceAddr = nullptr;
    void* bagSizeDeviceAddr = nullptr;
    void* maxIndicesDeviceAddr = nullptr;

    aclTensor* weight = nullptr;
    aclTensor* indices = nullptr;
    aclTensor* offsets = nullptr;
    aclTensor* perSampleWeights = nullptr;
    aclTensor* output = nullptr;
    aclTensor* offset2bag = nullptr;
    aclTensor* bagSize = nullptr;
    aclTensor* maxIndices = nullptr;

    std::vector<float> weightHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<int64_t> indicesHostData = {1, 2, 0, 2, 2, 1};
    std::vector<int64_t> offsetsHostData = {0, 2, 4, 5};
    std::vector<float> perSampleWeightsHostData = {1, 1, 1, 1, 1, 1};
    std::vector<float> outputHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int64_t> offset2bagHostData = {0, 0, 0, 0, 0, 0};
    std::vector<int64_t> bagSizeHostData = {0, 0, 0, 0};
    std::vector<int64_t> maxIndicesHostData = {0, 0, 0, 0};

    //创建weight aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    //创建indices aclTensor
    ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    //创建offsets aclTensor
    ret = CreateAclTensor(offsetsHostData, offsetsShape, &offsetsDeviceAddr, aclDataType::ACL_INT64, &offsets);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    //创建perSampleWeights aclTensor
    ret = CreateAclTensor(
        perSampleWeightsHostData, perSampleWeightsShape, &perSampleWeightsDeviceAddr, aclDataType::ACL_FLOAT,
        &perSampleWeights);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    //创建output aclTensor
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    //创建offset2bag aclTensor
    ret = CreateAclTensor(
        offset2bagHostData, offset2bagShape, &offset2bagDeviceAddr, aclDataType::ACL_INT64, &offset2bag);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    //创建bagSize aclTensor
    ret = CreateAclTensor(bagSizeHostData, bagSizeShape, &bagSizeDeviceAddr, aclDataType::ACL_INT64, &bagSize);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    //创建maxIndices aclTensor
    ret = CreateAclTensor(
        maxIndicesHostData, maxIndicesShape, &maxIndicesDeviceAddr, aclDataType::ACL_INT64, &maxIndices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    //非tensor参数
    bool scaleGradByFreq = false;
    int64_t mode = 0;
    bool sparse = false;
    bool includeLastOffset = false;
    int64_t paddingIdx = 1;

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnEmbeddingBag第一段接口
    ret = aclnnEmbeddingBagGetWorkspaceSize(
        weight, indices, offsets, scaleGradByFreq, mode, sparse, perSampleWeights, includeLastOffset, paddingIdx,
        output, offset2bag, bagSize, maxIndices, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEmbeddingBagGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnEmbeddingBag第二段接口
    ret = aclnnEmbeddingBag(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEmbeddingBag failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto outputSize = GetShapeSize(outputShape);
    std::vector<float> outputResultData(outputSize, 0);
    ret = aclrtMemcpy(
        outputResultData.data(), outputResultData.size() * sizeof(outputResultData[0]), outputDeviceAddr,
        outputSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < outputSize; i++) {
        LOG_PRINT("outputResult[%ld] is: %f\n", i, outputResultData[i]);
    }

    auto offset2bagSize = GetShapeSize(offset2bagShape);
    std::vector<int64_t> offset2bagResultData(offset2bagSize, 0);
    ret = aclrtMemcpy(
        offset2bagResultData.data(), offset2bagResultData.size() * sizeof(offset2bagResultData[0]),
        offset2bagDeviceAddr, offset2bagSize * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < offset2bagSize; i++) {
        LOG_PRINT("offset2bagResult[%ld] is: %ld\n", i, offset2bagResultData[i]);
    }

    auto bagSizeSize = GetShapeSize(bagSizeShape);
    std::vector<int64_t> bagSizeResultData(bagSizeSize, 0);
    ret = aclrtMemcpy(
        bagSizeResultData.data(), bagSizeResultData.size() * sizeof(bagSizeResultData[0]), bagSizeDeviceAddr,
        bagSizeSize * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < bagSizeSize; i++) {
        LOG_PRINT("bagSizeResult[%ld] is: %ld\n", i, bagSizeResultData[i]);
    }

    auto maxIndicesSize = GetShapeSize(maxIndicesShape);
    std::vector<int64_t> maxIndicesResultData(maxIndicesSize, 0);
    ret = aclrtMemcpy(
        maxIndicesResultData.data(), maxIndicesResultData.size() * sizeof(maxIndicesResultData[0]),
        maxIndicesDeviceAddr, maxIndicesSize * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < maxIndicesSize; i++) {
        LOG_PRINT("maxIndicesResult[%ld] is: %ld\n", i, maxIndicesResultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(weight);
    aclDestroyTensor(indices);
    aclDestroyTensor(offsets);
    aclDestroyTensor(perSampleWeights);
    aclDestroyTensor(output);
    aclDestroyTensor(offset2bag);
    aclDestroyTensor(bagSize);
    aclDestroyTensor(maxIndices);

    // 7. 释放device资源, 需要根据具体API的接口定义修改
    aclrtFree(weightDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(offsetsDeviceAddr);
    aclrtFree(perSampleWeightsDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclrtFree(offset2bagDeviceAddr);
    aclrtFree(bagSizeDeviceAddr);
    aclrtFree(maxIndicesDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}