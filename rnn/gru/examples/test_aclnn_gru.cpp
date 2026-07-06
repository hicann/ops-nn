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
 * @file test_aclnn_gru.cpp
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gru.h"

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

void PrintOutResult(const std::string& name, const std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s from device to host failed. ERROR: %d\n", name.c_str(), ret);
              return );
    LOG_PRINT("=== %s shape=[", name.c_str());
    for (size_t i = 0; i < shape.size(); i++) {
        LOG_PRINT("%ld%s", shape[i], (i + 1 < shape.size()) ? "," : "");
    }
    LOG_PRINT("] ===\n");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("  [%ld] = %f\n", i, resultData[i]);
    }
    LOG_PRINT("\n");
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，AscendCL初始化
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
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
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
int CreateAclTensorList(const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr, aclDataType dataType,
                        aclTensorList** tensor, T initVal = 1)
{
    int size = shapes.size();
    aclTensor* tensors[size];
    for (int i = 0; i < size; i++) {
        std::vector<T> hostData(GetShapeSize(shapes[i]), initVal);
        int ret = CreateAclTensor<float>(hostData, shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t timeStep = 2;
    int64_t batchSize = 3;
    int64_t inputSize = 4;
    int64_t hiddenSize = 5;
    int64_t gateNum = 3;
    int64_t numLayers = 1;
    bool isbias = false;
    bool batchFirst = false;
    bool bidirection = false;
    bool isTraining = true;
    double dropout = 0.0;
    int64_t dScale = bidirection ? 2 : 1;
    int64_t ldScale = numLayers * dScale;

    std::vector<int64_t> inputShape = {timeStep, batchSize, inputSize};
    std::vector<int64_t> outputShape = {timeStep, batchSize, dScale * hiddenSize};
    std::vector<int64_t> hyShape = {ldScale, batchSize, hiddenSize};
    std::vector<int64_t> hxShape = {ldScale, batchSize, hiddenSize};
    std::vector<std::vector<int64_t>> paramsListShape = {};

    auto curLayerInputSize = inputSize;
    for (int i = 0; i < numLayers; i++) {
        for (int64_t j = 0; j < dScale; j++) {
            paramsListShape.push_back({hiddenSize * gateNum, curLayerInputSize});
            paramsListShape.push_back({hiddenSize * gateNum, hiddenSize});
            if (isbias) {
                paramsListShape.push_back({hiddenSize * gateNum});
                paramsListShape.push_back({hiddenSize * gateNum});
            }
        }
        curLayerInputSize = dScale * hiddenSize;
    }

    std::vector<std::vector<int64_t>> rOutListShape;
    std::vector<std::vector<int64_t>> zOutListShape;
    std::vector<std::vector<int64_t>> nOutListShape;
    std::vector<std::vector<int64_t>> hnOutListShape;
    std::vector<std::vector<int64_t>> hOutListShape;

    if (isTraining) {
        for (int64_t i = 0; i < ldScale; i++) {
            rOutListShape.push_back({timeStep, batchSize, hiddenSize});
            zOutListShape.push_back({timeStep, batchSize, hiddenSize});
            nOutListShape.push_back({timeStep, batchSize, hiddenSize});
            hnOutListShape.push_back({timeStep, batchSize, hiddenSize});
            hOutListShape.push_back({timeStep, batchSize, hiddenSize});
        }
    }

    void* inputDeviceAddr = nullptr;
    std::vector<void*> paramsListDeviceAddr(paramsListShape.size(), nullptr);
    void* outputDeviceAddr = nullptr;
    void* hyDeviceAddr = nullptr;
    void* hxDeviceAddr = nullptr;

    std::vector<void*> rOutDeviceAddr;
    std::vector<void*> zOutDeviceAddr;
    std::vector<void*> nOutDeviceAddr;
    std::vector<void*> hnOutDeviceAddr;
    std::vector<void*> hOutDeviceAddr;

    aclTensor* input = nullptr;
    aclTensorList* params = nullptr;
    aclTensor* output = nullptr;
    aclTensor* hy = nullptr;
    aclTensor* hx = nullptr;

    aclTensorList* rOut = nullptr;
    aclTensorList* zOut = nullptr;
    aclTensorList* nOut = nullptr;
    aclTensorList* hnOut = nullptr;
    aclTensorList* hOut = nullptr;

    std::vector<float> inputHostData(GetShapeSize(inputShape), 1.0);
    ret = CreateAclTensor<float>(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(paramsListShape, paramsListDeviceAddr.data(), aclDataType::ACL_FLOAT, &params,
                                     1.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> outputHostData(GetShapeSize(outputShape), 0.0);
    ret = CreateAclTensor<float>(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> hyHostData(GetShapeSize(hyShape), 0.0);
    ret = CreateAclTensor<float>(hyHostData, hyShape, &hyDeviceAddr, aclDataType::ACL_FLOAT, &hy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    bool useHx = false;
    if (useHx) {
        std::vector<float> hxHostData(GetShapeSize(hxShape), 0.5);
        ret = CreateAclTensor<float>(hxHostData, hxShape, &hxDeviceAddr, aclDataType::ACL_FLOAT, &hx);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    if (isTraining) {
        rOutDeviceAddr.resize(ldScale, nullptr);
        zOutDeviceAddr.resize(ldScale, nullptr);
        nOutDeviceAddr.resize(ldScale, nullptr);
        hnOutDeviceAddr.resize(ldScale, nullptr);
        hOutDeviceAddr.resize(ldScale, nullptr);

        ret = CreateAclTensorList<float>(rOutListShape, rOutDeviceAddr.data(), aclDataType::ACL_FLOAT, &rOut, 0.0);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensorList<float>(zOutListShape, zOutDeviceAddr.data(), aclDataType::ACL_FLOAT, &zOut, 0.0);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensorList<float>(nOutListShape, nOutDeviceAddr.data(), aclDataType::ACL_FLOAT, &nOut, 0.0);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensorList<float>(hnOutListShape, hnOutDeviceAddr.data(), aclDataType::ACL_FLOAT, &hnOut, 0.0);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensorList<float>(hOutListShape, hOutDeviceAddr.data(), aclDataType::ACL_FLOAT, &hOut, 0.0);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 调用aclnnGRU第一段接口
    ret = aclnnGRUGetWorkspaceSize(
        input, params, (useHx ? hx : nullptr), nullptr, isbias, numLayers, dropout, isTraining, bidirection, batchFirst,
        output, hy, isTraining ? rOut : nullptr, isTraining ? zOut : nullptr, isTraining ? nOut : nullptr,
        isTraining ? hnOut : nullptr, isTraining ? hOut : nullptr, &workspaceSize, &executor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGRUGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnLSTM第二段接口
    ret = aclnnGRU(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGRU failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改

    PrintOutResult("output", outputShape, &outputDeviceAddr);
    PrintOutResult("hy", hyShape, &hyDeviceAddr);
    if (isTraining) {
        for (int64_t i = 0; i < ldScale; i++) {
            PrintOutResult("rOut[" + std::to_string(i) + "]", rOutListShape[i], &rOutDeviceAddr[i]);
            PrintOutResult("zOut[" + std::to_string(i) + "]", zOutListShape[i], &zOutDeviceAddr[i]);
            PrintOutResult("nOut[" + std::to_string(i) + "]", nOutListShape[i], &nOutDeviceAddr[i]);
            PrintOutResult("hnOut[" + std::to_string(i) + "]", hnOutListShape[i], &hnOutDeviceAddr[i]);
            PrintOutResult("hOut[" + std::to_string(i) + "]", hOutListShape[i], &hOutDeviceAddr[i]);
        }
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(input);
    aclDestroyTensorList(params);
    aclDestroyTensor(output);
    aclDestroyTensor(hy);
    if (hx != nullptr) {
        aclDestroyTensor(hx);
    }

    if (isTraining) {
        aclDestroyTensorList(rOut);
        aclDestroyTensorList(zOut);
        aclDestroyTensorList(nOut);
        aclDestroyTensorList(hnOut);
        aclDestroyTensorList(hOut);
    }

    //   // 7. 释放device资源
    aclrtFree(inputDeviceAddr);
    for (size_t i = 0; i < paramsListShape.size(); i++) {
        aclrtFree(paramsListDeviceAddr[i]);
    }
    aclrtFree(outputDeviceAddr);
    aclrtFree(hyDeviceAddr);
    if (hxDeviceAddr != nullptr) {
        aclrtFree(hxDeviceAddr);
    }
    if (isTraining) {
        for (size_t i = 0; i < rOutDeviceAddr.size(); i++) {
            aclrtFree(rOutDeviceAddr[i]);
            aclrtFree(zOutDeviceAddr[i]);
            aclrtFree(nOutDeviceAddr[i]);
            aclrtFree(hnOutDeviceAddr[i]);
            aclrtFree(hOutDeviceAddr[i]);
        }
    }

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}