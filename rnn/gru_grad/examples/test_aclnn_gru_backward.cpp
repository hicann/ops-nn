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
 * @file test_aclnn_gru_backward.cpp
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gru_backward.h"

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
              return);
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
    bool hasBias = false;
    bool batchFirst = false;
    bool bidirection = false;
    int64_t dScale = bidirection ? 2 : 1;
    int64_t ldScale = numLayers * dScale;

    std::vector<int64_t> inputShape = {timeStep, batchSize, inputSize};
    std::vector<int64_t> dyShape = {timeStep, batchSize, dScale * hiddenSize};
    std::vector<int64_t> dhShape = {ldScale, batchSize, hiddenSize};
    std::vector<int64_t> hxShape = {ldScale, batchSize, hiddenSize};
    std::vector<std::vector<int64_t>> paramsListShape = {};

    auto curLayerInputSize = inputSize;
    for (int i = 0; i < numLayers; i++) {
        for (int64_t j = 0; j < dScale; j++) {
            paramsListShape.push_back({hiddenSize * gateNum, curLayerInputSize});
            paramsListShape.push_back({hiddenSize * gateNum, hiddenSize});
            if (hasBias) {
                paramsListShape.push_back({hiddenSize * gateNum});
                paramsListShape.push_back({hiddenSize * gateNum});
            }
        }
        curLayerInputSize = dScale * hiddenSize;
    }

    // gate lists: r, z, n, hn, h each has ldScale tensors of [T, B, H]
    std::vector<std::vector<int64_t>> gateListShape;
    for (int64_t i = 0; i < ldScale; i++) {
        gateListShape.push_back({timeStep, batchSize, hiddenSize});
    }

    void* inputDeviceAddr = nullptr;
    std::vector<void*> paramsListDeviceAddr(paramsListShape.size(), nullptr);
    void* dyDeviceAddr = nullptr;
    void* dhDeviceAddr = nullptr;
    void* hxDeviceAddr = nullptr;

    std::vector<void*> rDeviceAddr;
    std::vector<void*> zDeviceAddr;
    std::vector<void*> nDeviceAddr;
    std::vector<void*> hnDeviceAddr;
    std::vector<void*> hDeviceAddr;

    // output
    void* dxDeviceAddr = nullptr;
    std::vector<void*> dparamsListDeviceAddr(paramsListShape.size(), nullptr);
    void* dhPrevDeviceAddr = nullptr;

    aclTensor* input = nullptr;
    aclTensorList* params = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* dh = nullptr;
    aclTensor* hx = nullptr;

    aclTensorList* r = nullptr;
    aclTensorList* z = nullptr;
    aclTensorList* n = nullptr;
    aclTensorList* hn = nullptr;
    aclTensorList* h = nullptr;

    aclTensor* dxOut = nullptr;
    aclTensor* dhPrevOut = nullptr;
    aclTensorList* dparamsOut = nullptr;

    // 构造输入
    std::vector<float> inputHostData(GetShapeSize(inputShape), 1.0);
    ret = CreateAclTensor<float>(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(paramsListShape, paramsListDeviceAddr.data(), aclDataType::ACL_FLOAT, &params,
                                     1.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> dyHostData(GetShapeSize(dyShape), 0.5);
    ret = CreateAclTensor<float>(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> dhHostData(GetShapeSize(dhShape), 0.1);
    ret = CreateAclTensor<float>(dhHostData, dhShape, &dhDeviceAddr, aclDataType::ACL_FLOAT, &dh);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> hxHostData(GetShapeSize(hxShape), 0.0);
    ret = CreateAclTensor<float>(hxHostData, hxShape, &hxDeviceAddr, aclDataType::ACL_FLOAT, &hx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 构造 gate lists (r, z, n, hn, h) - 前向计算的中间值
    rDeviceAddr.resize(ldScale, nullptr);
    zDeviceAddr.resize(ldScale, nullptr);
    nDeviceAddr.resize(ldScale, nullptr);
    hnDeviceAddr.resize(ldScale, nullptr);
    hDeviceAddr.resize(ldScale, nullptr);

    ret = CreateAclTensorList<float>(gateListShape, rDeviceAddr.data(), aclDataType::ACL_FLOAT, &r, 0.5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorList<float>(gateListShape, zDeviceAddr.data(), aclDataType::ACL_FLOAT, &z, 0.5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorList<float>(gateListShape, nDeviceAddr.data(), aclDataType::ACL_FLOAT, &n, 0.5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorList<float>(gateListShape, hnDeviceAddr.data(), aclDataType::ACL_FLOAT, &hn, 0.5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensorList<float>(gateListShape, hDeviceAddr.data(), aclDataType::ACL_FLOAT, &h, 0.5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 构造输出
    std::vector<float> dxHostData(GetShapeSize(inputShape), 0.0);
    ret = CreateAclTensor<float>(dxHostData, inputShape, &dxDeviceAddr, aclDataType::ACL_FLOAT, &dxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> dhPrevHostData(GetShapeSize(hxShape), 0.0);
    ret = CreateAclTensor<float>(dhPrevHostData, hxShape, &dhPrevDeviceAddr, aclDataType::ACL_FLOAT, &dhPrevOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(paramsListShape, dparamsListDeviceAddr.data(), aclDataType::ACL_FLOAT, &dparamsOut,
                                     0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 调用aclnnGRUBackward第一段接口
    ret = aclnnGRUBackwardGetWorkspaceSize(input, params, hx, dy, dh, r, z, n, hn, h, nullptr, hasBias, numLayers,
                                           bidirection, batchFirst, dxOut, dhPrevOut, dparamsOut, &workspaceSize,
                                           &executor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGRUBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnGRUBackward第二段接口
    ret = aclnnGRUBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGRUBackward failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧
    PrintOutResult("dxOut", inputShape, &dxDeviceAddr);
    PrintOutResult("dhPrevOut", hxShape, &dhPrevDeviceAddr);
    for (size_t i = 0; i < paramsListShape.size(); i++) {
        PrintOutResult("dparamsOut[" + std::to_string(i) + "]", paramsListShape[i], &dparamsListDeviceAddr[i]);
    }

    // 6. 释放aclTensor和aclTensorList
    aclDestroyTensor(input);
    aclDestroyTensorList(params);
    aclDestroyTensor(dy);
    aclDestroyTensor(dh);
    aclDestroyTensor(hx);

    aclDestroyTensorList(r);
    aclDestroyTensorList(z);
    aclDestroyTensorList(n);
    aclDestroyTensorList(hn);
    aclDestroyTensorList(h);

    aclDestroyTensor(dxOut);
    aclDestroyTensor(dhPrevOut);
    aclDestroyTensorList(dparamsOut);

    // 7. 释放device资源
    aclrtFree(inputDeviceAddr);
    for (size_t i = 0; i < paramsListShape.size(); i++) {
        aclrtFree(paramsListDeviceAddr[i]);
    }
    aclrtFree(dyDeviceAddr);
    aclrtFree(dhDeviceAddr);
    aclrtFree(hxDeviceAddr);

    for (size_t i = 0; i < rDeviceAddr.size(); i++) {
        aclrtFree(rDeviceAddr[i]);
        aclrtFree(zDeviceAddr[i]);
        aclrtFree(nDeviceAddr[i]);
        aclrtFree(hnDeviceAddr[i]);
        aclrtFree(hDeviceAddr[i]);
    }

    aclrtFree(dxDeviceAddr);
    aclrtFree(dhPrevDeviceAddr);
    for (size_t i = 0; i < dparamsListDeviceAddr.size(); i++) {
        aclrtFree(dparamsListDeviceAddr[i]);
    }

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
