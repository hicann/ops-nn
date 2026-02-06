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
 * @file test_aclnn_dynamic_rnn.cpp
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lstm.h"

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

void PrintOutResult(const std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorList(
    const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr, aclDataType dataType, aclTensorList** tensor,
    T initVal = 1)
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
    int time_step = 1;
    int batch_size = 1;
    int hidden_size = 1;
    int input_size = hidden_size;
    int64_t numLayers = 1;
    bool isbias = false;
    bool batchFirst = false;
    bool bidirection = false;
    bool isTraining = true;
    int64_t d_scale = bidirection == true ? 2 : 1;

    std::vector<int64_t> inputShape = {time_step, batch_size, input_size};
    std::vector<int64_t> outputShape = {time_step, batch_size, d_scale * hidden_size};
    std::vector<int64_t> hycyShape = {numLayers * d_scale, batch_size, hidden_size};
    std::vector<std::vector<int64_t>> paramsListShape = {};

    std::vector<std::vector<int64_t>> outIListShape = {};
    std::vector<std::vector<int64_t>> outJListShape = {};
    std::vector<std::vector<int64_t>> outFListShape = {};
    std::vector<std::vector<int64_t>> outOListShape = {};
    std::vector<std::vector<int64_t>> outHListShape = {};
    std::vector<std::vector<int64_t>> outCListShape = {};
    std::vector<std::vector<int64_t>> outTanhCListShape = {};

    auto cur_input_size = input_size;
    for (int i = 0; i < numLayers; i++) {
        paramsListShape.push_back({hidden_size * 4, cur_input_size});
        paramsListShape.push_back({hidden_size * 4, hidden_size});

        outIListShape.push_back({time_step, batch_size, hidden_size});
        outJListShape.push_back({time_step, batch_size, hidden_size});
        outFListShape.push_back({time_step, batch_size, hidden_size});
        outOListShape.push_back({time_step, batch_size, hidden_size});
        if (isTraining == true) {
            outHListShape.push_back({time_step, batch_size, hidden_size});
            outCListShape.push_back({time_step, batch_size, hidden_size});
        } else {
            outHListShape.push_back({batch_size, hidden_size});
            outCListShape.push_back({batch_size, hidden_size});
        }
        outTanhCListShape.push_back({time_step, batch_size, hidden_size});
        cur_input_size = hidden_size;
    }

    void* inputDeviceAddr = nullptr;
    void* paramsListDeviceAddr[2 * numLayers];

    void* outputDeviceAddr = nullptr;
    void* hyDeviceAddr = nullptr;
    void* cyDeviceAddr = nullptr;
    void* outIListDeviceAddr[numLayers];
    void* outJListDeviceAddr[numLayers];
    void* outFListDeviceAddr[numLayers];
    void* outOListDeviceAddr[numLayers];
    void* outHListDeviceAddr[numLayers];
    void* outCListDeviceAddr[numLayers];
    void* outTanhCListDeviceAddr[numLayers];

    aclTensor* input = nullptr;
    aclTensorList* params = nullptr;

    aclTensor* output = nullptr;
    aclTensor* hy = nullptr;
    aclTensor* cy = nullptr;
    aclTensorList* outIList = nullptr;
    aclTensorList* outJList = nullptr;
    aclTensorList* outFList = nullptr;
    aclTensorList* outOList = nullptr;
    aclTensorList* outHList = nullptr;
    aclTensorList* outCList = nullptr;
    aclTensorList* outTanhCList = nullptr;

    std::vector<float> inputHostData(GetShapeSize(inputShape), 1);

    ret = CreateAclTensor<float>(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(paramsListShape, paramsListDeviceAddr, aclDataType::ACL_FLOAT, &params, 1.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> outputHostData(GetShapeSize(outputShape), 1);
    ret = CreateAclTensor<float>(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> hycyHostData(GetShapeSize(hycyShape), 1);
    ret = CreateAclTensor<float>(hycyHostData, hycyShape, &hyDeviceAddr, aclDataType::ACL_FLOAT, &hy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(hycyHostData, hycyShape, &cyDeviceAddr, aclDataType::ACL_FLOAT, &cy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outIListShape, outIListDeviceAddr, aclDataType::ACL_FLOAT, &outIList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outJListShape, outJListDeviceAddr, aclDataType::ACL_FLOAT, &outJList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outFListShape, outFListDeviceAddr, aclDataType::ACL_FLOAT, &outFList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outOListShape, outOListDeviceAddr, aclDataType::ACL_FLOAT, &outOList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outHListShape, outHListDeviceAddr, aclDataType::ACL_FLOAT, &outHList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outCListShape, outCListDeviceAddr, aclDataType::ACL_FLOAT, &outCList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(
        outTanhCListShape, outTanhCListDeviceAddr, aclDataType::ACL_FLOAT, &outTanhCList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnLSTM第一段接口
    ret = aclnnLSTMGetWorkspaceSize(
        input, params, nullptr, nullptr, isbias, numLayers, 0.0, isTraining, bidirection, batchFirst, output, hy, cy,
        outIList, outJList, outFList, outOList, outHList, outCList, outTanhCList, &workspaceSize, &executor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLSTMGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnLSTM第二段接口
    ret = aclnnLSTM(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLSTM failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改

    PrintOutResult(outputShape, &outputDeviceAddr);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(input);
    aclDestroyTensorList(params);
    aclDestroyTensor(output);
    aclDestroyTensor(hy);
    aclDestroyTensor(cy);
    aclDestroyTensorList(outIList);
    aclDestroyTensorList(outJList);
    aclDestroyTensorList(outFList);
    aclDestroyTensorList(outOList);
    aclDestroyTensorList(outHList);
    aclDestroyTensorList(outCList);
    aclDestroyTensorList(outTanhCList);

    //   // 7. 释放device资源
    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclrtFree(hyDeviceAddr);
    aclrtFree(cyDeviceAddr);
    for (int i = 0; i < numLayers; i++) {
        aclrtFree(outIListDeviceAddr[i]);
        aclrtFree(outJListDeviceAddr[i]);
        aclrtFree(outFListDeviceAddr[i]);
        aclrtFree(outOListDeviceAddr[i]);
        aclrtFree(outHListDeviceAddr[i]);
        aclrtFree(outCListDeviceAddr[i]);
        aclrtFree(outTanhCListDeviceAddr[i]);
    }

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}