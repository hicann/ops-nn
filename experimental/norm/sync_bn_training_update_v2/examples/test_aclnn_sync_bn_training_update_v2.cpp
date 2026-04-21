/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <inttypes.h>  // 新增：用于int64_t的格式化输出宏
#include "acl/acl.h"
#include "aclnn_sync_bn_training_update_v2.h"

constexpr float MOMENTUM = 0.9f;
constexpr float EPS = 1e-8f;
constexpr int64_t INPUT_N = 8;    // 输入shape[0]
constexpr int64_t INPUT_C = 8;    // 输入shape[1]
constexpr int64_t INPUT_H = 8;    // 输入shape[2]
constexpr int64_t INPUT_W = 8;    // 输入shape[3]
constexpr int64_t INPUT_NUM = INPUT_N * INPUT_C * INPUT_H * INPUT_W;

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
    std::vector<uint8_t> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        // 修正：使用PRId64格式化int64_t类型
        LOG_PRINT("result[%" PRId64 "] is: %d\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，初始化
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
    // 2. 申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 3. 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
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

void GenerateInputData(
    std::vector<float>& x1HostData,  // 输入特征x [8,8,8,8]
    std::vector<float>& x2HostData,  // running_mean [C]
    std::vector<float>& x3HostData   // running_var [C]
) {
    // 随机数生成器（模拟Python的np.random.uniform）
    std::random_device rd;
    std::mt19937 gen(rd());  // 随机种子，可固定为12345方便复现
    std::uniform_real_distribution<float> distX(0.0f, 10.0f);    // x: [0,10]
    std::uniform_real_distribution<float> distX1(1.0f, 5.0f);    // x1: [1,5]

    // 生成x [8,8,8,8]
    x1HostData.resize(INPUT_NUM);
    for (int64_t i = 0; i < INPUT_NUM; ++i) {
        x1HostData[i] = distX(gen);
    }

    // 生成x2 [C=8]
    x2HostData.resize(INPUT_C);
    for (int64_t i = 0; i < INPUT_C; ++i) {
        x2HostData[i] = distX1(gen);
    }

    x3HostData.resize(INPUT_C);
    for (int64_t i = 0; i < INPUT_C; ++i) {
        x3HostData[i] = distX1(gen);
    }

    // 打印输入信息（对齐Python日志）
    float xMin = *std::min_element(x1HostData.begin(), x1HostData.end());
    float xMax = *std::max_element(x1HostData.begin(), x1HostData.end());

    LOG_PRINT("生成的 x2 (running_mean): %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n", 
        x2HostData[0], x2HostData[1], x2HostData[2], x2HostData[3], x2HostData[4], x2HostData[5], x2HostData[6], x2HostData[7]);
    LOG_PRINT("生成的 x3 (running_var): %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n", 
        x3HostData[0], x3HostData[1], x3HostData[2], x3HostData[3], x3HostData[4], x3HostData[5], x3HostData[6], x3HostData[7]);    
    // 修正：使用PRId64格式化int64_t类型，解决%lld和long int不匹配的警告
    LOG_PRINT("输入shape: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "], 总元素数: %" PRId64 "\n", 
        INPUT_N, INPUT_C, INPUT_H, INPUT_W, INPUT_NUM);
    LOG_PRINT("momentum: %.1f, eps: %.1e\n", MOMENTUM, EPS);
    LOG_PRINT("x范围: [%.4f, %.4f]\n", xMin, xMax);
}

void ComputeGoldenData(
    const std::vector<float>& xHostData,
    const std::vector<float>& x1HostData,
    const std::vector<float>& x2HostData,
    std::vector<float>& goldenYHostData,  // 归一化结果y
    std::vector<float>& goldenY1HostData, // 更新后running_mean
    std::vector<float>& goldenY2HostData  // 更新后running_var
) {
    // 1. 按通道计算全局均值和方差
    std::vector<float> globalMean(INPUT_C, 0.0f);
    std::vector<float> globalVar(INPUT_C, 0.0f);
    int64_t elemPerChannel = INPUT_N * INPUT_H * INPUT_W;
    std::vector<float> Sum(INPUT_C, 0.0f);
    std::vector<float> Sum2(INPUT_C, 0.0f);
    // 计算每个通道的sum和sum²
    for (int64_t c = 0; c < INPUT_C; ++c) {
        for (int64_t n = 0; n < INPUT_N; ++n) {
            for (int64_t h = 0; h < INPUT_H; ++h) {
                for (int64_t w = 0; w < INPUT_W; ++w) {
                    int64_t idx = ((n * INPUT_C + c) * INPUT_H + h) * INPUT_W + w;
                    float val = xHostData[idx];
                    Sum[c] += val;
                    Sum2[c] += val * val;
                }
            }
        }
        globalMean[c] = Sum[c] / elemPerChannel;
        globalVar[c] = (Sum2[c] / elemPerChannel) - (globalMean[c] * globalMean[c]);

    }
    LOG_PRINT("Golden 通道的Sum: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",Sum[0], Sum[1], Sum[2], Sum[3], Sum[4], Sum[5], Sum[6], Sum[7]);
    LOG_PRINT("Golden 通道的Sum2: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",Sum2[0], Sum2[1], Sum2[2], Sum2[3], Sum2[4], Sum2[5], Sum2[6], Sum2[7]);
    LOG_PRINT("Golden 通道的Mean: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n", globalMean[0], globalMean[1], globalMean[2], globalMean[3],
        globalMean[4], globalMean[5], globalMean[6], globalMean[7]);
    LOG_PRINT("Golden 通道的Var: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n", globalVar[0], globalVar[1], globalVar[2], globalVar[3],
        globalVar[4], globalVar[5], globalVar[6], globalVar[7]);
    // 2. 计算归一化结果
    goldenYHostData.resize(INPUT_NUM);
    for (int64_t c = 0; c < INPUT_C; ++c) {
        float sqrtVarEps = std::sqrt(globalVar[c] + EPS);
        for (int64_t n = 0; n < INPUT_N; ++n) {
            for (int64_t h = 0; h < INPUT_H; ++h) {
                for (int64_t w = 0; w < INPUT_W; ++w) {
                    int64_t idx = ((n * INPUT_C + c) * INPUT_H + h) * INPUT_W + w;
                    goldenYHostData[idx] = (xHostData[idx] - globalMean[c]) / sqrtVarEps;
                }
            }
        }
    }

    // 3. 计算更新后的running_mean和running_var
    goldenY1HostData.resize(INPUT_C);
    goldenY2HostData.resize(INPUT_C);
    for (int64_t c = 0; c < INPUT_C; ++c) {
        goldenY1HostData[c] = MOMENTUM * x1HostData[c] + (1 - MOMENTUM) * globalMean[c];
        goldenY2HostData[c] = MOMENTUM * x2HostData[c] + (1 - MOMENTUM) * globalVar[c];
    }

    // 打印Golden信息
    float meanMin = *std::min_element(globalMean.begin(), globalMean.end());
    float meanMax = *std::max_element(globalMean.begin(), globalMean.end());
    float varMin = *std::min_element(globalVar.begin(), globalVar.end());
    float varMax = *std::max_element(globalVar.begin(), globalVar.end());
    
    LOG_PRINT("Golden 全局均值范围: [%.4f, %.4f]\n", meanMin, meanMax);
    LOG_PRINT("Golden 全局方差范围: [%.4f, %.4f]\n", varMin, varMax);
}

int VerifyResult(
    const std::vector<float>& goldenData,
    const std::vector<int64_t>& shape,
    void* deviceAddr,
    const std::string& name
) {
    int64_t dataSize = GetShapeSize(shape) * sizeof(float);
    std::vector<float> hostData(dataSize / sizeof(float));

    // 1. Device -> Host 拷贝结果
    aclError ret = aclrtMemcpy(
        hostData.data(), dataSize,
        deviceAddr, dataSize,
        ACL_MEMCPY_DEVICE_TO_HOST
    );
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Copy %s from device to host failed. ERROR: %d\n", name.c_str(), ret); return ret);

    // 2. 计算最大误差
    float maxError = 0.0f;
    for (size_t i = 0; i < hostData.size(); ++i) {
        float error = std::fabs(hostData[i] - goldenData[i]);
        if (error > maxError) {
            maxError = error;
        }
    }

    // 3. 打印验证结果
    LOG_PRINT("最大误差（%s）: %.6f\n", name.c_str(), maxError);
    return 0;
}

int main()
{
    // 1. 调用acl进行device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<float> selfX1HostData;
    std::vector<float> selfX2HostData;
    std::vector<float> selfX3HostData;
    GenerateInputData(selfX1HostData, selfX2HostData, selfX3HostData);

    std::vector<float> out1HostData(INPUT_NUM, 0.0f);
    std::vector<float> out2HostData(INPUT_C, 0.0f);
    std::vector<float> out3HostData(INPUT_C, 0.0f);

    std::vector<float> goldenY1HostData;
    std::vector<float> goldenY2HostData;
    std::vector<float> goldenY3HostData;
    ComputeGoldenData(selfX1HostData, selfX2HostData, selfX3HostData, goldenY1HostData, goldenY2HostData, goldenY3HostData);

    std::vector<int64_t> selfX1Shape = {INPUT_N, INPUT_C, INPUT_H, INPUT_W}; 
    std::vector<int64_t> selfX2Shape = {INPUT_C};     
    std::vector<int64_t> selfX3Shape = {INPUT_C};

    std::vector<int64_t> out1Shape = selfX1Shape;              // 归一化结果shape与x一致
    std::vector<int64_t> out2Shape = selfX2Shape; 
    std::vector<int64_t> out3Shape = selfX3Shape; 

    aclTensor* selfX1 = nullptr;
    aclTensor* selfX2 = nullptr;
    aclTensor* selfX3 = nullptr;
    aclTensor* out1 = nullptr;
    aclTensor* out2 = nullptr;
    aclTensor* out3 = nullptr;

    void* selfX1DeviceAddr = nullptr;
    void* selfX2DeviceAddr = nullptr;
    void* selfX3DeviceAddr = nullptr;
    void* out1DeviceAddr = nullptr;
    void* out2DeviceAddr = nullptr;
    void* out3DeviceAddr = nullptr;

    // 生成输入数据
    ret = CreateAclTensor(selfX1HostData, selfX1Shape, &selfX1DeviceAddr, aclDataType::ACL_FLOAT, &selfX1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(selfX2HostData, selfX2Shape, &selfX2DeviceAddr, aclDataType::ACL_FLOAT, &selfX2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(selfX3HostData, selfX3Shape, &selfX3DeviceAddr, aclDataType::ACL_FLOAT, &selfX3);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(out1HostData, out1Shape, &out1DeviceAddr, aclDataType::ACL_FLOAT, &out1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(out2HostData, out2Shape, &out2DeviceAddr, aclDataType::ACL_FLOAT, &out2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(out3HostData, out3Shape, &out3DeviceAddr, aclDataType::ACL_FLOAT, &out3);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    double moment = 0.9;
    
    // 4. 调用aclnnSyncBnTrainingUpdateV2第一段接口
    ret = aclnnSyncBnTrainingUpdateV2GetWorkspaceSize(selfX1, selfX2, selfX3, moment, out1, out2, out3, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSyncBnTrainingUpdateV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. 调用aclnnSyncBnTrainingUpdateV2第二段接口
    ret = aclnnSyncBnTrainingUpdateV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSyncBnTrainingUpdateV2 failed. ERROR: %d\n", ret); return ret);

    // 6. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);


    // 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    // auto size = GetShapeSize(out1Shape);
    // std::vector<float> resultData(size, 0);
    // ret = aclrtMemcpy(
    //     resultData.data(), resultData.size() * sizeof(resultData[0]), out1DeviceAddr, size * sizeof(resultData[0]),
    //     ACL_MEMCPY_DEVICE_TO_HOST);
    // CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    // for (int64_t i = 0; i < INPUT_C; i++) {
    //     LOG_PRINT("aclnnBn y result[%ld] is: %f\n", i, (float)resultData[i]);
    // }

    ret = VerifyResult(goldenY1HostData, out1Shape, out1DeviceAddr, "y1");
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = VerifyResult(goldenY2HostData, out2Shape, out2DeviceAddr, "y2");
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = VerifyResult(goldenY3HostData, out3Shape, out3DeviceAddr, "y3");
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 7. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(selfX1);
    aclDestroyTensor(selfX2);
    aclDestroyTensor(selfX3);
    aclDestroyTensor(out1);
    aclDestroyTensor(out2);
    aclDestroyTensor(out3);
    
    // 8. 释放device资源
    aclrtFree(selfX1DeviceAddr);
    aclrtFree(selfX2DeviceAddr);
    aclrtFree(selfX3DeviceAddr);
    aclrtFree(out1DeviceAddr);
    aclrtFree(out2DeviceAddr);
    aclrtFree(out3DeviceAddr);
    
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);

    // 9. acl去初始化
    aclFinalize();

    return 0;
}