/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_quant_matmul_v5.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
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

float Bf16ToFloat(uint16_t h)
{
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    return *reinterpret_cast<float*>(&bits);
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
    auto size = hostData.size() * sizeof(T);
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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int AclnnQuantMatmulV5DynamicScaleTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t m = 16;
    int64_t k = 64;
    int64_t n = 32;

    // K-C量化模式（pertoken-perchannel）：x1和x2为INT8，x1Scale和x2Scale为FLOAT32，out为BFLOAT16
    aclDataType x1Dtype = aclDataType::ACL_INT8;
    aclDataType x2Dtype = aclDataType::ACL_INT8;
    aclDataType x1ScaleDtype = aclDataType::ACL_FLOAT;
    aclDataType x2ScaleDtype = aclDataType::ACL_FLOAT;
    aclDataType outDtype = aclDataType::ACL_BF16;

    // 形状设置：x1Scale为(m,)，每个token一个scale；x2Scale为(n,)，每个channel一个scale
    std::vector<int64_t> x1Shape = {m, k};
    std::vector<int64_t> x2Shape = {n, k}; // transposeX2为true时是(n, k)
    std::vector<int64_t> x1ScaleShape = {m};
    std::vector<int64_t> x2ScaleShape = {n};
    std::vector<int64_t> outShape = {m, n};

    // 设备内存地址
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* x1ScaleDeviceAddr = nullptr;
    void* x2ScaleDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;

    // 张量指针
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* x1Scale = nullptr;
    aclTensor* x2Scale = nullptr;
    aclTensor* yScale = nullptr;   // 不使用
    aclTensor* x1Offset = nullptr; // 不使用
    aclTensor* x2Offset = nullptr; // 不使用
    aclTensor* yOffset = nullptr;  // 不使用
    aclTensor* bias = nullptr;     // 不使用
    aclTensor* out = nullptr;

    // 构造输入数据
    // 模拟FP16的activation数据，实际推理中activation由上游算子输出，不同token的数值范围不同
    std::vector<float> activationHostData(m * k);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < k; ++j) {
            activationHostData[i * k + j] = static_cast<float>(((i * 5 + j * 3) % 11) - 5) * (0.5f + (i % 3) * 0.5f);
        }
    }

    // x1为per-token动态量化结果：每个token（行）在线计算scale，scale = absmax / 127
    // 实际推理中x1Scale随输入数据动态变化，每次调用算子前更新scale取值后传入，接口用法与静态scale完全一致
    std::vector<float> x1ScaleHostData(m, 0.0f);
    std::vector<int8_t> x1HostData(m * k, 0);
    for (int64_t i = 0; i < m; ++i) {
        float absMax = 0.0f;
        for (int64_t j = 0; j < k; ++j) {
            absMax = std::max(absMax, std::fabs(activationHostData[i * k + j]));
        }
        x1ScaleHostData[i] = absMax / 127.0f;
        for (int64_t j = 0; j < k; ++j) {
            x1HostData[i * k + j] = static_cast<int8_t>(
                std::lround(activationHostData[i * k + j] / x1ScaleHostData[i]));
        }
    }

    // x2为weight的per-channel离线量化结果：每个channel（行）独立计算scale，离线量化后与scale一起固化
    std::vector<float> weightHostData(n * k);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < k; ++j) {
            weightHostData[i * k + j] = static_cast<float>(((i * 7 + j * 2) % 9) - 4) * (1.0f + (i % 2));
        }
    }
    std::vector<float> x2ScaleHostData(n, 0.0f);
    std::vector<int8_t> x2HostData(n * k, 0);
    for (int64_t i = 0; i < n; ++i) {
        float absMax = 0.0f;
        for (int64_t j = 0; j < k; ++j) {
            absMax = std::max(absMax, std::fabs(weightHostData[i * k + j]));
        }
        x2ScaleHostData[i] = absMax / 127.0f;
        for (int64_t j = 0; j < k; ++j) {
            x2HostData[i * k + j] = static_cast<int8_t>(std::lround(weightHostData[i * k + j] / x2ScaleHostData[i]));
        }
    }
    // out: BFLOAT16，用于接收结果
    std::vector<uint16_t> outHostData(m * n, 0);

    // 创建x1 aclTensor
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, x1Dtype, &x1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建x2 aclTensor
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, x2Dtype, &x2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建x1Scale aclTensor
    ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, x1ScaleDtype, &x1Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建x2Scale aclTensor
    ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, x2ScaleDtype, &x2Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, outDtype, &out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 转置设置
    bool transposeX1 = false;
    bool transposeX2 = true; // x2为(n, k)，转置后参与计算

    // groupSize设置：K-C量化模式下groupSize不生效，传0
    int64_t groupSize = 0;

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 调用aclnnQuantMatmulV5第一段接口
    ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, x2, x1Scale, x2Scale, yScale, x1Offset, x2Offset, yOffset, bias,
                                             transposeX1, transposeX2, groupSize, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed to allocate workspace. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 调用aclnnQuantMatmulV5第二段接口
    ret = aclnnQuantMatmulV5(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0); // C语言中无法直接打印bfloat16的数据，需要用uint16读出来
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // 打印部分结果
    LOG_PRINT("First 4 per-token scales (x1Scale):");
    for (int64_t i = 0; i < std::min<int64_t>(4, m); i++) {
        LOG_PRINT(" x1Scale[%ld]=%.6f", i, x1ScaleHostData[i]);
    }
    LOG_PRINT("\n");
    LOG_PRINT("First 10 results:\n");
    for (int64_t i = 0; i < std::min<int64_t>(10, size); i++) {
        LOG_PRINT("result[%ld] is: %.4f\n", i, Bf16ToFloat(resultData[i]));
    }

    return ACL_SUCCESS;
}

int main(int argc, char* argv[])
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = AclnnQuantMatmulV5DynamicScaleTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnQuantMatmulV5DynamicScaleTest failed. ERROR: %d\n", ret);
                   return ret);

    Finalize(deviceId, stream);
    return 0;
}
