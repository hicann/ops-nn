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
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dynamic_mx_quant_with_dual_axis_v2.h"

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

    int64_t
    GetShapeSize(const std::vector<int64_t>& shape)
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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

// 测试场景1: scaleAlg=0 (OCP算法) + FLOAT8_E4M3FN, dstTypeMax=0.0 (默认值，与V1行为一致)
int aclnnDynamicMxQuantWithDualAxisV2TestOcp(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    std::vector<int64_t> xShape = {1, 4};
    std::vector<int64_t> y1OutShape = {1, 4};
    std::vector<int64_t> y2OutShape = {1, 4};
    std::vector<int64_t> mxscale1OutShape = {1, 1, 2};
    std::vector<int64_t> mxscale2OutShape = {1, 4, 2};
    void* xDeviceAddr = nullptr;
    void* y1OutDeviceAddr = nullptr;
    void* mxscale1OutDeviceAddr = nullptr;
    void* y2OutDeviceAddr = nullptr;
    void* mxscale2OutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y1Out = nullptr;
    aclTensor* mxscale1Out = nullptr;
    aclTensor* y2Out = nullptr;
    aclTensor* mxscale2Out = nullptr;
    // BF16原始数据: 0.0, 8.0, 12.0, 16.0
    std::vector<uint16_t> xHostData = {0, 16640, 17024, 17408};
    std::vector<uint8_t> y1OutHostData(4, 0);
    std::vector<uint8_t> y2OutHostData(4, 0);
    std::vector<uint8_t> mxscale1OutHostData(2, 0);
    std::vector<uint8_t> mxscale2OutHostData(8, 0);
    char* roundModeOptional = const_cast<char*>("rint");
    int64_t dstType = 36;   // FLOAT8_E4M3FN
    int64_t scaleAlg = 0;   // OCP算法
    float dstTypeMax = 0.0;  // 默认值，与V1行为一致

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y1Out aclTensor
    ret = CreateAclTensor(y1OutHostData, y1OutShape, &y1OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y1Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y1OutTensorPtr(y1Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y1OutDeviceAddrPtr(y1OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建mxscale1Out aclTensor
    ret = CreateAclTensor(mxscale1OutHostData, mxscale1OutShape, &mxscale1OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale1Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscale1OutTensorPtr(mxscale1Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> mxscale1OutDeviceAddrPtr(mxscale1OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y2Out aclTensor
    ret = CreateAclTensor(y2OutHostData, y2OutShape, &y2OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y2Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y2OutTensorPtr(y2Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y2OutDeviceAddrPtr(y2OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建mxscale2Out aclTensor
    ret = CreateAclTensor(mxscale2OutHostData, mxscale2OutShape, &mxscale2OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale2Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscale2OutTensorPtr(mxscale2Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> mxscale2OutDeviceAddrPtr(mxscale2OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用aclnnDynamicMxQuantWithDualAxisV2第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnDynamicMxQuantWithDualAxisV2GetWorkspaceSize(x, roundModeOptional, dstType, scaleAlg, dstTypeMax,
        y1Out, mxscale1Out, y2Out, mxscale2Out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnDynamicMxQuantWithDualAxisV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 调用aclnnDynamicMxQuantWithDualAxisV2第二段接口
    ret = aclnnDynamicMxQuantWithDualAxisV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantWithDualAxisV2 failed. ERROR: %d\n", ret); return ret);

    // 4. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值
    auto size1 = GetShapeSize(y1OutShape);
    auto size2 = GetShapeSize(y2OutShape);
    std::vector<uint8_t> y1OutData(size1, 0);
    std::vector<uint8_t> y2OutData(size2, 0);
    ret = aclrtMemcpy(y1OutData.data(), y1OutData.size() * sizeof(y1OutData[0]), y1OutDeviceAddr,
                    size1 * sizeof(y1OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y1Out from device to host failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(y2OutData.data(), y2OutData.size() * sizeof(y2OutData[0]), y2OutDeviceAddr,
                    size2 * sizeof(y2OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y2Out from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("===== Test OCP (scaleAlg=0, FP8_E4M3FN, dstTypeMax=0.0) =====\n");
    for (int64_t i = 0; i < size1; i++) {
        LOG_PRINT("y1Out[%ld] is: %d\n", i, y1OutData[i]);
    }
    for (int64_t i = 0; i < size2; i++) {
        LOG_PRINT("y2Out[%ld] is: %d\n", i, y2OutData[i]);
    }
    size1 = GetShapeSize(mxscale1OutShape);
    size2 = GetShapeSize(mxscale2OutShape);
    std::vector<uint8_t> mxscale1OutData(size1, 0);
    std::vector<uint8_t> mxscale2OutData(size2, 0);
    ret = aclrtMemcpy(mxscale1OutData.data(), mxscale1OutData.size() * sizeof(mxscale1OutData[0]),
                    mxscale1OutDeviceAddr, size1 * sizeof(mxscale1OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy mxscale1Out from device to host failed. ERROR: %d\n", ret);
            return ret);
    ret = aclrtMemcpy(mxscale2OutData.data(), mxscale2OutData.size() * sizeof(mxscale2OutData[0]),
                    mxscale2OutDeviceAddr, size2 * sizeof(mxscale2OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy mxscale2Out from device to host failed. ERROR: %d\n", ret);
            return ret);
    for (int64_t i = 0; i < size1; i++) {
        LOG_PRINT("mxscale1Out[%ld] is: %d\n", i, mxscale1OutData[i]);
    }
    for (int64_t i = 0; i < size2; i++) {
        LOG_PRINT("mxscale2Out[%ld] is: %d\n", i, mxscale2OutData[i]);
    }
    return ACL_SUCCESS;
}

// 测试场景2: scaleAlg=2 (DynamicDtypeRange算法) + FLOAT4_E2M1, dstTypeMax=6.0 (V2新增能力)
int aclnnDynamicMxQuantWithDualAxisV2TestDynamicDefault(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    // FLOAT4_E2M1要求x的最后一维必须是偶数
    std::vector<int64_t> xShape = {1, 4};
    std::vector<int64_t> y1OutShape = {1, 4};
    std::vector<int64_t> y2OutShape = {1, 4};
    std::vector<int64_t> mxscale1OutShape = {1, 1, 2};
    std::vector<int64_t> mxscale2OutShape = {1, 4, 2};
    void* xDeviceAddr = nullptr;
    void* y1OutDeviceAddr = nullptr;
    void* mxscale1OutDeviceAddr = nullptr;
    void* y2OutDeviceAddr = nullptr;
    void* mxscale2OutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y1Out = nullptr;
    aclTensor* mxscale1Out = nullptr;
    aclTensor* y2Out = nullptr;
    aclTensor* mxscale2Out = nullptr;
    // BF16原始数据: 0.0, 8.0, 12.0, 16.0
    std::vector<uint16_t> xHostData = {0, 16640, 17024, 17408};
    // FLOAT4_E2M1: 每个uint8存储2个FP4值，实际存储大小为 shape_size / 2
    std::vector<uint8_t> y1OutHostData(2, 0);
    std::vector<uint8_t> y2OutHostData(2, 0);
    std::vector<uint8_t> mxscale1OutHostData(2, 0);
    std::vector<uint8_t> mxscale2OutHostData(8, 0);
    char* roundModeOptional = const_cast<char*>("rint");
    int64_t dstType = 40;       // FLOAT4_E2M1
    int64_t scaleAlg = 2;       // DynamicDtypeRange算法 (V2新增)
    float dstTypeMax = 6.0;     // 自定义量化范围

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y1Out aclTensor (FLOAT4_E2M1)
    ret = CreateAclTensor(y1OutHostData, y1OutShape, &y1OutDeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &y1Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y1OutTensorPtr(y1Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y1OutDeviceAddrPtr(y1OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建mxscale1Out aclTensor
    ret = CreateAclTensor(mxscale1OutHostData, mxscale1OutShape, &mxscale1OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale1Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscale1OutTensorPtr(mxscale1Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> mxscale1OutDeviceAddrPtr(mxscale1OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y2Out aclTensor (FLOAT4_E2M1)
    ret = CreateAclTensor(y2OutHostData, y2OutShape, &y2OutDeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &y2Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y2OutTensorPtr(y2Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y2OutDeviceAddrPtr(y2OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建mxscale2Out aclTensor
    ret = CreateAclTensor(mxscale2OutHostData, mxscale2OutShape, &mxscale2OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale2Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscale2OutTensorPtr(mxscale2Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> mxscale2OutDeviceAddrPtr(mxscale2OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用aclnnDynamicMxQuantWithDualAxisV2第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnDynamicMxQuantWithDualAxisV2GetWorkspaceSize(x, roundModeOptional, dstType, scaleAlg, dstTypeMax,
        y1Out, mxscale1Out, y2Out, mxscale2Out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnDynamicMxQuantWithDualAxisV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 调用aclnnDynamicMxQuantWithDualAxisV2第二段接口
    ret = aclnnDynamicMxQuantWithDualAxisV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantWithDualAxisV2 failed. ERROR: %d\n", ret); return ret);

    // 4. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值
    // FLOAT4_E2M1: 每个uint8存储2个FP4值，用uint8读出后自行解析
    auto size1 = GetShapeSize(y1OutShape) / 2;  // FP4: 2个元素占1字节
    auto size2 = GetShapeSize(y2OutShape) / 2;
    std::vector<uint8_t> y1OutData(size1, 0);
    std::vector<uint8_t> y2OutData(size2, 0);
    ret = aclrtMemcpy(y1OutData.data(), y1OutData.size() * sizeof(y1OutData[0]), y1OutDeviceAddr,
                    size1 * sizeof(y1OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y1Out from device to host failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(y2OutData.data(), y2OutData.size() * sizeof(y2OutData[0]), y2OutDeviceAddr,
                    size2 * sizeof(y2OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y2Out from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("===== Test DynamicDtypeRange (scaleAlg=2, FP4_E2M1, dstTypeMax=6.0) =====\n");
    for (int64_t i = 0; i < size1; i++) {
        LOG_PRINT("y1Out[%ld] is: 0x%02x\n", i, y1OutData[i]);
    }
    for (int64_t i = 0; i < size2; i++) {
        LOG_PRINT("y2Out[%ld] is: 0x%02x\n", i, y2OutData[i]);
    }
    size1 = GetShapeSize(mxscale1OutShape);
    size2 = GetShapeSize(mxscale2OutShape);
    std::vector<uint8_t> mxscale1OutData(size1, 0);
    std::vector<uint8_t> mxscale2OutData(size2, 0);
    ret = aclrtMemcpy(mxscale1OutData.data(), mxscale1OutData.size() * sizeof(mxscale1OutData[0]),
                    mxscale1OutDeviceAddr, size1 * sizeof(mxscale1OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy mxscale1Out from device to host failed. ERROR: %d\n", ret);
            return ret);
    ret = aclrtMemcpy(mxscale2OutData.data(), mxscale2OutData.size() * sizeof(mxscale2OutData[0]),
                    mxscale2OutDeviceAddr, size2 * sizeof(mxscale2OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy mxscale2Out from device to host failed. ERROR: %d\n", ret);
            return ret);
    for (int64_t i = 0; i < size1; i++) {
        LOG_PRINT("mxscale1Out[%ld] is: %d\n", i, mxscale1OutData[i]);
    }
    for (int64_t i = 0; i < size2; i++) {
        LOG_PRINT("mxscale2Out[%ld] is: %d\n", i, mxscale2OutData[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;

    // 测试场景1: scaleAlg=0 (OCP) + FP8, dstTypeMax=0.0
    auto ret = aclnnDynamicMxQuantWithDualAxisV2TestOcp(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnDynamicMxQuantWithDualAxisV2TestOcp failed. ERROR: %d\n", ret); return ret);
    Finalize(deviceId, stream);

    // 测试场景2: scaleAlg=2 (DynamicDtypeRange) + FP4_E2M1, dstTypeMax=6.0
    ret = aclnnDynamicMxQuantWithDualAxisV2TestDynamicDefault(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnDynamicMxQuantWithDualAxisV2TestDynamicDefault failed. ERROR: %d\n", ret); return ret);
    Finalize(deviceId, stream);

    return 0;
}
