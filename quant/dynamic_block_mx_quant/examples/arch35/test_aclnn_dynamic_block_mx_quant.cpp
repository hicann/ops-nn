/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <iostream>
#include <memory>
#include <vector>
#include <cstdio>
#include "acl/acl.h"
#include "aclnnop/aclnn_dynamic_block_mx_quant.h"

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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnDynamicBlockMxQuantTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {1, 4};
    std::vector<int64_t> yOutShape = {1, 4};
    std::vector<int64_t> scale1OutShape = {1, 1, 2};
    std::vector<int64_t> scale2OutShape = {1, 4, 2};
    void* xDeviceAddr = nullptr;
    void* yOutDeviceAddr = nullptr;
    void* scale1OutDeviceAddr = nullptr;
    void* scale2OutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* yOut = nullptr;
    aclTensor* scale1Out = nullptr;
    aclTensor* scale2Out = nullptr;
    std::vector<uint16_t> xHostData = {0, 16640, 17024, 17408};
    std::vector<uint8_t> yOutHostData = {0, 72, 96, 120};
    std::vector<uint8_t> scale1OutHostData = {128, 0};
    std::vector<uint8_t> scale2OutHostData = {128, 0, 128, 0, 128, 0, 128, 0};
    char* roundModeOptional = const_cast<char*>("rint");
    int64_t dstType = 36;
    int64_t scaleAlg = 0;
    float dstTypeMax = 0.0;
    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建yOut aclTensor
    ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &yOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yOutTensorPtr(yOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yOutDeviceAddrPtr(yOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建scale1Out aclTensor
    ret = CreateAclTensor(
        scale1OutHostData, scale1OutShape, &scale1OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale1Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scale1OutTensorPtr(scale1Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scale1OutDeviceAddrPtr(scale1OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建scale2Out aclTensor
    ret = CreateAclTensor(
        scale2OutHostData, scale2OutShape, &scale2OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale2Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scale2OutTensorPtr(scale2Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scale2OutDeviceAddrPtr(scale2OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnDynamicBlockMxQuant第一段接口
    ret = aclnnDynamicBlockMxQuantGetWorkspaceSize(
        x, roundModeOptional, dstType, scaleAlg, dstTypeMax, yOut, scale1Out, scale2Out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicBlockMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnDynamicBlockMxQuant第二段接口
    ret = aclnnDynamicBlockMxQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicBlockMxQuant failed. ERROR: %d\n", ret); return ret);

    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size1 = GetShapeSize(yOutShape);
    std::vector<uint8_t> yOutData(size1, 0); // C语言中无法直接打印fp4的数据，需要用uint8读出来，自行通过二进制转成fp4
    ret = aclrtMemcpy(
        yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr, size1 * sizeof(yOutData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size1; i++) {
        LOG_PRINT("yOut[%ld] is: %d\n", i, yOutData[i]);
    }
    auto size2 = GetShapeSize(scale1OutShape);
    auto size3 = GetShapeSize(scale2OutShape);
    std::vector<uint8_t> scale1OutData(
        size2, 0); // C语言中无法直接打印fp8的数据，需要用uint8读出来，自行通过二进制转成fp8
    std::vector<uint8_t> scale2OutData(
        size3, 0); // C语言中无法直接打印fp8的数据，需要用uint8读出来，自行通过二进制转成fp8
    ret = aclrtMemcpy(
        scale1OutData.data(), scale1OutData.size() * sizeof(scale1OutData[0]), scale1OutDeviceAddr,
        size2 * sizeof(scale1OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale1Out from device to host failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(
        scale2OutData.data(), scale2OutData.size() * sizeof(scale2OutData[0]), scale2OutDeviceAddr,
        size3 * sizeof(scale2OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale2Out from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size2; i++) {
        LOG_PRINT("scale1Out[%ld] is: %d\n", i, scale1OutData[i]);
    }
    for (int64_t i = 0; i < size3; i++) {
        LOG_PRINT("scale2Out[%ld] is: %d\n", i, scale2OutData[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnDynamicBlockMxQuantTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicBlockMxQuantTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}