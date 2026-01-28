/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "aclnnop/aclnn_dynamic_dual_level_mx_quant.h"

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

int aclnnDynamicDualLevelMxQuantTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {1, 512};
    std::vector<int64_t> smoothScaleOptionalShape = {1};
    std::vector<int64_t> yOutShape = {1, 512};
    std::vector<int64_t> level0ScaleOutShape = {1, 1};
    std::vector<int64_t> level1ScaleOutShape = {1, 8, 2};
    void* xDeviceAddr = nullptr;
    void* smoothScaleOptionalDeviceAddr = nullptr;
    void* yOutDeviceAddr = nullptr;
    void* level0ScaleOutDeviceAddr = nullptr;
    void* level1ScaleOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* smoothScaleOptional = nullptr;
    aclTensor* yOut = nullptr;
    aclTensor* level0ScaleOut = nullptr;
    aclTensor* level1ScaleOut = nullptr;

    // 对应 BF16 的值 (0->0, 16640->8, 17024->64, 17408->512)
    std::vector<uint16_t> xHostData(512, 16640);
    std::vector<uint16_t> smoothScaleOptionalHostData = {0};
    // 对应 float4_e2m1 的值 (0->0, 72->4, 96->32, 120->256)
    std::vector<uint8_t> yOutHostData(512, 0);
    // 对应 float32 的值 (0->0)
    std::vector<float> level0ScaleOutHostData = {{0}};
    //对应float8_e8m0的值(128->2)
    std::vector<std::vector<std::vector<uint8_t>>> level1ScaleOutHostData(1, std::vector<std::vector<uint8_t>>(8, std::vector<uint8_t>(2, 0)));
    const char* roundModeOptional = "rint";
    int64_t level0Blocksize = 512;
    int64_t level1Blocksize = 32;

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建smoothScaleOptional aclTensor
    ret = CreateAclTensor(smoothScaleOptionalHostData, smoothScaleOptionalShape, &smoothScaleOptionalDeviceAddr, aclDataType::ACL_BF16, &smoothScaleOptional);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> smoothScaleOptionalTensorPtr(smoothScaleOptional, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> smoothScaleOptionalDeviceAddrPtr(smoothScaleOptionalDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建yOut aclTensor
    ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &yOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yOutTensorPtr(yOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yOutDeviceAddrPtr(yOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建level0ScaleOut aclTensor
    ret = CreateAclTensor(level0ScaleOutHostData, level0ScaleOutShape, &level0ScaleOutDeviceAddr, aclDataType::ACL_FLOAT, &level0ScaleOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> level0ScaleOutTensorPtr(level0ScaleOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> level0ScaleOutDeviceAddrPtr(level0ScaleOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建level1ScaleOut aclTensor
    ret = CreateAclTensor(level1ScaleOutHostData, level1ScaleOutShape, &level1ScaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &level1ScaleOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> level1ScaleOutTensorPtr(level1ScaleOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> level1ScaleOutDeviceAddrPtr(level1ScaleOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnDynamicDualLevelMxQuant第一段接口
    ret = aclnnDynamicDualLevelMxQuantGetWorkspaceSize(x, smoothScaleOptional, (char *)roundModeOptional, level0Blocksize, level1Blocksize, yOut, level0ScaleOut, level1ScaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicDualLevelMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnDynamicDualLevelMxQuant第二段接口
    ret = aclnnDynamicDualLevelMxQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicDualLevelMxQuant failed. ERROR: %d\n", ret); return ret);

    //（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(yOutShape) / 2;
    std::vector<uint8_t> yOutData(
        size, 0);  // C语言中无法直接打印fp4的数据，需要用uint8读出来，自行通过二进制转成fp4
    ret = aclrtMemcpy(yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr,
                    size * sizeof(yOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret);
            return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("yOut[%ld] is: %d\n", i, yOutData[i]);
    }
    size = GetShapeSize(level0ScaleOutShape);
    std::vector<float> level0ScaleOutData(
        size, 0);
    ret = aclrtMemcpy(level0ScaleOutData.data(), level0ScaleOutData.size() * sizeof(level0ScaleOutData[0]), level0ScaleOutDeviceAddr,
                    size * sizeof(level0ScaleOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy level0ScaleOut from device to host failed. ERROR: %d\n", ret);
            return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("level0ScaleOut[%ld] is: %f\n", i, level0ScaleOutData[i]);
    }
    size = GetShapeSize(level1ScaleOutShape);
    std::vector<uint8_t> level1ScaleOutData(
        size, 0);  // C语言中无法直接打印fp8的数据，需要用uint8读出来，自行通过二进制转成fp8
    ret = aclrtMemcpy(level1ScaleOutData.data(), level1ScaleOutData.size() * sizeof(level1ScaleOutData[0]), level1ScaleOutDeviceAddr,
                    size * sizeof(level1ScaleOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy level1ScaleOut from device to host failed. ERROR: %d\n", ret);
            return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("level1ScaleOut[%ld] is: %d\n", i, level1ScaleOutData[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnDynamicDualLevelMxQuantTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicDualLevelMxQuantTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}