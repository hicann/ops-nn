/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_broadcast_gradient_args.h"

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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
              aclFinalize(); return ret);
    return 0;
}

bool CheckHardwareSupport()
{
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) {
        LOG_PRINT("Warning: Cannot get SOC name, skip hardware check\n");
        return true;
    }

    LOG_PRINT("Current SOC: %s\n", socName);
    if (strstr(socName, "Ascend950") != nullptr || strstr(socName, "ascend950") != nullptr) {
        return true;
    }

    LOG_PRINT("Warning: BroadcastGradientArgs only supports Ascend950, current SOC '%s' is not supported. Skip test.\n",
              socName);
    return false;
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

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    if (!CheckHardwareSupport()) {
        LOG_PRINT("\n=== Test SKIPPED (hardware not supported) ===\n");
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 0;
    }

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    //    x1 = {2, 1, 4, 1, 6}  (原始张量a的shape)
    //    x2 = {2, 3, 1, 5, 1}  (原始张量b的shape)
    //    输出按最大可能size申请，max(5, 5) = 5
    std::vector<int32_t> x1HostData = {2, 1, 4, 1, 6};
    std::vector<int32_t> x2HostData = {2, 3, 1, 5, 1};
    std::vector<int64_t> y1Shape = {5};
    std::vector<int64_t> y2Shape = {5};
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* y1DeviceAddr = nullptr;
    void* y2DeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* y1 = nullptr;
    aclTensor* y2 = nullptr;
    std::vector<int32_t> y1HostData(5, 0);
    std::vector<int32_t> y2HostData(5, 0);
    // 创建x1 aclTensor
    ret = CreateAclTensor(x1HostData, {5}, &x1DeviceAddr, aclDataType::ACL_INT32, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x2 aclTensor
    ret = CreateAclTensor(x2HostData, {5}, &x2DeviceAddr, aclDataType::ACL_INT32, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y1 aclTensor
    ret = CreateAclTensor(y1HostData, y1Shape, &y1DeviceAddr, aclDataType::ACL_INT32, &y1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y2 aclTensor
    ret = CreateAclTensor(y2HostData, y2Shape, &y2DeviceAddr, aclDataType::ACL_INT32, &y2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnBroadcastGradientArgs第一段接口
    ret = aclnnBroadcastGradientArgsGetWorkspaceSize(x1, x2, y1, y2, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBroadcastGradientArgsGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnBroadcastGradientArgs第二段接口
    ret = aclnnBroadcastGradientArgs(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBroadcastGradientArgs failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的实际shape（动态shape关键步骤），并将device侧结果拷贝至host侧
    int64_t* y1ViewDims = nullptr;
    uint64_t y1ViewDimsNum = 0;
    ret = aclGetViewShape(y1, &y1ViewDims, &y1ViewDimsNum);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclGetViewShape y1 failed. ERROR: %d\n", ret); return ret);
    int64_t y1Size = 1;
    for (uint64_t i = 0; i < y1ViewDimsNum; i++) {
        y1Size *= y1ViewDims[i];
    }
    if (y1Size > 0) {
        std::vector<int32_t> y1ResultData(y1Size, 0);
        ret = aclrtMemcpy(y1ResultData.data(), y1ResultData.size() * sizeof(int32_t), y1DeviceAddr,
                          y1Size * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y1 result failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < y1Size; i++) {
            LOG_PRINT("y1[%ld] is: %d\n", i, y1ResultData[i]);
        }
    } else {
        LOG_PRINT("y1 is empty (no broadcast axis)\n");
    }
    delete[] y1ViewDims;

    int64_t* y2ViewDims = nullptr;
    uint64_t y2ViewDimsNum = 0;
    ret = aclGetViewShape(y2, &y2ViewDims, &y2ViewDimsNum);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclGetViewShape y2 failed. ERROR: %d\n", ret); return ret);
    int64_t y2Size = 1;
    for (uint64_t i = 0; i < y2ViewDimsNum; i++) {
        y2Size *= y2ViewDims[i];
    }
    if (y2Size > 0) {
        std::vector<int32_t> y2ResultData(y2Size, 0);
        ret = aclrtMemcpy(y2ResultData.data(), y2ResultData.size() * sizeof(int32_t), y2DeviceAddr,
                          y2Size * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y2 result failed. ERROR: %d\n", ret); return ret);
        for (int64_t i = 0; i < y2Size; i++) {
            LOG_PRINT("y2[%ld] is: %d\n", i, y2ResultData[i]);
        }
    } else {
        LOG_PRINT("y2 is empty (no broadcast axis)\n");
    }
    delete[] y2ViewDims;

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(y1);
    aclDestroyTensor(y2);

    // 7. 释放device 资源
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(y1DeviceAddr);
    aclrtFree(y2DeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
