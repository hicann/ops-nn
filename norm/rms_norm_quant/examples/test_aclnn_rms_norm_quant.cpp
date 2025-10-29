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
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_rms_norm_quant.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
    // 1. （固定写法）device/stream初始化, 参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {3, 5};
    std::vector<int64_t> gammaShape = {1, 5};
    std::vector<int64_t> betaShape = {1, 5};
    std::vector<int64_t> scaleShape = {1};
    std::vector<int64_t> offsetShape = {1};
    std::vector<int64_t> yShape = {3, 5};
    void* xDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* y = nullptr;

    std::vector<aclFloat16> xHostData;
    std::vector<aclFloat16> gammaHostData;
    std::vector<aclFloat16> betaHostData;
    std::vector<aclFloat16> scaleHostData;
    float values = 0;
    for (int i = 0; i < 15; ++i) {
        values = i;
        xHostData.push_back(aclFloatToFloat16(values));
    }

    for (int i = 0; i < 5; ++i) {
        values = i;
        gammaHostData.push_back(aclFloatToFloat16(values));
        betaHostData.push_back(aclFloatToFloat16(values));
    }
    values = 1;
    scaleHostData.push_back(aclFloatToFloat16(values));

    std::vector<int8_t> offsetHostData(1, 1);
    std::vector<int8_t> yHostData(15, 0);
    double epsilon = 1e-6;
    // 创建self aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT16, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_INT8, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_INT8, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnRmsNormQuant第一段接口
    ret = aclnnRmsNormQuantGetWorkspaceSize(x, gamma, beta, scale, offset, epsilon, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnAddRmsNormQuant第二段接口
    ret = aclnnRmsNormQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormQuant failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(yShape);
    std::vector<int8_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(int8_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(gamma);
    aclDestroyTensor(beta);
    aclDestroyTensor(scale);
    aclDestroyTensor(offset);
    aclDestroyTensor(y);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(yDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}