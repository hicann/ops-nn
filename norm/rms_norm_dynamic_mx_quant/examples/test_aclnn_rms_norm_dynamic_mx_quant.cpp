/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_rms_norm_dynamic_mx_quant.h"

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

// Data type constants for dst_type attribute
#define GE_DT_FLOAT4_E2M1 40
#define GE_DT_FLOAT4_E1M2 41
#define GE_DT_FLOAT8_E4M3FN 36
#define GE_DT_FLOAT8_E5M2 35

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

bool CheckHardwareSupport()
{
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) {
        LOG_PRINT("Warning: Cannot get SOC name, skip hardware check\n");
        return true;
    }

    LOG_PRINT("Current SOC: %s\n", socName);

    // This operator only supports Ascend950
    if (strstr(socName, "Ascend950") != nullptr || strstr(socName, "ascend950") != nullptr) {
        return true;
    }

    LOG_PRINT(
        "Warning: This operator only supports Ascend950, current SOC '%s' is not supported. Skip test.\n", socName);
    return false;
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
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 检查硬件支持
    if (!CheckHardwareSupport()) {
        LOG_PRINT("\n=== Test SKIPPED (hardware not supported) ===\n");
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 0;
    }

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // Input shape: [batch_size, hidden_size]
    int64_t batchSize = 2;
    int64_t hiddenSize = 64;
    std::vector<int64_t> xShape = {batchSize, hiddenSize};
    std::vector<int64_t> gammaShape = {hiddenSize};
    std::vector<int64_t> yShape = {batchSize, hiddenSize};

    // MX scale shape: [batchSize, ceil(hiddenSize/32), 2]
    int64_t mxBlockNum = (hiddenSize + 31) / 32;
    int64_t mxscaleDim = (mxBlockNum + 1) / 2;
    std::vector<int64_t> mxscaleShape = {batchSize, mxscaleDim, 2};
    std::vector<int64_t> rstdShape = {batchSize, 1};

    void* xDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* mxscaleDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* workspaceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* y = nullptr;
    aclTensor* mxscale = nullptr;
    aclTensor* rstd = nullptr;

    // Initialize host data
    // Use FP16 values (1.0 in FP16 = 0x3C00)
    std::vector<uint16_t> xHostData(batchSize * hiddenSize, 0x3C00);
    std::vector<uint16_t> gammaHostData(hiddenSize, 0x3C00);
    std::vector<uint8_t> yHostData(batchSize * hiddenSize, 0);
    std::vector<uint8_t> mxscaleHostData(GetShapeSize(mxscaleShape), 0);
    std::vector<float> rstdHostData(batchSize, 0.0f);

    // Operator attributes
    double epsilon = 1e-6;
    int64_t scaleAlg = 0; // OCP algorithm
    char* roundMode = const_cast<char*>("rint");
    int64_t dstType = GE_DT_FLOAT8_E4M3FN;
    bool outputRstd = true;

    LOG_PRINT("RmsNormDynamicMxQuant Test\n");
    LOG_PRINT("Input shape: [%ld, %ld]\n", batchSize, hiddenSize);
    LOG_PRINT("MX block num: %ld, MX scale dim: %ld\n", mxBlockNum, mxscaleDim);

    // 创建x aclTensor (FP16输入)
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("CreateAclTensor x failed. ERROR: %d\n", ret);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }

    // 创建gamma aclTensor (FP16输入)
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("CreateAclTensor gamma failed. ERROR: %d\n", ret);
        aclDestroyTensor(x);
        aclrtFree(xDeviceAddr);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }

    // 创建y aclTensor (FP8 E4M3FN输出)
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("CreateAclTensor y failed. ERROR: %d\n", ret);
        aclDestroyTensor(x);
        aclDestroyTensor(gamma);
        aclrtFree(xDeviceAddr);
        aclrtFree(gammaDeviceAddr);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }

    // 创建mxscale aclTensor (FP8 E8M0输出)
    ret = CreateAclTensor(mxscaleHostData, mxscaleShape, &mxscaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("CreateAclTensor mxscale failed. ERROR: %d\n", ret);
        aclDestroyTensor(x);
        aclDestroyTensor(gamma);
        aclDestroyTensor(y);
        aclrtFree(xDeviceAddr);
        aclrtFree(gammaDeviceAddr);
        aclrtFree(yDeviceAddr);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }

    // 创建rstd aclTensor (float输出)
    ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("CreateAclTensor rstd failed. ERROR: %d\n", ret);
        aclDestroyTensor(x);
        aclDestroyTensor(gamma);
        aclDestroyTensor(y);
        aclDestroyTensor(mxscale);
        aclrtFree(xDeviceAddr);
        aclrtFree(gammaDeviceAddr);
        aclrtFree(yDeviceAddr);
        aclrtFree(mxscaleDeviceAddr);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 调用aclnnRmsNormDynamicMxQuant第一段接口
    ret = aclnnRmsNormDynamicMxQuantGetWorkspaceSize(
        x, gamma, nullptr, epsilon, scaleAlg, roundMode, dstType, outputRstd, y, mxscale, rstd, &workspaceSize,
        &executor);

    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnRmsNormDynamicMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
        aclDestroyTensor(x);
        aclDestroyTensor(gamma);
        aclDestroyTensor(y);
        aclDestroyTensor(mxscale);
        aclDestroyTensor(rstd);
        aclrtFree(xDeviceAddr);
        aclrtFree(gammaDeviceAddr);
        aclrtFree(yDeviceAddr);
        aclrtFree(mxscaleDeviceAddr);
        aclrtFree(rstdDeviceAddr);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }

    LOG_PRINT("Workspace size: %lu bytes\n", workspaceSize);

    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            aclDestroyTensor(x);
            aclDestroyTensor(gamma);
            aclDestroyTensor(y);
            aclDestroyTensor(mxscale);
            aclDestroyTensor(rstd);
            aclrtFree(xDeviceAddr);
            aclrtFree(gammaDeviceAddr);
            aclrtFree(yDeviceAddr);
            aclrtFree(mxscaleDeviceAddr);
            aclrtFree(rstdDeviceAddr);
            aclrtDestroyStream(stream);
            aclrtResetDevice(deviceId);
            aclFinalize();
            return ret;
        }
    }

    // 调用aclnnRmsNormDynamicMxQuant第二段接口
    ret = aclnnRmsNormDynamicMxQuant(workspaceAddr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnRmsNormDynamicMxQuant failed. ERROR: %d\n", ret);
        if (workspaceAddr)
            aclrtFree(workspaceAddr);
        aclDestroyTensor(x);
        aclDestroyTensor(gamma);
        aclDestroyTensor(y);
        aclDestroyTensor(mxscale);
        aclDestroyTensor(rstd);
        aclrtFree(xDeviceAddr);
        aclrtFree(gammaDeviceAddr);
        aclrtFree(yDeviceAddr);
        aclrtFree(mxscaleDeviceAddr);
        aclrtFree(rstdDeviceAddr);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        if (workspaceAddr)
            aclrtFree(workspaceAddr);
        aclDestroyTensor(x);
        aclDestroyTensor(gamma);
        aclDestroyTensor(y);
        aclDestroyTensor(mxscale);
        aclDestroyTensor(rstd);
        aclrtFree(xDeviceAddr);
        aclrtFree(gammaDeviceAddr);
        aclrtFree(yDeviceAddr);
        aclrtFree(mxscaleDeviceAddr);
        aclrtFree(rstdDeviceAddr);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    {
        auto size = GetShapeSize(yShape);
        std::vector<uint8_t> yResult(size, 0);
        ret = aclrtMemcpy(
            yResult.data(), yResult.size() * sizeof(uint8_t), yDeviceAddr, size * sizeof(uint8_t),
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("copy y from device to host failed. ERROR: %d\n", ret);
            if (workspaceAddr)
                aclrtFree(workspaceAddr);
            aclDestroyTensor(x);
            aclDestroyTensor(gamma);
            aclDestroyTensor(y);
            aclDestroyTensor(mxscale);
            aclDestroyTensor(rstd);
            aclrtFree(xDeviceAddr);
            aclrtFree(gammaDeviceAddr);
            aclrtFree(yDeviceAddr);
            aclrtFree(mxscaleDeviceAddr);
            aclrtFree(rstdDeviceAddr);
            aclrtDestroyStream(stream);
            aclrtResetDevice(deviceId);
            aclFinalize();
            return ret;
        }

        LOG_PRINT("Output y (first 10 values):\n");
        for (int64_t i = 0; i < std::min(size, (int64_t)10); i++) {
            LOG_PRINT("  y[%ld] = 0x%02x\n", i, yResult[i]);
        }

        size = GetShapeSize(mxscaleShape);
        std::vector<uint8_t> mxscaleResult(size, 0);
        ret = aclrtMemcpy(
            mxscaleResult.data(), mxscaleResult.size() * sizeof(uint8_t), mxscaleDeviceAddr, size * sizeof(uint8_t),
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("copy mxscale from device to host failed. ERROR: %d\n", ret);
            if (workspaceAddr)
                aclrtFree(workspaceAddr);
            aclDestroyTensor(x);
            aclDestroyTensor(gamma);
            aclDestroyTensor(y);
            aclDestroyTensor(mxscale);
            aclDestroyTensor(rstd);
            aclrtFree(xDeviceAddr);
            aclrtFree(gammaDeviceAddr);
            aclrtFree(yDeviceAddr);
            aclrtFree(mxscaleDeviceAddr);
            aclrtFree(rstdDeviceAddr);
            aclrtDestroyStream(stream);
            aclrtResetDevice(deviceId);
            aclFinalize();
            return ret;
        }

        LOG_PRINT("MX scale values (first 10):\n");
        for (int64_t i = 0; i < std::min(size, (int64_t)10); i++) {
            LOG_PRINT("  mxscale[%ld] = 0x%02x\n", i, mxscaleResult[i]);
        }

        size = GetShapeSize(rstdShape);
        std::vector<float> rstdResult(size, 0.0f);
        ret = aclrtMemcpy(
            rstdResult.data(), rstdResult.size() * sizeof(float), rstdDeviceAddr, size * sizeof(float),
            ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("copy rstd from device to host failed. ERROR: %d\n", ret);
            if (workspaceAddr)
                aclrtFree(workspaceAddr);
            aclDestroyTensor(x);
            aclDestroyTensor(gamma);
            aclDestroyTensor(y);
            aclDestroyTensor(mxscale);
            aclDestroyTensor(rstd);
            aclrtFree(xDeviceAddr);
            aclrtFree(gammaDeviceAddr);
            aclrtFree(yDeviceAddr);
            aclrtFree(mxscaleDeviceAddr);
            aclrtFree(rstdDeviceAddr);
            aclrtDestroyStream(stream);
            aclrtResetDevice(deviceId);
            aclFinalize();
            return ret;
        }

        LOG_PRINT("Rstd values:\n");
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("  rstd[%ld] = %f\n", i, rstdResult[i]);
        }
    }

    LOG_PRINT("\n=== RmsNormDynamicMxQuant Test PASSED ===\n");

    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(gamma);
    aclDestroyTensor(y);
    aclDestroyTensor(mxscale);
    aclDestroyTensor(rstd);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(mxscaleDeviceAddr);
    aclrtFree(rstdDeviceAddr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
