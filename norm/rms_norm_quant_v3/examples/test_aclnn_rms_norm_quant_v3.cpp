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
#include <cmath>
#include <cstring>
#include <memory>
#include "acl/acl.h"
#include "aclnnop/aclnn_rms_norm_quant_v3.h"

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
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);

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

    LOG_PRINT("Warning: This operator only supports Ascend950, current SOC '%s' is not supported. Skip test.\n",
              socName);
    return false;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    if (!CheckHardwareSupport()) {
        LOG_PRINT("\n=== Test SKIPPED (hardware not supported) ===\n");
        Finalize(deviceId, stream);
        return 0;
    }

    // 2. 构造输入与输出
    // x: [2, 64] FP16, gamma: [64] FP16, scale: [1] FP16, offset: [1] INT8, beta: [64] FP16
    // y: [2, 64] INT8, rstd: [2, 1] FLOAT32
    std::vector<int64_t> x_shape = {2, 64};
    std::vector<int64_t> gamma_shape = {64};
    std::vector<int64_t> scale_shape = {1};
    std::vector<int64_t> offset_shape = {1};
    std::vector<int64_t> beta_shape = {64};
    std::vector<int64_t> y_shape = {2, 64};
    std::vector<int64_t> rstd_shape = {2, 1};

    void* x_device_addr = nullptr;
    void* gamma_device_addr = nullptr;
    void* scale_device_addr = nullptr;
    void* offset_device_addr = nullptr;
    void* beta_device_addr = nullptr;
    void* y_device_addr = nullptr;
    void* rstd_device_addr = nullptr;
    void* workspace_addr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* y = nullptr;
    aclTensor* rstd = nullptr;

    // 初始化输入数据
    // x: FP16 0.5 = 0x3800
    std::vector<uint16_t> x_host_data(GetShapeSize(x_shape), 0x3800);
    // gamma: FP16 1.5 = 0x3e00
    std::vector<uint16_t> gamma_host_data(GetShapeSize(gamma_shape), 0x3e00);
    // scale: FP16 1.0 = 0x3C00
    std::vector<uint16_t> scale_host_data(GetShapeSize(scale_shape), 0x3C00);
    // offset: INT8 0
    std::vector<int8_t> offset_host_data(GetShapeSize(offset_shape), 0);
    // beta: FP16 0.0 = 0x0000
    std::vector<uint16_t> beta_host_data(GetShapeSize(beta_shape), 0);
    // 输出初始化
    std::vector<int8_t> y_host_data(GetShapeSize(y_shape), 0);
    std::vector<float> rstd_host_data(GetShapeSize(rstd_shape), 0.0f);

    double epsilon = 1e-6;
    bool divMode = true;
    bool outputRstd = true;

    LOG_PRINT("Input x shape: [2, 64], dtype: FP16\n");
    LOG_PRINT("Gamma shape: [64], dtype: FP16\n");
    LOG_PRINT("Scale shape: [1], dtype: FP16\n");
    LOG_PRINT("Offset shape: [1], dtype: INT8\n");
    LOG_PRINT("Beta shape: [64], dtype: FP16\n");
    LOG_PRINT("Output y shape: [2, 64], dtype: INT8\n");
    LOG_PRINT("Output rstd shape: [2, 1], dtype: FLOAT32\n");
    LOG_PRINT("epsilon: %e, divMode: %s, outputRstd: %s\n",
              epsilon, divMode ? "true" : "false", outputRstd ? "true" : "false");

    // 创建x aclTensor (FP16)
    ret = CreateAclTensor(x_host_data, x_shape, &x_device_addr, aclDataType::ACL_FLOAT16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(x_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor x failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建gamma aclTensor (FP16)
    ret = CreateAclTensor(gamma_host_data, gamma_shape, &gamma_device_addr, aclDataType::ACL_FLOAT16, &gamma);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> gammaTensorPtr(gamma, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> gammaDeviceAddrPtr(gamma_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor gamma failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建scale aclTensor (FP16)
    ret = CreateAclTensor(scale_host_data, scale_shape, &scale_device_addr, aclDataType::ACL_FLOAT16, &scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scaleTensorPtr(scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scaleDeviceAddrPtr(scale_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor scale failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建offset aclTensor (INT8)
    ret = CreateAclTensor(offset_host_data, offset_shape, &offset_device_addr, aclDataType::ACL_INT8, &offset);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> offsetTensorPtr(offset, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> offsetDeviceAddrPtr(offset_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor offset failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建beta aclTensor (FP16)
    ret = CreateAclTensor(beta_host_data, beta_shape, &beta_device_addr, aclDataType::ACL_FLOAT16, &beta);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> betaTensorPtr(beta, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> betaDeviceAddrPtr(beta_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor beta failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建y aclTensor (INT8输出)
    ret = CreateAclTensor(y_host_data, y_shape, &y_device_addr, aclDataType::ACL_INT8, &y);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(y_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor y failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建rstd aclTensor (FLOAT32输出)
    ret = CreateAclTensor(rstd_host_data, rstd_shape, &rstd_device_addr, aclDataType::ACL_FLOAT, &rstd);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> rstdTensorPtr(rstd, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> rstdDeviceAddrPtr(rstd_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor rstd failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 3. 调用CANN算子库API
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    // 调用aclnnRmsNormQuantV3第一段接口
    LOG_PRINT("Calling aclnnRmsNormQuantV3GetWorkspaceSize...\n");
    ret = aclnnRmsNormQuantV3GetWorkspaceSize(
        x, gamma, scale, offset, beta, epsilon, divMode, outputRstd,
        y, rstd, &workspace_size, &executor);

    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnRmsNormQuantV3GetWorkspaceSize failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    LOG_PRINT("Workspace size: %lu bytes (%.2f KB)\n", workspace_size, workspace_size / 1024.0);

    // 根据第一段接口计算出的workspaceSize申请device内存
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);
        workspaceAddrPtr.reset(workspace_addr);
    }

    // 调用aclnnRmsNormQuantV3第二段接口
    LOG_PRINT("Calling aclnnRmsNormQuantV3...\n");
    ret = aclnnRmsNormQuantV3(workspaceAddrPtr.get(), workspace_size, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnRmsNormQuantV3 failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 4. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
    {
        // 拷贝y结果 (INT8)
        auto size = GetShapeSize(y_shape);
        std::vector<int8_t> y_result(size, 0);
        ret = aclrtMemcpy(y_result.data(), y_result.size() * sizeof(y_result[0]),
                          yDeviceAddrPtr.get(), size * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("copy y from device to host failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);

        LOG_PRINT("Output y (first 10 values):\n");
        int64_t numTen = 10;
        for (int64_t i = 0; i < std::min(size, numTen); i++) {
            LOG_PRINT("  y[%ld] = %d\n", i, y_result[i]);
        }

        // 拷贝rstd结果 (FLOAT32)
        size = GetShapeSize(rstd_shape);
        std::vector<float> rstd_result(size, 0.0f);
        ret = aclrtMemcpy(rstd_result.data(), rstd_result.size() * sizeof(rstd_result[0]),
                          rstdDeviceAddrPtr.get(), size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("copy rstd from device to host failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);

        LOG_PRINT("Output rstd values:\n");
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("  rstd[%ld] = %f\n", i, rstd_result[i]);
        }
    }

    LOG_PRINT("\n=== RmsNormQuantV3 Test PASSED ===\n");

    // 6. 资源自动释放
    Finalize(deviceId, stream);

    return 0;
}
