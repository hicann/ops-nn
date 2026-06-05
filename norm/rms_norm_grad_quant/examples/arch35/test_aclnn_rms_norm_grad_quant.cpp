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
#include "aclnnop/aclnn_rms_norm_grad_quant.h"

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
    // 1. device/stream 初始化
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
    // RmsNormGradQuant: dy[M, D], x[M, D], rstd[M], gamma[D], scalesX[1], offsetX[1] -> dx[M, D], dgamma[D]
    std::vector<int64_t> dy_shape = {8, 64};
    std::vector<int64_t> x_shape = {8, 64};
    std::vector<int64_t> rstd_shape = {8};
    std::vector<int64_t> gamma_shape = {64};
    std::vector<int64_t> scales_x_shape = {1};
    std::vector<int64_t> offset_x_shape = {1};
    std::vector<int64_t> dx_shape = {8, 64};
    std::vector<int64_t> dgamma_shape = {64};

    void* dy_device_addr = nullptr;
    void* x_device_addr = nullptr;
    void* rstd_device_addr = nullptr;
    void* gamma_device_addr = nullptr;
    void* scales_x_device_addr = nullptr;
    void* offset_x_device_addr = nullptr;
    void* dx_device_addr = nullptr;
    void* dgamma_device_addr = nullptr;
    void* workspace_addr = nullptr;

    aclTensor* dy = nullptr;
    aclTensor* x = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* scalesX = nullptr;
    aclTensor* offsetX = nullptr;
    aclTensor* dxOut = nullptr;
    aclTensor* dgammaOut = nullptr;

    // 初始化输入数据
    // dy, x: FP16, 1.0 = 0x3C00
    std::vector<uint16_t> dy_host_data(GetShapeSize(dy_shape), 0x3C00);
    std::vector<uint16_t> x_host_data(GetShapeSize(x_shape), 0x3C00);
    // rstd: FP32, 1.0
    std::vector<float> rstd_host_data(GetShapeSize(rstd_shape), 1.0f);
    // gamma: FP32, 1.0
    std::vector<float> gamma_host_data(GetShapeSize(gamma_shape), 1.0f);
    // scalesX: FP32, 1.0
    std::vector<float> scales_x_host_data(GetShapeSize(scales_x_shape), 1.0f);
    // offsetX: INT32, 0
    std::vector<int32_t> offset_x_host_data(GetShapeSize(offset_x_shape), 0);
    // dx: INT8 输出
    std::vector<int8_t> dx_host_data(GetShapeSize(dx_shape), 0);
    // dgamma: FP32 输出
    std::vector<float> dgamma_host_data(GetShapeSize(dgamma_shape), 0.0f);

    char quantMode[] = "static";
    bool divMode = true;

    LOG_PRINT("Input shape: dy[8, 64], x[8, 64], rstd[8], gamma[64]\n");
    LOG_PRINT("Output shape: dx[8, 64](INT8), dgamma[64](FP32)\n");
    LOG_PRINT("quantMode: static, divMode: true\n");

    // 创建 dy aclTensor (FP16)
    ret = CreateAclTensor(dy_host_data, dy_shape, &dy_device_addr, aclDataType::ACL_FLOAT16, &dy);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> dyPtr(dy, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> dyDevPtr(dy_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor dy failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建 x aclTensor (FP16)
    ret = CreateAclTensor(x_host_data, x_shape, &x_device_addr, aclDataType::ACL_FLOAT16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDevPtr(x_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor x failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建 rstd aclTensor (FP32)
    ret = CreateAclTensor(rstd_host_data, rstd_shape, &rstd_device_addr, aclDataType::ACL_FLOAT, &rstd);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> rstdPtr(rstd, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> rstdDevPtr(rstd_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor rstd failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建 gamma aclTensor (FP32)
    ret = CreateAclTensor(gamma_host_data, gamma_shape, &gamma_device_addr, aclDataType::ACL_FLOAT, &gamma);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> gammaPtr(gamma, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> gammaDevPtr(gamma_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor gamma failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建 scalesX aclTensor (FP32)
    ret = CreateAclTensor(scales_x_host_data, scales_x_shape, &scales_x_device_addr, aclDataType::ACL_FLOAT, &scalesX);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scalesXPtr(scalesX, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scalesXDevPtr(scales_x_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor scalesX failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建 offsetX aclTensor (INT32)
    ret = CreateAclTensor(offset_x_host_data, offset_x_shape, &offset_x_device_addr, aclDataType::ACL_INT32, &offsetX);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> offsetXPtr(offsetX, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> offsetXDevPtr(offset_x_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor offsetX failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建 dxOut aclTensor (INT8)
    ret = CreateAclTensor(dx_host_data, dx_shape, &dx_device_addr, aclDataType::ACL_INT8, &dxOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> dxOutPtr(dxOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> dxDevPtr(dx_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor dxOut failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建 dgammaOut aclTensor (FP32)
    ret = CreateAclTensor(dgamma_host_data, dgamma_shape, &dgamma_device_addr, aclDataType::ACL_FLOAT, &dgammaOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> dgammaOutPtr(dgammaOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> dgammaDevPtr(dgamma_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor dgammaOut failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 3. 调用 CANN 算子库 API
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    // 调用 aclnnRmsNormGradQuant 第一段接口
    LOG_PRINT("Calling aclnnRmsNormGradQuantGetWorkspaceSize...\n");
    ret = aclnnRmsNormGradQuantGetWorkspaceSize(
        dy, x, rstd, gamma, scalesX, offsetX,
        quantMode, divMode,
        dxOut, dgammaOut, &workspace_size, &executor);

    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnRmsNormGradQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    LOG_PRINT("Workspace size: %lu bytes (%.2f KB)\n", workspace_size, workspace_size / 1024.0);

    // 根据第一段接口计算出的 workspaceSize 申请 device 内存
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);
        workspaceAddrPtr.reset(workspace_addr);
    }

    // 调用 aclnnRmsNormGradQuant 第二段接口
    LOG_PRINT("Calling aclnnRmsNormGradQuant...\n");
    ret = aclnnRmsNormGradQuant(workspaceAddrPtr.get(), workspace_size, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnRmsNormGradQuant failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 4. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 5. 获取输出的值，将 device 侧内存上的结果拷贝至 host 侧
    {
        // 拷贝 dx 结果 (INT8)
        auto size = GetShapeSize(dx_shape);
        std::vector<int8_t> dx_result(size, 0);
        ret = aclrtMemcpy(dx_result.data(), dx_result.size() * sizeof(dx_result[0]),
                          dxDevPtr.get(), size * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("copy dx from device to host failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);

        LOG_PRINT("Output dx (first 10 values):\n");
        for (int64_t i = 0; i < std::min(size, (int64_t)10); i++) {
            LOG_PRINT("  dx[%ld] = %d\n", i, dx_result[i]);
        }

        // 拷贝 dgamma 结果 (FP32)
        size = GetShapeSize(dgamma_shape);
        std::vector<float> dgamma_result(size, 0.0f);
        ret = aclrtMemcpy(dgamma_result.data(), dgamma_result.size() * sizeof(dgamma_result[0]),
                          dgammaDevPtr.get(), size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("copy dgamma from device to host failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);

        LOG_PRINT("Output dgamma (first 10 values):\n");
        for (int64_t i = 0; i < std::min(size, (int64_t)10); i++) {
            LOG_PRINT("  dgamma[%ld] = %f\n", i, dgamma_result[i]);
        }
    }

    LOG_PRINT("\n=== RmsNormGradQuant Test PASSED ===\n");

    // 6. 资源自动释放
    Finalize(deviceId, stream);

    return 0;
}
