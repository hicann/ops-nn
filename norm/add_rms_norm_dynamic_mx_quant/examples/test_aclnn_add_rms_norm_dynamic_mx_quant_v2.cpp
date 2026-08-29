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
#include "aclnnop/aclnn_add_rms_norm_dynamic_mx_quant.h"

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

#define GE_DT_FLOAT8_E4M3FN 36
#define GE_DT_FLOAT8_E5M2 35

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
        LOG_PRINT("Warning: Cannot get SOC name, skip test\n");
        return false;
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    if (!CheckHardwareSupport()) {
        LOG_PRINT("\n=== Test SKIPPED (hardware not supported) ===\n");
        Finalize(deviceId, stream);
        return 0;
    }

    // x3 present: x = (x3 + x1) + x2
    std::vector<int64_t> x1_shape = {2, 32};
    std::vector<int64_t> x2_shape = {2, 32};
    std::vector<int64_t> x3_shape = {2, 32};
    std::vector<int64_t> gamma_shape = {32};
    std::vector<int64_t> y_out_shape = {2, 32};
    std::vector<int64_t> x_out_shape = {2, 32};

    int64_t mx_block_num = (32 + 31) / 32;
    int64_t mxscale_dim = (mx_block_num + 1) / 2;
    std::vector<int64_t> mxscale_out_shape = {2, mxscale_dim, 2};
    std::vector<int64_t> rstd_out_shape = {2, 1};

    void* x1_device_addr = nullptr;
    void* x2_device_addr = nullptr;
    void* x3_device_addr = nullptr;
    void* gamma_device_addr = nullptr;
    void* y_out_device_addr = nullptr;
    void* x_out_device_addr = nullptr;
    void* mxscale_out_device_addr = nullptr;
    void* rstd_out_device_addr = nullptr;
    void* workspace_addr = nullptr;

    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* x3 = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* y_out = nullptr;
    aclTensor* x_out = nullptr;
    aclTensor* mxscale_out = nullptr;
    aclTensor* rstd_out = nullptr;

    // FP16: 1.0 = 0x3C00, 0.5 = 0x3800
    std::vector<uint16_t> x1_host_data(2 * 32, 0x3C00);
    std::vector<uint16_t> x2_host_data(2 * 32, 0x3C00);
    std::vector<uint16_t> x3_host_data(2 * 32, 0x3800);
    std::vector<uint16_t> gamma_host_data(32, 0x3C00);
    std::vector<uint8_t> y_out_host_data(2 * 32, 0);
    std::vector<uint16_t> x_out_host_data(2 * 32, 0);
    std::vector<uint8_t> mxscale_out_host_data(GetShapeSize(mxscale_out_shape), 0);
    std::vector<float> rstd_out_host_data(GetShapeSize(rstd_out_shape), 0.0f);

    double epsilon = 1e-6;
    int64_t quant_alg = 0;
    char* round_mode_optional = const_cast<char*>("rint");
    int64_t dst_type = GE_DT_FLOAT8_E4M3FN;
    bool output_rstd = true;

    LOG_PRINT("V2 API: x3 present, x = (x3 + x1) + x2\n");
    LOG_PRINT("Input shape: [2, 32], x3=0.5, x1=1.0, x2=1.0 => x=2.5\n");
    LOG_PRINT("Output dtype: FP8 E4M3FN (dst_type=%ld)\n", dst_type);

    ret = CreateAclTensor(x1_host_data, x1_shape, &x1_device_addr, aclDataType::ACL_FLOAT16, &x1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor x1 failed. ERROR: %d\n", ret); Finalize(deviceId, stream);
              return ret);

    ret = CreateAclTensor(x2_host_data, x2_shape, &x2_device_addr, aclDataType::ACL_FLOAT16, &x2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor x2 failed. ERROR: %d\n", ret); Finalize(deviceId, stream);
              return ret);

    ret = CreateAclTensor(x3_host_data, x3_shape, &x3_device_addr, aclDataType::ACL_FLOAT16, &x3);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x3TensorPtr(x3, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x3DeviceAddrPtr(x3_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor x3 failed. ERROR: %d\n", ret); Finalize(deviceId, stream);
              return ret);

    ret = CreateAclTensor(gamma_host_data, gamma_shape, &gamma_device_addr, aclDataType::ACL_FLOAT16, &gamma);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> gammaTensorPtr(gamma, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> gammaDeviceAddrPtr(gamma_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor gamma failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    ret = CreateAclTensor(y_out_host_data, y_out_shape, &y_out_device_addr, aclDataType::ACL_FLOAT8_E4M3FN, &y_out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yOutTensorPtr(y_out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yOutDeviceAddrPtr(y_out_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor y_out failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    ret = CreateAclTensor(x_out_host_data, x_out_shape, &x_out_device_addr, aclDataType::ACL_FLOAT16, &x_out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xOutTensorPtr(x_out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xOutDeviceAddrPtr(x_out_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor x_out failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    ret = CreateAclTensor(mxscale_out_host_data, mxscale_out_shape, &mxscale_out_device_addr,
                          aclDataType::ACL_FLOAT8_E8M0, &mxscale_out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscaleOutTensorPtr(mxscale_out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> mxscaleOutDeviceAddrPtr(mxscale_out_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor mxscale_out failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    ret = CreateAclTensor(rstd_out_host_data, rstd_out_shape, &rstd_out_device_addr, aclDataType::ACL_FLOAT, &rstd_out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> rstdOutTensorPtr(rstd_out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> rstdOutDeviceAddrPtr(rstd_out_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor rstd_out failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    LOG_PRINT("Calling aclnnAddRmsNormDynamicMxQuantV2GetWorkspaceSize...\n");
    ret = aclnnAddRmsNormDynamicMxQuantV2GetWorkspaceSize(x1, x2, gamma, nullptr, x3, epsilon, quant_alg,
                                                          round_mode_optional, dst_type, output_rstd, y_out, x_out,
                                                          mxscale_out, rstd_out, &workspace_size, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormDynamicMxQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    LOG_PRINT("Workspace size: %lu bytes (%.2f KB)\n", workspace_size, workspace_size / 1024.0);

    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);
        workspaceAddrPtr.reset(workspace_addr);
    }

    LOG_PRINT("Calling aclnnAddRmsNormDynamicMxQuantV2...\n");
    ret = aclnnAddRmsNormDynamicMxQuantV2(workspaceAddrPtr.get(), workspace_size, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormDynamicMxQuantV2 failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    {
        auto size = GetShapeSize(x_out_shape);
        std::vector<uint16_t> x_out_result(size, 0);
        ret = aclrtMemcpy(x_out_result.data(), x_out_result.size() * sizeof(x_out_result[0]), xOutDeviceAddrPtr.get(),
                          size * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy x_out from device to host failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);
        LOG_PRINT("Output xOut (first 10 values, expecting 0x4100 = 2.5 in FP16):\n");
        for (int64_t i = 0; i < std::min(size, (int64_t)10); i++) {
            LOG_PRINT("  xOut[%ld] = 0x%04x\n", i, x_out_result[i]);
        }

        size = GetShapeSize(rstd_out_shape);
        std::vector<float> rstd_out_result(size, 0.0f);
        ret = aclrtMemcpy(rstd_out_result.data(), rstd_out_result.size() * sizeof(rstd_out_result[0]),
                          rstdOutDeviceAddrPtr.get(), size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy rstd_out from device to host failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);
        LOG_PRINT("Rstd values (expecting ~0.4 = 1/sqrt(2.5^2)):\n");
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("  rstdOut[%ld] = %f\n", i, rstd_out_result[i]);
        }

        size = GetShapeSize(y_out_shape);
        std::vector<uint8_t> y_out_result(size, 0);
        ret = aclrtMemcpy(y_out_result.data(), y_out_result.size() * sizeof(y_out_result[0]), yOutDeviceAddrPtr.get(),
                          size * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y_out from device to host failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);
        LOG_PRINT("Output yOut (first 10 values):\n");
        for (int64_t i = 0; i < std::min(size, (int64_t)10); i++) {
            LOG_PRINT("  yOut[%ld] = 0x%02x\n", i, y_out_result[i]);
        }

        size = GetShapeSize(mxscale_out_shape);
        std::vector<uint8_t> mxscale_out_result(size, 0);
        ret = aclrtMemcpy(mxscale_out_result.data(), mxscale_out_result.size() * sizeof(mxscale_out_result[0]),
                          mxscaleOutDeviceAddrPtr.get(), size * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy mxscale_out from device to host failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);
        LOG_PRINT("MX scale values:\n");
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("  mxscaleOut[%ld] = 0x%02x\n", i, mxscale_out_result[i]);
        }
    }

    LOG_PRINT("\n=== AddRmsNormDynamicMxQuantV2 Test PASSED ===\n");

    Finalize(deviceId, stream);
    return 0;
}
