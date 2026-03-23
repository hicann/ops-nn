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
#include <cmath>
#include <cstring>
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

// Data type definitions for dstType parameter (matching ge::DataType enum)
// Refer to rms_norm_dynamic_mx_quant_base_tiling.cpp for mapping:
// FLOAT8_E5M2=35, FLOAT8_E4M3FN=36, FLOAT4_E2M1=40, FLOAT4_E1M2=41
#define GE_DT_FLOAT8_E5M2 35
#define GE_DT_FLOAT8_E4M3FN 36
#define GE_DT_FLOAT4_E2M1 40
#define GE_DT_FLOAT4_E1M2 41

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

// Check if current hardware is supported
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

    LOG_PRINT("Warning: This operator only supports Ascend950, current SOC '%s' is not supported. Skip test.\n", socName);
    return false;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
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
        Finalize(deviceId, stream);
        return 0;  // 返回0表示测试通过（被跳过）
    }

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // 注意：当前算子仅支持输出FP8类型（E4M3FN或E5M2）
    std::vector<int64_t> x_shape = {2, 32};
    std::vector<int64_t> gamma_shape = {32};
    std::vector<int64_t> y_out_shape = {2, 32};

    // MX scale shape: [H, numBlocks, 2] where numBlocks = ceil(N/32)
    int64_t mx_block_num = (32 + 31) / 32;  // = 1
    int64_t mxscale_dim = (mx_block_num + 1) / 2;  // = 1
    std::vector<int64_t> mxscale_out_shape = {2, mxscale_dim, 2};
    std::vector<int64_t> rstd_out_shape = {2};

    void* x_device_addr = nullptr;
    void* gamma_device_addr = nullptr;
    void* y_out_device_addr = nullptr;
    void* mxscale_out_device_addr = nullptr;
    void* rstd_out_device_addr = nullptr;
    void* workspace_addr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* y_out = nullptr;
    aclTensor* mxscale_out = nullptr;
    aclTensor* rstd_out = nullptr;

    constexpr int DIMSIZE = 32;
    constexpr int NUM_TWO = 2;
    // Initialize input data (FP16): 1.0 = 0x3C00 in FP16
    std::vector<uint16_t> x_host_data(DIMSIZE * NUM_TWO, 0x3C00);
    std::vector<uint16_t> gamma_host_data(DIMSIZE, 0x3C00);
    std::vector<uint16_t> beta_host_data(DIMSIZE, 0);  // beta optional, init with zeros
    // Output FP8 data
    std::vector<uint8_t> y_out_host_data(DIMSIZE * NUM_TWO, 0);
    std::vector<uint8_t> mxscale_out_host_data(GetShapeSize(mxscale_out_shape), 0);
    std::vector<float> rstd_out_host_data(GetShapeSize(rstd_out_shape), 0.0f);

    double epsilon = 1e-6;
    int64_t quant_alg = 0;  // OCP algorithm
    char* round_mode_optional = const_cast<char*>("rint");
    int64_t dst_type = GE_DT_FLOAT8_E4M3FN;  // FP8 E4M3FN output
    bool output_rstd = true;

    LOG_PRINT("Input shape: [2, 32], Total elements: %ld\n", static_cast<int64_t>(64));
    LOG_PRINT("MX block size: 32, Num blocks: %ld\n", mx_block_num);
    LOG_PRINT("MX scale shape: [%ld, %ld, 2]\n", static_cast<int64_t>(2), mxscale_dim);
    LOG_PRINT("Output dtype: FP8 E4M3FN (dst_type=%ld)\n", dst_type);

    // 创建x aclTensor (FP16输入)
    ret = CreateAclTensor(x_host_data, x_shape, &x_device_addr, aclDataType::ACL_FLOAT16, &x);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("CreateAclTensor x failed. ERROR: %d\n", ret);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }

    // 创建gamma aclTensor (FP16输入)
    ret = CreateAclTensor(gamma_host_data, gamma_shape, &gamma_device_addr, aclDataType::ACL_FLOAT16, &gamma);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("CreateAclTensor gamma failed. ERROR: %d\n", ret);
        aclDestroyTensor(x);
        aclrtFree(x_device_addr);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }

    // 创建beta aclTensor (FP16输入，可选)
    void* beta_device_addr = nullptr;
    aclTensor* beta = nullptr;
    std::vector<int64_t> beta_shape = {32};
    ret = CreateAclTensor(beta_host_data, beta_shape, &beta_device_addr, aclDataType::ACL_FLOAT16, &beta);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("CreateAclTensor beta failed. ERROR: %d\n", ret);
        aclDestroyTensor(x);
        aclDestroyTensor(gamma);
        aclDestroyTensor(beta);
        aclrtFree(x_device_addr);
        aclrtFree(gamma_device_addr);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }

    // 创建y_out aclTensor (FP8 E4M3FN输出)
    ret = CreateAclTensor(y_out_host_data, y_out_shape, &y_out_device_addr, aclDataType::ACL_FLOAT8_E4M3FN, &y_out);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("CreateAclTensor y_out failed. ERROR: %d\n", ret);
        aclDestroyTensor(x);
        aclDestroyTensor(gamma);
        aclDestroyTensor(beta);
        aclrtFree(x_device_addr);
        aclrtFree(gamma_device_addr);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }

    // 创建mxscale_out aclTensor (FP8 E8M0输出)
    ret = CreateAclTensor(mxscale_out_host_data, mxscale_out_shape, &mxscale_out_device_addr, 
                          aclDataType::ACL_FLOAT8_E8M0, &mxscale_out);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("CreateAclTensor mxscale_out failed. ERROR: %d\n", ret);
        aclDestroyTensor(x);
        aclDestroyTensor(gamma);
        aclDestroyTensor(y_out);
        aclrtFree(x_device_addr);
        aclrtFree(gamma_device_addr);
        aclrtFree(beta_device_addr);
        aclrtFree(y_out_device_addr);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }

    // 创建rstd_out aclTensor (float输出)
    ret = CreateAclTensor(rstd_out_host_data, rstd_out_shape, &rstd_out_device_addr, aclDataType::ACL_FLOAT, &rstd_out);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("CreateAclTensor rstd_out failed. ERROR: %d\n", ret);
        aclDestroyTensor(x);
        aclDestroyTensor(gamma);
        aclDestroyTensor(y_out);
        aclDestroyTensor(mxscale_out);
        aclrtFree(x_device_addr);
        aclrtFree(gamma_device_addr);
        aclrtFree(beta_device_addr);
        aclrtFree(y_out_device_addr);
        aclrtFree(mxscale_out_device_addr);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    // 调用aclnnRmsNormDynamicMxQuant第一段接口
    LOG_PRINT("Calling aclnnRmsNormDynamicMxQuantGetWorkspaceSize...\n");
    ret = aclnnRmsNormDynamicMxQuantGetWorkspaceSize(
        x, gamma, beta, epsilon, quant_alg, round_mode_optional, dst_type, output_rstd,
        y_out, mxscale_out, rstd_out, &workspace_size, &executor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnRmsNormDynamicMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
        aclDestroyTensor(x);
        aclDestroyTensor(gamma);
        aclDestroyTensor(y_out);
        aclDestroyTensor(mxscale_out);
        aclDestroyTensor(rstd_out);
        aclrtFree(x_device_addr);
        aclrtFree(gamma_device_addr);
        aclrtFree(beta_device_addr);
        aclrtFree(y_out_device_addr);
        aclrtFree(mxscale_out_device_addr);
        aclrtFree(rstd_out_device_addr);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }
    LOG_PRINT("Workspace size: %lu bytes (%.2f KB)\n", workspace_size, workspace_size / 1024.0);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            aclDestroyTensor(x);
            aclDestroyTensor(gamma);
            aclDestroyTensor(y_out);
            aclDestroyTensor(mxscale_out);
            aclDestroyTensor(rstd_out);
            aclrtFree(x_device_addr);
            aclrtFree(gamma_device_addr);
            aclrtFree(y_out_device_addr);
            aclrtFree(mxscale_out_device_addr);
            aclrtFree(rstd_out_device_addr);
            aclrtDestroyStream(stream);
            aclrtResetDevice(deviceId);
            aclFinalize();
            return ret;
        }
    }
    // 调用aclnnRmsNormDynamicMxQuant第二段接口
    LOG_PRINT("Calling aclnnRmsNormDynamicMxQuant...\n");
    ret = aclnnRmsNormDynamicMxQuant(workspace_addr, workspace_size, executor, stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnRmsNormDynamicMxQuant failed. ERROR: %d\n", ret);
        if (workspace_addr) {aclrtFree(workspace_addr);};
        aclDestroyTensor(x);
        aclDestroyTensor(gamma);
        aclDestroyTensor(y_out);
        aclDestroyTensor(mxscale_out);
        aclDestroyTensor(rstd_out);
        aclrtFree(x_device_addr);
        aclrtFree(gamma_device_addr);
        aclrtFree(beta_device_addr);
        aclrtFree(y_out_device_addr);
        aclrtFree(mxscale_out_device_addr);
        aclrtFree(rstd_out_device_addr);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        if (workspace_addr) {aclrtFree(workspace_addr);};
        aclDestroyTensor(x);
        aclDestroyTensor(gamma);
        aclDestroyTensor(y_out);
        aclDestroyTensor(mxscale_out);
        aclDestroyTensor(rstd_out);
        aclrtFree(x_device_addr);
        aclrtFree(gamma_device_addr);
        aclrtFree(beta_device_addr);
        aclrtFree(y_out_device_addr);
        aclrtFree(mxscale_out_device_addr);
        aclrtFree(rstd_out_device_addr);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    {
        // 拷贝y_out结果
        auto size = GetShapeSize(y_out_shape);
        std::vector<uint8_t> y_out_result(size, 0);
        ret = aclrtMemcpy(
            y_out_result.data(), y_out_result.size() * sizeof(y_out_result[0]), 
            y_out_device_addr, size * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("copy y_out from device to host failed. ERROR: %d\n", ret);
            if (workspace_addr) {aclrtFree(workspace_addr);};
            aclDestroyTensor(x);
            aclDestroyTensor(gamma);
            aclDestroyTensor(y_out);
            aclDestroyTensor(mxscale_out);
            aclDestroyTensor(rstd_out);
            aclrtFree(x_device_addr);
            aclrtFree(gamma_device_addr);
            aclrtFree(y_out_device_addr);
            aclrtFree(mxscale_out_device_addr);
            aclrtFree(rstd_out_device_addr);
            aclrtDestroyStream(stream);
            aclrtResetDevice(deviceId);
            aclFinalize();
            return ret;
        }
        LOG_PRINT("Output yOut (first 10 values):\n");
        for (int64_t i = 0; i < std::min(size, static_cast<int64_t>(10)); i++) {
            LOG_PRINT("  yOut[%ld] = 0x%02x\n", i, y_out_result[i]);
        }
        // 拷贝mxscale_out结果
        size = GetShapeSize(mxscale_out_shape);
        std::vector<uint8_t> mxscale_out_result(size, 0);
        ret = aclrtMemcpy(
            mxscale_out_result.data(), mxscale_out_result.size() * sizeof(mxscale_out_result[0]), 
            mxscale_out_device_addr, size * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("copy mxscale_out from device to host failed. ERROR: %d\n", ret);
            if (workspace_addr) {aclrtFree(workspace_addr);};
            aclDestroyTensor(x);
            aclDestroyTensor(gamma);
            aclDestroyTensor(y_out);
            aclDestroyTensor(mxscale_out);
            aclDestroyTensor(rstd_out);
            aclrtFree(x_device_addr);
            aclrtFree(gamma_device_addr);
            aclrtFree(y_out_device_addr);
            aclrtFree(mxscale_out_device_addr);
            aclrtFree(rstd_out_device_addr);
            aclrtDestroyStream(stream);
            aclrtResetDevice(deviceId);
            aclFinalize();
            return ret;
        }
        LOG_PRINT("MX scale values:\n");
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("  mxscaleOut[%ld] = 0x%02x\n", i, mxscale_out_result[i]);
        }
        // 拷贝rstd_out结果
        size = GetShapeSize(rstd_out_shape);
        std::vector<float> rstd_out_result(size, 0.0f);
        ret = aclrtMemcpy(
            rstd_out_result.data(), rstd_out_result.size() * sizeof(rstd_out_result[0]), 
            rstd_out_device_addr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("copy rstd_out from device to host failed. ERROR: %d\n", ret);
            if (workspace_addr) {aclrtFree(workspace_addr);};
            aclDestroyTensor(x);
            aclDestroyTensor(gamma);
            aclDestroyTensor(y_out);
            aclDestroyTensor(mxscale_out);
            aclDestroyTensor(rstd_out);
            aclrtFree(x_device_addr);
            aclrtFree(gamma_device_addr);
            aclrtFree(y_out_device_addr);
            aclrtFree(mxscale_out_device_addr);
            aclrtFree(rstd_out_device_addr);
            aclrtDestroyStream(stream);
            aclrtResetDevice(deviceId);
            aclFinalize();
            return ret;
        }
        LOG_PRINT("Rstd values:\n");
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("  rstdOut[%ld] = %f\n", i, rstd_out_result[i]);
        }
    }
    LOG_PRINT("\n=== RmsNormDynamicMxQuant Test PASSED ===\n");
    if (workspace_addr) {
        aclrtFree(workspace_addr);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(gamma);
    aclDestroyTensor(beta);
    aclDestroyTensor(y_out);
    aclDestroyTensor(mxscale_out);
    aclDestroyTensor(rstd_out);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(x_device_addr);
    aclrtFree(gamma_device_addr);
    aclrtFree(beta_device_addr);
    aclrtFree(y_out_device_addr);
    aclrtFree(mxscale_out_device_addr);
    aclrtFree(rstd_out_device_addr);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return (ret == ACL_SUCCESS) ? 0 : ret;
}
