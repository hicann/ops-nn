/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * @file test_aclnn_apply_gradient_descent.cpp
 * @brief aclnnApplyGradientDescent 调用示例。
 *
 * 功能：梯度下降单步参数更新（inplace）。
 * 公式：var = var - alpha * delta
 * 注意：本算子为 inplace 语义，var 既是输入也是输出，与其 Device 内存共享。
 */
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_apply_gradient_descent.h"

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
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
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
    // 1. （固定写法）device/stream 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    //    var/delta shape 必须完全一致，alpha 为 1 元素标量 Tensor，dtype 与 var 一致。
    std::vector<int64_t> shape = {4, 2};
    std::vector<int64_t> alphaShape = {1};
    void* varDeviceAddr = nullptr;
    void* alphaDeviceAddr = nullptr;
    void* deltaDeviceAddr = nullptr;
    aclTensor* var = nullptr;
    aclTensor* alpha = nullptr;
    aclTensor* delta = nullptr;

    std::vector<float> varHostData = {1.0f, 2.0f, -1.0f, 0.5f, 0.0f, -0.5f, 1.5f, -2.0f};
    std::vector<float> alphaHostData = {0.1f};
    std::vector<float> deltaHostData = {0.5f, -0.3f, 0.1f, -0.2f, 0.4f, -0.1f, 0.3f, -0.4f};

    // var 既是输入也是输出（inplace）
    ret = CreateAclTensor(varHostData, shape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(alphaHostData, alphaShape, &alphaDeviceAddr, aclDataType::ACL_FLOAT, &alpha);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(deltaHostData, shape, &deltaDeviceAddr, aclDataType::ACL_FLOAT, &delta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用 CANN 算子库 API（两段式接口）
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnApplyGradientDescentGetWorkspaceSize(var, alpha, delta, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyGradientDescentGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnApplyGradientDescent(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyGradientDescent failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出（inplace：从 var Device 地址读回）
    auto size = GetShapeSize(shape);
    std::vector<float> varResult(size, 0);
    ret = aclrtMemcpy(varResult.data(), varResult.size() * sizeof(float), varDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy var result failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("var_out[%ld] = %.6f\n", i, varResult[i]);
    }

    // 6. 释放 aclTensor
    aclDestroyTensor(var);
    aclDestroyTensor(alpha);
    aclDestroyTensor(delta);

    // 7. 释放 Device 资源
    aclrtFree(varDeviceAddr);
    aclrtFree(alphaDeviceAddr);
    aclrtFree(deltaDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
