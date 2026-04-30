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
#include "acl/acl.h"
#include "aclnn_l1_loss_backward.h"

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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);

    // 调用 aclrtMalloc 申请 device 侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用 aclrtMemcpy 将 host 侧数据拷贝到 device 侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续 tensor 的 strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用 aclCreateTensor 接口创建 aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    std::vector<int64_t> commonShape = {2, 2};

    void* gradsDeviceAddr = nullptr;
    void* predictDeviceAddr = nullptr;
    void* labelDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;

    aclTensor* grads = nullptr;
    aclTensor* predict = nullptr;
    aclTensor* label = nullptr;
    aclTensor* y = nullptr;

    // 构造测试数据
    std::vector<float> gradsHostData = {1.0, 1.0, 1.0, 1.0};
    std::vector<float> predictHostData = {2.0, 3.0, 0.0, 1.0};
    std::vector<float> labelHostData = {1.0, 1.0, 1.0, 1.0};
    std::vector<float> yHostData = {0.0, 0.0, 0.0, 0.0};

    // 创建输入 Tensors
    ret = CreateAclTensor(gradsHostData, commonShape, &gradsDeviceAddr, aclDataType::ACL_FLOAT, &grads);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(predictHostData, commonShape, &predictDeviceAddr, aclDataType::ACL_FLOAT, &predict);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(labelHostData, commonShape, &labelDeviceAddr, aclDataType::ACL_FLOAT, &label);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输出 Tensor
    ret = CreateAclTensor(yHostData, commonShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 准备 Attribute
    int64_t reduction = 1; // 0: none, 1: mean, 2: sum

    // 3. 调用 CANN 算子库 API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 获取 Workspace 大小（输入顺序：grads, predict, label）
    ret = aclnnL1LossBackwardGetWorkspaceSize(grads, predict, label, reduction, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnL1LossBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 申请 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 执行算子
    ret = aclnnL1LossBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnL1LossBackward failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出结果
    auto size = GetShapeSize(commonShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // 打印结果
    LOG_PRINT("Input Grads: ");
    for (auto v : gradsHostData)
        LOG_PRINT("%.1f ", v);
    LOG_PRINT("\nInput Predict: ");
    for (auto v : predictHostData)
        LOG_PRINT("%.1f ", v);
    LOG_PRINT("\nInput Label: ");
    for (auto v : labelHostData)
        LOG_PRINT("%.1f ", v);
    LOG_PRINT("\nReduction: %ld\n", reduction);

    LOG_PRINT("\nResult Y: \n");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("Index[%ld]: %f\n", i, resultData[i]);
    }

    // 6. 释放资源
    aclDestroyTensor(grads);
    aclDestroyTensor(predict);
    aclDestroyTensor(label);
    aclDestroyTensor(y);

    aclrtFree(gradsDeviceAddr);
    aclrtFree(predictDeviceAddr);
    aclrtFree(labelDeviceAddr);
    aclrtFree(yDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}