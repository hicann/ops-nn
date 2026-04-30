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
#include "aclnn_smooth_l1_loss_backward.h"
#include "aclnn/aclnn_base.h"

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
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 输入 shape
    std::vector<int64_t> predictShape = {2, 2};
    std::vector<int64_t> labelShape = {2, 2};
    // dout 的形状：reduction="none" 时与 predict 相同，这里简单测试 "none"
    std::vector<int64_t> doutShape = {2, 2};
    // 梯度输出 shape 必须与 predict 相同
    std::vector<int64_t> gradientShape = predictShape;

    void* predictDeviceAddr = nullptr;
    void* labelDeviceAddr = nullptr;
    void* doutDeviceAddr = nullptr;
    void* gradientDeviceAddr = nullptr;

    aclTensor* predictTensor = nullptr;
    aclTensor* labelTensor = nullptr;
    aclTensor* doutTensor = nullptr;
    aclTensor* gradientTensor = nullptr;

    std::vector<float> predictHostData = {0.6537, 1.1573, 0.1225, 0.2654};
    std::vector<float> labelHostData = {-0.7470, 0.3864, 0.8421, 0.5000};
    std::vector<float> doutHostData = {1.1, 1.2, 1.3, 1.4};
    std::vector<float> gradientHostData = {0, 0, 0, 0};

    // 创建 aclTensor
    ret = CreateAclTensor(predictHostData, predictShape, &predictDeviceAddr, aclDataType::ACL_FLOAT, &predictTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(labelHostData, labelShape, &labelDeviceAddr, aclDataType::ACL_FLOAT, &labelTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(doutHostData, doutShape, &doutDeviceAddr, aclDataType::ACL_FLOAT, &doutTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret =
        CreateAclTensor(gradientHostData, gradientShape, &gradientDeviceAddr, aclDataType::ACL_FLOAT, &gradientTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 算子属性
    double sigma = 1.0; // 原始 beta 参数
    float beta = static_cast<float>(sigma);
    int64_t reduction = 0; // 0: "none", 1: "mean", 2: "sum"

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    std::cout << "lnlnln";

    // 调用获取 workspace 大小接口
    ret = aclnnSmoothL1LossBackwardGetWorkspaceSize(
        doutTensor, predictTensor, labelTensor, reduction, beta, gradientTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSmoothL1LossBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 执行算子
    ret = aclnnSmoothL1LossBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSmoothL1LossBackward failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 将结果从 device 拷贝回 host
    auto size = GetShapeSize(gradientShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), gradientDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    // 打印结果
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("aclnnSmoothL1LossGradV2 gradient[%ld] = %f\n", i, resultData[i]);
    }

    // 释放资源
    aclDestroyTensor(predictTensor);
    aclDestroyTensor(labelTensor);
    aclDestroyTensor(doutTensor);
    aclDestroyTensor(gradientTensor);

    aclrtFree(predictDeviceAddr);
    aclrtFree(labelDeviceAddr);
    aclrtFree(doutDeviceAddr);
    aclrtFree(gradientDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}