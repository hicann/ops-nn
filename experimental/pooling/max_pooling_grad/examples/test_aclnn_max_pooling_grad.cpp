/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_max_pooling_grad.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <unistd.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_pooling_grad.h"

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
    // 1. device/stream 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造测试数据: shape = [64, 512]
    //    演示梯度选择性传递: dx[i] = (x[i] == y[i]) ? dy[i] : 0
    //    前5个元素自定义, 其余 x==y=0, dy=0
    std::vector<int64_t> shape = {64, 512};
    int64_t numElements = GetShapeSize(shape);

    std::vector<float> dyHostData(numElements, 0.0f);
    std::vector<float> xHostData(numElements, 0.0f);
    std::vector<float> yHostData(numElements, 0.0f);
    std::vector<float> dxHostData(numElements, 0.0f);

    // 自定义前5个元素:
    dyHostData[0] = 0.5f;
    xHostData[0] = 2.0f;
    yHostData[0] = 2.0f; // x==y → dx=0.5
    dyHostData[1] = 1.5f;
    xHostData[1] = 1.0f;
    yHostData[1] = 1.0f; // x==y → dx=1.5
    dyHostData[2] = 3.0f;
    xHostData[2] = 5.0f;
    yHostData[2] = 5.0f; // x==y → dx=3.0
    dyHostData[3] = 2.0f;
    xHostData[3] = 0.0f;
    yHostData[3] = 0.0f; // x==y → dx=2.0
    dyHostData[4] = 9.9f;
    xHostData[4] = 1.0f;
    yHostData[4] = 2.0f; // x!=y → dx=0

    void *dyDeviceAddr = nullptr, *xDeviceAddr = nullptr;
    void *yDeviceAddr = nullptr, *dxDeviceAddr = nullptr;
    aclTensor *dyTensor = nullptr, *xTensor = nullptr;
    aclTensor *yTensor = nullptr, *dxTensor = nullptr;

    ret = CreateAclTensor(dyHostData, shape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dyTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xHostData, shape, &xDeviceAddr, aclDataType::ACL_FLOAT, &xTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, shape, &yDeviceAddr, aclDataType::ACL_FLOAT, &yTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dxHostData, shape, &dxDeviceAddr, aclDataType::ACL_FLOAT, &dxTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用 ACLNN 两阶段 API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnMaxPoolingGradGetWorkspaceSize(dyTensor, xTensor, yTensor, dxTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxPoolingGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnMaxPoolingGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxPoolingGrad failed. ERROR: %d\n", ret); return ret);

    // 4. 同步等待任务完成
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出结果
    std::vector<float> dxResult(numElements, 0);
    ret = aclrtMemcpy(dxResult.data(), dxResult.size() * sizeof(float), dxDeviceAddr, numElements * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dx result from device to host failed. ERROR: %d\n", ret); return ret);

    // 打印前几个结果验证 (x==y 时 dx 应等于 dy)
    LOG_PRINT("MaxPoolingGrad result (first 5 elements):\n");
    for (int64_t i = 0; i < 5 && i < numElements; i++) {
        LOG_PRINT("  dx[%ld] = %f  (dy=%f)\n", i, dxResult[i], dyHostData[i]);
    }

    // 6. 释放资源
    aclDestroyTensor(dyTensor);
    aclDestroyTensor(xTensor);
    aclDestroyTensor(yTensor);
    aclDestroyTensor(dxTensor);
    aclrtFree(dyDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(dxDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
