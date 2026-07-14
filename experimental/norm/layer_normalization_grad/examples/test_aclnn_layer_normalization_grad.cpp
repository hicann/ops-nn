/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
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
#include "aclnn_layer_normalization_grad.h"

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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr, const std::vector<float>& dyHostData)
{
    auto size = std::min(GetShapeSize(shape), static_cast<int64_t>(10));
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    LOG_PRINT("Notice: Only printing the first 10 elements.\n");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("dy[%ld] is: %f, dx[%ld] is: %f\n", i, dyHostData[i], i, resultData[i]);
    }
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
    // 1. 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入和输出 -- LayerNormalizationGrad: dy[N,D], x[N,D], gamma[D], mean[N], rstd[N]
    int64_t N = 16;
    int64_t D = 1024;
    std::vector<int64_t> ndShape = {N, D};
    std::vector<int64_t> dShape = {D};
    std::vector<int64_t> nShape = {N};

    // dy: [N, D]
    aclTensor* dy = nullptr;
    void* dyDeviceAddr = nullptr;
    std::vector<float> dyHostData(N * D, 1.0f);
    ret = CreateAclTensor(dyHostData, ndShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // x: [N, D]
    aclTensor* x = nullptr;
    void* xDeviceAddr = nullptr;
    std::vector<float> xHostData(N * D, 1.0f);
    ret = CreateAclTensor(xHostData, ndShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // gamma: [D]
    aclTensor* gamma = nullptr;
    void* gammaDeviceAddr = nullptr;
    std::vector<float> gammaHostData(D, 1.0f);
    ret = CreateAclTensor(gammaHostData, dShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // mean: [N]
    aclTensor* mean = nullptr;
    void* meanDeviceAddr = nullptr;
    std::vector<float> meanHostData(N, 0.0f);
    ret = CreateAclTensor(meanHostData, nShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // rstd: [N]
    aclTensor* rstd = nullptr;
    void* rstdDeviceAddr = nullptr;
    std::vector<float> rstdHostData(N, 1.0f);
    ret = CreateAclTensor(rstdHostData, nShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // dx: [N, D] (output)
    aclTensor* dx = nullptr;
    void* dxDeviceAddr = nullptr;
    std::vector<float> dxHostData(N * D, 0.0f);
    ret = CreateAclTensor(dxHostData, ndShape, &dxDeviceAddr, aclDataType::ACL_FLOAT, &dx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // dgamma: [D] (output)
    aclTensor* dgamma = nullptr;
    void* dgammaDeviceAddr = nullptr;
    std::vector<float> dgammaHostData(D, 0.0f);
    ret = CreateAclTensor(dgammaHostData, dShape, &dgammaDeviceAddr, aclDataType::ACL_FLOAT, &dgamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // dbeta: [D] (output)
    aclTensor* dbeta = nullptr;
    void* dbetaDeviceAddr = nullptr;
    std::vector<float> dbetaHostData(D, 0.0f);
    ret = CreateAclTensor(dbetaHostData, dShape, &dbetaDeviceAddr, aclDataType::ACL_FLOAT, &dbeta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用第一段接口获取 workspace 大小
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnLayerNormalizationGradGetWorkspaceSize(dy, x, gamma, mean, rstd, dx, dgamma, dbeta, &workspaceSize,
                                                      &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNormalizationGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 4. 申请 workspace 内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. 调用第二段接口执行算子
    ret = aclnnLayerNormalizationGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNormalizationGrad failed. ERROR: %d\n", ret); return ret);

    // 6. 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. 打印结果
    LOG_PRINT("dx output:\n");
    PrintOutResult(ndShape, &dxDeviceAddr, dyHostData);
    LOG_PRINT("dgamma output:\n");
    PrintOutResult(dShape, &dgammaDeviceAddr, gammaHostData);
    LOG_PRINT("dbeta output:\n");
    PrintOutResult(dShape, &dbetaDeviceAddr, gammaHostData);

    // 8. 释放资源
    aclDestroyTensor(dy);
    aclDestroyTensor(x);
    aclDestroyTensor(gamma);
    aclDestroyTensor(mean);
    aclDestroyTensor(rstd);
    aclDestroyTensor(dx);
    aclDestroyTensor(dgamma);
    aclDestroyTensor(dbeta);

    aclrtFree(dyDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(rstdDeviceAddr);
    aclrtFree(dxDeviceAddr);
    aclrtFree(dgammaDeviceAddr);
    aclrtFree(dbetaDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
