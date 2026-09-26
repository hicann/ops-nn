/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_npu_scatter_add_bwd.h"

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
    // 1.（固定写法）device/stream初始化，参考acl API手册
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    // y_grad: (D, H) 上游梯度, x: (N, H) 源张量, s: (N,) 缩放因子, indices: (N,) 目标行索引
    // x_grad: (N, H) x 的梯度, s_grad: (N,) s 的梯度
    int64_t D = 4;
    int64_t N = 8;
    int64_t H = 4;
    std::vector<int64_t> yGradShape = {D, H};
    std::vector<int64_t> xShape = {N, H};
    std::vector<int64_t> sShape = {N};
    std::vector<int64_t> indicesShape = {N};
    void* yGradDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* sDeviceAddr = nullptr;
    void* indicesDeviceAddr = nullptr;
    void* xGradDeviceAddr = nullptr;
    void* sGradDeviceAddr = nullptr;
    aclTensor* yGrad = nullptr;
    aclTensor* x = nullptr;
    aclTensor* s = nullptr;
    aclTensor* indices = nullptr;
    aclTensor* xGrad = nullptr;
    aclTensor* sGrad = nullptr;

    // y_grad 第 i 行所有元素为 i+1, x 第 i 行所有元素为 i+1, 缩放因子全1
    std::vector<aclFloat16> yGradHostData;
    for (int64_t i = 0; i < D * H; i++) {
        yGradHostData.push_back(aclFloatToFloat16(static_cast<float>(i / H + 1)));
    }
    std::vector<aclFloat16> xHostData;
    for (int64_t i = 0; i < N * H; i++) {
        xHostData.push_back(aclFloatToFloat16(static_cast<float>(i / H + 1)));
    }
    std::vector<aclFloat16> sHostData(N, aclFloatToFloat16(1));
    std::vector<int32_t> indicesHostData = {3, 0, 2, 1, 3, 0, 1, 2};

    ret = CreateAclTensor(yGradHostData, yGradShape, &yGradDeviceAddr, aclDataType::ACL_FLOAT16, &yGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sHostData, sShape, &sDeviceAddr, aclDataType::ACL_FLOAT16, &s);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<aclFloat16> xGradHostData(N * H, aclFloatToFloat16(0));
    ret = CreateAclTensor(xGradHostData, xShape, &xGradDeviceAddr, aclDataType::ACL_FLOAT16, &xGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<aclFloat16> sGradHostData(N, aclFloatToFloat16(0));
    ret = CreateAclTensor(sGradHostData, sShape, &sGradDeviceAddr, aclDataType::ACL_FLOAT16, &sGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnNpuScatterAddBwdGetWorkspaceSize(yGrad, x, s, indices, xGrad, sGrad, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuScatterAddBwdGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnNpuScatterAddBwd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuScatterAddBwd failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值
    // 预期结果：x_grad[i] = indices[i] + 1, s_grad[i] = 4 * (i + 1) * (indices[i] + 1)
    auto xGradSize = GetShapeSize(xShape);
    std::vector<aclFloat16> xGradResult(xGradSize, 0);
    ret = aclrtMemcpy(xGradResult.data(), xGradResult.size() * sizeof(xGradResult[0]), xGradDeviceAddr,
                      xGradSize * sizeof(xGradResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy x_grad from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < xGradSize; i++) {
        LOG_PRINT("x_grad[%ld][%ld] is: %f\n", i / H, i % H, aclFloat16ToFloat(xGradResult[i]));
    }
    auto sGradSize = GetShapeSize(sShape);
    std::vector<aclFloat16> sGradResult(sGradSize, 0);
    ret = aclrtMemcpy(sGradResult.data(), sGradResult.size() * sizeof(sGradResult[0]), sGradDeviceAddr,
                      sGradSize * sizeof(sGradResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy s_grad from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < sGradSize; i++) {
        LOG_PRINT("s_grad[%ld] is: %f\n", i, aclFloat16ToFloat(sGradResult[i]));
    }

    // 6. 释放aclTensor
    aclDestroyTensor(yGrad);
    aclDestroyTensor(x);
    aclDestroyTensor(s);
    aclDestroyTensor(indices);
    aclDestroyTensor(xGrad);
    aclDestroyTensor(sGrad);

    // 7. 释放device资源
    aclrtFree(yGradDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(sDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(xGradDeviceAddr);
    aclrtFree(sGradDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
