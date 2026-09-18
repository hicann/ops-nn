/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnn_fused_matmul_silu.h"

#define CHECK_RET(expr)                                                       \
    do {                                                                      \
        const auto ret = (expr);                                              \
        if (ret != ACL_SUCCESS) {                                             \
            std::printf("%s failed, ret=%d\n", #expr, static_cast<int>(ret)); \
            return static_cast<int>(ret);                                     \
        }                                                                     \
    } while (0)

namespace {
constexpr int64_t kM = 2;
constexpr int64_t kK = 64;
constexpr int64_t kN = 4;
constexpr uint16_t kBf16One = 0x3F80;
constexpr uint16_t kExpectedBf16Output = 0x4280;

aclTensor* CreateTensor(void* deviceAddr, const int64_t* shape, size_t dimNum)
{
    std::array<int64_t, 2> stride = {1, 1};
    for (size_t i = dimNum - 1; i > 0; --i) {
        stride[i - 1] = stride[i] * shape[i];
    }
    return aclCreateTensor(shape, dimNum, ACL_BF16, stride.data(), 0, ACL_FORMAT_ND, shape, dimNum, deviceAddr);
}

void DestroyTensor(aclTensor*& tensor)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
        tensor = nullptr;
    }
}

void FreeDevice(void*& deviceAddr)
{
    if (deviceAddr != nullptr) {
        aclrtFree(deviceAddr);
        deviceAddr = nullptr;
    }
}
} // namespace

int main()
{
    const int64_t xShape[] = {kM, kK};
    const int64_t weightShape[] = {kN, kK};
    const int64_t biasShape[] = {kN};
    const int64_t yShape[] = {kM, kN};
    std::vector<uint16_t> xHost(kM * kK, kBf16One);
    std::vector<uint16_t> weightHost(kN * kK, kBf16One);
    std::vector<uint16_t> biasHost(kN, 0);
    std::vector<uint16_t> yHost(kM * kN, 0);
    aclrtStream stream = nullptr;
    void* xDevice = nullptr;
    void* weightDevice = nullptr;
    void* biasDevice = nullptr;
    void* yDevice = nullptr;
    void* workspace = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* y = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    CHECK_RET(aclInit(nullptr));
    CHECK_RET(aclrtSetDevice(0));
    CHECK_RET(aclrtCreateStream(&stream));
    CHECK_RET(aclrtMalloc(&xDevice, xHost.size() * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_RET(aclrtMalloc(&weightDevice, weightHost.size() * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_RET(aclrtMalloc(&biasDevice, biasHost.size() * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_RET(aclrtMalloc(&yDevice, yHost.size() * sizeof(uint16_t), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_RET(aclrtMemcpy(xDevice, xHost.size() * sizeof(uint16_t), xHost.data(), xHost.size() * sizeof(uint16_t),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_RET(aclrtMemcpy(weightDevice, weightHost.size() * sizeof(uint16_t), weightHost.data(),
                          weightHost.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_RET(aclrtMemcpy(biasDevice, biasHost.size() * sizeof(uint16_t), biasHost.data(),
                          biasHost.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));

    x = CreateTensor(xDevice, xShape, 2);
    weight = CreateTensor(weightDevice, weightShape, 2);
    bias = CreateTensor(biasDevice, biasShape, 1);
    y = CreateTensor(yDevice, yShape, 2);
    if (x == nullptr || weight == nullptr || bias == nullptr || y == nullptr) {
        std::printf("aclCreateTensor failed\n");
        return 1;
    }
    CHECK_RET(aclnnFusedMatmulSiluGetWorkspaceSize(x, weight, bias, y, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        CHECK_RET(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_RET(aclnnFusedMatmulSilu(workspace, workspaceSize, executor, stream));
    CHECK_RET(aclrtSynchronizeStream(stream));
    CHECK_RET(aclrtMemcpy(yHost.data(), yHost.size() * sizeof(uint16_t), yDevice, yHost.size() * sizeof(uint16_t),
                          ACL_MEMCPY_DEVICE_TO_HOST));

    bool resultMatches = true;
    for (uint16_t value : yHost) {
        if (value != kExpectedBf16Output) {
            resultMatches = false;
            break;
        }
    }

    FreeDevice(workspace);
    DestroyTensor(x);
    DestroyTensor(weight);
    DestroyTensor(bias);
    DestroyTensor(y);
    FreeDevice(xDevice);
    FreeDevice(weightDevice);
    FreeDevice(biasDevice);
    FreeDevice(yDevice);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    if (!resultMatches) {
        std::printf("aclnnFusedMatmulSilu result check failed.\n");
        return 1;
    }
    std::printf("aclnnFusedMatmulSilu example passed.\n");
    return 0;
}
