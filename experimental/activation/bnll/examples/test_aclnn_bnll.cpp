/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file test_aclnn_bnll.cpp
 * @brief ACLNN invocation example for BNLL operator
 *
 * Demonstrates the two-phase aclnn calling pattern:
 *   1. aclnnBnllGetWorkspaceSize - get workspace size and executor
 *   2. aclnnBnll - execute computation
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "aclnn/aclnn_base.h"
#include "aclnn/acl_meta.h"
#include "acl/acl_rt.h"
#include "aclnn_bnll.h"

#define CHECK_ACL(expr)                                                                                 \
    do {                                                                                                \
        auto __ret = (expr);                                                                            \
        int32_t __code = static_cast<int32_t>(__ret);                                                   \
        if (__code != 0) {                                                                              \
            fprintf(stderr, "[ERROR] %s failed at %s:%d, ret=%d\n", #expr, __FILE__, __LINE__, __code); \
            exit(1);                                                                                    \
        }                                                                                               \
    } while (0)

static float bnllReference(float x)
{
    if (x > 0.0f) {
        return x + std::log(1.0f + std::exp(-x));
    } else {
        return std::log(1.0f + std::exp(x));
    }
}

int32_t main(int32_t argc, char** argv)
{
    printf("=== BNLL aclnn invocation example ===\n");

    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    CHECK_ACL(aclnnInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));

    const std::vector<int64_t> shape = {4, 256};
    const int64_t elementCount = shape[0] * shape[1];
    const size_t bufferSize = elementCount * sizeof(float);

    void* xDeviceMem = nullptr;
    CHECK_ACL(aclrtMalloc(&xDeviceMem, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor* xTensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0,
                                         ACL_FORMAT_ND, shape.data(), shape.size(), xDeviceMem);

    void* yDeviceMem = nullptr;
    CHECK_ACL(aclrtMalloc(&yDeviceMem, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor* yTensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0,
                                         ACL_FORMAT_ND, shape.data(), shape.size(), yDeviceMem);

    std::vector<float> xHostData(elementCount);
    for (int64_t i = 0; i < elementCount; i++) {
        xHostData[i] = -5.0f + 10.0f * static_cast<float>(i) / static_cast<float>(elementCount - 1);
    }

    std::vector<float> goldenData(elementCount);
    for (int64_t i = 0; i < elementCount; i++) {
        goldenData[i] = bnllReference(xHostData[i]);
    }

    CHECK_ACL(aclrtMemcpy(xDeviceMem, bufferSize, xHostData.data(), bufferSize,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACL(aclnnBnllGetWorkspaceSize(xTensor, yTensor, &workspaceSize, &executor));

    void* workspaceDeviceMem = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspaceDeviceMem, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    CHECK_ACL(aclnnBnll(workspaceDeviceMem, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<float> yHostData(elementCount);
    CHECK_ACL(aclrtMemcpy(yHostData.data(), bufferSize, yDeviceMem, bufferSize,
                          ACL_MEMCPY_DEVICE_TO_HOST));

    printf("Input/Output preview (first 10 elements):\n");
    const int64_t previewCount = std::min<int64_t>(elementCount, 10);
    for (int64_t i = 0; i < previewCount; i++) {
        printf("  x[%ld]=%.4f  y[%ld]=%.4f  golden=%.4f\n",
               (long)i, xHostData[i], (long)i, yHostData[i], goldenData[i]);
    }

    const float rtol = 1e-3f;
    const float atol = 1e-3f;
    bool pass = true;
    for (int64_t i = 0; i < elementCount; i++) {
        float diff = std::fabs(yHostData[i] - goldenData[i]);
        float threshold = atol + rtol * std::fabs(goldenData[i]);
        if (diff > threshold) {
            printf("[MISMATCH] index=%ld, y=%.6f, golden=%.6f, diff=%.6e\n",
                   (long)i, yHostData[i], goldenData[i], diff);
            pass = false;
        }
    }
    printf("\ntest %s\n", pass ? "pass" : "failed");

    aclDestroyTensor(xTensor);
    aclDestroyTensor(yTensor);
    CHECK_ACL(aclrtFree(xDeviceMem));
    CHECK_ACL(aclrtFree(yDeviceMem));
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtFree(workspaceDeviceMem));
    }
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclnnFinalize());

    return pass ? 0 : 1;
}
