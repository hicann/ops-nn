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
 * @file test_aclnn_softsign.cpp
 * @brief ACLNN invocation example for Softsign operator
 *
 * Demonstrates the two-phase aclnn API:
 *   1. aclnnSoftsignGetWorkspaceSize - compute workspace size, create executor
 *   2. aclnnSoftsign - execute computation on NPU
 *
 * Build and run:
 *   cd examples && bash run.sh
 *
 * Expected output:
 *   Softsign([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]) =
 *           [-0.6667, -0.5000, -0.3333, 0.0000, 0.3333, 0.5000, 0.6667, 0.7500]
 *   test pass
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "acl/acl_rt.h"
#include "aclnn/aclnn_base.h"
#include "aclnn/acl_meta.h"
#include "aclnn_softsign.h"

#define CHECK_ACL(expr)                                                                                 \
    do {                                                                                                \
        auto __ret = (expr);                                                                            \
        int32_t __code = static_cast<int32_t>(__ret);                                                   \
        if (__code != 0) {                                                                              \
            fprintf(stderr, "[ERROR] %s failed at %s:%d, ret=%d\n", #expr, __FILE__, __LINE__, __code); \
            exit(1);                                                                                    \
        }                                                                                               \
    } while (0)

/**
 * @brief CPU reference: softsign(x) = x / (1 + |x|)
 */
static float SoftsignRef(float x)
{
    return x / (1.0f + std::fabs(x));
}

int32_t main(int32_t argc, char** argv)
{
    // =========================================================================
    // 1. Initialize ACL environment
    // =========================================================================
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    CHECK_ACL(aclnnInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));

    // =========================================================================
    // 2. Prepare input data
    // =========================================================================
    const std::vector<float> inputData = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};
    const std::vector<int64_t> shape = {static_cast<int64_t>(inputData.size())};
    const int64_t elementCount = static_cast<int64_t>(inputData.size());
    const size_t bufferSize = elementCount * sizeof(float);

    // =========================================================================
    // 3. Allocate device memory and create tensors
    // =========================================================================
    // Input tensor
    void* inputDeviceMem = nullptr;
    CHECK_ACL(aclrtMalloc(&inputDeviceMem, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor* inputTensor = aclCreateTensor(
        shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
        shape.data(), shape.size(), inputDeviceMem);

    // Output tensor
    void* outputDeviceMem = nullptr;
    CHECK_ACL(aclrtMalloc(&outputDeviceMem, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor* outputTensor = aclCreateTensor(
        shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
        shape.data(), shape.size(), outputDeviceMem);

    // =========================================================================
    // 4. Copy input data to device
    // =========================================================================
    CHECK_ACL(aclrtMemcpy(inputDeviceMem, bufferSize, inputData.data(),
                          bufferSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // =========================================================================
    // 5. Call two-phase aclnn API
    // =========================================================================
    // Phase 1: Get workspace size and create executor
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACL(aclnnSoftsignGetWorkspaceSize(inputTensor, outputTensor,
                                            &workspaceSize, &executor));

    // Allocate workspace on device (if needed)
    void* workspaceDeviceMem = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspaceDeviceMem, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // Phase 2: Execute computation
    CHECK_ACL(aclnnSoftsign(workspaceDeviceMem, workspaceSize, executor, stream));

    // =========================================================================
    // 6. Synchronize and copy result back to host
    // =========================================================================
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<float> outputData(elementCount, 0.0f);
    CHECK_ACL(aclrtMemcpy(outputData.data(), bufferSize, outputDeviceMem,
                          bufferSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // =========================================================================
    // 7. Verify result against CPU reference
    // =========================================================================
    printf("Softsign([");
    for (int64_t i = 0; i < elementCount; i++) {
        printf("%.1f%s", inputData[i], (i < elementCount - 1) ? ", " : "");
    }
    printf("]) =\n        [");
    for (int64_t i = 0; i < elementCount; i++) {
        printf("%.4f%s", outputData[i], (i < elementCount - 1) ? ", " : "");
    }
    printf("]\n");

    bool pass = true;
    const float tolerance = 1e-3f;
    for (int64_t i = 0; i < elementCount; i++) {
        float expected = SoftsignRef(inputData[i]);
        float diff = std::fabs(outputData[i] - expected);
        if (diff > tolerance) {
            fprintf(stderr, "[FAIL] index=%ld, output=%.6f, expected=%.6f, diff=%.6f\n",
                    i, outputData[i], expected, diff);
            pass = false;
        }
    }
    printf("test %s\n", pass ? "pass" : "failed");

    // =========================================================================
    // 8. Cleanup
    // =========================================================================
    aclDestroyTensor(inputTensor);
    aclDestroyTensor(outputTensor);
    CHECK_ACL(aclrtFree(inputDeviceMem));
    CHECK_ACL(aclrtFree(outputDeviceMem));
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtFree(workspaceDeviceMem));
    }
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclnnFinalize());

    return pass ? 0 : 1;
}
