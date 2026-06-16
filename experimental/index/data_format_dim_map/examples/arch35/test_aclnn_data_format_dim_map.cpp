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
 * @file test_aclnn_data_format_dim_map.cpp
 * @brief DataFormatDimMap ACLNN two-stage invocation example.
 *
 * Creates an input tensor [0, 1, 2, 3] (INT32) representing dimension indices
 * in NHWC format, maps them to NCHW format, and prints the result.
 * Expected output: [0, 2, 3, 1]
 *
 * Build prerequisites:
 *   - CANN Toolkit installed and environment variables sourced (set_env.sh)
 *   - Custom operator package deployed via `bash build.sh --soc=ascend950`
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnn_data_format_dim_map.h"

#define ACL_CHECK(expr)                                                    \
    do {                                                                   \
        auto _ret = (expr);                                                \
        if (_ret != 0) {                                                   \
            printf("[ERROR] %s:%d  %s  failed, ret = %d\n",               \
                   __FILE__, __LINE__, #expr, static_cast<int>(_ret));    \
            goto cleanup;                                                  \
        }                                                                  \
    } while (0)

static aclTensor* CreateAclTensor(const std::vector<int64_t>& shape,
                                   aclDataType dtype,
                                   void* devAddr)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return aclCreateTensor(shape.data(), static_cast<uint64_t>(shape.size()),
                           dtype, strides.data(), 0,
                           aclFormat::ACL_FORMAT_ND,
                           shape.data(), static_cast<uint64_t>(shape.size()),
                           devAddr);
}

int main()
{
    // ---- 0. Resource handles (for goto cleanup) ----
    aclrtStream stream = nullptr;
    void* devX = nullptr;
    void* devY = nullptr;
    void* workspace = nullptr;
    aclTensor* tensorX = nullptr;
    aclTensor* tensorY = nullptr;
    aclOpExecutor* executor = nullptr;
    int ret = 1;

    // ---- 1. ACL Init ----
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    ACL_CHECK(aclrtCreateStream(&stream));

    {
        // ---- 2. Prepare input data ----
        const int32_t hostX[] = {0, 1, 2, 3};
        const int64_t numElements = 4;
        const int64_t byteSize = numElements * sizeof(int32_t);
        std::vector<int64_t> shape = {numElements};

        ACL_CHECK(aclrtMalloc(&devX, byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMalloc(&devY, byteSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMemcpy(devX, byteSize, hostX, byteSize,
                               ACL_MEMCPY_HOST_TO_DEVICE));

        // ---- 3. Create aclTensors ----
        tensorX = CreateAclTensor(shape, ACL_INT32, devX);
        tensorY = CreateAclTensor(shape, ACL_INT32, devY);
        if (tensorX == nullptr || tensorY == nullptr) {
            printf("[ERROR] Failed to create aclTensor.\n");
            goto cleanup;
        }

        // ---- 4. Stage 1: GetWorkspaceSize ----
        char srcFormat[] = "NHWC";
        char dstFormat[] = "NCHW";
        uint64_t workspaceSize = 0;

        ACL_CHECK(aclnnDataFormatDimMapGetWorkspaceSize(
            tensorX, srcFormat, dstFormat, tensorY,
            &workspaceSize, &executor));

        // ---- 5. Allocate workspace (if needed) ----
        if (workspaceSize > 0) {
            ACL_CHECK(aclrtMalloc(&workspace, workspaceSize,
                                   ACL_MEM_MALLOC_HUGE_FIRST));
        }

        // ---- 6. Stage 2: Execute ----
        ACL_CHECK(aclnnDataFormatDimMap(workspace, workspaceSize,
                                         executor, stream));
        ACL_CHECK(aclrtSynchronizeStream(stream));

        // ---- 7. Copy result back and print ----
        int32_t hostY[4] = {0};
        ACL_CHECK(aclrtMemcpy(hostY, byteSize, devY, byteSize,
                               ACL_MEMCPY_DEVICE_TO_HOST));

        printf("Input  (NHWC dim indices): [%d, %d, %d, %d]\n",
               hostX[0], hostX[1], hostX[2], hostX[3]);
        printf("Output (NCHW dim indices): [%d, %d, %d, %d]\n",
               hostY[0], hostY[1], hostY[2], hostY[3]);

        // Verify result
        const int32_t expected[] = {0, 2, 3, 1};
        bool pass = true;
        for (int i = 0; i < 4; ++i) {
            if (hostY[i] != expected[i]) {
                pass = false;
                break;
            }
        }
        printf("Result: %s\n", pass ? "PASS" : "FAIL");
        ret = pass ? 0 : 1;
    }

cleanup:
    // ---- 8. Release resources ----
    if (tensorX != nullptr) { aclDestroyTensor(tensorX); }
    if (tensorY != nullptr) { aclDestroyTensor(tensorY); }
    if (workspace != nullptr) { aclrtFree(workspace); }
    if (devX != nullptr) { aclrtFree(devX); }
    if (devY != nullptr) { aclrtFree(devY); }
    if (stream != nullptr) { aclrtDestroyStream(stream); }
    aclrtResetDevice(0);
    aclFinalize();
    return ret;
}
