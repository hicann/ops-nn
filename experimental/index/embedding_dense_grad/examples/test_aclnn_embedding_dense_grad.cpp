/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_embedding_dense_grad.h"

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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
              return ret);
    return 0;
}

bool CheckHardwareSupport()
{
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) {
        LOG_PRINT("Warning: Cannot get SOC name, skip test\n");
        return false;
    }

    LOG_PRINT("Current SOC: %s\n", socName);

    if (strstr(socName, "Ascend310P") != nullptr || strstr(socName, "ascend310p") != nullptr) {
        return true;
    }

    LOG_PRINT("Warning: This operator only supports Ascend310P, current SOC '%s' is not supported. Skip test.\n",
              socName);
    return false;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); aclrtFree(*deviceAddr);
              *deviceAddr = nullptr; return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = (int64_t)shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor returned nullptr\n"); aclrtFree(*deviceAddr);
              *deviceAddr = nullptr; return -1);
    return 0;
}

int main()
{
    // 1. Initialize device and stream.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    if (!CheckHardwareSupport()) {
        LOG_PRINT("\n=== Test SKIPPED (hardware not supported) ===\n");
        Finalize(deviceId, stream);
        return 0;
    }

    // 2. Create input and output tensors.
    // grad:     [4, 3] float32
    // indices:  [4]    int32   values: {1, 2, 3, 4}
    // num_weights = 10, padding_idx = -1, scale_grad_by_freq = false
    // expected y[1] = {0.1, 0.2, 0.3}
    //         y[2] = {0.4, 0.5, 0.6}
    //         y[3] = {0.7, 0.8, 0.9}
    //         y[4] = {1.0, 1.1, 1.2}
    //         other rows are 0
    int64_t numWeights = 10;
    int64_t paddingIdx = -1;
    bool scaleGradByFreq = false;

    std::vector<int64_t> gradShape = {4, 3};
    std::vector<int64_t> indicesShape = {4};
    std::vector<int64_t> outShape = {numWeights, 3};

    std::vector<float> gradHost = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
    std::vector<int32_t> indicesHost = {1, 2, 3, 4};
    std::vector<float> outHost((size_t)GetShapeSize(outShape), 0.0f);

    void* gradDev = nullptr;
    void* indicesDev = nullptr;
    void* outDev = nullptr;
    aclTensor* gradDesc = nullptr;
    aclTensor* indicesDesc = nullptr;
    aclTensor* outDesc = nullptr;

    CHECK_RET(CreateAclTensor(gradHost, gradShape, &gradDev, ACL_FLOAT, &gradDesc) == 0,
              LOG_PRINT("Create gradDesc failed\n");
              return -1);
    CHECK_RET(CreateAclTensor(indicesHost, indicesShape, &indicesDev, ACL_INT32, &indicesDesc) == 0,
              LOG_PRINT("Create indicesDesc failed\n");
              return -1);
    CHECK_RET(CreateAclTensor(outHost, outShape, &outDev, ACL_FLOAT, &outDesc) == 0,
              LOG_PRINT("Create outDesc failed\n");
              return -1);

    // 3. Call CANN operator API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnEmbeddingDenseGradGetWorkspaceSize(gradDesc, indicesDesc, numWeights, paddingIdx, scaleGradByFreq,
                                                  outDesc, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEmbeddingDenseGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    LOG_PRINT("workspaceSize = %lu\n", workspaceSize);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnEmbeddingDenseGrad(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnEmbeddingDenseGrad failed. ERROR: %d\n", ret); return ret);

    // 4. Wait for the task to finish.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Copy result from device to host and verify.
    auto outBytes = outHost.size() * sizeof(float);
    ret = aclrtMemcpy(outHost.data(), outBytes, outDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    printf("==== y[%lld][3] ====\n", (long long)numWeights);
    int bad = 0;
    for (int64_t w = 0; w < numWeights; w++) {
        for (int64_t d = 0; d < 3; d++) {
            float v = outHost[w * 3 + d];
            printf("%6.3f ", v);
            if (w >= 1 && w <= 4) {
                float expect = gradHost[(w - 1) * 3 + d];
                if (fabsf(v - expect) > 1e-5f) {
                    bad++;
                }
            } else {
                if (fabsf(v) > 1e-5f) {
                    bad++;
                }
            }
        }
        printf("\n");
    }

    if (bad == 0) {
        printf("[PASS] EmbeddingDenseGrad produced expected output.\n");
    } else {
        printf("[FAIL] %d mismatched elements.\n", bad);
    }

    // 6. Release resources.
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclrtFree(gradDev);
    aclrtFree(indicesDev);
    aclrtFree(outDev);
    aclDestroyTensor(gradDesc);
    aclDestroyTensor(indicesDesc);
    aclDestroyTensor(outDesc);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return bad == 0 ? 0 : 1;
}
