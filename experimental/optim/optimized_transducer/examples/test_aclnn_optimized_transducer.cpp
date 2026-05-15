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
#include <random>
#include <cstring>

#include "acl/acl.h"
#include "aclnn_optimized_transducer.h"

#define CHECK_ACL(expr)                                                    \
    do {                                                                   \
        aclError ret = (expr);                                             \
        if (ret != ACL_SUCCESS) {                                          \
            std::cerr << "ACL Error " << ret << " at line " << __LINE__    \
                      << ": " << #expr << std::endl;                       \
            return -1;                                                     \
        }                                                                  \
    } while (0)

#define CHECK_PTR(ptr, msg)                                                \
    do {                                                                   \
        if ((ptr) == nullptr) {                                            \
            std::cerr << "Error: " << msg << " at line " << __LINE__ << std::endl; \
            return -1;                                                     \
        }                                                                  \
    } while (0)

int main(int argc, char* argv[]) {
    int32_t deviceId = 0;
    if (argc > 1) {
        deviceId = std::atoi(argv[1]);
    }

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));

    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));

    const int32_t batchSize = 2;
    const int32_t maxT = 1; 
    const int32_t maxU = 2;
    const int32_t V = 2;

    int64_t logitsElements = batchSize * maxT * maxU * V;
    int64_t logitsSize = logitsElements * sizeof(float);
    int64_t targetsElements = batchSize * (maxU - 1);
    int64_t targetsSize = targetsElements * sizeof(int32_t);
    int64_t lengthsSize = batchSize * sizeof(int32_t);
    int64_t lossSize = batchSize * sizeof(float);

    void *logitsHost = nullptr, *targetsHost = nullptr;
    void *logitLensHost = nullptr, *targetLensHost = nullptr;
    void *lossHost = nullptr;
    void *gradHost = nullptr;

    CHECK_ACL(aclrtMallocHost(&logitsHost, logitsSize));
    CHECK_ACL(aclrtMallocHost(&targetsHost, targetsSize));
    CHECK_ACL(aclrtMallocHost(&logitLensHost, lengthsSize));
    CHECK_ACL(aclrtMallocHost(&targetLensHost, lengthsSize));
    CHECK_ACL(aclrtMallocHost(&lossHost, lossSize));
    CHECK_ACL(aclrtMallocHost(&gradHost, logitsSize));

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> floatDist(-1.0f, 1.0f);
    std::uniform_int_distribution<int32_t> intDist(1, V - 1);

    float* logitsPtr = static_cast<float*>(logitsHost);
    for (int64_t i = 0; i < logitsElements; ++i) {
        logitsPtr[i] = floatDist(gen);
    }

    int32_t* targetsPtr = static_cast<int32_t*>(targetsHost);
    for (int64_t i = 0; i < targetsElements; ++i) {
        targetsPtr[i] = intDist(gen);
    }

    int32_t* logitLensPtr = static_cast<int32_t*>(logitLensHost);
    int32_t* targetLensPtr = static_cast<int32_t*>(targetLensHost);
    for (int32_t b = 0; b < batchSize; ++b) {
        logitLensPtr[b] = maxT;      
        targetLensPtr[b] = maxU - 1;
    }

    void *logitsDev = nullptr, *targetsDev = nullptr;
    void *logitLensDev = nullptr, *targetLensDev = nullptr;
    void *lossDev = nullptr, *gradsDev = nullptr;

    CHECK_ACL(aclrtMalloc(&logitsDev, logitsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&targetsDev, targetsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&logitLensDev, lengthsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&targetLensDev, lengthsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&lossDev, lossSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&gradsDev, logitsSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(logitsDev, logitsSize, logitsHost, logitsSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(targetsDev, targetsSize, targetsHost, targetsSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(logitLensDev, lengthsSize, logitLensHost, lengthsSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(targetLensDev, lengthsSize, targetLensHost, lengthsSize, ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<int64_t> logitsShape = {batchSize, maxT, maxU, V};
    std::vector<int64_t> logitsStrides = {
        (int64_t)maxT * maxU * V,
        (int64_t)maxU * V,
        (int64_t)V,
        1
    };

    aclTensor* logitsTensor = aclCreateTensor(
        logitsShape.data(), logitsShape.size(),
        ACL_FLOAT, logitsStrides.data(), 0,
        ACL_FORMAT_ND, logitsShape.data(), logitsShape.size(),
        logitsDev);
    CHECK_PTR(logitsTensor, "Failed to create logits tensor");

    std::vector<int64_t> targetsShape = {batchSize, maxU - 1};
    std::vector<int64_t> targetsStrides = {maxU - 1, 1};

    aclTensor* targetsTensor = aclCreateTensor(
        targetsShape.data(), targetsShape.size(),
        ACL_INT32, targetsStrides.data(), 0,
        ACL_FORMAT_ND, targetsShape.data(), targetsShape.size(),
        targetsDev);
    CHECK_PTR(targetsTensor, "Failed to create targets tensor");

    std::vector<int64_t> lengthsShape = {batchSize};
    std::vector<int64_t> lengthsStrides = {1};

    aclTensor* logitLensTensor = aclCreateTensor(
        lengthsShape.data(), lengthsShape.size(),
        ACL_INT32, lengthsStrides.data(), 0,
        ACL_FORMAT_ND, lengthsShape.data(), lengthsShape.size(),
        logitLensDev);
    CHECK_PTR(logitLensTensor, "Failed to create logitLengths tensor");

    aclTensor* targetLensTensor = aclCreateTensor(
        lengthsShape.data(), lengthsShape.size(),
        ACL_INT32, lengthsStrides.data(), 0,
        ACL_FORMAT_ND, lengthsShape.data(), lengthsShape.size(),
        targetLensDev);
    CHECK_PTR(targetLensTensor, "Failed to create targetLengths tensor");

    std::vector<int64_t> lossShape = {batchSize};
    std::vector<int64_t> lossStrides = {1};

    aclTensor* lossTensor = aclCreateTensor(
        lossShape.data(), lossShape.size(),
        ACL_FLOAT, lossStrides.data(), 0,
        ACL_FORMAT_ND, lossShape.data(), lossShape.size(),
        lossDev);
    CHECK_PTR(lossTensor, "Failed to create loss tensor");

    aclTensor* gradsTensor = aclCreateTensor(
        logitsShape.data(), logitsShape.size(),
        ACL_FLOAT, logitsStrides.data(), 0,
        ACL_FORMAT_ND, logitsShape.data(), logitsShape.size(),
        gradsDev);
    CHECK_PTR(gradsTensor, "Failed to create grads tensor");

    std::cout << "[INFO] Tensors created" << std::endl;

    int64_t blank = 0;
    double clamp = -1;
    bool fused_log_softmax = true;
    bool requires_grad = true;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 第一步: 获取workspace大小
    aclError ret = aclnnOptimizedTransducerGetWorkspaceSize(
        logitsTensor,       // logits
        targetsTensor,      // targets
        logitLensTensor,    // logitLengths
        targetLensTensor,   // targetLengths
        blank,              // blank id
        clamp,              // clamp值
        fused_log_softmax,  // fused_log_softmax标志
        requires_grad,      // requires_grad
        lossTensor,         // loss输出
        gradsTensor,        // grads输出
        &workspaceSize,
        &executor);

    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclnnOptimizedTransducerGetWorkspaceSize failed: " << ret << std::endl;
        return -1;
    }

    std::cout << "[INFO] Workspace size: " << workspaceSize << " bytes" << std::endl;

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    ret = aclnnOptimizedTransducer(workspace, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[ERROR] aclnnOptimizedTransducer execution failed: " << ret << std::endl;
        return -1;
    }

    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::cout << "[INFO] Operator executed successfully" << std::endl;

    CHECK_ACL(aclrtMemcpy(lossHost, lossSize, lossDev, lossSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(gradHost, logitsSize, gradsDev, logitsSize, ACL_MEMCPY_DEVICE_TO_HOST));

    float* lossResult = static_cast<float*>(lossHost);
    float *gradResult = static_cast<float*>(gradHost);
    std::cout << "\n========== Results ==========" << std::endl;
    for (int32_t b = 0; b < batchSize; ++b) {
        std::cout << " batch " << b << " loss: " << lossResult[b] << std::endl;
    }

    aclDestroyTensor(logitsTensor);
    aclDestroyTensor(targetsTensor);
    aclDestroyTensor(logitLensTensor);
    aclDestroyTensor(targetLensTensor);
    aclDestroyTensor(lossTensor);
    aclDestroyTensor(gradsTensor);

    if (workspace) {
        aclrtFree(workspace);
    }
    aclrtFree(logitsDev);
    aclrtFree(targetsDev);
    aclrtFree(logitLensDev);
    aclrtFree(targetLensDev);
    aclrtFree(lossDev);
    aclrtFree(gradsDev);

    aclrtFreeHost(logitsHost);
    aclrtFreeHost(targetsHost);
    aclrtFreeHost(logitLensHost);
    aclrtFreeHost(targetLensHost);
    aclrtFreeHost(lossHost);
    aclrtFreeHost(gradHost);

    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();

    std::cout << "\n[INFO] Test completed successfully!" << std::endl;
    return 0;
}