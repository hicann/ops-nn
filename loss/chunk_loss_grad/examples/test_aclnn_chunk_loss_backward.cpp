/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_chunk_loss_backward.h"

#define LOG_PRINT(...) std::printf(__VA_ARGS__)
#define CHECK_RET(cond, action) \
    do {                        \
        if (!(cond)) {          \
            action;             \
        }                       \
    } while (0)

struct Resources {
    bool initialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    std::vector<aclTensor*> tensors;
    std::vector<void*> allocations;
    ~Resources()
    {
        if (stream != nullptr) {
            aclrtSynchronizeStream(stream);
        }
        for (auto tensor : tensors) {
            aclDestroyTensor(tensor);
        }
        for (auto address : allocations) {
            aclrtFree(address);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(0);
        }
        if (initialized) {
            aclFinalize();
        }
    }
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

bool Init(Resources& resources)
{
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return false);
    resources.initialized = true;
    CHECK_RET(aclrtSetDevice(0) == ACL_SUCCESS, return false);
    resources.deviceSet = true;
    return aclrtCreateStream(&resources.stream) == ACL_SUCCESS;
}

template <typename T>
aclTensor* CreateAclTensor(const std::vector<T>& data, const std::vector<int64_t>& shape, aclDataType dtype,
                           Resources& resources, void** deviceAddress = nullptr)
{
    auto bytes = GetShapeSize(shape) * sizeof(T);
    void* address = nullptr;
    CHECK_RET(aclrtMalloc(&address, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return nullptr);
    resources.allocations.push_back(address);
    CHECK_RET(aclrtMemcpy(address, bytes, data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
              return nullptr);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    auto tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                  shape.size(), address);
    if (tensor != nullptr) {
        resources.tensors.push_back(tensor);
    }
    if (deviceAddress != nullptr) {
        *deviceAddress = address;
    }
    return tensor;
}

int main()
{
    // 1. 初始化Device和Stream，资源在退出时统一释放。
    Resources resources;
    CHECK_RET(Init(resources), LOG_PRINT("ACL initialization failed\n"); return 1);
    // 2. 构造N=1、H=1、V=2的例子。uint16_t保存FLOAT16的原始位表示。
    auto hidden = CreateAclTensor<uint16_t>({0x3c00}, {1, 1}, ACL_FLOAT16, resources); // 1.0
    auto weight = CreateAclTensor<uint16_t>({0, 0}, {2, 1}, ACL_FLOAT16, resources);
    auto labels = CreateAclTensor<int64_t>({0}, {1}, ACL_INT64, resources);
    const double alpha = 1.0;
    // logits全为0，因此softmaxMax=0、softmaxSum=exp(0)+exp(0)=2。
    auto softmaxMax = CreateAclTensor<float>({0.0f}, {1}, ACL_FLOAT, resources);
    auto softmaxSum = CreateAclTensor<float>({2.0f}, {1}, ACL_FLOAT, resources);
    auto gradOutput = CreateAclTensor<float>({1.0f}, {1}, ACL_FLOAT, resources);
    void* outputAddress = nullptr;
    auto output = CreateAclTensor<uint16_t>({0, 0}, {2, 1}, ACL_FLOAT16, resources, &outputAddress);
    CHECK_RET(hidden && weight && labels && softmaxMax && softmaxSum && gradOutput && output,
              LOG_PRINT("Tensor construction failed\n");
              return 1);
    // 3. 查询workspace并创建执行器。
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnChunkLossBackwardGetWorkspaceSize(hidden, weight, labels, alpha, softmaxMax, softmaxSum,
                                                         gradOutput, -100, output, &workspaceSize, &executor);
    CHECK_RET(status == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed: %d\n", status); return 1);
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_RET(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return 1);
        resources.allocations.push_back(workspace);
    }
    // 4. 异步执行并等待结果。
    status = aclnnChunkLossBackward(workspace, workspaceSize, executor, resources.stream);
    CHECK_RET(status == ACL_SUCCESS, LOG_PRINT("Backward failed: %d\n", status); return 1);
    CHECK_RET(aclrtSynchronizeStream(resources.stream) == ACL_SUCCESS, return 1);
    // 5. 拷贝输出，预期FLOAT16位表示为[-0.5, 0.5]，即[0xb800, 0x3800]。
    uint16_t result[2] = {};
    CHECK_RET(
        aclrtMemcpy(result, sizeof(result), outputAddress, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS,
        return 1);
    LOG_PRINT("dW bits: [0x%04x, 0x%04x], expected [0xb800, 0x3800]\n", result[0], result[1]);
    return result[0] == 0xb800 && result[1] == 0x3800 ? 0 : 1;
}
