/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_hardsigmoid_backward.h"

namespace {

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape)
{
    if (shape.empty()) {
        return {};
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
}

aclError CreateAclTensor(const std::vector<int64_t>& shape, aclDataType dtype, void* device_addr, aclTensor** tensor)
{
    std::vector<int64_t> strides = MakeStrides(shape);
    const int64_t* shape_ptr = shape.empty() ? nullptr : shape.data();
    const int64_t* strides_ptr = strides.empty() ? nullptr : strides.data();
    *tensor = aclCreateTensor(shape_ptr, shape.size(), dtype, strides_ptr, 0, ACL_FORMAT_ND, shape_ptr, shape.size(),
                              device_addr);
    return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

} // namespace

int main()
{
    aclError ret = ACL_SUCCESS;
    constexpr int32_t kDeviceId = 0;
    int exitCode = 0;
    bool aclInitialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    void* grad_device = nullptr;
    void* self_device = nullptr;
    void* out_device = nullptr;
    void* workspace = nullptr;
    aclTensor* grad_tensor = nullptr;
    aclTensor* self_tensor = nullptr;
    aclTensor* out_tensor = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspace_size = 0;

    const std::vector<int64_t> shape = {4, 2};
    const std::vector<float> grad_host = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
    const std::vector<float> self_host = {-4.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f};
    std::vector<float> out_host(static_cast<size_t>(GetShapeSize(shape)), 0.f);
    const size_t bytes = out_host.size() * sizeof(float);

    ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    aclInitialized = true;

    ret = aclrtSetDevice(kDeviceId);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }
    deviceSet = true;

    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }

    ret = aclrtMalloc(&grad_device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }
    ret = aclrtMalloc(&self_device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }
    ret = aclrtMalloc(&out_device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }

    ret = aclrtMemcpy(grad_device, bytes, grad_host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }
    ret = aclrtMemcpy(self_device, bytes, self_host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }
    ret = aclrtMemset(out_device, bytes, 0, bytes);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }

    ret = CreateAclTensor(shape, ACL_FLOAT, grad_device, &grad_tensor);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }
    ret = CreateAclTensor(shape, ACL_FLOAT, self_device, &self_tensor);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }
    ret = CreateAclTensor(shape, ACL_FLOAT, out_device, &out_tensor);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }

    ret = aclnnHardsigmoidBackwardGetWorkspaceSize(grad_tensor, self_tensor, out_tensor, &workspace_size, &executor);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            exitCode = ret;
            goto CLEANUP;
        }
    }

    ret = aclnnHardsigmoidBackward(workspace, workspace_size, executor, stream);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }
    ret = aclrtMemcpy(out_host.data(), bytes, out_device, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        exitCode = ret;
        goto CLEANUP;
    }

    const std::vector<float> expected = {0.f, 2.f / 6.f, 3.f / 6.f, 4.f / 6.f, 5.f / 6.f, 1.f, 0.f, 0.f};
    for (size_t i = 0; i < expected.size(); ++i) {
        std::printf("out[%zu] = %.6f\n", i, out_host[i]);
        if (std::abs(out_host[i] - expected[i]) > 1e-5f) {
            std::fprintf(stderr, "mismatch at %zu: got %.6f expected %.6f\n", i, out_host[i], expected[i]);
            exitCode = 1;
            goto CLEANUP;
        }
    }

CLEANUP:
    if (grad_tensor != nullptr) {
        aclDestroyTensor(grad_tensor);
    }
    if (self_tensor != nullptr) {
        aclDestroyTensor(self_tensor);
    }
    if (out_tensor != nullptr) {
        aclDestroyTensor(out_tensor);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    if (grad_device != nullptr) {
        aclrtFree(grad_device);
    }
    if (self_device != nullptr) {
        aclrtFree(self_device);
    }
    if (out_device != nullptr) {
        aclrtFree(out_device);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    if (deviceSet) {
        aclrtResetDevice(kDeviceId);
    }
    if (aclInitialized) {
        aclFinalize();
    }
    if (exitCode == 0) {
        std::printf("hard_sigmoid_grad_v3 example passed\n");
    }
    return exitCode;
}
