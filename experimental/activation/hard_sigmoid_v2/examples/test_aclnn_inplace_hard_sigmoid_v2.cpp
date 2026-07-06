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
#include <cstdlib>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_hardsigmoid_v2.h"

namespace {

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (int64_t dim : shape) {
        shape_size *= dim;
    }
    return shape_size;
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
    const std::vector<int64_t> shape = {4, 2};
    std::vector<float> self_host = {-4.0f, -3.0f, -2.0f, 0.0f, 1.0f, 2.0f, 4.0f, 5.0f};
    const size_t bytes = static_cast<size_t>(GetShapeSize(shape)) * sizeof(float);

    int final_ret = ACL_SUCCESS;
    bool acl_initialized = false;
    bool device_set = false;
    int32_t device_id = 0;
    if (const char* device_id_env = std::getenv("ASCEND_DEVICE_ID")) {
        device_id = std::atoi(device_id_env);
    }

    aclrtStream stream = nullptr;
    void* self_device = nullptr;
    void* workspace = nullptr;
    aclTensor* self_tensor = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspace_size = 0;

    auto cleanup = [&]() -> int {
        if (self_tensor != nullptr) {
            aclDestroyTensor(self_tensor);
        }
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        if (self_device != nullptr) {
            aclrtFree(self_device);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (device_set) {
            aclrtResetDevice(device_id);
        }
        if (acl_initialized) {
            aclFinalize();
        }
        return final_ret;
    };

    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        final_ret = ret;
        return cleanup();
    }
    acl_initialized = true;

    ret = aclrtSetDevice(device_id);
    if (ret != ACL_SUCCESS) {
        final_ret = ret;
        return cleanup();
    }
    device_set = true;

    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        final_ret = ret;
        return cleanup();
    }

    ret = aclrtMalloc(&self_device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        final_ret = ret;
        return cleanup();
    }

    ret = aclrtMemcpy(self_device, bytes, self_host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        final_ret = ret;
        return cleanup();
    }

    ret = CreateAclTensor(shape, ACL_FLOAT, self_device, &self_tensor);
    if (ret != ACL_SUCCESS) {
        final_ret = ret;
        return cleanup();
    }

    ret = aclnnInplaceHardsigmoidV2GetWorkspaceSize(self_tensor, &workspace_size, &executor);
    if (ret != ACL_SUCCESS) {
        final_ret = ret;
        return cleanup();
    }

    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            final_ret = ret;
            return cleanup();
        }
    }

    ret = aclnnInplaceHardsigmoidV2(workspace, workspace_size, executor, stream);
    if (ret != ACL_SUCCESS) {
        final_ret = ret;
        return cleanup();
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        final_ret = ret;
        return cleanup();
    }

    ret = aclrtMemcpy(self_host.data(), bytes, self_device, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        final_ret = ret;
        return cleanup();
    }

    const std::vector<float> expected = {0.0f, 0.0f, 1.0f / 6.0f, 0.5f, 2.0f / 3.0f, 5.0f / 6.0f, 1.0f, 1.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(self_host[i] - expected[i]) > 1e-5f) {
            std::fprintf(stderr, "inplace check failed at %zu: got %.8f expected %.8f\n", i, self_host[i], expected[i]);
            final_ret = 1;
            return cleanup();
        }
    }

    std::printf("inplace example passed\n");
    return cleanup();
}
