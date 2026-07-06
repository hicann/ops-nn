/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_relu.h"

#define CHECK_RET(cond, expr) \
    do {                      \
        if (!(cond)) {        \
            expr;             \
        }                     \
    } while (0)

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
    const int32_t device_id = 0;
    std::vector<int64_t> shape = {2, 4};
    std::vector<float> host_data = {-4.0f, -1.0f, 0.0f, 2.0f, 3.0f, -5.0f, 6.0f, -7.0f};
    const size_t bytes = host_data.size() * sizeof(float);
    std::vector<float> result(host_data.size(), 0.0f);
    const std::vector<float> expected = {0.0f, 0.0f, 0.0f, 2.0f, 3.0f, 0.0f, 6.0f, 0.0f};
    uint64_t workspace_size = 0;
    aclrtStream stream = nullptr;
    void* device_addr = nullptr;
    void* workspace = nullptr;
    aclTensor* self = nullptr;
    aclOpExecutor* executor = nullptr;
    bool acl_initialized = false;
    bool device_set = false;

    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    acl_initialized = true;
    ret = aclrtSetDevice(device_id);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    device_set = true;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    ret = aclrtMalloc(&device_addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    ret = aclrtMemcpy(device_addr, bytes, host_data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    ret = CreateAclTensor(shape, ACL_FLOAT, device_addr, &self);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    ret = aclnnInplaceReluGetWorkspaceSize(self, &workspace_size, &executor);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    }

    ret = aclnnInplaceRelu(workspace, workspace_size, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    ret = aclrtMemcpy(result.data(), bytes, device_addr, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(result[i] - expected[i]) > 1e-6f) {
            std::fprintf(stderr, "check failed at %zu: got %.6f expected %.6f\n", i, result[i], expected[i]);
            ret = ACL_ERROR_FAILURE;
            goto cleanup;
        }
    }
    std::printf("inplace relu example passed\n");

cleanup:
    if (self != nullptr) {
        aclDestroyTensor(self);
    }
    if (device_addr != nullptr) {
        aclrtFree(device_addr);
    }
    if (workspace != nullptr) {
        aclrtFree(workspace);
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
    return ret;
}
