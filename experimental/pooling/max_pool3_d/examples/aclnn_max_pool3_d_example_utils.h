/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAX_POOL3_D_EXAMPLES_ACLNN_MAX_POOL3_D_EXAMPLE_UTILS_H_
#define MAX_POOL3_D_EXAMPLES_ACLNN_MAX_POOL3_D_EXAMPLE_UTILS_H_

#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

namespace max_pool3_d_example {
inline int64_t ElementCount(const std::vector<int64_t>& shape)
{
    int64_t count = 1;
    for (const int64_t dim : shape) {
        count *= dim;
    }
    return count;
}

struct AclResources {
    ~AclResources()
    {
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (x != nullptr) {
            aclDestroyTensor(x);
        }
        if (y != nullptr) {
            aclDestroyTensor(y);
        }
        if (ksize != nullptr) {
            aclDestroyIntArray(ksize);
        }
        if (strides != nullptr) {
            aclDestroyIntArray(strides);
        }
        if (pads != nullptr) {
            aclDestroyIntArray(pads);
        }
        if (dilation != nullptr) {
            aclDestroyIntArray(dilation);
        }
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        if (xDevice != nullptr) {
            aclrtFree(xDevice);
        }
        if (yDevice != nullptr) {
            aclrtFree(yDevice);
        }
        if (deviceSet) {
            aclrtResetDevice(deviceId);
        }
        if (initialized) {
            aclFinalize();
        }
    }

    aclError Initialize(int32_t id)
    {
        deviceId = id;
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
        initialized = true;
        ret = aclrtSetDevice(deviceId);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
        deviceSet = true;
        return aclrtCreateStream(&stream);
    }

    aclError CreateFloatTensor(const std::vector<float>& hostData, const std::vector<int64_t>& shape,
                               void** deviceAddress, aclTensor** tensor)
    {
        const size_t bytes = hostData.size() * sizeof(float);
        aclError ret = aclrtMalloc(deviceAddress, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
        ret = aclrtMemcpy(*deviceAddress, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
        std::vector<int64_t> tensorStrides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            tensorStrides[i] = shape[i + 1] * tensorStrides[i + 1];
        }
        *tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, tensorStrides.data(), 0, ACL_FORMAT_ND,
                                  shape.data(), shape.size(), *deviceAddress);
        return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
    }

    int32_t deviceId = 0;
    bool initialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    void* xDevice = nullptr;
    void* yDevice = nullptr;
    void* workspace = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclIntArray* ksize = nullptr;
    aclIntArray* strides = nullptr;
    aclIntArray* pads = nullptr;
    aclIntArray* dilation = nullptr;
};
} // namespace max_pool3_d_example

#endif // MAX_POOL3_D_EXAMPLES_ACLNN_MAX_POOL3_D_EXAMPLE_UTILS_H_
