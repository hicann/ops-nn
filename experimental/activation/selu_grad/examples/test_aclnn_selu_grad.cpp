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

#include "acl/acl.h"
#include "aclnnop/aclnn_selu_backward.h"

#define CHECK_RET(condition, action)                                                                                  \
    do {                                                                                                              \
        if (!(condition)) {                                                                                           \
            std::cerr << "Check failed: " << #condition << " at line " << __LINE__ << ", error=" << ret << std::endl; \
            action;                                                                                                   \
        }                                                                                                             \
    } while (0)

namespace {
int64_t GetElementNum(const std::vector<int64_t>& shape)
{
    int64_t elementNum = 1;
    for (const int64_t dim : shape) {
        elementNum *= dim;
    }
    return elementNum;
}

int InitResource(int32_t deviceId, aclrtStream& stream)
{
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return ACL_SUCCESS;
}

int CreateAclTensor(const std::vector<float>& hostData, const std::vector<int64_t>& shape, void*& deviceAddress,
                    aclTensor*& tensor)
{
    const size_t dataSize = static_cast<size_t>(GetElementNum(shape)) * sizeof(float);
    aclError ret = aclrtMalloc(&deviceAddress, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclrtMemcpy(deviceAddress, dataSize, hostData.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(deviceAddress); deviceAddress = nullptr; return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), deviceAddress);
    CHECK_RET(tensor != nullptr, aclrtFree(deviceAddress); deviceAddress = nullptr; return ACL_ERROR_INTERNAL_ERROR);
    return ACL_SUCCESS;
}

void DestroyTensor(aclTensor*& tensor, void*& deviceAddress)
{
    if (tensor != nullptr) {
        aclDestroyTensor(tensor);
        tensor = nullptr;
    }
    if (deviceAddress != nullptr) {
        aclrtFree(deviceAddress);
        deviceAddress = nullptr;
    }
}
} // namespace

int main()
{
    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    int ret = InitResource(deviceId, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const std::vector<int64_t> shape = {2, 4};
    const std::vector<float> gradientsData = {1.0F, 1.0F, 2.0F, 2.0F, -1.0F, -1.0F, 0.5F, 0.5F};
    const std::vector<float> outputsData = {-1.5F, -0.5F, -0.0F, 0.0F, 0.5F, 1.0F, -1.0F, 2.0F};
    std::vector<float> resultData(static_cast<size_t>(GetElementNum(shape)), 0.0F);

    void* gradientsAddress = nullptr;
    void* outputsAddress = nullptr;
    void* yAddress = nullptr;
    aclTensor* gradients = nullptr;
    aclTensor* outputs = nullptr;
    aclTensor* y = nullptr;

    ret = CreateAclTensor(gradientsData, shape, gradientsAddress, gradients);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    ret = CreateAclTensor(outputsData, shape, outputsAddress, outputs);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    ret = CreateAclTensor(resultData, shape, yAddress, y);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        ret = aclnnSeluBackwardGetWorkspaceSize(gradients, outputs, y, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

        void* workspace = nullptr;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
        }

        ret = aclnnSeluBackward(workspace, workspaceSize, executor, stream);
        if (ret == ACL_SUCCESS) {
            ret = aclrtSynchronizeStream(stream);
        }
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    }

    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), yAddress, resultData.size() * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    for (size_t i = 0; i < resultData.size(); ++i) {
        std::cout << "y[" << i << "] = " << resultData[i] << std::endl;
    }

cleanup:
    DestroyTensor(gradients, gradientsAddress);
    DestroyTensor(outputs, outputsAddress);
    DestroyTensor(y, yAddress);
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ret;
}
