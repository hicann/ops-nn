/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <vector>

#include "aclnn_max_pool3_d_example_utils.h"
#include "aclnnop/aclnn_max_pool3_d.h"

int main()
{
    using max_pool3_d_example::AclResources;
    using max_pool3_d_example::ElementCount;

    AclResources resources;
    aclError ret = resources.Initialize(0);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    const std::vector<int64_t> xShape = {1, 2, 4, 4, 4};
    const std::vector<int64_t> yShape = {1, 2, 2, 2, 2};
    std::vector<float> xHost(ElementCount(xShape), 2.0F);
    std::vector<float> yHost(ElementCount(yShape), 0.0F);
    ret = resources.CreateFloatTensor(xHost, xShape, &resources.xDevice, &resources.x);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = resources.CreateFloatTensor(yHost, yShape, &resources.yDevice, &resources.y);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    const std::vector<int64_t> ksizeValue = {1, 1, 2, 2, 2};
    const std::vector<int64_t> stridesValue = {1, 1, 2, 2, 2};
    const std::vector<int64_t> padsValue = {0, 0, 0, 0, 0, 0};
    const std::vector<int64_t> dilationValue = {1, 1, 1, 1, 1};
    resources.ksize = aclCreateIntArray(ksizeValue.data(), ksizeValue.size());
    resources.strides = aclCreateIntArray(stridesValue.data(), stridesValue.size());
    resources.pads = aclCreateIntArray(padsValue.data(), padsValue.size());
    resources.dilation = aclCreateIntArray(dilationValue.data(), dilationValue.size());
    if (resources.ksize == nullptr || resources.strides == nullptr || resources.pads == nullptr ||
        resources.dilation == nullptr) {
        return ACL_ERROR_FAILURE;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnMaxPool3DGetWorkspaceSize(resources.x, resources.ksize, resources.strides, "VALID", resources.pads,
                                         resources.dilation, 0, "NCDHW", resources.y, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&resources.workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
    }
    ret = aclnnMaxPool3D(resources.workspace, workspaceSize, executor, resources.stream);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtSynchronizeStream(resources.stream);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(yHost.data(), yHost.size() * sizeof(float), resources.yDevice, yHost.size() * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    const bool passed = std::all_of(yHost.begin(), yHost.end(), [](float value) { return value == 2.0F; });
    return passed ? 0 : 1;
}
