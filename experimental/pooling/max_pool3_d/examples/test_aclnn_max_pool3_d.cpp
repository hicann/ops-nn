/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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

    const std::vector<int64_t> xShape = {1, 4, 4, 4, 2};
    const std::vector<int64_t> yShape = {1, 2, 2, 2, 2};
    std::vector<float> xHost(ElementCount(xShape));
    for (size_t i = 0; i < xHost.size(); ++i) {
        xHost[i] = static_cast<float>(i);
    }
    std::vector<float> yHost(ElementCount(yShape), 0.0F);
    ret = resources.CreateFloatTensor(xHost, xShape, &resources.xDevice, &resources.x);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = resources.CreateFloatTensor(yHost, yShape, &resources.yDevice, &resources.y);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    const std::vector<int64_t> ksizeValue = {1, 2, 2, 2, 1};
    const std::vector<int64_t> stridesValue = {1, 2, 2, 2, 1};
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
                                         resources.dilation, 0, "NDHWC", resources.y, &workspaceSize, &executor);
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

    bool passed = true;
    size_t outputIndex = 0;
    for (int64_t od = 0; od < 2; ++od) {
        for (int64_t oh = 0; oh < 2; ++oh) {
            for (int64_t ow = 0; ow < 2; ++ow) {
                for (int64_t c = 0; c < 2; ++c) {
                    const int64_t id = od * 2 + 1;
                    const int64_t ih = oh * 2 + 1;
                    const int64_t iw = ow * 2 + 1;
                    const size_t inputIndex = static_cast<size_t>((((id * 4 + ih) * 4 + iw) * 2) + c);
                    passed = passed && yHost[outputIndex++] == xHost[inputIndex];
                }
            }
        }
    }
    return passed ? 0 : 1;
}
