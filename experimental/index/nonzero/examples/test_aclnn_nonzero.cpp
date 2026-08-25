/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Example: aclnnNonzero usage
 *   Input:  x = [[1, 0, 2], [0, 3, 0]] (float32, shape [2, 3])
 *   Output: y = [[0, 0], [0, 2], [1, 1]] (int64, shape [3, 2])
 */
#include <cstdio>
#include <vector>

#include "aclnn_nonzero_example_utils.h"
#include "aclnnop/aclnn_nonzero.h"

int main()
{
    using nonzero_example::AclResources;
    using nonzero_example::ElementCount;

    AclResources resources;
    aclError ret = resources.Initialize(0);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    const std::vector<int64_t> xShape = {2, 3};
    const std::vector<int64_t> yShape = {6, 2};
    std::vector<float> xHost = {1.0F, 0.0F, 2.0F, 0.0F, 3.0F, 0.0F};
    std::vector<int64_t> yHost(ElementCount(yShape), 0);
    ret = resources.CreateTensor(xHost, xShape, ACL_FLOAT, &resources.xDevice, &resources.x);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = resources.CreateTensor(yHost, yShape, ACL_INT64, &resources.yDevice, &resources.y);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnNonzeroGetWorkspaceSize(resources.x, resources.y, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&resources.workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            return ret;
        }
    }
    ret = aclnnNonzero(resources.workspace, workspaceSize, executor, resources.stream);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtSynchronizeStream(resources.stream);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(yHost.data(), yHost.size() * sizeof(int64_t), resources.yDevice, yHost.size() * sizeof(int64_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    const std::vector<int64_t> expected = {0, 0, 0, 2, 1, 1};
    bool passed = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        passed = passed && yHost[i] == expected[i];
    }
    if (passed) {
        std::printf("Nonzero example passed.\n");
        return 0;
    }
    std::printf("Nonzero example failed.\n");
    return 1;
}
