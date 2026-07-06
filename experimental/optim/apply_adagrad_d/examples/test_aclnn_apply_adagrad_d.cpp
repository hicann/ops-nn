/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_apply_adagrad_d.cpp
 * \brief Eager (aclnn) runner example for ApplyAdagradD (experimental/optim).
 *
 * Runs a simple shape={4,4} test with float32 to verify the operator end-to-end.
 * Expected result (update_slots=true, var=1.0, accum=1.0, lr=0.1, grad=2.0):
 *   accum_out = accum + grad*grad = 1.0 + 4.0 = 5.0
 *   var_out   = var - lr*grad/sqrt(accum_out) = 1.0 - 0.1*2.0/sqrt(5.0) ≈ 0.9107
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_apply_adagrad_d.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // In-place op: var and accum are updated in place, no separate output tensors.
    // Inputs: var, accum, grad share same shape; lr is scalar {1}
    std::vector<int64_t> varShape = {4, 4};
    std::vector<int64_t> lrShape = {1};

    void* varDeviceAddr = nullptr;
    void* accumDeviceAddr = nullptr;
    void* lrDeviceAddr = nullptr;
    void* gradDeviceAddr = nullptr;

    aclTensor* varTensor = nullptr;
    aclTensor* accumTensor = nullptr;
    aclTensor* lrTensor = nullptr;
    aclTensor* gradTensor = nullptr;

    int64_t elemCount = GetShapeSize(varShape);

    std::vector<float> varHostData(elemCount, 1.0f);
    std::vector<float> accumHostData(elemCount, 1.0f);
    std::vector<float> lrHostData = {0.1f};
    std::vector<float> gradHostData(elemCount, 2.0f);

    ret = CreateAclTensor(varHostData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &varTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(accumHostData, varShape, &accumDeviceAddr, aclDataType::ACL_FLOAT, &accumTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(lrHostData, lrShape, &lrDeviceAddr, aclDataType::ACL_FLOAT, &lrTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradHostData, varShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &gradTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // attrs
    bool updateSlots = true;
    bool useLocking = false;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // Signature: (var, accum, lr, grad, updateSlots, useLocking, &workspaceSize, &executor)
    ret = aclnnApplyAdagradDGetWorkspaceSize(varTensor, accumTensor, lrTensor, gradTensor, updateSlots, useLocking,
                                             &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdagradDGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnApplyAdagradD(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdagradD failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Read back var (in-place updated, expect ≈ 0.9107)
    std::vector<float> varResult(elemCount, 0.0f);
    ret = aclrtMemcpy(varResult.data(), varResult.size() * sizeof(float), varDeviceAddr, elemCount * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy var failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("[var] after update (expect ≈ 0.9107):\n");
    for (int64_t i = 0; i < elemCount && i < 4; i++) {
        LOG_PRINT("  var[%ld] = %f\n", i, varResult[i]);
    }

    // Read back accum (in-place updated, expect = 5.0)
    std::vector<float> accumResult(elemCount, 0.0f);
    ret = aclrtMemcpy(accumResult.data(), accumResult.size() * sizeof(float), accumDeviceAddr,
                      elemCount * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy accum failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("[accum] after update (expect = 5.0):\n");
    for (int64_t i = 0; i < elemCount && i < 4; i++) {
        LOG_PRINT("  accum[%ld] = %f\n", i, accumResult[i]);
    }

    // Cleanup
    aclDestroyTensor(varTensor);
    aclDestroyTensor(accumTensor);
    aclDestroyTensor(lrTensor);
    aclDestroyTensor(gradTensor);

    aclrtFree(varDeviceAddr);
    aclrtFree(accumDeviceAddr);
    aclrtFree(lrDeviceAddr);
    aclrtFree(gradDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
