/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_quant_batch_matmul_inplace_add.h"

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
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        aclFinalize();
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        aclrtResetDevice(deviceId);
        aclFinalize();
    }
    return ret;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    // FP4 hostData contains packed bytes; aclTensor shape and strides count logical FP4 elements.
    const size_t byteSize = hostData.size() * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, byteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, byteSize, hostData.data(), byteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return 1);
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int AclnnQuantBatchMatmulInplaceAddMxfp4Demo(aclrtStream stream)
{
    const int64_t m = 4;
    const int64_t k = 64;
    const int64_t n = 6;
    const int64_t scaleGroups = (k + 63) / 64;
    const std::vector<int64_t> x1Shape = {k, m};
    const std::vector<int64_t> x2Shape = {k, n};
    const std::vector<int64_t> x1ScaleShape = {scaleGroups, m, 2};
    const std::vector<int64_t> x2ScaleShape = {scaleGroups, n, 2};
    const std::vector<int64_t> yRefShape = {m, n};

    // Each 0x22 byte contains two E2M1 values of 1.0; E8M0 encoding 127 represents 1.0.
    const std::vector<uint8_t> x1HostData(GetShapeSize(x1Shape) / 2, 0x22);
    const std::vector<uint8_t> x2HostData(GetShapeSize(x2Shape) / 2, 0x22);
    const std::vector<uint8_t> x1ScaleHostData(GetShapeSize(x1ScaleShape), 127);
    const std::vector<uint8_t> x2ScaleHostData(GetShapeSize(x2ScaleShape), 127);
    const std::vector<float> yRefHostData(GetShapeSize(yRefShape), 1.0f);

    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* x1ScaleDeviceAddr = nullptr;
    void* x2ScaleDeviceAddr = nullptr;
    void* yRefDeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* x1Scale = nullptr;
    aclTensor* x2Scale = nullptr;
    aclTensor* yRef = nullptr;

    auto ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, ACL_FLOAT4_E2M1, &x1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, ACL_FLOAT4_E2M1, &x2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, ACL_FLOAT8_E8M0, &x1Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, ACL_FLOAT8_E8M0, &x2Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(yRefHostData, yRefShape, &yRefDeviceAddr, ACL_FLOAT, &yRef);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yRefTensorPtr(yRef, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yRefDeviceAddrPtr(yRefDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const bool transposeX1 = true;
    const bool transposeX2 = false;
    const int64_t groupSize = 4295032864;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize(x1, x2, x1Scale, x2Scale, yRef, transposeX1, transposeX2,
                                                          groupSize, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnQuantBatchMatmulInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantBatchMatmulInplaceAdd failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    std::vector<float> resultData(GetShapeSize(yRefShape), 0.0f);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), yRefDeviceAddr,
                      resultData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);
    const float expected = static_cast<float>(k) + 1.0f;
    for (size_t i = 0; i < resultData.size(); ++i) {
        LOG_PRINT("result[%zu] is: %f\n", i, resultData[i]);
        CHECK_RET(resultData[i] == expected, LOG_PRINT("Expected %f, got %f.\n", expected, resultData[i]); return 1);
    }
    LOG_PRINT("MXFP4 E2M1 example passed: all %zu outputs equal %f.\n", resultData.size(), expected);
    return ACL_SUCCESS;
}

int main(int argc, char* argv[])
{
    CHECK_RET(argc <= 2, LOG_PRINT("Usage: %s [deviceId]\n", argv[0]); return 1);
    int32_t deviceId = 0;
    if (argc == 2) {
        char* end = nullptr;
        const long value = std::strtol(argv[1], &end, 10);
        CHECK_RET(end != argv[1] && *end == '\0' && value >= 0 && value <= INT32_MAX,
                  LOG_PRINT("Invalid deviceId: %s\n", argv[1]);
                  return 1);
        deviceId = static_cast<int32_t>(value);
    }
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = AclnnQuantBatchMatmulInplaceAddMxfp4Demo(stream);
    Finalize(deviceId, stream);
    return ret;
}
