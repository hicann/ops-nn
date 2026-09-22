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
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_unpool2d.h"

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

int RunMaxUnpool2d(const aclTensor* self, const aclTensor* indices, const aclIntArray* outputSize, aclTensor* out,
                   aclrtStream stream, void* outDeviceAddr, const std::vector<int64_t>& outShape)
{
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnMaxUnpool2dGetWorkspaceSize(self, indices, outputSize, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool2dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnMaxUnpool2d(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxUnpool2d failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    std::vector<uint16_t> expectedData(size, 0);
    expectedData[3] = 0x3f80;
    expectedData[8] = 0x4000;
    expectedData[11] = 0x4040;
    expectedData[13] = 0x4080;
    bool matches = true;
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] BF16 bits: 0x%04x\n", i, static_cast<unsigned int>(resultData[i]));
        if (resultData[i] != expectedData[i]) {
            LOG_PRINT("expected[%ld] BF16 bits: 0x%04x\n", i, static_cast<unsigned int>(expectedData[i]));
            matches = false;
        }
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return matches ? 0 : 1;
}

int main()
{
    // 1. Initialize the device and stream.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Create input and output tensors.
    std::vector<int64_t> selfShape = {1, 1, 2, 2};
    std::vector<int64_t> outShape = {1, 1, 4, 4};
    void* selfDeviceAddr = nullptr;
    void* indicesDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* indices = nullptr;
    aclTensor* out = nullptr;
    // BF16 bit patterns for 1.0, 2.0, 3.0 and 4.0.
    std::vector<uint16_t> selfHostData = {0x3f80, 0x4000, 0x4040, 0x4080};
    std::vector<int32_t> indicesHostData = {3, 8, 11, 13};
    std::vector<uint16_t> outHostData(16, 0);

    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, ACL_BF16, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indicesHostData, selfShape, &indicesDeviceAddr, ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, ACL_BF16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> arraySize = {4, 4};
    const aclIntArray* outputSize = aclCreateIntArray(arraySize.data(), arraySize.size());
    CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 3. Call the two-stage API, wait for completion, and print the result.
    ret = RunMaxUnpool2d(self, indices, outputSize, out, stream, outDeviceAddr, outShape);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 4. Release tensors, attributes, and device resources.
    aclDestroyTensor(self);
    aclDestroyTensor(indices);
    aclDestroyTensor(out);
    aclDestroyIntArray(outputSize);

    // 5. Release device resources.
    aclrtFree(selfDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
