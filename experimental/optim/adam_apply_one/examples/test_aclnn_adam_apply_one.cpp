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
#include "aclnnop/aclnn_adam_apply_one.h"

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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
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

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> shape = {64, 8};

    std::vector<float> input0HostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> input1HostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> input2HostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> input3HostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> input4HostData = {0, 1, 2, 3, 4, 5, 6, 7};

    std::vector<float> mul0_xHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> mul1_xHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> mul2_xHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> mul3_xHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> add2_yHostData = {0, 1, 2, 3, 4, 5, 6, 7};

    std::vector<float> output0HostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> output1HostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> output2HostData = {0, 1, 2, 3, 4, 5, 6, 7};

    void* input0DeviceAddr = nullptr;
    void* input1DeviceAddr = nullptr;
    void* input2DeviceAddr = nullptr;
    void* input3DeviceAddr = nullptr;
    void* input4DeviceAddr = nullptr;
    void* mul0_xDeviceAddr = nullptr;
    void* mul1_xDeviceAddr = nullptr;
    void* mul2_xDeviceAddr = nullptr;
    void* mul3_xDeviceAddr = nullptr;
    void* add2_yDeviceAddr = nullptr;
    void* output0DeviceAddr = nullptr;
    void* output1DeviceAddr = nullptr;
    void* output2DeviceAddr = nullptr;

    aclTensor* input0 = nullptr;
    aclTensor* input1 = nullptr;
    aclTensor* input2 = nullptr;
    aclTensor* input3 = nullptr;
    aclTensor* input4 = nullptr;
    aclTensor* mul0_x = nullptr;
    aclTensor* mul1_x = nullptr;
    aclTensor* mul2_x = nullptr;
    aclTensor* mul3_x = nullptr;
    aclTensor* add2_y = nullptr;
    aclTensor* output0 = nullptr;
    aclTensor* output1 = nullptr;
    aclTensor* output2 = nullptr;

    ret = CreateAclTensor(input0HostData, shape, &input0DeviceAddr, aclDataType::ACL_FLOAT, &input0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input1HostData, shape, &input1DeviceAddr, aclDataType::ACL_FLOAT, &input1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input2HostData, shape, &input2DeviceAddr, aclDataType::ACL_FLOAT, &input2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input3HostData, shape, &input3DeviceAddr, aclDataType::ACL_FLOAT, &input3);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input4HostData, shape, &input4DeviceAddr, aclDataType::ACL_FLOAT, &input4);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mul0_xHostData, shape, &mul0_xDeviceAddr, aclDataType::ACL_FLOAT, &mul0_x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mul1_xHostData, shape, &mul1_xDeviceAddr, aclDataType::ACL_FLOAT, &mul1_x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mul2_xHostData, shape, &mul2_xDeviceAddr, aclDataType::ACL_FLOAT, &mul2_x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mul3_xHostData, shape, &mul3_xDeviceAddr, aclDataType::ACL_FLOAT, &mul3_x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(add2_yHostData, shape, &add2_yDeviceAddr, aclDataType::ACL_FLOAT, &add2_y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(output0HostData, shape, &output0DeviceAddr, aclDataType::ACL_FLOAT, &output0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(output1HostData, shape, &output1DeviceAddr, aclDataType::ACL_FLOAT, &output1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(output2HostData, shape, &output2DeviceAddr, aclDataType::ACL_FLOAT, &output2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnAdamApplyOneGetWorkspaceSize(
        input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, add2_y, output0, output1, output2,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdamApplyOneGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnAdamApplyOne(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdamApplyOne failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    PrintOutResult(shape, &output0DeviceAddr);
    PrintOutResult(shape, &output1DeviceAddr);
    PrintOutResult(shape, &output2DeviceAddr);

    aclDestroyTensor(input0);
    aclDestroyTensor(input1);
    aclDestroyTensor(input2);
    aclDestroyTensor(input3);
    aclDestroyTensor(input4);
    aclDestroyTensor(mul0_x);
    aclDestroyTensor(mul1_x);
    aclDestroyTensor(mul2_x);
    aclDestroyTensor(mul3_x);
    aclDestroyTensor(add2_y);
    aclDestroyTensor(output0);
    aclDestroyTensor(output1);
    aclDestroyTensor(output2);

    aclrtFree(input0DeviceAddr);
    aclrtFree(input1DeviceAddr);
    aclrtFree(input2DeviceAddr);
    aclrtFree(input3DeviceAddr);
    aclrtFree(input4DeviceAddr);
    aclrtFree(mul0_xDeviceAddr);
    aclrtFree(mul1_xDeviceAddr);
    aclrtFree(mul2_xDeviceAddr);
    aclrtFree(mul3_xDeviceAddr);
    aclrtFree(add2_yDeviceAddr);
    aclrtFree(output0DeviceAddr);
    aclrtFree(output1DeviceAddr);
    aclrtFree(output2DeviceAddr);
    
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}