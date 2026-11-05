/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_expand_into_jagged_permute.h"

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
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        (void)aclFinalize();
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        (void)aclrtResetDevice(deviceId);
        (void)aclFinalize();
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        (void)aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ret;
    }

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    if (*tensor == nullptr) {
        LOG_PRINT("aclCreateTensor failed.\n");
        (void)aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ACL_ERROR_FAILURE;
    }
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化, 参考acl API
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. 构造输入与输出, 需要根据API的接口自定义构造
    std::vector<int64_t> permuteShape = {3};
    std::vector<int64_t> inputOffsetsShape = {4};
    std::vector<int64_t> outputOffsetsShape = {4};
    std::vector<int64_t> outputPermuteShape = {6};
    void* permuteDeviceAddr = nullptr;
    void* inputOffsetsDeviceAddr = nullptr;
    void* outputOffsetsDeviceAddr = nullptr;
    void* outputPermuteDeviceAddr = nullptr;

    aclTensor* permute = nullptr;
    aclTensor* inputOffsets = nullptr;
    aclTensor* outputOffsets = nullptr;
    aclTensor* outputPermute = nullptr;
    int64_t outputSize = 6;
    uint64_t workspaceSize = 0;
    void* workspaceAddr = nullptr;
    aclOpExecutor* executor = nullptr;
    int result = ACL_SUCCESS;

    std::vector<int32_t> permuteHostData = {1, 0, 2};
    std::vector<int32_t> inputOffsetsHostData = {0, 3, 5, 8};
    std::vector<int32_t> outputOffsetsHostData = {0, 2, 4, 6};
    std::vector<int32_t> outputPermuteHostData = {3, 4, 0, 1, 5, 6};

    auto cleanup = [&]() {
        if (permute != nullptr) {
            aclDestroyTensor(permute);
        }
        if (inputOffsets != nullptr) {
            aclDestroyTensor(inputOffsets);
        }
        if (outputOffsets != nullptr) {
            aclDestroyTensor(outputOffsets);
        }
        if (outputPermute != nullptr) {
            aclDestroyTensor(outputPermute);
        }
        if (permuteDeviceAddr != nullptr) {
            (void)aclrtFree(permuteDeviceAddr);
        }
        if (inputOffsetsDeviceAddr != nullptr) {
            (void)aclrtFree(inputOffsetsDeviceAddr);
        }
        if (outputOffsetsDeviceAddr != nullptr) {
            (void)aclrtFree(outputOffsetsDeviceAddr);
        }
        if (outputPermuteDeviceAddr != nullptr) {
            (void)aclrtFree(outputPermuteDeviceAddr);
        }
        if (workspaceAddr != nullptr) {
            (void)aclrtFree(workspaceAddr);
        }
        if (stream != nullptr) {
            (void)aclrtDestroyStream(stream);
        }
        (void)aclrtResetDevice(deviceId);
        (void)aclFinalize();
    };

    do {
        ret = CreateAclTensor(permuteHostData, permuteShape, &permuteDeviceAddr, aclDataType::ACL_INT32, &permute);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("create permute tensor failed. ERROR: %d\n", ret);
            result = ret;
            break;
        }
        ret = CreateAclTensor(inputOffsetsHostData, inputOffsetsShape, &inputOffsetsDeviceAddr, aclDataType::ACL_INT32,
                              &inputOffsets);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("create inputOffsets tensor failed. ERROR: %d\n", ret);
            result = ret;
            break;
        }
        ret = CreateAclTensor(outputOffsetsHostData, outputOffsetsShape, &outputOffsetsDeviceAddr,
                              aclDataType::ACL_INT32, &outputOffsets);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("create outputOffsets tensor failed. ERROR: %d\n", ret);
            result = ret;
            break;
        }
        ret = CreateAclTensor(outputPermuteHostData, outputPermuteShape, &outputPermuteDeviceAddr,
                              aclDataType::ACL_INT32, &outputPermute);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("create outputPermute tensor failed. ERROR: %d\n", ret);
            result = ret;
            break;
        }

        // 3. 调用CANN算子库API, 需要修改为具体的Api名称
        ret = aclnnExpandIntoJaggedPermuteGetWorkspaceSize(permute, inputOffsets, outputOffsets, outputSize,
                                                           outputPermute, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("aclnnExpandIntoJaggedPermuteGetWorkspaceSize failed. ERROR: %d\n", ret);
            result = ret;
            break;
        }

        // 根据第一段接口计算出的workspaceSize申请device内存
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                result = ret;
                break;
            }
        }

        // 调用aclnnExpandIntoJaggedPermute第二段接口
        ret = aclnnExpandIntoJaggedPermute(workspaceAddr, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("aclnnExpandIntoJaggedPermute failed. ERROR: %d\n", ret);
            result = ret;
            break;
        }

        // 4. （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            result = ret;
            break;
        }

        // 5.获取输出的值,将device侧内存上的结果拷贝至host侧,需要根据具体API的接口定义修改
        const int64_t size = GetShapeSize(outputPermuteShape);
        std::vector<int32_t> resultData(size, 0);
        ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputPermuteDeviceAddr,
                          size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
            result = ret;
            break;
        }
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
        }
    } while (false);

    cleanup();
    return result;
}
