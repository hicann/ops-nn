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
#include "acl/acl.h"
#include "aclnnop/aclnn_foreach_zero_inplace.h"

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

void LogCleanupFailure(const char* operation, aclError status)
{
    if (status != ACL_SUCCESS) {
        LOG_PRINT("%s failed. ERROR: %d\n", operation, status);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 初始化ACL资源
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
              LogCleanupFailure("aclFinalize", aclFinalize()); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
              LogCleanupFailure("aclrtResetDevice", aclrtResetDevice(deviceId));
              LogCleanupFailure("aclFinalize", aclFinalize()); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    *deviceAddr = nullptr;
    *tensor = nullptr;
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return ACL_ERROR_FAILURE);
    return 0;
}

int main()
{
    // 1. 初始化device和stream
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入TensorList
    std::vector<int64_t> selfShape1 = {2, 3};
    std::vector<int64_t> selfShape2 = {1, 3};
    void* input1DeviceAddr = nullptr;
    void* input2DeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* input1 = nullptr;
    aclTensor* input2 = nullptr;
    aclTensorList* tensorListInput = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool streamHasPendingWork = false;

    auto cleanup = [&](int status) {
        bool deviceWasReset = false;
        auto recordCleanupStatus = [&](const char* operation, aclError cleanupStatus) {
            if (cleanupStatus != ACL_SUCCESS) {
                LOG_PRINT("%s failed. ERROR: %d\n", operation, cleanupStatus);
                if (status == ACL_SUCCESS) {
                    status = cleanupStatus;
                }
            }
        };
        if (stream != nullptr) {
            const aclError streamStatus = streamHasPendingWork ? aclrtDestroyStreamForce(stream) :
                                                                 aclrtDestroyStream(stream);
            recordCleanupStatus(streamHasPendingWork ? "aclrtDestroyStreamForce" : "aclrtDestroyStream", streamStatus);
            if (streamHasPendingWork && streamStatus != ACL_SUCCESS) {
                const aclError resetStatus = aclrtResetDeviceForce(deviceId);
                recordCleanupStatus("aclrtResetDeviceForce", resetStatus);
                if (resetStatus != ACL_SUCCESS) {
                    LOG_PRINT("Device reset failed; retain in-flight resources until process exit.\n");
                    return status;
                }
                deviceWasReset = true;
                workspaceAddr = nullptr;
                input1DeviceAddr = nullptr;
                input2DeviceAddr = nullptr;
            }
            stream = nullptr;
            streamHasPendingWork = false;
        }
        if (tensorListInput != nullptr) {
            recordCleanupStatus("aclDestroyTensorList", aclDestroyTensorList(tensorListInput));
            tensorListInput = nullptr;
        } else {
            if (input1 != nullptr) {
                recordCleanupStatus("aclDestroyTensor(input1)", aclDestroyTensor(input1));
                input1 = nullptr;
            }
            if (input2 != nullptr) {
                recordCleanupStatus("aclDestroyTensor(input2)", aclDestroyTensor(input2));
                input2 = nullptr;
            }
        }
        if (!deviceWasReset && workspaceAddr != nullptr) {
            recordCleanupStatus("aclrtFree(workspace)", aclrtFree(workspaceAddr));
            workspaceAddr = nullptr;
        }
        if (!deviceWasReset && input1DeviceAddr != nullptr) {
            recordCleanupStatus("aclrtFree(input1)", aclrtFree(input1DeviceAddr));
            input1DeviceAddr = nullptr;
        }
        if (!deviceWasReset && input2DeviceAddr != nullptr) {
            recordCleanupStatus("aclrtFree(input2)", aclrtFree(input2DeviceAddr));
            input2DeviceAddr = nullptr;
        }
        if (!deviceWasReset) {
            recordCleanupStatus("aclrtResetDevice", aclrtResetDevice(deviceId));
        }
        recordCleanupStatus("aclFinalize", aclFinalize());
        return status;
    };

    std::vector<float> input1HostData = {1, 2, 3, 4, 5, 6};
    std::vector<float> input2HostData = {7, 8, 9};

    // 创建input1 aclTensor
    ret = CreateAclTensor(input1HostData, selfShape1, &input1DeviceAddr, aclDataType::ACL_FLOAT, &input1);
    CHECK_RET(ret == ACL_SUCCESS, return cleanup(ret));

    // 创建input2 aclTensor
    ret = CreateAclTensor(input2HostData, selfShape2, &input2DeviceAddr, aclDataType::ACL_FLOAT, &input2);
    CHECK_RET(ret == ACL_SUCCESS, return cleanup(ret));

    std::vector<aclTensor*> tempInput{input1, input2};
    tensorListInput = aclCreateTensorList(tempInput.data(), tempInput.size());
    CHECK_RET(tensorListInput != nullptr, LOG_PRINT("aclCreateTensorList failed\n"); return cleanup(ACL_ERROR_FAILURE));

    // 3. 调用两段式接口
    // 调用aclnnForeachZeroInplace第一段接口
    ret = aclnnForeachZeroInplaceGetWorkspaceSize(tensorListInput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachZeroInplaceGetWorkspaceSize failed. ERROR: %d\n", ret);
              return cleanup(ret));
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return cleanup(ret));
    }
    // 调用aclnnForeachZeroInplace第二段接口
    ret = aclnnForeachZeroInplace(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachZeroInplace failed. ERROR: %d\n", ret); return cleanup(ret));
    streamHasPendingWork = true;

    // 4. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return cleanup(ret));
    streamHasPendingWork = false;

    // 5. 将原地更新后的输入复制回Host
    auto size = GetShapeSize(selfShape1);
    std::vector<float> self1Data(size, 0);
    ret = aclrtMemcpy(self1Data.data(), self1Data.size() * sizeof(self1Data[0]), input1DeviceAddr,
                      size * sizeof(self1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              return cleanup(ret));
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out1 result[%ld] is: %f\n", i, self1Data[i]);
    }

    size = GetShapeSize(selfShape2);
    std::vector<float> self2Data(size, 0);
    ret = aclrtMemcpy(self2Data.data(), self2Data.size() * sizeof(self2Data[0]), input2DeviceAddr,
                      size * sizeof(self2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              return cleanup(ret));
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out2 result[%ld] is: %f\n", i, self2Data[i]);
    }

    // 6. 释放aclTensor和device资源
    return cleanup(ACL_SUCCESS);
}
