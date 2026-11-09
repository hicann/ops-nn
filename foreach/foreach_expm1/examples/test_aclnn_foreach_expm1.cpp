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
#include "aclnnop/aclnn_foreach_expm1.h"

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

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
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
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    void* input1DeviceAddr = nullptr;
    void* input2DeviceAddr = nullptr;
    void* out1DeviceAddr = nullptr;
    void* out2DeviceAddr = nullptr;
    void* workspaceAddr = nullptr;
    aclTensor* input1 = nullptr;
    aclTensor* input2 = nullptr;
    aclTensor* out1 = nullptr;
    aclTensor* out2 = nullptr;
    aclTensorList* tensorListInput = nullptr;
    aclTensorList* tensorListOutput = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    int ret = ACL_SUCCESS;
    bool aclInited = false;
    bool deviceSet = false;
    std::vector<int64_t> selfShape1 = {2, 3};
    std::vector<int64_t> selfShape2 = {1, 3};
    std::vector<int64_t> outShape1 = {2, 3};
    std::vector<int64_t> outShape2 = {1, 3};
    std::vector<float> input1HostData = {1, 2, 3, 4, 5, 6};
    std::vector<float> input2HostData = {7, 8, 9};
    std::vector<float> out1HostData(6, 0);
    std::vector<float> out2HostData(3, 0);
    std::vector<float> out1Data;
    std::vector<float> out2Data;
    std::vector<aclTensor*> tempInput;
    std::vector<aclTensor*> tempOutput;
    int64_t size = 0;

    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        goto cleanup;
    }
    aclInited = true;
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        goto cleanup;
    }
    deviceSet = true;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        goto cleanup;
    }

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // 创建input1 aclTensor
    ret = CreateAclTensor(input1HostData, selfShape1, &input1DeviceAddr, aclDataType::ACL_FLOAT, &input1);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    // 创建input2 aclTensor
    ret = CreateAclTensor(input2HostData, selfShape2, &input2DeviceAddr, aclDataType::ACL_FLOAT, &input2);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    // 创建out1 aclTensor
    ret = CreateAclTensor(out1HostData, outShape1, &out1DeviceAddr, aclDataType::ACL_FLOAT, &out1);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    // 创建out2 aclTensor
    ret = CreateAclTensor(out2HostData, outShape2, &out2DeviceAddr, aclDataType::ACL_FLOAT, &out2);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    tempInput = {input1, input2};
    tensorListInput = aclCreateTensorList(tempInput.data(), tempInput.size());
    CHECK_RET(tensorListInput != nullptr, LOG_PRINT("aclCreateTensorList for input failed.\n"); ret = ACL_ERROR_FAILURE;
              goto cleanup);
    tempOutput = {out1, out2};
    tensorListOutput = aclCreateTensorList(tempOutput.data(), tempOutput.size());
    CHECK_RET(tensorListOutput != nullptr, LOG_PRINT("aclCreateTensorList for output failed.\n");
              ret = ACL_ERROR_FAILURE; goto cleanup);

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    // 调用aclnnForeachExpm1第一段接口
    ret = aclnnForeachExpm1GetWorkspaceSize(tensorListInput, tensorListOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachExpm1GetWorkspaceSize failed. ERROR: %d\n", ret);
              goto cleanup);
    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); goto cleanup);
    }
    // 调用aclnnForeachExpm1第二段接口
    ret = aclnnForeachExpm1(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachExpm1 failed. ERROR: %d\n", ret); goto cleanup);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); goto cleanup);

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    size = GetShapeSize(outShape1);
    out1Data.assign(size, 0);
    ret = aclrtMemcpy(out1Data.data(), out1Data.size() * sizeof(out1Data[0]), out1DeviceAddr,
                      size * sizeof(out1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); goto cleanup);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out1 result[%ld] is: %f\n", i, out1Data[i]);
    }

    size = GetShapeSize(outShape2);
    out2Data.assign(size, 0);
    ret = aclrtMemcpy(out2Data.data(), out2Data.size() * sizeof(out2Data[0]), out2DeviceAddr,
                      size * sizeof(out2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); goto cleanup);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out2 result[%ld] is: %f\n", i, out2Data[i]);
    }

cleanup:
    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    if (tensorListInput != nullptr) {
        aclDestroyTensorList(tensorListInput);
    } else {
        if (input1 != nullptr) {
            aclDestroyTensor(input1);
        }
        if (input2 != nullptr) {
            aclDestroyTensor(input2);
        }
    }
    if (tensorListOutput != nullptr) {
        aclDestroyTensorList(tensorListOutput);
    } else {
        if (out1 != nullptr) {
            aclDestroyTensor(out1);
        }
        if (out2 != nullptr) {
            aclDestroyTensor(out2);
        }
    }

    // 7.释放device资源，需要根据具体API的接口定义修改
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    if (input1DeviceAddr != nullptr) {
        aclrtFree(input1DeviceAddr);
    }
    if (input2DeviceAddr != nullptr) {
        aclrtFree(input2DeviceAddr);
    }
    if (out1DeviceAddr != nullptr) {
        aclrtFree(out1DeviceAddr);
    }
    if (out2DeviceAddr != nullptr) {
        aclrtFree(out2DeviceAddr);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    if (deviceSet) {
        aclrtResetDevice(deviceId);
    }
    if (aclInited) {
        aclFinalize();
    }
    return ret;
}
