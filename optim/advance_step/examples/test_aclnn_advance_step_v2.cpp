/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_advance_step_v2.h"//不确定头文件名字
#define CHECK_RET(cond, return_expr) \
    do {                               \
    if (!(cond)) {                   \
        return_expr;                   \
    }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
    printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
    shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
    auto size = GetShapeSize(shape);
    std::vector<int64_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                        *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %ld\n", i, resultData[i]);
    }
}

int Init(int64_t deviceId, aclrtStream* stream) {
    // 固定写法，资源初始化
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
                    aclDataType dataType, aclTensor** tensor) {
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
    return 0;
}

int main() {
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> input1Shape = {16}; 
    std::vector<int64_t> input2Shape = {8,2}; 
    std::vector<int64_t> input3Shape = {8,1000}; 
    std::vector<int64_t> input4Shape = {8,1}; 
    std::vector<int64_t> input5Shape = {8}; 
    std::vector<int64_t> input1HostData = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int64_t> input2HostData = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int64_t> input3HostData(8000, 7);
    std::vector<int64_t> input4HostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int64_t> input5HostData = {0, 1, 2, 3, 4, 5, 6, 7};

    void* input1DeviceAddr = nullptr;
    aclTensor* input1 = nullptr;
    void* input2DeviceAddr = nullptr;
    aclTensor* input2 = nullptr;
    void* input3DeviceAddr = nullptr;
    aclTensor* input3 = nullptr;
    void* input4DeviceAddr = nullptr;
    aclTensor* input4 = nullptr;
    void* input5DeviceAddr = nullptr;
    aclTensor* input5 = nullptr;
    void* input6DeviceAddr = nullptr;
    aclTensor* input6 = nullptr;
    void* input7DeviceAddr = nullptr;
    aclTensor* input7 = nullptr;
    void* input8DeviceAddr = nullptr;
    aclTensor* input8 = nullptr;
    // 创建input aclTensor
    ret = CreateAclTensor(input1HostData, input1Shape, &input1DeviceAddr, aclDataType::ACL_INT64, &input1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input2HostData, input2Shape, &input2DeviceAddr, aclDataType::ACL_INT64, &input2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input1HostData, input1Shape, &input3DeviceAddr, aclDataType::ACL_INT64, &input3);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input1HostData, input1Shape, &input4DeviceAddr, aclDataType::ACL_INT64, &input4);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input1HostData, input1Shape, &input5DeviceAddr, aclDataType::ACL_INT64, &input5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input3HostData, input3Shape, &input6DeviceAddr, aclDataType::ACL_INT64, &input6);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input4HostData, input4Shape, &input5DeviceAddr, aclDataType::ACL_INT64, &input7);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input5HostData, input5Shape, &input6DeviceAddr, aclDataType::ACL_INT64, &input8);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int64_t numseq = 8;
    int64_t specnum = 8;
    int64_t blocksize = 8;

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 16 * 1024 * 1024;
    aclOpExecutor* executor;

    // 调用aclnnAdvanceStepV2第一段接口
    ret = aclnnAdvanceStepV2GetWorkspaceSize(
    input1,input2,input3,input4,input5,input6,input7,input8,
    numseq,numseq,blocksize,
    &workspaceSize,
    &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdvanceStepV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnAdvanceStepV2第二段接口
    ret = aclnnAdvanceStepV2(
    workspaceAddr,
    workspaceSize,
    executor,
    stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdvanceStepV2 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    PrintOutResult(input1Shape, &input1DeviceAddr);
    PrintOutResult(input2Shape, &input2DeviceAddr);
    PrintOutResult(input3Shape, &input3DeviceAddr);
    PrintOutResult(input4Shape, &input4DeviceAddr);
    PrintOutResult(input5Shape, &input5DeviceAddr);

    // 6. 释放aclTensor和aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(input1);
    aclDestroyTensor(input2);
    aclDestroyTensor(input3);
    aclDestroyTensor(input4);
    aclDestroyTensor(input5);
    aclDestroyTensor(input6);
    aclDestroyTensor(input7);
    aclDestroyTensor(input8);

    // 7.释放device资源，需要根据具体API的接口定义修改
    aclrtFree(input1DeviceAddr);
    aclrtFree(input2DeviceAddr);
    aclrtFree(input3DeviceAddr);
    aclrtFree(input4DeviceAddr);
    aclrtFree(input5DeviceAddr);
    aclrtFree(input6DeviceAddr);
    aclrtFree(input7DeviceAddr);
    aclrtFree(input8DeviceAddr);

    if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}