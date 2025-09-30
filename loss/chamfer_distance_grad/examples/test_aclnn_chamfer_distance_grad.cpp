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
#include <unistd.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_chamfer_distance_backward.h"

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

int Init(int32_t deviceId, aclrtStream* stream) {
    // 固定写法，AscendCL初始化
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

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
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
    // 1.(固定写法)device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2.构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xyz1Shape = {2, 2, 2};
    std::vector<int64_t> xyz2Shape = {2, 2, 2};
    std::vector<int64_t> idx1Shape = {2, 2};
    std::vector<int64_t> idx2Shape = {2, 2};
    std::vector<int64_t> gradDist1Shape = {2, 2};
    std::vector<int64_t> gradDist2Shape = {2, 2};
    std::vector<int64_t> gradXyz1Shape = {2, 2, 2};
    std::vector<int64_t> gradXyz2Shape = {2, 2, 2};
    void* xyz1DeviceAddr = nullptr;
    void* xyz2DeviceAddr = nullptr;
    void* idx1DeviceAddr = nullptr;
    void* idx2DeviceAddr = nullptr;
    void* gradDist1DeviceAddr = nullptr;
    void* gradDist2DeviceAddr = nullptr;
    void* gradXyz1DeviceAddr = nullptr;
    void* gradXyz2DeviceAddr = nullptr;
    aclTensor* xyz1 = nullptr;
    aclTensor* xyz2 = nullptr;
    aclTensor* idx1 = nullptr;
    aclTensor* idx2 = nullptr;
    aclTensor* gradDist1 = nullptr;
    aclTensor* gradDist2 = nullptr;
    aclTensor* gradXyz1 = nullptr;
    aclTensor* gradXyz2 = nullptr;
    std::vector<float> xyz1HostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> xyz2HostData = {1, 1, 1, 2, 2, 2, 3, 3};
    std::vector<int32_t> idx1HostData = {0, 1, 2, 3};
    std::vector<int32_t> idx2HostData = {0, 1, 2, 3};
    std::vector<float> gradDist1HostData = {0, 1, 2, 3};
    std::vector<float> gradDist2HostData = {0, 1, 2, 3};
    std::vector<float> gradXyz1HostData = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> gradXyz2HostData = {0, 0, 0, 0, 0, 0, 0, 0};
    // 创建xyz1 aclTensor
    ret = CreateAclTensor(xyz1HostData, xyz1Shape, &xyz1DeviceAddr, aclDataType::ACL_FLOAT, &xyz1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建xyz2 aclTensor
    ret = CreateAclTensor(xyz2HostData, xyz2Shape, &xyz2DeviceAddr, aclDataType::ACL_FLOAT, &xyz2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建idx1 aclTensor
    ret = CreateAclTensor(idx1HostData, idx1Shape, &idx1DeviceAddr, aclDataType::ACL_INT32, &idx1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建idx2 aclTensor
    ret = CreateAclTensor(idx2HostData, idx2Shape, &idx2DeviceAddr, aclDataType::ACL_INT32, &idx2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradDist1 aclTensor
    ret = CreateAclTensor(gradDist1HostData, gradDist1Shape, &gradDist1DeviceAddr, aclDataType::ACL_FLOAT, &gradDist1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradDist2 aclTensor
    ret = CreateAclTensor(gradDist2HostData, gradDist2Shape, &gradDist2DeviceAddr, aclDataType::ACL_FLOAT, &gradDist2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradXyz1 aclTensor
    ret = CreateAclTensor(gradXyz1HostData, gradXyz1Shape, &gradXyz1DeviceAddr, aclDataType::ACL_FLOAT, &gradXyz1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradXyz2 aclTensor
    ret = CreateAclTensor(gradXyz2HostData, gradXyz2Shape, &gradXyz2DeviceAddr, aclDataType::ACL_FLOAT, &gradXyz2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3.调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnChamferDistanceBackward第一段接口
    ret = aclnnChamferDistanceBackwardGetWorkspaceSize(xyz1, xyz2, idx1, idx2, gradDist1, gradDist2, gradXyz1, gradXyz2,
                                                       &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnChamferDistanceBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnChamferDistanceBackward第二段接口
    ret = aclnnChamferDistanceBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnChamferDistanceBackward failed. ERROR: %d\n", ret); return ret);

    // 4.(固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(gradXyz1Shape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradXyz1DeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result1[%ld] is: %f\n", i, resultData[i]);
    }

    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradXyz2DeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result2[%ld] is: %f\n", i, resultData[i]);
    }

    // 6.释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(xyz1);
    aclDestroyTensor(xyz2);
    aclDestroyTensor(idx1);
    aclDestroyTensor(idx2);
    aclDestroyTensor(gradDist1);
    aclDestroyTensor(gradDist2);
    aclDestroyTensor(gradXyz1);
    aclDestroyTensor(gradXyz2);

    // 7.释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xyz1DeviceAddr);
    aclrtFree(xyz2DeviceAddr);
    aclrtFree(idx1DeviceAddr);
    aclrtFree(idx2DeviceAddr);
    aclrtFree(gradDist1DeviceAddr);
    aclrtFree(gradDist2DeviceAddr);
    aclrtFree(gradXyz1DeviceAddr);
    aclrtFree(gradXyz2DeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}