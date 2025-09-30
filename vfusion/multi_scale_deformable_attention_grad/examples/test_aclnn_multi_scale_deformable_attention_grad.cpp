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
#include "aclnnop/aclnn_multi_scale_deformable_attention_grad.h"

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

    std::vector<int64_t> valueShape = {1, 1, 1, 8};
    std::vector<int64_t> spatialShapeShape = {1, 2};
    std::vector<int64_t> levelStartIndexShape = {1};
    std::vector<int64_t> locationShape = {1, 32, 1, 1, 1, 2};
    std::vector<int64_t> attnWeightShape = {1, 32, 1, 1, 1};
    std::vector<int64_t> gradOutputShape = {1, 32, 1, 8};
    std::vector<int64_t> gradValueShape = {1, 1, 1, 8};
    std::vector<int64_t> gradLocationShape = {1, 32, 1, 1, 1, 2};
    std::vector<int64_t> gradAttnWeightShape = {1, 32, 1, 1, 1};
    void* valueDeviceAddr = nullptr;
    void* spatialShapeDeviceAddr = nullptr;
    void* levelStartIndexDeviceAddr = nullptr;
    void* locationDeviceAddr = nullptr;
    void* attnWeightDeviceAddr = nullptr;
    void* gradOutputDeviceAddr = nullptr;
    void* gradValueDeviceAddr = nullptr;
    void* gradLocationDeviceAddr = nullptr;
    void* gradAttnWeightDeviceAddr = nullptr;
    aclTensor* value = nullptr;
    aclTensor* spatialShape = nullptr;
    aclTensor* levelStartIndex = nullptr;
    aclTensor* location = nullptr;
    aclTensor* attnWeight = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* gradValue = nullptr;
    aclTensor* gradLocation = nullptr;
    aclTensor* gradAttnWeight = nullptr;
    std::vector<float> valueHostData = {1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> spatialShapeHostData = {1, 1};
    std::vector<float> levelStartIndexHostData = {0};
    std::vector<float> gradValueHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> gradLocationHostData(GetShapeSize(gradLocationShape), 0);
    std::vector<float> gradAttnWeightHostData(GetShapeSize(gradAttnWeightShape), 0);
    std::vector<float> locationHostData(GetShapeSize(locationShape), 0);
    std::vector<float> attnWeightHostData(GetShapeSize(attnWeightShape), 1);
    std::vector<float> gradOutputHostData(GetShapeSize(gradOutputShape), 1);

    // value aclTensor
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建spatialShape aclTensor
    ret = CreateAclTensor(spatialShapeHostData, spatialShapeShape, &spatialShapeDeviceAddr, aclDataType::ACL_INT32, &spatialShape);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建levelStartIndex aclTensor
    ret = CreateAclTensor(levelStartIndexHostData, levelStartIndexShape, &levelStartIndexDeviceAddr, aclDataType::ACL_INT32, &levelStartIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建location aclTensor
    ret = CreateAclTensor(locationHostData, locationShape, &locationDeviceAddr, aclDataType::ACL_FLOAT, &location);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建attnWeight aclTensor
    ret = CreateAclTensor(attnWeightHostData, attnWeightShape, &attnWeightDeviceAddr, aclDataType::ACL_FLOAT, &attnWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // gradOutput aclTensor
    ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradValue aclTensor
    ret = CreateAclTensor(gradValueHostData, gradValueShape, &gradValueDeviceAddr, aclDataType::ACL_FLOAT, &gradValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradLocation aclTensor
    ret = CreateAclTensor(gradLocationHostData, gradLocationShape, &gradLocationDeviceAddr, aclDataType::ACL_FLOAT, &gradLocation);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradAttnWeight aclTensor
    ret = CreateAclTensor(gradAttnWeightHostData, gradAttnWeightShape, &gradAttnWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradAttnWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3.调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnMultiScaleDeformableAttentionGrad第一段接口
    ret = aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize(value, spatialShape, levelStartIndex, location, attnWeight, gradOutput, gradValue, gradLocation, gradAttnWeight, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnMultiScaleDeformableAttentionGrad第二段接口
    ret = aclnnMultiScaleDeformableAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultiScaleDeformableAttentionGrad failed. ERROR: %d\n", ret); return ret);
    // 4.(固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(gradValueShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradValueDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6.释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(value);
    aclDestroyTensor(spatialShape);
    aclDestroyTensor(levelStartIndex);
    aclDestroyTensor(location);
    aclDestroyTensor(attnWeight);
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(gradValue);
    aclDestroyTensor(gradLocation);
    aclDestroyTensor(gradAttnWeight);

    // 7.释放device资源，需要根据具体API的接口定义修改
    aclrtFree(valueDeviceAddr);
    aclrtFree(spatialShapeDeviceAddr);
    aclrtFree(levelStartIndexDeviceAddr);
    aclrtFree(locationDeviceAddr);
    aclrtFree(attnWeightDeviceAddr);
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(gradValueDeviceAddr);
    aclrtFree(gradLocationDeviceAddr);
    aclrtFree(gradAttnWeightDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}