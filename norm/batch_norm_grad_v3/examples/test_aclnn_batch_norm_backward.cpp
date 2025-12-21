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
#include "aclnnop/aclnn_batch_norm_backward.h"

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
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> gradOutShape = {1, 2, 4};
    std::vector<int64_t> selfShape = {1, 2, 4};
    std::vector<int64_t> weightShape = {2};
    std::vector<int64_t> rMeanShape = {2};
    std::vector<int64_t> rVarShape = {2};
    std::vector<int64_t> sMeanShape = {2};
    std::vector<int64_t> sVarShape = {2};
    std::vector<int64_t> gradInShape = {1, 2, 4};
    std::vector<int64_t> gradWeightShape = {2};
    std::vector<int64_t> gradBiasShape = {2};
    void* gradOutDeviceAddr = nullptr;
    void* selfDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* rMeanDeviceAddr = nullptr;
    void* rVarDeviceAddr = nullptr;
    void* sMeanDeviceAddr = nullptr;
    void* sVarDeviceAddr = nullptr;
    void* outMaskDeviceAddr = nullptr;
    void* gradInDeviceAddr = nullptr;
    void* gradWeightDeviceAddr = nullptr;
    void* gradBiasDeviceAddr = nullptr;
    aclTensor* gradOut = nullptr;
    aclTensor* self = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* rMean = nullptr;
    aclTensor* rVar = nullptr;
    aclTensor* sMean = nullptr;
    aclTensor* sVar = nullptr;
    aclBoolArray* outMask = nullptr;
    aclTensor* gradIn = nullptr;
    aclTensor* gradWeight = nullptr;
    aclTensor* gradBias = nullptr;
    std::vector<float> gradOutHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> weightHostData = {1, 1};
    std::vector<float> rMeanHostData = {0, 0};
    std::vector<float> rVarHostData = {1, 1};
    std::vector<float> sMeanHostData = {0, 0};
    std::vector<float> sVarHostData = {1, 1};
    std::vector<float> gradInHostData(8, 0);
    std::vector<float> gradWeightHostData(2, 0);
    std::vector<float> gradBiasHostData(2, 0);
    ;
    bool training = true;
    double eps = 1e-5;
    // 创建gradOut aclTensor
    ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rMean aclTensor
    ret = CreateAclTensor(rMeanHostData, rMeanShape, &rMeanDeviceAddr, aclDataType::ACL_FLOAT, &rMean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rVar aclTensor
    ret = CreateAclTensor(rVarHostData, rVarShape, &rVarDeviceAddr, aclDataType::ACL_FLOAT, &rVar);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建sMean aclTensor
    ret = CreateAclTensor(sMeanHostData, sMeanShape, &sMeanDeviceAddr, aclDataType::ACL_FLOAT, &sMean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建sVar aclTensor
    ret = CreateAclTensor(sVarHostData, sVarShape, &sVarDeviceAddr, aclDataType::ACL_FLOAT, &sVar);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建outMask aclBoolArray
    bool maskData[3] = {true, true, true};
    outMask = aclCreateBoolArray(&(maskData[0]), 3);
    // 创建gradIn aclTensor
    ret = CreateAclTensor(gradInHostData, gradInShape, &gradInDeviceAddr, aclDataType::ACL_FLOAT, &gradIn);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradWeight aclTensor
    ret = CreateAclTensor(
        gradWeightHostData, gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradBias aclTensor
    ret = CreateAclTensor(gradBiasHostData, gradBiasShape, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnBatchNormBackward接口调用示例
    // 3. 调用CANN算子库API，需要修改为具体的API名称
    // 调用aclnnBatchNormBackward第一段接口
    ret = aclnnBatchNormBackwardGetWorkspaceSize(
        gradOut, self, weight, rMean, rVar, sMean, sVar, training, eps, outMask, gradIn, gradWeight, gradBias,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnBatchNormBackward第二段接口
    ret = aclnnBatchNormBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormBackward failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(gradInShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), gradInDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(gradOut);
    aclDestroyTensor(self);
    aclDestroyTensor(weight);
    aclDestroyTensor(rMean);
    aclDestroyTensor(rVar);
    aclDestroyTensor(sMean);
    aclDestroyTensor(sVar);
    aclDestroyBoolArray(outMask);
    aclDestroyTensor(gradIn);
    aclDestroyTensor(gradWeight);
    aclDestroyTensor(gradBias);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(gradOutDeviceAddr);
    aclrtFree(selfDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(rMeanDeviceAddr);
    aclrtFree(rVarDeviceAddr);
    aclrtFree(sMeanDeviceAddr);
    aclrtFree(sVarDeviceAddr);
    aclrtFree(outMaskDeviceAddr);
    aclrtFree(gradInDeviceAddr);
    aclrtFree(gradWeightDeviceAddr);
    aclrtFree(gradBiasDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}