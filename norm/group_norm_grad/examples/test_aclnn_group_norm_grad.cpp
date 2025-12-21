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
#include "aclnnop/aclnn_group_norm_backward.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，acl初始化
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
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> gradOutShape = {2, 3, 4};
    std::vector<int64_t> inputShape = {2, 3, 4};
    std::vector<int64_t> meanShape = {2, 1};
    std::vector<int64_t> rstdShape = {2, 1};
    std::vector<int64_t> gammaShape = {3};
    std::vector<int64_t> gradInputShape = {2, 3, 4};
    std::vector<int64_t> gradGammaOutShape = {3};
    std::vector<int64_t> gradBetaOutShape = {3};
    void* gradOutDeviceAddr = nullptr;
    void* inputDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    void* gradGammaOutDeviceAddr = nullptr;
    void* gradBetaOutDeviceAddr = nullptr;
    aclTensor* gradOut = nullptr;
    aclTensor* input = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* gradInput = nullptr;
    aclTensor* gradGammaOut = nullptr;
    aclTensor* gradBetaOut = nullptr;
    std::vector<float> gradOutHostData = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    std::vector<float> inputHostData = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                                        13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    std::vector<float> meanHostData = {6.5, 18.5};
    std::vector<float> rstdHostData = {0.2896827, 0.2896827};
    std::vector<float> gammaHostData = {1.0, 1.0, 1.0};
    std::vector<float> gradInputHostData = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> gradGammaOutHostData = {0.0, 0.0, 0.0};
    std::vector<float> gradBetaOutHostData = {0.0, 0.0, 0.0};
    int64_t N = 2;
    int64_t C = 3;
    int64_t HxW = 4;
    int64_t group = 1;
    std::array<bool, 3> outputMaskData = {true, true, true};
    // 创建gradOut aclTensor
    ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建input aclTensor
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建mean aclTensor
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rstd aclTensor
    ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gamma aclTensor
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    auto outputMask = aclCreateBoolArray(outputMaskData.data(), outputMaskData.size());
    CHECK_RET(outputMask != nullptr, return ACL_ERROR_INTERNAL_ERROR);
    // 创建gradInput aclTensor
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradGammaOut aclTensor
    ret = CreateAclTensor(
        gradGammaOutHostData, gradGammaOutShape, &gradGammaOutDeviceAddr, aclDataType::ACL_FLOAT, &gradGammaOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradBetaOut aclTensor
    ret = CreateAclTensor(
        gradBetaOutHostData, gradBetaOutShape, &gradBetaOutDeviceAddr, aclDataType::ACL_FLOAT, &gradBetaOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的HostApi
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnGroupNormBackward第一段接口
    ret = aclnnGroupNormBackwardGetWorkspaceSize(
        gradOut, input, mean, rstd, gamma, N, C, HxW, group, outputMask, gradInput, gradGammaOut, gradBetaOut,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnGroupNormBackward第二段接口
    ret = aclnnGroupNormBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormBackward failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> gradInputResultData(size, 0);
    ret = aclrtMemcpy(
        gradInputResultData.data(), gradInputResultData.size() * sizeof(gradInputResultData[0]), gradInputDeviceAddr,
        size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradInputResultData[%ld] is: %f\n", i, gradInputResultData[i]);
    }

    size = GetShapeSize(gradGammaOutShape);
    std::vector<float> gradGammaOutResultData(size, 0);
    ret = aclrtMemcpy(
        gradGammaOutResultData.data(), gradGammaOutResultData.size() * sizeof(gradGammaOutResultData[0]),
        gradGammaOutDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradGammaOutResultData[%ld] is: %f\n", i, gradGammaOutResultData[i]);
    }

    size = GetShapeSize(gradBetaOutShape);
    std::vector<float> gradBetaOutResultData(size, 0);
    ret = aclrtMemcpy(
        gradBetaOutResultData.data(), gradBetaOutResultData.size() * sizeof(gradBetaOutResultData[0]),
        gradBetaOutDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradBetaOutResultData[%ld] is: %f\n", i, gradBetaOutResultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(gradOut);
    aclDestroyTensor(input);
    aclDestroyTensor(mean);
    aclDestroyTensor(rstd);
    aclDestroyTensor(gamma);
    aclDestroyTensor(gradInput);
    aclDestroyTensor(gradGammaOut);
    aclDestroyTensor(gradBetaOut);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(gradOutDeviceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(rstdDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    aclrtFree(gradGammaOutDeviceAddr);
    aclrtFree(gradBetaOutDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}