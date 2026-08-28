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
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_group_grad.h"

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
    for (auto dim : shape) {
        shapeSize *= dim;
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
    return ACL_SUCCESS;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
}

bool CheckHardwareSupport()
{
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) {
        LOG_PRINT("Warning: Cannot get SOC name, skip hardware check\n");
        return true;
    }

    LOG_PRINT("Current SOC: %s\n", socName);
    if (strstr(socName, "Ascend950") != nullptr || strstr(socName, "ascend950") != nullptr) {
        return true;
    }

    LOG_PRINT("Warning: SwigluGroupGrad only supports Ascend950, current SOC '%s' is not supported. Skip test.\n",
              socName);
    return false;
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

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    if (!CheckHardwareSupport()) {
        LOG_PRINT("\n=== Test SKIPPED (hardware not supported) ===\n");
        Finalize(deviceId, stream);
        return ACL_SUCCESS;
    }

    // 参数设置（对齐UT测试 l2_normal_FLOAT_ND_with_all_options）
    int64_t T = 4;
    int64_t H = 16;
    int64_t twoH = 2 * H;
    float clampLimit = 3.0f;

    std::vector<int64_t> gradYShape = {T, H};
    std::vector<int64_t> xShape = {T, twoH};
    std::vector<int64_t> weightShape = {T, 1};
    std::vector<int64_t> yOriginShape = {T, H};
    std::vector<int64_t> groupIndexShape = {1};
    std::vector<int64_t> gradXOutShape = {T, twoH};
    std::vector<int64_t> gradWeightOutShape = {T, 1};

    void* gradYDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* yOriginDeviceAddr = nullptr;
    void* groupIndexDeviceAddr = nullptr;
    void* gradXOutDeviceAddr = nullptr;
    void* gradWeightOutDeviceAddr = nullptr;

    aclTensor* gradY = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* yOrigin = nullptr;
    aclTensor* groupIndex = nullptr;
    aclTensor* gradXOut = nullptr;
    aclTensor* gradWeightOut = nullptr;

    // 创建 gradY tensor (FLOAT)
    std::vector<float> hostGradY(GetShapeSize(gradYShape), 1.0f);
    ret = CreateAclTensor(hostGradY, gradYShape, &gradYDeviceAddr, ACL_FLOAT, &gradY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 x tensor (FLOAT)
    std::vector<float> hostX(GetShapeSize(xShape), 0.5f);
    ret = CreateAclTensor(hostX, xShape, &xDeviceAddr, ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 weight tensor (FLOAT)
    std::vector<float> hostWeight(GetShapeSize(weightShape), 1.0f);
    ret = CreateAclTensor(hostWeight, weightShape, &weightDeviceAddr, ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 yOrigin tensor (FLOAT)
    std::vector<float> hostYOrigin(GetShapeSize(yOriginShape), 0.3f);
    ret = CreateAclTensor(hostYOrigin, yOriginShape, &yOriginDeviceAddr, ACL_FLOAT, &yOrigin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 groupIndex tensor (INT64)
    std::vector<int64_t> hostGroupIndex = {T};
    ret = CreateAclTensor(hostGroupIndex, groupIndexShape, &groupIndexDeviceAddr, ACL_INT64, &groupIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 gradXOut output tensor (FLOAT)
    std::vector<float> hostGradXOut(GetShapeSize(gradXOutShape), 0.0f);
    ret = CreateAclTensor(hostGradXOut, gradXOutShape, &gradXOutDeviceAddr, ACL_FLOAT, &gradXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 gradWeightOut output tensor (FLOAT)
    std::vector<float> hostGradWeightOut(GetShapeSize(gradWeightOutShape), 0.0f);
    ret = CreateAclTensor(hostGradWeightOut, gradWeightOutShape, &gradWeightOutDeviceAddr, ACL_FLOAT, &gradWeightOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 1. 获取 workspace 大小
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    ret = aclnnSwigluGroupGradGetWorkspaceSize(gradY, x, weight, yOrigin, groupIndex, clampLimit, gradXOut,
                                               gradWeightOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 2. 申请 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 3. 执行计算
    ret = aclnnSwigluGroupGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupGrad failed. ERROR: %d\n", ret); return ret);

    // 4. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出结果
    auto gradXOutSize = GetShapeSize(gradXOutShape);
    std::vector<float> gradXOutData(gradXOutSize, 0.0f);
    ret = aclrtMemcpy(gradXOutData.data(), gradXOutData.size() * sizeof(float), gradXOutDeviceAddr,
                      gradXOutSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradXOut from device to host failed. ERROR: %d\n", ret); return ret);

    auto gradWeightOutSize = GetShapeSize(gradWeightOutShape);
    std::vector<float> gradWeightOutData(gradWeightOutSize, 0.0f);
    ret = aclrtMemcpy(gradWeightOutData.data(), gradWeightOutData.size() * sizeof(float), gradWeightOutDeviceAddr,
                      gradWeightOutSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradWeightOut from device to host failed. ERROR: %d\n", ret);
              return ret);

    // 打印前10个结果
    for (int64_t i = 0; i < 10 && i < gradXOutSize; i++) {
        LOG_PRINT("gradXOut[%ld] is: %f\n", i, gradXOutData[i]);
    }
    for (int64_t i = 0; i < 10 && i < gradWeightOutSize; i++) {
        LOG_PRINT("gradWeightOut[%ld] is: %f\n", i, gradWeightOutData[i]);
    }

    // 6. 释放资源
    aclDestroyTensor(gradY);
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(yOrigin);
    aclDestroyTensor(groupIndex);
    aclDestroyTensor(gradXOut);
    aclDestroyTensor(gradWeightOut);

    aclrtFree(gradYDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(yOriginDeviceAddr);
    aclrtFree(groupIndexDeviceAddr);
    aclrtFree(gradXOutDeviceAddr);
    aclrtFree(gradWeightOutDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}
