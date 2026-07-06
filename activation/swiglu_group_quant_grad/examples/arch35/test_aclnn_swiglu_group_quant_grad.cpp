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
#include "aclnnop/aclnn_swiglu_group_quant_grad.h"

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

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorWithValue(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                             aclTensor** tensor, T value)
{
    int64_t shapeSize = GetShapeSize(shape);
    std::vector<T> hostData(shapeSize, value);
    return CreateAclTensor(hostData, shape, deviceAddr, dataType, tensor);
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> gradYShape = {512, 512};
    std::vector<int64_t> xShape = {512, 1024};
    std::vector<int64_t> weightShape = {512, 1};
    std::vector<int64_t> yOriginShape = {512, 512};
    std::vector<int64_t> groupIndexShape = {256};
    std::vector<int64_t> gradXShape = {512, 1024};
    std::vector<int64_t> gradWeightShape = {512, 1};

    void* gradYDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* yOriginDeviceAddr = nullptr;
    void* groupIndexDeviceAddr = nullptr;
    void* gradXDeviceAddr = nullptr;
    void* gradWeightDeviceAddr = nullptr;

    aclTensor* gradYTensor = nullptr;
    aclTensor* xTensor = nullptr;
    aclTensor* weightTensor = nullptr;
    aclTensor* yOriginTensor = nullptr;
    aclTensor* groupIndexTensor = nullptr;
    aclTensor* gradXTensor = nullptr;
    aclTensor* gradWeightTensor = nullptr;

    int64_t gradYSize = GetShapeSize(gradYShape);
    std::vector<float> gradYHostData(gradYSize, 1.0f);
    for (int64_t i = 0; i < gradYSize; i++) {
        gradYHostData[i] = static_cast<float>(i % 10) * 0.1f;
    }

    int64_t xSize = GetShapeSize(xShape);
    std::vector<float> xHostData(xSize, 1.0f);
    for (int64_t i = 0; i < xSize; i++) {
        xHostData[i] = static_cast<float>((i % 20) - 10) * 0.5f;
    }

    int64_t weightSize = GetShapeSize(weightShape);
    std::vector<float> weightHostData(weightSize, 1.0f);
    for (int64_t i = 0; i < weightSize; i++) {
        weightHostData[i] = static_cast<float>((i % 5) + 1) * 0.2f;
    }

    int64_t yOriginSize = GetShapeSize(yOriginShape);
    std::vector<float> yOriginHostData(yOriginSize, 1.0f);
    for (int64_t i = 0; i < yOriginSize; i++) {
        yOriginHostData[i] = static_cast<float>((i % 8) + 1) * 0.3f;
    }

    int64_t groupIndexSize = GetShapeSize(groupIndexShape);
    std::vector<int64_t> groupIndexHostData(groupIndexSize, 0);
    int64_t groupStride = 512 / 256;
    for (int64_t i = 0; i < groupIndexSize; i++) {
        groupIndexHostData[i] = i * groupStride;
    }

    ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT16, &gradYTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &xTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weightTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(yOriginHostData, yOriginShape, &yOriginDeviceAddr, aclDataType::ACL_FLOAT16, &yOriginTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(groupIndexHostData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT64,
                          &groupIndexTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorWithValue<float>(gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT16, &gradXTensor, 0.0f);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorWithValue<float>(gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT,
                                          &gradWeightTensor, 0.0f);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    float clampLimit = 1.0f;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnSwigluGroupQuantGradGetWorkspaceSize(gradYTensor, xTensor, weightTensor, yOriginTensor, groupIndexTensor,
                                                    clampLimit, gradXTensor, gradWeightTensor, &workspaceSize,
                                                    &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupQuantGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnSwigluGroupQuantGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupQuantGrad failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto gradXResultSize = GetShapeSize(gradXShape);
    std::vector<float> gradXResultData(gradXResultSize, 0);
    ret = aclrtMemcpy(gradXResultData.data(), gradXResultData.size() * sizeof(float), gradXDeviceAddr,
                      gradXResultSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradX result from device to host failed. ERROR: %d\n", ret);
              return ret);

    LOG_PRINT("gradX output (first 10 elements):\n");
    for (int64_t i = 0; i < 10 && i < gradXResultSize; i++) {
        LOG_PRINT("gradX[%ld] = %f\n", i, gradXResultData[i]);
    }

    auto gradWeightResultSize = GetShapeSize(gradWeightShape);
    std::vector<float> gradWeightResultData(gradWeightResultSize, 0);
    ret = aclrtMemcpy(gradWeightResultData.data(), gradWeightResultData.size() * sizeof(float), gradWeightDeviceAddr,
                      gradWeightResultSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradWeight result from device to host failed. ERROR: %d\n", ret);
              return ret);

    LOG_PRINT("gradWeight output (first 10 elements):\n");
    for (int64_t i = 0; i < 10 && i < gradWeightResultSize; i++) {
        LOG_PRINT("gradWeight[%ld] = %f\n", i, gradWeightResultData[i]);
    }

    aclDestroyTensor(gradYTensor);
    aclDestroyTensor(xTensor);
    aclDestroyTensor(weightTensor);
    aclDestroyTensor(yOriginTensor);
    aclDestroyTensor(groupIndexTensor);
    aclDestroyTensor(gradXTensor);
    aclDestroyTensor(gradWeightTensor);

    aclrtFree(gradYDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(yOriginDeviceAddr);
    aclrtFree(groupIndexDeviceAddr);
    aclrtFree(gradXDeviceAddr);
    aclrtFree(gradWeightDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}