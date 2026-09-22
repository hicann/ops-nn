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
#include "aclnnop/aclnn_swiglu_backward_group_quant_with_dual_axis.h"

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

    std::vector<int64_t> gradYShape = {128, 64};
    std::vector<int64_t> xShape = {128, 128};
    std::vector<int64_t> weightShape = {128};
    std::vector<int64_t> groupIndexShape = {2};
    std::vector<int64_t> y1Shape = {128, 128};
    std::vector<int64_t> scale1Shape = {128, 2, 2};
    std::vector<int64_t> scale2Shape = {4, 128, 2};

    void* gradYDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* yOriginDeviceAddr = nullptr;
    void* groupIndexDeviceAddr = nullptr;
    void* y1DeviceAddr = nullptr;
    void* scale1DeviceAddr = nullptr;
    void* y2DeviceAddr = nullptr;
    void* scale2DeviceAddr = nullptr;
    void* gradWeightDeviceAddr = nullptr;

    aclTensor* gradYTensor = nullptr;
    aclTensor* xTensor = nullptr;
    aclTensor* weightTensor = nullptr;
    aclTensor* yOriginTensor = nullptr;
    aclTensor* groupIndexTensor = nullptr;
    aclTensor* y1Tensor = nullptr;
    aclTensor* scale1Tensor = nullptr;
    aclTensor* y2Tensor = nullptr;
    aclTensor* scale2Tensor = nullptr;
    aclTensor* gradWeightTensor = nullptr;

    int64_t gradYSize = GetShapeSize(gradYShape);
    std::vector<aclFloat16> gradYHostData(gradYSize, aclFloatToFloat16(0.0f));
    for (int64_t i = 0; i < gradYSize; i++) {
        gradYHostData[i] = aclFloatToFloat16(static_cast<float>(i % 10) * 0.1f);
    }

    int64_t xSize = GetShapeSize(xShape);
    std::vector<aclFloat16> xHostData(xSize, aclFloatToFloat16(0.0f));
    for (int64_t i = 0; i < xSize; i++) {
        xHostData[i] = aclFloatToFloat16(static_cast<float>((i % 20) - 10) * 0.5f);
    }

    int64_t weightSize = GetShapeSize(weightShape);
    std::vector<float> weightHostData(weightSize, 0.0f);
    for (int64_t i = 0; i < weightSize; i++) {
        weightHostData[i] = static_cast<float>((i % 5) + 1) * 0.2f;
    }

    int64_t yOriginSize = GetShapeSize(gradYShape);
    std::vector<aclFloat16> yOriginHostData(yOriginSize, aclFloatToFloat16(0.0f));
    for (int64_t i = 0; i < yOriginSize; i++) {
        yOriginHostData[i] = aclFloatToFloat16(static_cast<float>((i % 8) + 1) * 0.3f);
    }

    int64_t groupIndexSize = GetShapeSize(groupIndexShape);
    std::vector<int64_t> groupIndexHostData(groupIndexSize, 0);
    int64_t groupStride = gradYShape[0] / groupIndexSize;
    for (int64_t i = 0; i < groupIndexSize; i++) {
        groupIndexHostData[i] = (i + 1) * groupStride;
    }

    ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT16, &gradYTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &xTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weightTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(yOriginHostData, gradYShape, &yOriginDeviceAddr, aclDataType::ACL_FLOAT16, &yOriginTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(groupIndexHostData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT64,
                          &groupIndexTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorWithValue<uint8_t>(y1Shape, &y1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y1Tensor, 0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorWithValue<uint8_t>(scale1Shape, &scale1DeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale1Tensor,
                                            0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorWithValue<uint8_t>(y1Shape, &y2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y2Tensor, 0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorWithValue<uint8_t>(scale2Shape, &scale2DeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale2Tensor,
                                            0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorWithValue<float>(weightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeightTensor,
                                          0.0f);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    double clampLimit = -1.0f;
    double alpha = 1.702f;
    double bias = 0.0f;
    int64_t quantMode = 1;
    int64_t dstType = 36;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnSwigluBackwardGroupQuantWithDualAxisGetWorkspaceSize(
        gradYTensor, xTensor, weightTensor, yOriginTensor, groupIndexTensor, clampLimit, alpha, bias, quantMode,
        dstType, y1Tensor, scale1Tensor, y2Tensor, scale2Tensor, gradWeightTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnSwigluBackwardGroupQuantWithDualAxisGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnSwigluBackwardGroupQuantWithDualAxis(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluBackwardGroupQuantWithDualAxis failed. ERROR: %d\n", ret);
              return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto y1ResultSize = GetShapeSize(y1Shape);
    std::vector<uint8_t> y1ResultData(y1ResultSize, 0);
    ret = aclrtMemcpy(y1ResultData.data(), y1ResultData.size() * sizeof(uint8_t), y1DeviceAddr,
                      y1ResultSize * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y1 result from device to host failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("y1 output (first 10 elements):\n");
    for (int64_t i = 0; i < 10 && i < y1ResultSize; i++) {
        LOG_PRINT("y1[%ld] = %d\n", i, static_cast<int>(y1ResultData[i]));
    }

    auto scale1ResultSize = GetShapeSize(scale1Shape);
    std::vector<uint8_t> scale1ResultData(scale1ResultSize, 0);
    ret = aclrtMemcpy(scale1ResultData.data(), scale1ResultData.size() * sizeof(uint8_t), scale1DeviceAddr,
                      scale1ResultSize * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale1 result from device to host failed. ERROR: %d\n", ret);
              return ret);

    LOG_PRINT("scale1 output (first 10 elements):\n");
    for (int64_t i = 0; i < 10 && i < scale1ResultSize; i++) {
        LOG_PRINT("scale1[%ld] = %d\n", i, static_cast<int>(scale1ResultData[i]));
    }

    auto y2ResultSize = GetShapeSize(y1Shape);
    std::vector<uint8_t> y2ResultData(y2ResultSize, 0);
    ret = aclrtMemcpy(y2ResultData.data(), y2ResultData.size() * sizeof(uint8_t), y2DeviceAddr,
                      y2ResultSize * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y2 result from device to host failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("y2 output (first 10 elements):\n");
    for (int64_t i = 0; i < 10 && i < y2ResultSize; i++) {
        LOG_PRINT("y2[%ld] = %d\n", i, static_cast<int>(y2ResultData[i]));
    }

    auto scale2ResultSize = GetShapeSize(scale2Shape);
    std::vector<uint8_t> scale2ResultData(scale2ResultSize, 0);
    ret = aclrtMemcpy(scale2ResultData.data(), scale2ResultData.size() * sizeof(uint8_t), scale2DeviceAddr,
                      scale2ResultSize * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale2 result from device to host failed. ERROR: %d\n", ret);
              return ret);

    LOG_PRINT("scale2 output (first 10 elements):\n");
    for (int64_t i = 0; i < 10 && i < scale2ResultSize; i++) {
        LOG_PRINT("scale2[%ld] = %d\n", i, static_cast<int>(scale2ResultData[i]));
    }

    auto gradWeightResultSize = GetShapeSize(weightShape);
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
    aclDestroyTensor(y1Tensor);
    aclDestroyTensor(scale1Tensor);
    aclDestroyTensor(y2Tensor);
    aclDestroyTensor(scale2Tensor);
    aclDestroyTensor(gradWeightTensor);

    aclrtFree(gradYDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(yOriginDeviceAddr);
    aclrtFree(groupIndexDeviceAddr);
    aclrtFree(y1DeviceAddr);
    aclrtFree(scale1DeviceAddr);
    aclrtFree(y2DeviceAddr);
    aclrtFree(scale2DeviceAddr);
    aclrtFree(gradWeightDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
