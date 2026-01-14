/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file test_aclnn_extend_conv2d.cpp
 * \brief
 */

#include <iostream>
#include <memory>
#include <vector>
#include <bitset>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_convolution.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) 
{
    int64_t shape_size = 1;
    for (auto i: shape) {
        shape_size *= i;
    }
    return shape_size;
}

float Float16BitsToFloat(uint16_t float16Bits) 
{
    uint32_t sign = (float16Bits & 0x8000) << 16;
    uint32_t exp16 = (float16Bits & 0x7C00) >> 10;
    uint32_t mantissa16 = float16Bits & 0x03FF;

    if (exp16 == 0 && mantissa16 == 0) {
        return 0.0f;
    }
    if (exp16 == 0x1F && mantissa16 == 0) {
        return sign ? -INFINITY : INFINITY;
    }
    if (exp16 == 0x1F && mantissa16 != 0) {
        return NAN;
    }

    uint32_t exp32 = exp16 - 15 + 127;
    uint32_t mantissa32 = mantissa16 << 13;
    uint32_t float32Bits = sign | (exp32 << 23) | mantissa32;

    return *reinterpret_cast<float*>(&float32Bits);
}

uint64_t MaskFloat32(float deqScale) {
    uint32_t deqScaleUint32 = *reinterpret_cast<uint32_t*>(&deqScale);
    uint32_t maskedUint32 = deqScaleUint32 & 0xffffe000;
    uint64_t result = static_cast<uint64_t>(maskedUint32);
    
    return result;
}

std::vector<uint64_t> ConvertScaleData(const std::vector<float>& scaleData) {
    std::vector<uint64_t> scaleDataUint64;
    scaleDataUint64.reserve(scaleData.size());
    
    for (size_t i = 0; i < scaleData.size(); ++i) {
        uint64_t convertedVal = MaskFloat32(scaleData[i]);
        scaleDataUint64.push_back(convertedVal);
    }
    
    return scaleDataUint64;
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
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW, 
        shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorND(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, 
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用 aclrtMalloc 申请 device 侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用 aclrtMemcpy 将 host 侧数据拷贝到 device 侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream& stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnQuantConvolutionTest(int32_t deviceId, aclrtStream& stream, std::vector<aclDataType> dtypesInfo)
{
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据 API 的接口自定义构造
    std::vector<int64_t> shapeInput = {1, 2, 8, 8};
    std::vector<int64_t> shapeWeight = {3, 2, 3, 3};
    std::vector<int64_t> shapeScale = {3};
    std::vector<int64_t> shapeBias = {3};
    std::vector<int64_t> shapeResult = {1, 3, 8, 8};
    std::vector<int64_t> convStrides = {1, 1};
    std::vector<int64_t> convPads = {1, 1};
    std::vector<int64_t> convOutPads = {0, 0};
    std::vector<int64_t> convDilations = {1, 1};

    void* deviceDataA = nullptr;
    void* deviceDataB = nullptr;
    void* deviceDataScale = nullptr;
    void* deviceDataBias = nullptr;
    void* deviceDataResult = nullptr;

    aclTensor* input = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* scale= nullptr;
    aclTensor* bias= nullptr;
    aclTensor* result = nullptr;
    std::vector<int8_t> inputData(GetShapeSize(shapeInput), 1);
    std::vector<int8_t> weightData(GetShapeSize(shapeWeight), 1);
    std::vector<int32_t> biasData(GetShapeSize(shapeBias), 1);
    std::vector<float> scaleData(GetShapeSize(shapeScale), 1);
    std::vector<uint16_t> outputData(GetShapeSize(shapeResult), 1);
    aclDataType inputDtype = dtypesInfo[0];
    aclDataType weightDtype = dtypesInfo[1];
    aclDataType biasDtype = dtypesInfo[2];
    aclDataType scaleDtype = dtypesInfo[3];
    aclDataType outputDtype = dtypesInfo[4];
    // 创建input aclTensor
    ret = CreateAclTensor(inputData, shapeInput, &deviceDataA, inputDtype, &input);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(input, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> deviceDataAPtr(deviceDataA, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建weight aclTensor
    ret = CreateAclTensor(weightData, shapeWeight, &deviceDataB, weightDtype, &weight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> deviceDataBPtr(deviceDataB, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建scale，此处转换float的scale放入uint64的Tensor
    ret = CreateAclTensorND(ConvertScaleData(scaleData), shapeScale, &deviceDataScale, scaleDtype, &scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> deviceDataScalePtr(deviceDataScale, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建bias
    ret = CreateAclTensorND(biasData, shapeBias, &deviceDataBias, biasDtype, &bias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> deviceDataBiasPtr(deviceDataBias, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建out aclTensor
    ret = CreateAclTensor(outputData, shapeResult, &deviceDataResult, outputDtype, &result);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(result, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> deviceDataResultPtr(deviceDataResult, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    aclIntArray *strides = aclCreateIntArray(convStrides.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> stridesPtr(strides, aclDestroyIntArray);
    CHECK_FREE_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);
    aclIntArray *pads = aclCreateIntArray(convPads.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> padsPtr(pads, aclDestroyIntArray);
    CHECK_FREE_RET(pads != nullptr, return ACL_ERROR_INTERNAL_ERROR);
    aclIntArray *outPads = aclCreateIntArray(convOutPads.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> outPadsPtr(outPads, aclDestroyIntArray);
    CHECK_FREE_RET(outPads != nullptr, return ACL_ERROR_INTERNAL_ERROR);
    aclIntArray *dilations = aclCreateIntArray(convDilations.data(), 2);
    std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> dilationsPtr(dilations, aclDestroyIntArray);
    CHECK_FREE_RET(dilations != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 3. 调用 CANN 算子库 API，需要修改为具体的 API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnConvolution第一段接口
    ret = aclnnQuantConvolutionGetWorkspaceSize(input, weight, bias, scale, nullptr, strides, pads, dilations,
                                                false, outPads, 1, 0, nullptr, result, &workspaceSize, &executor);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantConvolutionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnConvolution第二段接口
    ret = aclnnQuantConvolution(workspaceAddr, workspaceSize, executor, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantConvolution failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将 device 侧内存上的结果拷贝至 host 侧，需要根据具体 API 的接口定义修改
    auto size = GetShapeSize(shapeResult);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceDataResult,
                        size * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    int64_t printSize = size > 10 ? 10 : size;
    for (int64_t i = 0; i < printSize; i++) {
        LOG_PRINT("result[%ld] is: %.4f\n", i, Float16BitsToFloat(resultData[i]));
    }

    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    std::vector<aclDataType> dtypesInfo = {aclDataType::ACL_INT8, aclDataType::ACL_INT8, aclDataType::ACL_INT32,
        aclDataType::ACL_INT64, aclDataType::ACL_FLOAT16}; // 分别是input/weight/bias/scale/output的datatype
    auto ret = aclnnQuantConvolutionTest(deviceId, stream, dtypesInfo);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantConvolutionTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}