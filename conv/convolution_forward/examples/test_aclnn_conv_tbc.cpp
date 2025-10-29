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
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_convolution.h"

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
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
    aclTensor** tensor, aclFormat dataFormat)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用 aclrtMalloc 申请 device 侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用 aclrtMemcpy 将 host 侧数据拷贝到 device 侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续 tensor 的 strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用 aclCreateTensor 接口创建 aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, dataFormat, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream& stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnConvTbcTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    // check 根据自己的需要处理
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据 API 的接口自定义构造
    std::vector<int64_t> shapeInput = {2, 2, 3};
    std::vector<int64_t> shapeWeight = {3, 3, 4};
    std::vector<int64_t> shapeBias = {4};
    std::vector<int64_t> shapeResult = {2, 2, 4};

    void* deviceDataA = nullptr;
    void* deviceDataB = nullptr;
    void* deviceDataC = nullptr;
    void* deviceDataResult = nullptr;

    aclTensor* input = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* result = nullptr;
    std::vector<float> inputData(GetShapeSize(shapeInput), 1);
    std::vector<float> weightData(GetShapeSize(shapeWeight), 1);
    std::vector<float> biasData(GetShapeSize(shapeBias), 1);
    std::vector<float> outputData(GetShapeSize(shapeResult), 1);

    // 创建 input aclTensor
    ret =
        CreateAclTensor(inputData, shapeInput, &deviceDataA, aclDataType::ACL_FLOAT, &input, aclFormat::ACL_FORMAT_NCL);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> inputTensorPtr(input, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> deviceDataAPtr(deviceDataA, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建 weight aclTensor
    ret = CreateAclTensor(
        weightData, shapeWeight, &deviceDataB, aclDataType::ACL_FLOAT, &weight, aclFormat::ACL_FORMAT_NCL);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> deviceDataBPtr(deviceDataB, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建 bias aclTensor
    ret = CreateAclTensor(biasData, shapeBias, &deviceDataC, aclDataType::ACL_FLOAT, &bias, aclFormat::ACL_FORMAT_ND);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> biasTensorPtr(bias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> deviceDataCPtr(deviceDataC, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建 result aclTensor
    ret = CreateAclTensor(
        outputData, shapeResult, &deviceDataResult, aclDataType::ACL_FLOAT, &result, aclFormat::ACL_FORMAT_NCL);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outputTensorPtr(result, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> deviceDataResultPtr(deviceDataResult, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用 CANN 算子库 API，需要修改为具体的 Api 名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用 aclnnConvTbc 第一段接口
    ret = aclnnConvTbcGetWorkspaceSize(input, weight, bias, 1, result, 1, &workspaceSize, &executor);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的 workspaceSize 申请 device 内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // 调用 aclnnConvTbc 第二段接口
    ret = aclnnConvTbc(workspaceAddr, workspaceSize, executor, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbc failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将 device 侧内存上的结果拷贝至 host 侧，需要根据具体 API 的接口定义修改
    auto size = GetShapeSize(shapeResult);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), deviceDataResult, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(
        ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    int64_t printSize = size > 10 ? 10 : size;
    for (int64_t i = 0; i < printSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream 初始化，参考 acl API 手册
    // 根据自己的实际 device 填写 deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnConvTbcTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}