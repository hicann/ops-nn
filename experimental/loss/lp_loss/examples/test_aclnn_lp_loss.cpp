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
#include "aclnn_l1_loss.h"

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
    if (*tensor == nullptr) {
        LOG_PRINT("aclCreateTensor returned nullptr.\n");
        return -1;
    }
    return 0;
}

static void ComputeExpected(
    const std::vector<float>& predictHostData, const std::vector<float>& labelHostData, int64_t reduction,
    std::vector<float>& expected)
{
    expected.resize(predictHostData.size());
    float sum = 0.0f;
    for (size_t i = 0; i < predictHostData.size(); ++i) {
        float diff = predictHostData[i] - labelHostData[i];
        float val = diff < 0.0f ? -diff : diff;
        expected[i] = val;
        sum += val;
    }
    if (reduction != 0) {
        expected.resize(1);
        if (reduction == 1) {
            expected[0] = (predictHostData.size() > 0) ? (sum / static_cast<float>(predictHostData.size())) : 0.0f;
        } else { // sum
            expected[0] = sum;
        }
    }
}

static const char* ReductionToString(int64_t reduction)
{
    if (reduction == 0) {
        return "none";
    }
    if (reduction == 1) {
        return "mean";
    }
    return "sum";
}

static void PrintVector(const char* tag, const std::vector<float>& values, int64_t reduction)
{
    for (size_t i = 0; i < values.size(); ++i) {
        LOG_PRINT("%s[%zu] (reduction=%ld) = %f\n", tag, i, reduction, values[i]);
    }
}

static void PrintExpectedPreview()
{
    std::vector<float> predictHostData = {0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> labelHostData = {2.0f, 0.5f, 2.0f, 2.5f};

    std::vector<float> expectedNone;
    ComputeExpected(predictHostData, labelHostData, 0, expectedNone);
    LOG_PRINT("\n==== Run case: reduction=%s ===\n", ReductionToString(0));
    PrintVector("expected", expectedNone, 0);

    std::vector<float> expectedMean;
    ComputeExpected(predictHostData, labelHostData, 1, expectedMean);
    LOG_PRINT("==== Run case: reduction=%s ===\n", ReductionToString(1));
    PrintVector("expected", expectedMean, 1);
}

static bool gExpectedPreviewPrinted = false;

static int RunL1LossCase(aclrtStream stream, int64_t reduction)
{
    std::vector<int64_t> inputShape = {2, 2};
    void* predictDeviceAddr = nullptr;
    void* labelDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* predict = nullptr;
    aclTensor* label = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> predictHostData = {0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<float> labelHostData = {2.0f, 0.5f, 2.0f, 2.5f};

    std::vector<int64_t> outShape;
    if (reduction == 0) {
        outShape = inputShape;
    } else {
        outShape = {};
    }
    std::vector<float> outHostData(GetShapeSize(outShape), 0.0f);

    auto ret = CreateAclTensor(predictHostData, inputShape, &predictDeviceAddr, aclDataType::ACL_FLOAT, &predict);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(labelHostData, inputShape, &labelDeviceAddr, aclDataType::ACL_FLOAT, &label);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnL1LossGetWorkspaceSize(predict, label, reduction, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnL1LossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    if (reduction != 0) {
        auto size = GetShapeSize(outShape) * sizeof(float);
        ret = aclrtMemset(outDeviceAddr, size, 0, size);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemset failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnL1Loss(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnL1Loss failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    if (!gExpectedPreviewPrinted) {
        PrintExpectedPreview();
        gExpectedPreviewPrinted = true;
    }

    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    PrintVector("aclnnL1Loss result", resultData, reduction);

    aclDestroyTensor(predict);
    aclDestroyTensor(label);
    aclDestroyTensor(out);
    aclrtFree(predictDeviceAddr);
    aclrtFree(labelDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    ret = RunL1LossCase(stream, 0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = RunL1LossCase(stream, 1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}