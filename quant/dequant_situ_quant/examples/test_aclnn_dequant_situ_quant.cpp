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
#include <cmath>
#include <cstring>
#include <algorithm>
#include "acl/acl.h"
#include "aclnnop/aclnn_dequant_situ_quant.h"

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

// Truncate towards zero (matching kernel's CAST_TRUNC for fp16→int8)
int8_t TruncToInt8(float val)
{
    int32_t intVal = static_cast<int32_t>(val); // truncation towards zero
    intVal = std::max(-128, std::min(127, intVal));
    return static_cast<int8_t>(intVal);
}

// CPU reference: DequantSituQuant static mode
void CpuStaticQuant(const std::vector<int8_t>& x, const std::vector<float>& dequantScale,
                    const std::vector<float>& quantScale, const std::vector<float>& quantOffset, int64_t rowLen,
                    int64_t inDimy, int64_t outDimy, double beta, double linearBeta, bool activateLeft,
                    std::vector<int8_t>& yOut)
{
    for (int64_t r = 0; r < rowLen; r++) {
        for (int64_t c = 0; c < outDimy; c++) {
            int64_t gateIdx = r * inDimy + (activateLeft ? c : c + outDimy);
            int64_t upIdx = r * inDimy + (activateLeft ? c + outDimy : c);

            float gate = static_cast<float>(x[gateIdx]) * dequantScale[gateIdx % dequantScale.size()];
            float up = static_cast<float>(x[upIdx]) * dequantScale[upIdx % dequantScale.size()];

            float invBeta = 1.0f / static_cast<float>(beta);
            float tanhResult = std::tanh(gate * invBeta) * static_cast<float>(beta);
            float sigmoidResult = 1.0f / (1.0f + std::exp(-gate));
            float situA = tanhResult * sigmoidResult;

            if (linearBeta > 0.0) {
                float invLb = 1.0f / static_cast<float>(linearBeta);
                up = std::tanh(up * invLb) * static_cast<float>(linearBeta);
            }

            float situOut = situA * up;

            float qs = quantScale.size() == 1 ? quantScale[0] : quantScale[c];
            float qo = quantOffset.empty() ? 0.0f : (quantOffset.size() == 1 ? quantOffset[0] : quantOffset[c]);
            float quantized = situOut / qs + qo;

            yOut[r * outDimy + c] = TruncToInt8(quantized);
        }
    }
}

// CPU reference: DequantSituQuant dynamic mode
void CpuDynamicQuant(const std::vector<int8_t>& x, const std::vector<float>& dequantScale, int64_t rowLen,
                     int64_t inDimy, int64_t outDimy, double beta, double linearBeta, bool activateLeft,
                     std::vector<int8_t>& yOut, std::vector<float>& yScaleOut)
{
    std::vector<float> situOutBuf(rowLen * outDimy, 0.0f);

    for (int64_t r = 0; r < rowLen; r++) {
        for (int64_t c = 0; c < outDimy; c++) {
            int64_t gateIdx = r * inDimy + (activateLeft ? c : c + outDimy);
            int64_t upIdx = r * inDimy + (activateLeft ? c + outDimy : c);

            float gate = static_cast<float>(x[gateIdx]) * dequantScale[gateIdx % dequantScale.size()];
            float up = static_cast<float>(x[upIdx]) * dequantScale[upIdx % dequantScale.size()];

            float invBeta = 1.0f / static_cast<float>(beta);
            float tanhResult = std::tanh(gate * invBeta) * static_cast<float>(beta);
            float sigmoidResult = 1.0f / (1.0f + std::exp(-gate));
            float situA = tanhResult * sigmoidResult;

            if (linearBeta > 0.0) {
                float invLb = 1.0f / static_cast<float>(linearBeta);
                up = std::tanh(up * invLb) * static_cast<float>(linearBeta);
            }

            situOutBuf[r * outDimy + c] = situA * up;
        }

        float absMax = 0.0f;
        for (int64_t c = 0; c < outDimy; c++) {
            absMax = std::max(absMax, std::abs(situOutBuf[r * outDimy + c]));
        }
        float scale = absMax / 127.0f;
        if (scale == 0.0f)
            scale = 1.0f;
        float invScale = 1.0f / scale;
        yScaleOut[r] = scale;

        for (int64_t c = 0; c < outDimy; c++) {
            yOut[r * outDimy + c] = TruncToInt8(situOutBuf[r * outDimy + c] * invScale);
        }
    }
}

// ============ Test 1: static quant, scalar quant_scale, no bias ============
int TestStaticQuantScalar(int32_t deviceId, aclrtStream stream)
{
    LOG_PRINT("\n========== Test 1: Static Quant (scalar scale, no bias) ==========\n");

    int64_t rowLen = 16;
    int64_t inDimy = 64;
    int64_t outDimy = 32;
    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = false;

    std::vector<int64_t> xShape = {rowLen, inDimy};
    std::vector<int64_t> dequantScaleShape = {inDimy};
    std::vector<int64_t> quantScaleShape = {1};
    std::vector<int64_t> quantOffsetShape = {1};
    std::vector<int64_t> yShape = {rowLen, outDimy};
    std::vector<int64_t> yScaleOutShape = {rowLen};

    auto xSize = GetShapeSize(xShape);
    auto dsSize = GetShapeSize(dequantScaleShape);
    auto qsSize = GetShapeSize(quantScaleShape);
    auto qoSize = GetShapeSize(quantOffsetShape);
    auto ySize = GetShapeSize(yShape);
    auto yScaleSize = GetShapeSize(yScaleOutShape);

    std::vector<int8_t> xHostData(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        xHostData[i] = static_cast<int8_t>((i * 7 + 3) % 100);
    }
    std::vector<float> dequantScaleHostData(dsSize, 0.1f);
    std::vector<float> quantScaleHostData(qsSize, 1.0f);
    std::vector<float> quantOffsetHostData(qoSize, 0.0f);
    std::vector<int8_t> yHostData(ySize, 0);
    std::vector<float> yScaleOutHostData(yScaleSize, 0.0f);

    void* xDeviceAddr = nullptr;
    void* dsDeviceAddr = nullptr;
    void* qsDeviceAddr = nullptr;
    void* qoDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* yScaleDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* dequantScale = nullptr;
    aclTensor* quantScale = nullptr;
    aclTensor* quantOffset = nullptr;
    aclTensor* y = nullptr;
    aclTensor* yScaleOut = nullptr;

    auto ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dequantScaleHostData, dequantScaleShape, &dsDeviceAddr, aclDataType::ACL_FLOAT,
                          &dequantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(quantScaleHostData, quantScaleShape, &qsDeviceAddr, aclDataType::ACL_FLOAT, &quantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(quantOffsetHostData, quantOffsetShape, &qoDeviceAddr, aclDataType::ACL_FLOAT, &quantOffset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_INT8, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yScaleOutHostData, yScaleOutShape, &yScaleDeviceAddr, aclDataType::ACL_FLOAT, &yScaleOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnDequantSituQuantGetWorkspaceSize(x, dequantScale, nullptr, nullptr, quantScale, quantOffset, nullptr,
                                                beta, linearBeta, activateLeft, const_cast<char*>("static"), y,
                                                yScaleOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnDequantSituQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSituQuant failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    std::vector<int8_t> npuYResult(ySize, 0);
    ret = aclrtMemcpy(npuYResult.data(), npuYResult.size() * sizeof(int8_t), yDeviceAddr, ySize * sizeof(int8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y result failed. ERROR: %d\n", ret); return ret);

    std::vector<int8_t> cpuYResult(ySize, 0);
    CpuStaticQuant(xHostData, dequantScaleHostData, quantScaleHostData, quantOffsetHostData, rowLen, inDimy, outDimy,
                   beta, linearBeta, activateLeft, cpuYResult);

    int64_t matchCount = 0;
    int64_t closeCount = 0;
    int64_t mismatchCount = 0;
    int maxDiff = 0;
    for (int64_t i = 0; i < ySize; i++) {
        int diff = std::abs(static_cast<int>(npuYResult[i]) - static_cast<int>(cpuYResult[i]));
        if (diff == 0)
            matchCount++;
        if (diff <= 1)
            closeCount++;
        else
            mismatchCount++;
        maxDiff = std::max(maxDiff, diff);
    }

    LOG_PRINT("Static quant: total=%ld, exact_match=%ld, within_1=%ld, mismatch=%ld, max_diff=%d\n", ySize, matchCount,
              closeCount, mismatchCount, maxDiff);
    LOG_PRINT("NPU y (first 16): ");
    for (int64_t i = 0; i < std::min<int64_t>(16, ySize); i++)
        LOG_PRINT("%d ", npuYResult[i]);
    LOG_PRINT("\nCPU y (first 16): ");
    for (int64_t i = 0; i < std::min<int64_t>(16, ySize); i++)
        LOG_PRINT("%d ", cpuYResult[i]);
    LOG_PRINT("\n");

    aclDestroyTensor(x);
    aclDestroyTensor(dequantScale);
    aclDestroyTensor(quantScale);
    aclDestroyTensor(quantOffset);
    aclDestroyTensor(y);
    aclDestroyTensor(yScaleOut);
    aclrtFree(xDeviceAddr);
    aclrtFree(dsDeviceAddr);
    aclrtFree(qsDeviceAddr);
    aclrtFree(qoDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(yScaleDeviceAddr);
    if (workspaceSize > 0)
        aclrtFree(workspaceAddr);

    bool passed = (maxDiff <= 2);
    LOG_PRINT("Test 1 %s (max_diff=%d, tolerance=2)\n", passed ? "PASSED" : "FAILED", maxDiff);
    return passed ? 0 : 1;
}

// ============ Test 2: dynamic quant, no bias, no smooth ============
int TestDynamicQuant(int32_t deviceId, aclrtStream stream)
{
    LOG_PRINT("\n========== Test 2: Dynamic Quant (no bias, no smooth) ==========\n");

    int64_t rowLen = 32;
    int64_t inDimy = 128;
    int64_t outDimy = 64;
    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = false;

    std::vector<int64_t> xShape = {rowLen, inDimy};
    std::vector<int64_t> dequantScaleShape = {inDimy};
    std::vector<int64_t> yShape = {rowLen, outDimy};
    std::vector<int64_t> yScaleOutShape = {rowLen};

    auto xSize = GetShapeSize(xShape);
    auto dsSize = GetShapeSize(dequantScaleShape);
    auto ySize = GetShapeSize(yShape);
    auto yScaleSize = GetShapeSize(yScaleOutShape);

    std::vector<int8_t> xHostData(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        xHostData[i] = static_cast<int8_t>((i * 7 + 3) % 100);
    }
    std::vector<float> dequantScaleHostData(dsSize, 0.1f);
    std::vector<int8_t> yHostData(ySize, 0);
    std::vector<float> yScaleOutHostData(yScaleSize, 0.0f);

    void* xDeviceAddr = nullptr;
    void* dsDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* yScaleDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* dequantScale = nullptr;
    aclTensor* y = nullptr;
    aclTensor* yScaleOut = nullptr;

    auto ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dequantScaleHostData, dequantScaleShape, &dsDeviceAddr, aclDataType::ACL_FLOAT,
                          &dequantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_INT8, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yScaleOutHostData, yScaleOutShape, &yScaleDeviceAddr, aclDataType::ACL_FLOAT, &yScaleOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnDequantSituQuantGetWorkspaceSize(x, dequantScale, nullptr, nullptr, nullptr, nullptr, nullptr, beta,
                                                linearBeta, activateLeft, const_cast<char*>("dynamic"), y, yScaleOut,
                                                &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnDequantSituQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSituQuant failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    std::vector<int8_t> npuYResult(ySize, 0);
    ret = aclrtMemcpy(npuYResult.data(), npuYResult.size() * sizeof(int8_t), yDeviceAddr, ySize * sizeof(int8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y result failed. ERROR: %d\n", ret); return ret);

    std::vector<float> npuScaleResult(yScaleSize, 0.0f);
    ret = aclrtMemcpy(npuScaleResult.data(), npuScaleResult.size() * sizeof(float), yScaleDeviceAddr,
                      yScaleSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale result failed. ERROR: %d\n", ret); return ret);

    std::vector<int8_t> cpuYResult(ySize, 0);
    std::vector<float> cpuScaleResult(yScaleSize, 0.0f);
    CpuDynamicQuant(xHostData, dequantScaleHostData, rowLen, inDimy, outDimy, beta, linearBeta, activateLeft,
                    cpuYResult, cpuScaleResult);

    int64_t matchCount = 0;
    int64_t closeCount = 0;
    int maxDiff = 0;
    for (int64_t i = 0; i < ySize; i++) {
        int diff = std::abs(static_cast<int>(npuYResult[i]) - static_cast<int>(cpuYResult[i]));
        if (diff == 0)
            matchCount++;
        if (diff <= 1)
            closeCount++;
        maxDiff = std::max(maxDiff, diff);
    }

    int64_t scaleMatch = 0;
    float maxScaleDiff = 0.0f;
    for (int64_t i = 0; i < yScaleSize; i++) {
        float diff = std::abs(npuScaleResult[i] - cpuScaleResult[i]);
        if (diff < 1e-3f)
            scaleMatch++;
        maxScaleDiff = std::max(maxScaleDiff, diff);
    }

    LOG_PRINT("Dynamic quant y: total=%ld, exact_match=%ld, within_1=%ld, max_diff=%d\n", ySize, matchCount, closeCount,
              maxDiff);
    LOG_PRINT("Dynamic quant scale: total=%ld, match=%ld, max_diff=%f\n", yScaleSize, scaleMatch, maxScaleDiff);
    LOG_PRINT("NPU y (first 16): ");
    for (int64_t i = 0; i < std::min<int64_t>(16, ySize); i++)
        LOG_PRINT("%d ", npuYResult[i]);
    LOG_PRINT("\nCPU y (first 16): ");
    for (int64_t i = 0; i < std::min<int64_t>(16, ySize); i++)
        LOG_PRINT("%d ", cpuYResult[i]);
    LOG_PRINT("\n");
    LOG_PRINT("NPU scale (first 5): ");
    for (int64_t i = 0; i < std::min<int64_t>(5, yScaleSize); i++)
        LOG_PRINT("%f ", npuScaleResult[i]);
    LOG_PRINT("\nCPU scale (first 5): ");
    for (int64_t i = 0; i < std::min<int64_t>(5, yScaleSize); i++)
        LOG_PRINT("%f ", cpuScaleResult[i]);
    LOG_PRINT("\n");

    aclDestroyTensor(x);
    aclDestroyTensor(dequantScale);
    aclDestroyTensor(y);
    aclDestroyTensor(yScaleOut);
    aclrtFree(xDeviceAddr);
    aclrtFree(dsDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(yScaleDeviceAddr);
    if (workspaceSize > 0)
        aclrtFree(workspaceAddr);

    bool passed = (maxDiff <= 2) && (maxScaleDiff < 1e-2f);
    LOG_PRINT("Test 2 %s (max_y_diff=%d, max_scale_diff=%f)\n", passed ? "PASSED" : "FAILED", maxDiff, maxScaleDiff);
    return passed ? 0 : 1;
}

int main()
{
    LOG_PRINT("=== DequantSituQuant Precision Test ===\n");

    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);

    int ret1 = TestStaticQuantScalar(deviceId, stream);
    int ret2 = TestDynamicQuant(deviceId, stream);

    LOG_PRINT("\n=== Summary ===\n");
    LOG_PRINT("Test 1 (Static Quant):  %s\n", ret1 == 0 ? "PASSED" : "FAILED");
    LOG_PRINT("Test 2 (Dynamic Quant): %s\n", ret2 == 0 ? "PASSED" : "FAILED");

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    if (ret1 == 0 && ret2 == 0) {
        LOG_PRINT("\nAll tests PASSED!\n");
    } else {
        LOG_PRINT("\nSome tests FAILED!\n");
    }
    return (ret1 == 0 && ret2 == 0) ? 0 : 1;
}
