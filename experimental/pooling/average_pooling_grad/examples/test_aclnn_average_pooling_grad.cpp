/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_average_pooling_grad.h"

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

struct PoolGradCase {
    const char* name;
    std::vector<int64_t> origInputShape;
    std::vector<int64_t> gradOutputShape;
    std::vector<float> gradOutputData;
    int64_t kernelH;
    int64_t kernelW;
    int64_t strideH;
    int64_t strideW;
    int64_t padTop;
    int64_t padBottom;
    int64_t padLeft;
    int64_t padRight;
    bool ceilMode;
    bool countIncludePad;
    int64_t divisorOverride;
};

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

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
    aclDataType dataType, aclTensor** tensor)
{
    const auto size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
        shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

int32_t FloorDiv(int32_t a, int32_t b)
{
    if (a >= 0) {
        return a / b;
    }
    return -((-a + b - 1) / b);
}

int64_t Offset4D(const std::vector<int64_t>& shape, int64_t n, int64_t c, int64_t h, int64_t w)
{
    return ((n * shape[1] + c) * shape[2] + h) * shape[3] + w;
}

std::vector<float> GenerateGolden(const PoolGradCase& testCase)
{
    std::vector<float> expected(static_cast<size_t>(GetShapeSize(testCase.origInputShape)), 0.0f);
    const int64_t nDim = testCase.origInputShape[0];
    const int64_t cDim = testCase.origInputShape[1];
    const int64_t hIn = testCase.origInputShape[2];
    const int64_t wIn = testCase.origInputShape[3];
    const int64_t hOut = testCase.gradOutputShape[2];
    const int64_t wOut = testCase.gradOutputShape[3];

    for (int64_t n = 0; n < nDim; ++n) {
        for (int64_t c = 0; c < cDim; ++c) {
            for (int64_t ih = 0; ih < hIn; ++ih) {
                for (int64_t iw = 0; iw < wIn; ++iw) {
                    const int32_t ohStart = std::max(FloorDiv(static_cast<int32_t>(ih + testCase.padTop - testCase.kernelH),
                                                         static_cast<int32_t>(testCase.strideH)) + 1, 0);
                    const int32_t owStart = std::max(FloorDiv(static_cast<int32_t>(iw + testCase.padLeft - testCase.kernelW),
                                                         static_cast<int32_t>(testCase.strideW)) + 1, 0);
                    const int32_t ohEnd = std::min(FloorDiv(static_cast<int32_t>(ih + testCase.padTop),
                                                       static_cast<int32_t>(testCase.strideH)) + 1,
                        static_cast<int32_t>(hOut));
                    const int32_t owEnd = std::min(FloorDiv(static_cast<int32_t>(iw + testCase.padLeft),
                                                       static_cast<int32_t>(testCase.strideW)) + 1,
                        static_cast<int32_t>(wOut));

                    float acc = 0.0f;
                    for (int32_t oh = ohStart; oh < ohEnd; ++oh) {
                        for (int32_t ow = owStart; ow < owEnd; ++ow) {
                            int64_t divisor = testCase.divisorOverride;
                            if (divisor == 0) {
                                if (testCase.countIncludePad) {
                                    divisor = testCase.kernelH * testCase.kernelW;
                                } else {
                                    const int64_t hStart = oh * testCase.strideH - testCase.padTop;
                                    const int64_t wStart = ow * testCase.strideW - testCase.padLeft;
                                    const int64_t validH = std::min(hStart + testCase.kernelH, hIn) - std::max(hStart, 0L);
                                    const int64_t validW = std::min(wStart + testCase.kernelW, wIn) - std::max(wStart, 0L);
                                    divisor = validH * validW;
                                }
                            }
                            if (divisor <= 0) {
                                continue;
                            }
                            const int64_t gradOffset = Offset4D(testCase.gradOutputShape, n, c, oh, ow);
                            acc += testCase.gradOutputData[static_cast<size_t>(gradOffset)] / static_cast<float>(divisor);
                        }
                    }
                    expected[static_cast<size_t>(Offset4D(testCase.origInputShape, n, c, ih, iw))] = acc;
                }
            }
        }
    }
    return expected;
}

void PrintVector(const char* name, const char* label, const std::vector<float>& data)
{
    LOG_PRINT("%s %s=[", name, label);
    for (size_t i = 0; i < data.size(); ++i) {
        LOG_PRINT("%.6f%s", data[i], i + 1 == data.size() ? "" : ", ");
    }
    LOG_PRINT("]\n");
}

int CheckResult(const char* name, void* outDeviceAddr, const std::vector<float>& expected)
{
    std::vector<float> actual(expected.size(), 0.0f);
    auto ret = aclrtMemcpy(actual.data(), actual.size() * sizeof(float), outDeviceAddr,
        expected.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);

    PrintVector(name, "actual", actual);
    PrintVector(name, "expected", expected);

    bool allMatch = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > 1e-5f) {
            LOG_PRINT("%s MISMATCH out[%zu]=%f expected=%f\n", name, i, actual[i], expected[i]);
            allMatch = false;
        }
    }

    if (!allMatch) {
        LOG_PRINT("%s TEST FAILED.\n", name);
        return ACL_ERROR_FAILURE;
    }

    LOG_PRINT("%s ALL PASSED.\n", name);
    return ACL_SUCCESS;
}

int RunCase(const PoolGradCase& testCase, aclrtStream stream)
{
    std::vector<float> gradInputHostData(GetShapeSize(testCase.origInputShape), 0.0f);
    const std::vector<float> expected = GenerateGolden(testCase);

    void* gradOutputDeviceAddr = nullptr;
    aclTensor* gradOutput = nullptr;
    auto ret = CreateAclTensor(testCase.gradOutputData, testCase.gradOutputShape, &gradOutputDeviceAddr,
        ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    void* gradInputDeviceAddr = nullptr;
    aclTensor* gradInput = nullptr;
    ret = CreateAclTensor(gradInputHostData, testCase.origInputShape, &gradInputDeviceAddr, ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclIntArray* origInputShapeArray = aclCreateIntArray(testCase.origInputShape.data(), testCase.origInputShape.size());
    CHECK_RET(origInputShapeArray != nullptr, LOG_PRINT("aclCreateIntArray failed.\n"); return ACL_ERROR_FAILURE);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAveragePoolingGradGetWorkspaceSize(origInputShapeArray, gradOutput,
        testCase.kernelH, testCase.kernelW, testCase.strideH, testCase.strideW,
        testCase.padTop, testCase.padBottom, testCase.padLeft, testCase.padRight,
        testCase.ceilMode, testCase.countIncludePad, testCase.divisorOverride,
        gradInput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("%s aclnnAveragePoolingGradGetWorkspaceSize failed. ERROR: %d\n", testCase.name, ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnAveragePoolingGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s aclnnAveragePoolingGrad failed. ERROR: %d\n", testCase.name, ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = CheckResult(testCase.name, gradInputDeviceAddr, expected);

    aclDestroyTensor(gradOutput);
    aclDestroyTensor(gradInput);
    aclDestroyIntArray(origInputShapeArray);
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return ret;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const std::vector<PoolGradCase> testCases = {
        {"overlap_stride1", {1, 1, 3, 3}, {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f},
            2, 2, 1, 1, 0, 0, 0, 0, false, true, 0},
        {"no_overlap_stride2", {1, 1, 4, 4}, {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f},
            2, 2, 2, 2, 0, 0, 0, 0, false, true, 0},
        {"padded_exclude_pad", {1, 1, 3, 3}, {1, 1, 3, 3},
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f},
            2, 2, 1, 1, 1, 0, 1, 0, false, false, 0},
        {"divisor_override", {1, 1, 3, 3}, {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f},
            2, 2, 1, 1, 0, 0, 0, 0, false, true, 2},
        {"ceil_mode_overlap", {1, 1, 4, 4}, {1, 1, 3, 3},
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f},
            3, 3, 2, 2, 1, 1, 1, 1, true, true, 0},
    };

    for (const auto& testCase : testCases) {
        ret = RunCase(testCase, stream);
        CHECK_RET(ret == ACL_SUCCESS, aclrtDestroyStream(stream); aclrtResetDevice(deviceId); aclFinalize(); return ret);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return ACL_SUCCESS;
}
