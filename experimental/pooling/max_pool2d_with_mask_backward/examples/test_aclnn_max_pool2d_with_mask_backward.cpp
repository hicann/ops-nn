// clang-format off
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// clang-format on

// Standalone backward example. The fixed K>=2 fixtures store per-channel INT32
// argmax positions in the prefix of the public INT8 container. Production callers
// must use indices from the matching forward operator.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_max_pool2d_with_mask_backward.h"

struct Case {
    const char* name;
    int64_t h;
    int64_t w;
    int64_t k;
    int64_t s;
    int64_t p;
    std::vector<float> input;
    std::vector<float> gradOutput;
    std::vector<int32_t> argmax;
    std::vector<float> expected;
};

struct DeviceTensor {
    void* addr = nullptr;
    aclTensor* tensor = nullptr;
};

static bool CheckAcl(aclError status, const char* expression)
{
    if (status == ACL_SUCCESS) {
        return true;
    }
    std::printf("%s failed, status=%d\n", expression, static_cast<int>(status));
    return false;
}

static bool CreateTensor(const void* hostData, size_t bytes, const std::vector<int64_t>& shape, aclDataType dtype,
                         DeviceTensor& result)
{
    if (!CheckAcl(aclrtMalloc(&result.addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc") ||
        !CheckAcl(aclrtMemcpy(result.addr, bytes, hostData, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D")) {
        return false;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    result.tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_NCHW, shape.data(),
                                    shape.size(), result.addr);
    return result.tensor != nullptr;
}

static void DestroyTensor(DeviceTensor& value)
{
    if (value.tensor != nullptr) {
        aclDestroyTensor(value.tensor);
    }
    if (value.addr != nullptr) {
        aclrtFree(value.addr);
    }
}

static bool Compare(const std::vector<float>& actual, const std::vector<float>& expected)
{
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > 1e-5F) {
            std::printf("gradInput mismatch[%zu]: actual=%f expected=%f\n", i, actual[i], expected[i]);
            return false;
        }
    }
    return true;
}

static bool RunCase(const Case& tc, aclrtStream stream)
{
    const int64_t ho = (tc.h + 2 * tc.p - tc.k) / tc.s + 1;
    const int64_t wo = (tc.w + 2 * tc.p - tc.k) / tc.s + 1;
    const int64_t maskWidth = ((ho * wo + 15) / 16 + 1) * 2 * 16;
    const std::vector<int64_t> inputShape = {1, 1, tc.h, tc.w};
    const std::vector<int64_t> outputShape = {1, 1, ho, wo};
    const std::vector<int64_t> indicesShape = {1, 1, tc.k * tc.k, maskWidth};
    const std::vector<int64_t> kernel = {tc.k, tc.k};
    const std::vector<int64_t> stride = {tc.s, tc.s};
    const std::vector<int64_t> padding = {tc.p, tc.p};
    const std::vector<int64_t> dilation = {1, 1};
    std::vector<int8_t> indices(static_cast<size_t>(tc.k * tc.k * maskWidth), 0);
    const size_t argmaxBytes = tc.argmax.size() * sizeof(int32_t);
    if (tc.input.size() != static_cast<size_t>(tc.h * tc.w) || tc.expected.size() != tc.input.size() ||
        tc.gradOutput.size() != static_cast<size_t>(ho * wo) || tc.argmax.size() != tc.gradOutput.size() ||
        argmaxBytes > indices.size()) {
        std::printf("Invalid fixture: %s\n", tc.name);
        return false;
    }
    std::memcpy(indices.data(), tc.argmax.data(), argmaxBytes);
    std::vector<float> gradInputInit(tc.expected.size(), -999.0F);
    DeviceTensor inputTensor;
    DeviceTensor indicesTensor;
    DeviceTensor gradOutputTensor;
    DeviceTensor gradInputTensor;
    aclIntArray* kernelArray = nullptr;
    aclIntArray* strideArray = nullptr;
    aclIntArray* paddingArray = nullptr;
    aclIntArray* dilationArray = nullptr;
    void* workspace = nullptr;
    bool ok = false;

    if (!CreateTensor(tc.input.data(), tc.input.size() * sizeof(float), inputShape, ACL_FLOAT, inputTensor) ||
        !CreateTensor(indices.data(), indices.size(), indicesShape, ACL_INT8, indicesTensor) ||
        !CreateTensor(tc.gradOutput.data(), tc.gradOutput.size() * sizeof(float), outputShape, ACL_FLOAT,
                      gradOutputTensor) ||
        !CreateTensor(gradInputInit.data(), gradInputInit.size() * sizeof(float), inputShape, ACL_FLOAT,
                      gradInputTensor)) {
        goto cleanup;
    }
    kernelArray = aclCreateIntArray(kernel.data(), kernel.size());
    strideArray = aclCreateIntArray(stride.data(), stride.size());
    paddingArray = aclCreateIntArray(padding.data(), padding.size());
    dilationArray = aclCreateIntArray(dilation.data(), dilation.size());
    if (kernelArray == nullptr || strideArray == nullptr || paddingArray == nullptr || dilationArray == nullptr) {
        std::printf("aclCreateIntArray failed\n");
        goto cleanup;
    }
    {
        constexpr aclnnStatus kNullPointerStatus = 161001; // ACLNN_ERR_PARAM_NULLPTR.
        struct NullOutputCase {
            const char* name;
            bool nullWorkspaceSize;
            bool nullExecutor;
        };
        const NullOutputCase nullCases[] = {
            {"null_workspace_size", true, false},
            {"null_executor", false, true},
            {"both_outputs_null", true, true},
        };
        for (const auto& nullCase : nullCases) {
            uint64_t untouchedWorkspaceSize = 123;
            aclOpExecutor* untouchedExecutor = nullptr;
            const aclnnStatus status = aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize(
                gradOutputTensor.tensor, inputTensor.tensor, indicesTensor.tensor, kernelArray, strideArray,
                paddingArray, dilationArray, false, gradInputTensor.tensor,
                nullCase.nullWorkspaceSize ? nullptr : &untouchedWorkspaceSize,
                nullCase.nullExecutor ? nullptr : &untouchedExecutor);
            if (status != kNullPointerStatus || untouchedWorkspaceSize != 123 || untouchedExecutor != nullptr) {
                std::printf("[%s/%s] FAIL: status=%d\n", tc.name, nullCase.name, static_cast<int>(status));
                goto cleanup;
            }
            std::printf("[%s/%s] PASS: null output rejected\n", tc.name, nullCase.name);
        }
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        aclnnStatus status = aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize(
            gradOutputTensor.tensor, inputTensor.tensor, indicesTensor.tensor, kernelArray, strideArray, paddingArray,
            dilationArray, false, gradInputTensor.tensor, &workspaceSize, &executor);
        if (!CheckAcl(status, "backward GetWorkspaceSize")) {
            goto cleanup;
        }
        if (workspaceSize > 0 &&
            !CheckAcl(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "workspace malloc")) {
            goto cleanup;
        }
        if (!CheckAcl(aclnnMaxPool2dWithMaskBackward(workspace, workspaceSize, executor, stream), "backward execute") ||
            !CheckAcl(aclrtSynchronizeStream(stream), "backward synchronize")) {
            goto cleanup;
        }
        std::vector<float> actual(tc.expected.size());
        if (!CheckAcl(aclrtMemcpy(actual.data(), actual.size() * sizeof(float), gradInputTensor.addr,
                                  actual.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST),
                      "gradInput memcpy D2H")) {
            goto cleanup;
        }
        ok = Compare(actual, tc.expected);
    }

cleanup:
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    if (kernelArray != nullptr) {
        aclDestroyIntArray(kernelArray);
    }
    if (strideArray != nullptr) {
        aclDestroyIntArray(strideArray);
    }
    if (paddingArray != nullptr) {
        aclDestroyIntArray(paddingArray);
    }
    if (dilationArray != nullptr) {
        aclDestroyIntArray(dilationArray);
    }
    DestroyTensor(inputTensor);
    DestroyTensor(indicesTensor);
    DestroyTensor(gradOutputTensor);
    DestroyTensor(gradInputTensor);
    std::printf("[%s] backward: %s\n", tc.name, ok ? "PASS" : "FAIL");
    return ok;
}

int main()
{
    constexpr int32_t deviceId = 0;
    if (!CheckAcl(aclInit(nullptr), "aclInit")) {
        return 1;
    }
    if (!CheckAcl(aclrtSetDevice(deviceId), "aclrtSetDevice")) {
        aclFinalize();
        return 1;
    }
    aclrtStream stream = nullptr;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) {
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }
    const std::vector<Case> cases = {
        {"k1s1_identity", 1, 1, 1, 1, 0, {2}, {3}, {0}, {3}},
        {"k2s2_non_overlap",
         4,
         4,
         2,
         2,
         0,
         {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
         {1, 2, 3, 4},
         {5, 7, 13, 15},
         {0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 3, 0, 4}},
        // All four windows select input[4], so its gradient is 1+2+3+4.
        {"k2s1_overlap",
         3,
         3,
         2,
         1,
         0,
         {1, 2, 3, 4, 9, 5, 6, 7, 8},
         {1, 2, 3, 4},
         {4, 4, 4, 4},
         {0, 0, 0, 0, 10, 0, 0, 0, 0}},
        // All nine padded windows select the center; the gradient sum is 45.
        {"k3s1p1_overlap",
         3,
         3,
         3,
         1,
         1,
         {1, 2, 3, 4, 9, 5, 6, 7, 8},
         {1, 2, 3, 4, 5, 6, 7, 8, 9},
         {4, 4, 4, 4, 4, 4, 4, 4, 4},
         {0, 0, 0, 0, 45, 0, 0, 0, 0}},
    };
    int failed = 0;
    for (const Case& tc : cases) {
        if (!RunCase(tc, stream)) {
            ++failed;
        }
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    std::printf("backward summary: %zu/%zu PASS\n", cases.size() - failed, cases.size());
    return failed == 0 ? 0 : 1;
}
