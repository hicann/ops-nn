/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
 * - Su Tonghua <@sutonghua>
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
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_group_normalization_grad.h"

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

// Compute host-side reference for GroupNormalizationGrad.
// dx = (rstd / M) * gamma * (M * dy - s1 - x_hat * s2)
// where s1 = sum(dy*gamma), s2 = sum(dy*gamma*x_hat), x_hat = (x - mean) * rstd, M = groupElemNum.
std::vector<float> ComputeExpected(const std::vector<int64_t>& shape, const std::vector<float>& x,
                                   const std::vector<float>& dy, const std::vector<float>& gamma,
                                   const std::vector<float>& mean, const std::vector<float>& rstd)
{
    int64_t groupCount = shape[0] * shape[1];
    int64_t groupElemNum = 1;
    for (size_t d = 2; d < shape.size(); d++) {
        groupElemNum *= shape[d];
    }
    int64_t totalSize = groupCount * groupElemNum;
    std::vector<float> expected(totalSize);

    float M = static_cast<float>(groupElemNum);
    for (int64_t g = 0; g < groupCount; g++) {
        float meanVal = mean[g];
        float rstdVal = rstd[g];
        int64_t base = g * groupElemNum;

        // reduction sums for this group
        float s1 = 0.0f;
        float s2 = 0.0f;
        for (int64_t i = 0; i < groupElemNum; i++) {
            float xhat = (x[base + i] - meanVal) * rstdVal;
            float dyg = dy[base + i] * gamma[base + i];
            s1 += dyg;
            s2 += dyg * xhat;
        }

        float coeff = rstdVal / M;
        for (int64_t i = 0; i < groupElemNum; i++) {
            float xhat = (x[base + i] - meanVal) * rstdVal;
            expected[base + i] = coeff * gamma[base + i] * (M * dy[base + i] - s1 - xhat * s2);
        }
    }
    return expected;
}

struct ErrorInfo {
    int64_t index;
    float computed;
    float expected;
    float absDiff;
};

void CompareAndReport(const std::vector<float>& computed, const std::vector<float>& expected, float threshold = 1e-5f)
{
    int64_t size = computed.size();
    float maxAbsErr = 0.0f;
    double sumAbsErr = 0.0;
    int64_t maxErrIdx = -1;

    std::vector<ErrorInfo> errors;
    for (int64_t i = 0; i < size; i++) {
        float diff = std::fabs(computed[i] - expected[i]);
        sumAbsErr += diff;
        if (diff > maxAbsErr) {
            maxAbsErr = diff;
            maxErrIdx = i;
        }
        if (diff > threshold) {
            errors.push_back({i, computed[i], expected[i], diff});
        }
    }

    float meanAbsErr = static_cast<float>(sumAbsErr / static_cast<double>(size));

    if (maxAbsErr <= threshold) {
        LOG_PRINT("\n[PASS] max_abs_err=%f, mean_abs_err=%f, threshold=%f\n", maxAbsErr, meanAbsErr, threshold);
    } else {
        LOG_PRINT("\n[FAIL] max_abs_err=%f at index=%ld, mean_abs_err=%f, threshold=%f, num_errors=%ld\n", maxAbsErr,
                  maxErrIdx, meanAbsErr, threshold, static_cast<int64_t>(errors.size()));

        std::sort(errors.begin(), errors.end(),
                  [](const ErrorInfo& a, const ErrorInfo& b) { return a.absDiff > b.absDiff; });

        int64_t topN = errors.size() < 10 ? static_cast<int64_t>(errors.size()) : 10;
        LOG_PRINT("Top-%ld worst errors:\n", topN);
        for (int64_t i = 0; i < topN; i++) {
            LOG_PRINT("  idx=%ld: computed=%f, expected=%f, diff=%f\n", errors[i].index, errors[i].computed,
                      errors[i].expected, errors[i].absDiff);
        }
    }
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> xShape = {2, 4, 128};
    std::vector<int64_t> meanShape = {2, 4};
    void* xDeviceAddr = nullptr;
    void* dyDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* dxDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* dx = nullptr;

    auto size = GetShapeSize(xShape);
    std::vector<float> xHostData(size);
    std::vector<float> dyHostData(size);
    std::vector<float> gammaHostData(size);
    std::vector<float> dxHostData(size, 0.0f);
    auto meanSize = GetShapeSize(meanShape);
    std::vector<float> meanHostData(meanSize);
    std::vector<float> rstdHostData(meanSize);
    for (int64_t i = 0; i < size; i++) {
        xHostData[i] = static_cast<float>(i % 128) / 128.0f;
        dyHostData[i] = 0.5f;
        gammaHostData[i] = 1.0f;
    }
    for (int64_t i = 0; i < meanSize; i++) {
        meanHostData[i] = 0.0f;
        rstdHostData[i] = 1.0f;
    }

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dyHostData, xShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, xShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(rstdHostData, meanShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dxHostData, xShape, &dxDeviceAddr, aclDataType::ACL_FLOAT, &dx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnGroupNormalizationGradGetWorkspaceSize(x, dy, gamma, mean, rstd, dx, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormalizationGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnGroupNormalizationGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormalizationGrad failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    size = GetShapeSize(xShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), dxDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);

    std::vector<float> expectedData = ComputeExpected(xShape, xHostData, dyHostData, gammaHostData, meanHostData,
                                                      rstdHostData);

    // preview: first N and last N results
    int64_t printCount = size < 10 ? size : 10;
    int64_t headCount = (printCount + 1) / 2;
    int64_t tailCount = printCount - headCount;
    for (int64_t i = 0; i < headCount; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    if (size > printCount) {
        LOG_PRINT("... (%ld results omitted) ...\n", size - printCount);
    }
    for (int64_t i = size - tailCount; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    CompareAndReport(resultData, expectedData);

    aclDestroyTensor(x);
    aclDestroyTensor(dy);
    aclDestroyTensor(gamma);
    aclDestroyTensor(mean);
    aclDestroyTensor(rstd);
    aclDestroyTensor(dx);
    aclrtFree(xDeviceAddr);
    aclrtFree(dyDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(rstdDeviceAddr);
    aclrtFree(dxDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
