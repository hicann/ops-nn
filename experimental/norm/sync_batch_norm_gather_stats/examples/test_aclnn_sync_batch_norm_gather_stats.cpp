/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_sync_batch_norm_gather_stats.h"

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
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
        return ret;
    }
    return 0;
}

size_t DtypeSize(aclDataType dataType)
{
    if (dataType == ACL_FLOAT) {
        return sizeof(float);
    }
    if (dataType == ACL_FLOAT16) {
        return sizeof(uint16_t);
    }
    if (dataType == ACL_INT32) {
        return sizeof(int32_t);
    }
    return 0;
}

std::vector<uint8_t> PackData(const std::vector<float>& data, aclDataType dataType)
{
    std::vector<uint8_t> buffer(data.size() * DtypeSize(dataType));
    for (size_t i = 0; i < data.size(); ++i) {
        if (dataType == ACL_FLOAT) {
            float value = data[i];
            std::memcpy(buffer.data() + i * sizeof(float), &value, sizeof(float));
        } else if (dataType == ACL_FLOAT16) {
            aclFloat16 value = aclFloatToFloat16(data[i]);
            std::memcpy(buffer.data() + i * sizeof(uint16_t), &value, sizeof(uint16_t));
        } else if (dataType == ACL_INT32) {
            int32_t value = static_cast<int32_t>(data[i]);
            std::memcpy(buffer.data() + i * sizeof(int32_t), &value, sizeof(int32_t));
        }
    }
    return buffer;
}

std::vector<float> UnpackData(const std::vector<uint8_t>& buffer, int64_t count, aclDataType dataType)
{
    std::vector<float> data(count, 0.0f);
    for (int64_t i = 0; i < count; ++i) {
        if (dataType == ACL_FLOAT) {
            float value = 0.0f;
            std::memcpy(&value, buffer.data() + i * sizeof(float), sizeof(float));
            data[i] = value;
        } else if (dataType == ACL_FLOAT16) {
            aclFloat16 value;
            std::memcpy(&value, buffer.data() + i * sizeof(uint16_t), sizeof(uint16_t));
            data[i] = aclFloat16ToFloat(value);
        }
    }
    return data;
}

int CreateAclTensor(const std::vector<float>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto buffer = PackData(hostData, dataType);
    auto ret = aclrtMalloc(deviceAddr, buffer.size(), ACL_MEM_MALLOC_HUGE_FIRST);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddr, buffer.size(), buffer.data(), buffer.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    if (!(*tensor != nullptr)) {
        LOG_PRINT("aclCreateTensor failed.\n");
        return -1;
    }
    return 0;
}

std::vector<float> CopyTensorToHost(void* deviceAddr, int64_t count, aclDataType dataType)
{
    std::vector<uint8_t> buffer(count * DtypeSize(dataType), 0);
    auto ret = aclrtMemcpy(buffer.data(), buffer.size(), deviceAddr, buffer.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy tensor from device failed. ERROR: %d\n", ret);
        return {};
    }
    return UnpackData(buffer, count, dataType);
}

struct CaseConfig {
    std::string id;
    int64_t n;
    int64_t c;
    aclDataType valueDtype;
    aclDataType sampleCountDtype;
    float momentum;
    float eps;
    float tolerance;
};

int RunCase(const CaseConfig& cfg, aclrtStream stream)
{
    std::vector<int64_t> matrixShape = {cfg.n, cfg.c};
    std::vector<int64_t> countShape = {cfg.n};
    std::vector<int64_t> vectorShape = {cfg.c};
    const float countPerRow = 4.0f;
    const float countAll = countPerRow * static_cast<float>(cfg.n);

    std::vector<float> totalSum(cfg.n * cfg.c, 0.0f);
    std::vector<float> totalSquareSum(cfg.n * cfg.c, 0.0f);
    std::vector<float> sampleCount(cfg.n, countPerRow);
    std::vector<float> mean(cfg.c, 1.0f);
    std::vector<float> variance(cfg.c, 2.0f);
    std::vector<float> batchMean(cfg.c, 0.0f);
    std::vector<float> batchInvstd(cfg.c, 0.0f);

    for (int64_t c = 0; c < cfg.c; ++c) {
        const float expectedMean = static_cast<float>((c % 9) - 4) * 0.125f;
        const float expectedMeanSquare = expectedMean * expectedMean + 1.0f;
        for (int64_t n = 0; n < cfg.n; ++n) {
            totalSum[n * cfg.c + c] = countPerRow * expectedMean;
            totalSquareSum[n * cfg.c + c] = countPerRow * expectedMeanSquare;
        }
    }

    void *dTotalSum = nullptr, *dTotalSquareSum = nullptr, *dSampleCount = nullptr;
    void *dMean = nullptr, *dVariance = nullptr, *dBatchMean = nullptr, *dBatchInvstd = nullptr;
    aclTensor *tTotalSum = nullptr, *tTotalSquareSum = nullptr, *tSampleCount = nullptr;
    aclTensor *tMean = nullptr, *tVariance = nullptr, *tBatchMean = nullptr, *tBatchInvstd = nullptr;

    CHECK_RET(CreateAclTensor(totalSum, matrixShape, &dTotalSum, cfg.valueDtype, &tTotalSum) == 0, return -1);
    CHECK_RET(CreateAclTensor(totalSquareSum, matrixShape, &dTotalSquareSum, cfg.valueDtype, &tTotalSquareSum) == 0,
              return -1);
    CHECK_RET(CreateAclTensor(sampleCount, countShape, &dSampleCount, cfg.sampleCountDtype, &tSampleCount) == 0,
              return -1);
    CHECK_RET(CreateAclTensor(mean, vectorShape, &dMean, cfg.valueDtype, &tMean) == 0, return -1);
    CHECK_RET(CreateAclTensor(variance, vectorShape, &dVariance, cfg.valueDtype, &tVariance) == 0, return -1);
    CHECK_RET(CreateAclTensor(batchMean, vectorShape, &dBatchMean, cfg.valueDtype, &tBatchMean) == 0, return -1);
    CHECK_RET(CreateAclTensor(batchInvstd, vectorShape, &dBatchInvstd, cfg.valueDtype, &tBatchInvstd) == 0, return -1);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnSyncBatchNormGatherStatsGetWorkspaceSize(tTotalSum, tTotalSquareSum, tSampleCount, tMean, tVariance,
                                                             cfg.momentum, cfg.eps, tBatchMean, tBatchInvstd,
                                                             &workspaceSize, &executor);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("[%s] aclnnSyncBatchNormGatherStatsGetWorkspaceSize failed. ERROR: %d\n", cfg.id.c_str(), ret);
        return ret;
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!(ret == ACL_SUCCESS)) {
            LOG_PRINT("[%s] allocate workspace failed. ERROR: %d\n", cfg.id.c_str(), ret);
            return ret;
        }
    }

    ret = aclnnSyncBatchNormGatherStats(workspaceAddr, workspaceSize, executor, stream);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("[%s] aclnnSyncBatchNormGatherStats failed. ERROR: %d\n", cfg.id.c_str(), ret);
        return ret;
    }
    ret = aclrtSynchronizeStream(stream);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("[%s] aclrtSynchronizeStream failed. ERROR: %d\n", cfg.id.c_str(), ret);
        return ret;
    }

    auto outBatchMean = CopyTensorToHost(dBatchMean, cfg.c, cfg.valueDtype);
    auto outBatchInvstd = CopyTensorToHost(dBatchInvstd, cfg.c, cfg.valueDtype);
    auto outMean = CopyTensorToHost(dMean, cfg.c, cfg.valueDtype);
    auto outVariance = CopyTensorToHost(dVariance, cfg.c, cfg.valueDtype);
    CHECK_RET(!outBatchMean.empty() && !outBatchInvstd.empty() && !outMean.empty() && !outVariance.empty(), return -1);

    int failures = 0;
    auto near = [&cfg](float actual, float expected) {
        return std::fabs(actual - expected) <= cfg.tolerance * (1.0f + std::fabs(expected));
    };
    for (int64_t c = 0; c < cfg.c; ++c) {
        const float expectedMean = static_cast<float>((c % 9) - 4) * 0.125f;
        const float expectedVariance = 1.0f;
        const float expectedInvstd = 1.0f / std::sqrt(expectedVariance + cfg.eps);
        const float expectedRunningMean = 1.0f * (1.0f - cfg.momentum) + cfg.momentum * expectedMean;
        const float expectedRunningVar = 2.0f * (1.0f - cfg.momentum) +
                                         cfg.momentum * expectedVariance * countAll / (countAll - 1.0f);
        if (!near(outBatchMean[c], expectedMean)) {
            ++failures;
            LOG_PRINT("[%s] batchMean[%ld]=%f expect %f\n", cfg.id.c_str(), c, outBatchMean[c], expectedMean);
        }
        if (!near(outBatchInvstd[c], expectedInvstd)) {
            ++failures;
            LOG_PRINT("[%s] batchInvstd[%ld]=%f expect %f\n", cfg.id.c_str(), c, outBatchInvstd[c], expectedInvstd);
        }
        if (!near(outMean[c], expectedRunningMean)) {
            ++failures;
            LOG_PRINT("[%s] mean[%ld]=%f expect %f\n", cfg.id.c_str(), c, outMean[c], expectedRunningMean);
        }
        if (!near(outVariance[c], expectedRunningVar)) {
            ++failures;
            LOG_PRINT("[%s] variance[%ld]=%f expect %f\n", cfg.id.c_str(), c, outVariance[c], expectedRunningVar);
        }
    }

    LOG_PRINT("[%s] public aclnn sampleCount dtype check %s, failures=%d\n", cfg.id.c_str(),
              failures == 0 ? "PASS" : "FAIL", failures);

    aclDestroyTensor(tTotalSum);
    aclDestroyTensor(tTotalSquareSum);
    aclDestroyTensor(tSampleCount);
    aclDestroyTensor(tMean);
    aclDestroyTensor(tVariance);
    aclDestroyTensor(tBatchMean);
    aclDestroyTensor(tBatchInvstd);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclrtFree(dTotalSum);
    aclrtFree(dTotalSquareSum);
    aclrtFree(dSampleCount);
    aclrtFree(dMean);
    aclrtFree(dVariance);
    aclrtFree(dBatchMean);
    aclrtFree(dBatchInvstd);
    return failures;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    std::vector<CaseConfig> cases = {
        {"G03_countf16_f32_N2_C63", 2, 63, ACL_FLOAT, ACL_FLOAT16, 0.5f, 1e-5f, 1e-3f},
        {"G04_countf32_f32_N2_C64", 2, 64, ACL_FLOAT, ACL_FLOAT, 0.5f, 1e-5f, 1e-3f},
        {"G14_countf16_f16_N128_C128", 128, 128, ACL_FLOAT16, ACL_FLOAT16, 0.5f, 1e-5f, 2e-2f},
    };

    int totalFailures = 0;
    for (const auto& cfg : cases) {
        int failures = RunCase(cfg, stream);
        if (failures < 0) {
            LOG_PRINT("[%s] run error\n", cfg.id.c_str());
            ++totalFailures;
        } else {
            totalFailures += failures;
        }
    }

    LOG_PRINT("public aclnn sampleCount dtype gap cases overall %s, totalFailures=%d\n",
              totalFailures == 0 ? "PASS" : "FAIL", totalFailures);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return totalFailures == 0 ? 0 : 1;
}
