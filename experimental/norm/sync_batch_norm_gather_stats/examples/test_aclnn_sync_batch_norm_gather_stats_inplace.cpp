/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// SyncBatchNormGatherStats inplace 回写自检：一次跑 I01-I07 配置，校验 4 个输出含原地更新的 running mean/var
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

// Pack logical float values into a host byte buffer of the requested dtype.
std::vector<uint8_t> Pack(const std::vector<float>& vals, aclDataType dt)
{
    std::vector<uint8_t> buf;
    if (dt == ACL_FLOAT) {
        buf.resize(vals.size() * sizeof(float));
        for (size_t i = 0; i < vals.size(); ++i) {
            float v = vals[i];
            std::memcpy(&buf[i * sizeof(float)], &v, sizeof(float));
        }
    } else if (dt == ACL_FLOAT16) {
        buf.resize(vals.size() * sizeof(uint16_t));
        for (size_t i = 0; i < vals.size(); ++i) {
            aclFloat16 v = aclFloatToFloat16(vals[i]);
            std::memcpy(&buf[i * sizeof(uint16_t)], &v, sizeof(uint16_t));
        }
    } else if (dt == ACL_INT32) {
        buf.resize(vals.size() * sizeof(int32_t));
        for (size_t i = 0; i < vals.size(); ++i) {
            int32_t v = static_cast<int32_t>(vals[i]);
            std::memcpy(&buf[i * sizeof(int32_t)], &v, sizeof(int32_t));
        }
    }
    return buf;
}

// Unpack a host byte buffer of the given dtype back to logical floats.
std::vector<float> Unpack(const std::vector<uint8_t>& buf, int64_t n, aclDataType dt)
{
    std::vector<float> out(n);
    if (dt == ACL_FLOAT) {
        for (int64_t i = 0; i < n; ++i) {
            float v;
            std::memcpy(&v, &buf[i * sizeof(float)], sizeof(float));
            out[i] = v;
        }
    } else if (dt == ACL_FLOAT16) {
        for (int64_t i = 0; i < n; ++i) {
            aclFloat16 v;
            std::memcpy(&v, &buf[i * sizeof(uint16_t)], sizeof(uint16_t));
            out[i] = aclFloat16ToFloat(v);
        }
    }
    return out;
}

size_t DtBytes(aclDataType dt)
{
    if (dt == ACL_FLOAT)
        return 4;
    if (dt == ACL_FLOAT16)
        return 2;
    if (dt == ACL_INT32)
        return 4;
    return 0;
}

int CreateTensor(const std::vector<float>& vals, const std::vector<int64_t>& shape, aclDataType dt, void** deviceAddr,
                 aclTensor** tensor)
{
    auto buf = Pack(vals, dt);
    auto size = buf.size();
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddr, size, buf.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
        return ret;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return 0;
}

constexpr int64_t kBatchMeanPeriod = 8;
constexpr float kSamplesPerDevice = 4.0f;

// 解析式每通道 batch mean；取模刻意为之，勿改成线性（防 FLOAT16 量化使 golden 先于 kernel 塌陷）
float BatchMeanOf(int64_t c) { return 3.0f * static_cast<float>((c % kBatchMeanPeriod) + 1) / 8.0f; }

struct Config {
    std::string id;
    aclDataType valDt;
    aclDataType cntDt;
    float momentum;
    float eps;
    float relTol; // relative tolerance for compare
    int64_t n;    // device count (leading dim of totalSum / totalSquareSum)
    int64_t c;    // channel count; drives blockDim on the kernel side
};

int RunOne(const Config& cfg, aclrtStream stream)
{
    const int64_t kN = cfg.n;
    const int64_t kC = cfg.c;
    const float countAll = kSamplesPerDevice * static_cast<float>(kN);
    std::vector<int64_t> sumShape = {kN, kC};
    std::vector<int64_t> vecShape = {kC};
    std::vector<int64_t> cntShape = {kN};

    std::vector<float> totalSum(kN * kC), totalSquareSum(kN * kC);
    std::vector<float> counts(kN, kSamplesPerDevice);
    std::vector<float> mean(kC, 1.0f), var(kC, 2.0f);
    std::vector<float> outMean(kC, 0.0f), outInvstd(kC, 0.0f);
    // 设备 n 权重为 (n+1)，各设备贡献不同 partial sum
    const float sumW = static_cast<float>(kN * (kN + 1) / 2);
    for (int64_t c = 0; c < kC; ++c) {
        const float bm = BatchMeanOf(c);
        for (int64_t n = 0; n < kN; ++n) {
            totalSum[n * kC + c] = countAll * bm * static_cast<float>(n + 1) / sumW;
            totalSquareSum[n * kC + c] = countAll * (bm * bm + 1.0f) / static_cast<float>(kN);
        }
    }

    void *dSum = nullptr, *dSq = nullptr, *dCnt = nullptr, *dMean = nullptr, *dVar = nullptr, *dBm = nullptr,
         *dBi = nullptr;
    aclTensor *tSum = nullptr, *tSq = nullptr, *tCnt = nullptr, *tMean = nullptr, *tVar = nullptr, *tBm = nullptr,
              *tBi = nullptr;
    CHECK_RET(CreateTensor(totalSum, sumShape, cfg.valDt, &dSum, &tSum) == 0, return -1);
    CHECK_RET(CreateTensor(totalSquareSum, sumShape, cfg.valDt, &dSq, &tSq) == 0, return -1);
    CHECK_RET(CreateTensor(counts, cntShape, cfg.cntDt, &dCnt, &tCnt) == 0, return -1);
    CHECK_RET(CreateTensor(mean, vecShape, cfg.valDt, &dMean, &tMean) == 0, return -1);
    CHECK_RET(CreateTensor(var, vecShape, cfg.valDt, &dVar, &tVar) == 0, return -1);
    CHECK_RET(CreateTensor(outMean, vecShape, cfg.valDt, &dBm, &tBm) == 0, return -1);
    CHECK_RET(CreateTensor(outInvstd, vecShape, cfg.valDt, &dBi, &tBi) == 0, return -1);

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    auto ret = aclnnSyncBatchNormGatherStatsGetWorkspaceSize(tSum, tSq, tCnt, tMean, tVar, cfg.momentum, cfg.eps, tBm,
                                                             tBi, &wsSize, &exec);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("[%s] GetWorkspaceSize failed. ERROR: %d\n", cfg.id.c_str(), ret);
        return ret;
    }
    void* ws = nullptr;
    if (wsSize > 0) {
        CHECK_RET(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return -1);
    }
    ret = aclnnSyncBatchNormGatherStats(ws, wsSize, exec, stream);
    if (!(ret == ACL_SUCCESS)) {
        LOG_PRINT("[%s] aclnn execute failed. ERROR: %d\n", cfg.id.c_str(), ret);
        return ret;
    }
    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, return -1);

    size_t vb = DtBytes(cfg.valDt);
    std::vector<uint8_t> bMean(kC * vb), bVar(kC * vb), bBm(kC * vb), bBi(kC * vb);
    aclrtMemcpy(bMean.data(), bMean.size(), dMean, bMean.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(bVar.data(), bVar.size(), dVar, bVar.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(bBm.data(), bBm.size(), dBm, bBm.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(bBi.data(), bBi.size(), dBi, bBi.size(), ACL_MEMCPY_DEVICE_TO_HOST);
    auto hMean = Unpack(bMean, kC, cfg.valDt);
    auto hVar = Unpack(bVar, kC, cfg.valDt);
    auto hBm = Unpack(bBm, kC, cfg.valDt);
    auto hBi = Unpack(bBi, kC, cfg.valDt);

    int fails = 0;
    for (int64_t c = 0; c < kC; ++c) {
        const float bm = BatchMeanOf(c);
        const float bv = 1.0f;
        const float bi = 1.0f / std::sqrt(bv + cfg.eps);
        const float em = 1.0f * (1 - cfg.momentum) + cfg.momentum * bm;
        const float ev = 2.0f * (1 - cfg.momentum) + cfg.momentum * bv * countAll / (countAll - 1);
        const float rt = cfg.relTol;
        auto near = [rt](float a, float b) { return std::fabs(a - b) <= rt * (1.0f + std::fabs(b)); };
        if (!near(hBm[c], bm)) {
            ++fails;
            LOG_PRINT("[%s] batchMean[%ld]=%f expect %f\n", cfg.id.c_str(), c, hBm[c], bm);
        }
        if (!near(hBi[c], bi)) {
            ++fails;
            LOG_PRINT("[%s] batchInvstd[%ld]=%f expect %f\n", cfg.id.c_str(), c, hBi[c], bi);
        }
        if (!near(hMean[c], em)) {
            ++fails;
            LOG_PRINT("[%s] mean[%ld]=%f expect %f\n", cfg.id.c_str(), c, hMean[c], em);
        }
        if (!near(hVar[c], ev)) {
            ++fails;
            LOG_PRINT("[%s] var[%ld]=%f expect %f\n", cfg.id.c_str(), c, hVar[c], ev);
        }
    }
    LOG_PRINT("[%s] inplace check %s, fails=%d\n", cfg.id.c_str(), fails == 0 ? "PASS" : "FAIL", fails);

    aclDestroyTensor(tSum);
    aclDestroyTensor(tSq);
    aclDestroyTensor(tCnt);
    aclDestroyTensor(tMean);
    aclDestroyTensor(tVar);
    aclDestroyTensor(tBm);
    aclDestroyTensor(tBi);
    if (ws != nullptr) {
        aclrtFree(ws);
    }
    aclrtFree(dSum);
    aclrtFree(dSq);
    aclrtFree(dCnt);
    aclrtFree(dMean);
    aclrtFree(dVar);
    aclrtFree(dBm);
    aclrtFree(dBi);
    return fails;
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

    // C=8 单核回归基线；C>=33 触发多核整 cache line 回写
    std::vector<Config> configs = {
        {"I01", ACL_FLOAT, ACL_INT32, 0.5f, 1e-5f, 1e-3f, 2, 8},
        {"I02", ACL_FLOAT16, ACL_INT32, 0.5f, 1e-5f, 2e-2f, 2, 8},
        {"I03", ACL_FLOAT, ACL_FLOAT16, 0.5f, 1e-5f, 1e-3f, 2, 8},
        {"I04", ACL_FLOAT, ACL_FLOAT, 0.99f, 1e-10f, 1e-3f, 2, 8},
        {"I05", ACL_FLOAT, ACL_INT32, 0.5f, 1e-5f, 1e-3f, 2, 64},
        {"I06", ACL_FLOAT16, ACL_INT32, 0.5f, 1e-5f, 2e-2f, 2, 128},
        {"I07", ACL_FLOAT, ACL_INT32, 0.5f, 1e-5f, 1e-3f, 2, 33},
    };

    int totalFails = 0;
    for (const auto& cfg : configs) {
        int f = RunOne(cfg, stream);
        if (f < 0) {
            LOG_PRINT("[%s] run error\n", cfg.id.c_str());
            totalFails += 1;
        } else {
            totalFails += f;
        }
    }
    LOG_PRINT("inplace I01-I07 overall %s, totalFails=%d\n", totalFails == 0 ? "PASS" : "FAIL", totalFails);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return totalFails == 0 ? 0 : 1;
}
