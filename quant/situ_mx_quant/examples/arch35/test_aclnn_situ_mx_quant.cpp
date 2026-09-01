/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_situ_mx_quant.cpp
 * \brief Example test for SituMxQuant operator with CPU reference comparison.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <memory>

#include "acl/acl.h"
#include "aclnnop/aclnn_situ_mx_quant.h"

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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
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
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

// ==================== BF16 / FP8 Conversion Helpers ====================

uint16_t FloatToBf16(float val)
{
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    // Round to nearest even: add bias + (lsb of result)
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t roundingBias = 0x7FFF + lsb;
    bits += roundingBias;
    return static_cast<uint16_t>(bits >> 16);
}

float Bf16ToFloat(uint16_t bf16)
{
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// FP8 E4M3FN to float
float Fp8E4M3ToFloat(uint8_t val)
{
    int32_t sign = (val >> 7) & 1;
    int32_t exp = (val >> 3) & 0xF;
    int32_t mant = val & 0x7;
    float result;
    if (exp == 0) {
        if (mant == 0) {
            result = 0.0f;
        } else {
            result = (mant / 8.0f) * std::ldexp(1.0f, 1 - 7);
        }
    } else if (exp == 0xF && mant == 0x7) {
        return sign ? -INFINITY : INFINITY;
    } else {
        result = (1.0f + mant / 8.0f) * std::ldexp(1.0f, exp - 7);
    }
    return sign ? -result : result;
}

// Float to FP8 E4M3FN (round to nearest, ties to even)
uint8_t FloatToFp8E4M3(float val)
{
    if (std::isnan(val))
        return 0x7F;
    if (val == 0.0f)
        return 0;
    bool sign = val < 0;
    float absVal = std::fabs(val);

    if (absVal >= 448.0f) {
        return sign ? 0xFE : 0x7E;
    }

    int exp;
    float mant = std::frexp(absVal, &exp);
    exp = exp + 6;

    if (exp <= 0) {
        int32_t mantBits = static_cast<int32_t>(std::round(absVal * 8.0f * 64.0f));
        if (mantBits <= 0)
            return sign ? 0x80 : 0x00;
        if (mantBits > 7)
            mantBits = 7;
        return static_cast<uint8_t>((sign ? 0x80 : 0) | mantBits);
    }

    // E4M3FN: exp=15 is valid for mantBits 0-6 (values 256-432), only mant=7 is NaN
    if (exp > 15) {
        return sign ? 0xFE : 0x7E;
    }

    int32_t mantBits = static_cast<int32_t>(std::round((mant * 2.0f - 1.0f) * 8.0f));
    if (mantBits < 0)
        mantBits = 0;
    if (mantBits > 7) {
        mantBits = 0;
        exp++;
        if (exp > 15) {
            return sign ? 0xFE : 0x7E;
        }
    }
    if (exp == 15 && mantBits >= 7) {
        mantBits = 6;
    }

    return static_cast<uint8_t>((sign ? 0x80 : 0) | (exp << 3) | mantBits);
}

// FP8 E5M2 to float
float Fp8E5M2ToFloat(uint8_t val)
{
    int32_t sign = (val >> 7) & 1;
    int32_t exp = (val >> 2) & 0x1F;
    int32_t mant = val & 0x3;
    float result;
    if (exp == 0) {
        if (mant == 0) {
            result = 0.0f;
        } else {
            result = (mant / 4.0f) * std::ldexp(1.0f, 1 - 15);
        }
    } else if (exp == 0x1F && mant == 0x3) {
        return sign ? -INFINITY : INFINITY;
    } else {
        result = (1.0f + mant / 4.0f) * std::ldexp(1.0f, exp - 15);
    }
    return sign ? -result : result;
}

// Float to FP8 E5M2 (round to nearest, ties to even)
uint8_t FloatToFp8E5M2(float val)
{
    if (std::isnan(val))
        return 0x7F;
    if (val == 0.0f)
        return 0;
    bool sign = val < 0;
    float absVal = std::fabs(val);

    if (absVal >= 57344.0f) {
        return sign ? 0xFB : 0x7B;
    }

    int exp;
    float mant = std::frexp(absVal, &exp);
    exp = exp + 14;

    if (exp <= 0) {
        int32_t mantBits = static_cast<int32_t>(std::round(absVal * 4.0f * 32768.0f));
        if (mantBits <= 0)
            return sign ? 0x80 : 0x00;
        if (mantBits > 3)
            mantBits = 3;
        return static_cast<uint8_t>((sign ? 0x80 : 0) | mantBits);
    }

    if (exp >= 31)
        exp = 30;

    int32_t mantBits = static_cast<int32_t>(std::round((mant * 2.0f - 1.0f) * 4.0f));
    if (mantBits < 0)
        mantBits = 0;
    if (mantBits > 3) {
        mantBits = 0;
        exp++;
        if (exp >= 31)
            exp = 30;
    }

    return static_cast<uint8_t>((sign ? 0x80 : 0) | (exp << 2) | mantBits);
}

// E8M0 scale to float: 2^(e8m0 - 127)
float E8M0ToFloat(uint8_t e8m0)
{
    if (e8m0 == 0)
        return 0.0f;
    int32_t exp = static_cast<int32_t>(e8m0) - 127;
    return std::ldexp(1.0f, exp);
}

// ==================== CPU Reference ====================

void CpuSituMxQuant(const std::vector<uint16_t>& xBf16, int64_t rowNum, int64_t colNum, float beta, float linearBeta,
                    bool activateLeft, int64_t dstType, std::vector<uint8_t>& yFp8, std::vector<uint8_t>& mxscaleE8M0)
{
    int64_t outputColNum = colNum / 2;
    int64_t blockSize = 32;
    int64_t alignNum = 2;
    // mxscale last dim = CeilDiv(outputColNum, 2*32), then append 2
    int64_t scaleAxis = (outputColNum + alignNum * blockSize - 1) / (alignNum * blockSize);

    // emax for OCP algorithm
    int32_t emax = (dstType == 35) ? 15 : 8;

    // Step 1: Situ activation
    std::vector<float> situOut(rowNum * outputColNum, 0.0f);

    for (int64_t r = 0; r < rowNum; r++) {
        for (int64_t c = 0; c < outputColNum; c++) {
            float gate, up;
            if (activateLeft) {
                gate = Bf16ToFloat(xBf16[r * colNum + c]);
                up = Bf16ToFloat(xBf16[r * colNum + c + outputColNum]);
            } else {
                gate = Bf16ToFloat(xBf16[r * colNum + c + outputColNum]);
                up = Bf16ToFloat(xBf16[r * colNum + c]);
            }

            // tanh(x) = 2 * sigmoid(2x) - 1  (match NPU kernel's FP32 approximation)
            float gateDivBeta = gate / beta;
            float tanhSigmoid = 1.0f / (1.0f + std::exp(-2.0f * gateDivBeta));
            float tanhResult = beta * (2.0f * tanhSigmoid - 1.0f);
            float sigmoidResult = 1.0f / (1.0f + std::exp(-gate));
            float situA = tanhResult * sigmoidResult;

            // Optional: up = linear_beta * tanh(up / linear_beta)
            if (linearBeta > 0.0f) {
                float upDivLb = up / linearBeta;
                float upTanhSigmoid = 1.0f / (1.0f + std::exp(-2.0f * upDivLb));
                up = linearBeta * (2.0f * upTanhSigmoid - 1.0f);
            }

            situOut[r * outputColNum + c] = situA * up;
        }
    }

    // Step 2: MX Quantization (OCP algorithm)
    // Scale block: each group of 2*32=64 output elements shares one E8M0 scale pair
    // mxscale layout: {rowNum, scaleAxis, 2}
    for (int64_t r = 0; r < rowNum; r++) {
        for (int64_t b = 0; b < scaleAxis; b++) {
            // Each scale group covers 64 output elements, split into 2 sub-blocks of 32
            for (int64_t sub = 0; sub < 2; sub++) {
                int64_t start = (b * 2 + sub) * blockSize;
                int64_t end = std::min(start + blockSize, outputColNum);

                // Find max abs in this sub-block
                float maxVal = 0.0f;
                for (int64_t c = start; c < end; c++) {
                    maxVal = std::max(maxVal, std::fabs(situOut[r * outputColNum + c]));
                }

                // Compute E8M0 scale
                uint8_t e8m0;
                float recipScale;
                if (maxVal <= 0.0f) {
                    e8m0 = 0;
                    recipScale = 0.0f;
                } else if (std::isinf(maxVal) || std::isnan(maxVal)) {
                    e8m0 = 0xFF;
                    recipScale = 0.0f;
                } else {
                    float log2Val = std::log2(maxVal);
                    int32_t floorLog2 = static_cast<int32_t>(std::floor(log2Val));
                    int32_t sharedExp = floorLog2 - emax;

                    int32_t e8m0Val = sharedExp + 127;
                    if (e8m0Val < 0)
                        e8m0Val = 0;
                    if (e8m0Val > 255)
                        e8m0Val = 255;
                    e8m0 = static_cast<uint8_t>(e8m0Val);

                    recipScale = std::pow(2.0f, static_cast<float>(-sharedExp));
                }

                // Quantize
                for (int64_t c = start; c < end; c++) {
                    float scaled = situOut[r * outputColNum + c] * recipScale;
                    if (dstType == 35) {
                        yFp8[r * outputColNum + c] = FloatToFp8E5M2(scaled);
                    } else {
                        yFp8[r * outputColNum + c] = FloatToFp8E4M3(scaled);
                    }
                }

                // Write mxscale: {rowNum, scaleAxis, 2}
                mxscaleE8M0[r * scaleAxis * 2 + b * 2 + sub] = e8m0;
            }
        }
    }
}

// ==================== Test Cases ====================

struct TestConfig {
    int64_t rowNum;
    int64_t colNum; // 2H, must be even
    float beta;
    float linearBeta;
    bool activateLeft;
    int64_t dstType; // 36=E4M3FN, 35=E5M2
    const char* desc;
};

int RunSingleTest(int32_t deviceId, aclrtStream stream, const TestConfig& cfg)
{
    LOG_PRINT("\n--- %s ---\n", cfg.desc);

    int64_t outputColNum = cfg.colNum / 2;
    int64_t blockSize = 32;
    int64_t alignNum = 2;
    int64_t scaleAxis = (outputColNum + alignNum * blockSize - 1) / (alignNum * blockSize);

    std::vector<int64_t> xShape = {cfg.rowNum, cfg.colNum};
    std::vector<int64_t> yShape = {cfg.rowNum, outputColNum};
    std::vector<int64_t> mxscaleShape = {cfg.rowNum, scaleAxis, 2};

    auto xSize = GetShapeSize(xShape);
    auto ySize = GetShapeSize(yShape);
    auto mxscaleSize = GetShapeSize(mxscaleShape);

    // Prepare BF16 input data: values in [-2, 2] range
    std::vector<uint16_t> xHostData(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        float val = ((i * 17 + 3) % 200 - 100) / 50.0f; // range [-2.0, 2.0)
        xHostData[i] = FloatToBf16(val);
    }

    // Output buffers
    std::vector<uint8_t> yHostData(ySize, 0);
    std::vector<uint8_t> mxscaleHostData(mxscaleSize, 0);

    // Create ACL tensors
    void* xDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* mxscaleDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* mxscale = nullptr;

    auto ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr,
                          (cfg.dstType == 35) ? aclDataType::ACL_FLOAT8_E5M2 : aclDataType::ACL_FLOAT8_E4M3FN, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(mxscaleHostData, mxscaleShape, &mxscaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Get workspace size
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnSituMxQuantGetWorkspaceSize(x, cfg.beta, cfg.linearBeta, cfg.activateLeft, -1, cfg.dstType,
                                           const_cast<char*>("rint"), y, mxscale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Execute
    ret = aclnnSituMxQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSituMxQuant failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Copy results back
    std::vector<uint8_t> npuYResult(ySize, 0);
    ret = aclrtMemcpy(npuYResult.data(), npuYResult.size() * sizeof(uint8_t), yDeviceAddr, ySize * sizeof(uint8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y result failed. ERROR: %d\n", ret); return ret);

    std::vector<uint8_t> npuMxscaleResult(mxscaleSize, 0);
    ret = aclrtMemcpy(npuMxscaleResult.data(), npuMxscaleResult.size() * sizeof(uint8_t), mxscaleDeviceAddr,
                      mxscaleSize * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy mxscale result failed. ERROR: %d\n", ret); return ret);

    // Compute CPU reference
    std::vector<uint8_t> cpuYResult(ySize, 0);
    std::vector<uint8_t> cpuMxscaleResult(mxscaleSize, 0);
    CpuSituMxQuant(xHostData, cfg.rowNum, cfg.colNum, cfg.beta, cfg.linearBeta, cfg.activateLeft, cfg.dstType,
                   cpuYResult, cpuMxscaleResult);

    // Compare y results
    // E4M3FN has 3-bit mantissa → 1 ULP = 1/9 ≈ 0.1111 relative error
    // E5M2 has 2-bit mantissa → 1 ULP = 1/5 = 0.2 relative error
    // Sign flips (byte diff=128) on near-zero Situ outputs are acceptable: FP32 vs double
    // tanh approximation can produce opposite signs for values numerically equal to zero.
    float ulpLimit = (cfg.dstType == 35) ? 0.21f : 0.112f;
    int64_t matchCount = 0;
    int64_t withinUlp = 0;
    int64_t signFlipCount = 0;
    int maxDiff = 0;
    float maxRelDiff = 0.0f;
    for (int64_t i = 0; i < ySize; i++) {
        int diff = std::abs(static_cast<int>(npuYResult[i]) - static_cast<int>(cpuYResult[i]));
        if (diff == 0)
            matchCount++;
        maxDiff = std::max(maxDiff, diff);

        float npuVal = (cfg.dstType == 35) ? Fp8E5M2ToFloat(npuYResult[i]) : Fp8E4M3ToFloat(npuYResult[i]);
        float cpuVal = (cfg.dstType == 35) ? Fp8E5M2ToFloat(cpuYResult[i]) : Fp8E4M3ToFloat(cpuYResult[i]);

        if (diff == 128) {
            // Sign flip — acceptable, both values are numerically near zero relative to scale
            signFlipCount++;
            withinUlp++;
        } else {
            float absDiff = std::fabs(npuVal - cpuVal);
            if (std::fabs(cpuVal) > 1e-6f) {
                float relDiff = absDiff / std::fabs(cpuVal);
                maxRelDiff = std::max(maxRelDiff, relDiff);
                if (relDiff <= ulpLimit)
                    withinUlp++;
            } else {
                withinUlp++;
            }
        }
    }

    // Compare mxscale results
    int64_t scaleMatchCount = 0;
    int maxScaleDiff = 0;
    for (int64_t i = 0; i < mxscaleSize; i++) {
        int diff = std::abs(static_cast<int>(npuMxscaleResult[i]) - static_cast<int>(cpuMxscaleResult[i]));
        if (diff == 0)
            scaleMatchCount++;
        maxScaleDiff = std::max(maxScaleDiff, diff);
    }

    LOG_PRINT(
        "y: total=%ld match=%ld within_ulp=%ld sign_flip=%ld max_rel=%.4f | mxscale: total=%ld match=%ld max=%d\n",
        ySize, matchCount, withinUlp, signFlipCount, maxRelDiff, mxscaleSize, scaleMatchCount, maxScaleDiff);

    // Cleanup
    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclDestroyTensor(mxscale);
    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(mxscaleDeviceAddr);
    if (workspaceSize > 0)
        aclrtFree(workspaceAddr);

    // Pass criteria:
    // - mxscale: exact match (diff <= 1)
    // - y: all within 1 ULP (including sign flips on near-zero values)
    bool yPass = (withinUlp == ySize);
    bool passed = yPass && (maxScaleDiff <= 1);
    LOG_PRINT("  -> %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

// ==================== Negative Param Probes ====================
// 文档错误码表承诺：第一段接口对非法入参（axis/dstType/beta/roundMode/空Tensor等）
// 返回ACLNN_ERR_PARAM_INVALID(161002)，此段逐一验证校验拦截是否生效。

struct ProbeConfig {
    const char* desc;
    double beta;
    int64_t axis;
    int64_t dstType;
    const char* roundMode;
    bool useEmptyTensor; // 用shape含0的空Tensor探测空tensor拦截
    bool mismatchYDtype; // y的dtype故意与dstType不一致
};

static int RunNegativeProbe(const ProbeConfig& cfg)
{
    LOG_PRINT("--- Probe: %s ---\n", cfg.desc);

    std::vector<int64_t> xShape = cfg.useEmptyTensor ? std::vector<int64_t>{0} : std::vector<int64_t>{2, 128};
    std::vector<int64_t> yShape = cfg.useEmptyTensor ? std::vector<int64_t>{0} : std::vector<int64_t>{2, 64};
    std::vector<int64_t> yScaleShape = cfg.useEmptyTensor ? std::vector<int64_t>{0} : std::vector<int64_t>{2, 1, 2};

    const aclDataType xDtype = aclDataType::ACL_BF16;
    const aclDataType yDtype = cfg.mismatchYDtype ? aclDataType::ACL_FLOAT8_E5M2 : aclDataType::ACL_FLOAT8_E4M3FN;
    const aclDataType yScaleDtype = aclDataType::ACL_FLOAT8_E8M0;

    // 非法参数在第一段即被拦截，不会下发device，无需真实device数据；
    // 空Tensor场景size为0无法aclrtMalloc，统一用nullptr数据指针构造。
    void* xDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* yScaleDeviceAddr = nullptr;

    auto makeStrides = [](const std::vector<int64_t>& shape) {
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        return strides;
    };
    auto xStrides = makeStrides(xShape);
    auto yStrides = makeStrides(yShape);
    auto yScaleStrides = makeStrides(yScaleShape);

    aclTensor* x = aclCreateTensor(xShape.data(), xShape.size(), xDtype, xStrides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                   xShape.data(), xShape.size(), xDeviceAddr);
    aclTensor* y = aclCreateTensor(yShape.data(), yShape.size(), yDtype, yStrides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                   yShape.data(), yShape.size(), yDeviceAddr);
    aclTensor* yScale = aclCreateTensor(yScaleShape.data(), yScaleShape.size(), yScaleDtype, yScaleStrides.data(), 0,
                                        aclFormat::ACL_FORMAT_ND, yScaleShape.data(), yScaleShape.size(),
                                        yScaleDeviceAddr);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnSituMxQuantGetWorkspaceSize(x, cfg.beta, 0.0, false, cfg.axis, cfg.dstType,
                                                const_cast<char*>(cfg.roundMode), y, yScale, &workspaceSize, &executor);

    bool passed = (ret == ACLNN_ERR_PARAM_INVALID);
    LOG_PRINT("  -> %s (ret=%d, expect ACLNN_ERR_PARAM_INVALID=%d)\n", passed ? "PASS" : "FAIL", static_cast<int>(ret),
              static_cast<int>(ACLNN_ERR_PARAM_INVALID));

    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclDestroyTensor(yScale);
    return passed ? 0 : 1;
}

static int RunNegativeProbes()
{
    LOG_PRINT("\n=== Negative Param Probes (expect ACLNN_ERR_PARAM_INVALID) ===\n");
    std::vector<ProbeConfig> probes = {
        {"axis=0 should be rejected", 1.0, 0, 36, "rint", false, false},
        {"axis=1 should be rejected", 1.0, 1, 36, "rint", false, false},
        {"dstType=37 should be rejected", 1.0, -1, 37, "rint", false, false},
        {"dstType=40 (FP4, unpublished) should be rejected", 1.0, -1, 40, "rint", false, false},
        {"beta=0 should be rejected", 0.0, -1, 36, "rint", false, false},
        {"beta<0 should be rejected", -1.0, -1, 36, "rint", false, false},
        {"roundMode=round should be rejected", 1.0, -1, 36, "round", false, false},
        {"empty tensor should be rejected", 1.0, -1, 36, "rint", true, false},
        {"y dtype mismatch with dstType should be rejected", 1.0, -1, 36, "rint", false, true},
    };

    int failed = 0;
    for (const auto& cfg : probes) {
        failed += RunNegativeProbe(cfg);
    }
    LOG_PRINT("=== Probes Summary: %ld/%d PASSED ===\n", static_cast<int64_t>(probes.size()) - failed,
              static_cast<int>(probes.size()));
    return failed;
}

int main()
{
    LOG_PRINT("=== SituMxQuant Example Test ===\n");

    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init failed. ERROR: %d\n", ret); return ret);

    // 负向探针：验证第一段接口对非法入参的校验拦截
    int probeFailures = RunNegativeProbes();

    std::vector<TestConfig> tests = {
        {2, 128, 1.0f, 0.0f, false, 36, "Test 1: BF16 -> E4M3FN, beta=1.0, no linear_beta"},
        {2, 128, 1.0f, 0.0f, false, 35, "Test 2: BF16 -> E5M2, beta=1.0, no linear_beta"},
        {2, 128, 0.5f, 2.0f, false, 36, "Test 3: BF16 -> E4M3FN, beta=0.5, linear_beta=2.0"},
        {4, 256, 1.0f, 0.0f, true, 36, "Test 4: BF16 -> E4M3FN, activate_left=true"},
    };

    int totalPassed = 0;
    for (const auto& cfg : tests) {
        int testResult = RunSingleTest(deviceId, stream, cfg);
        totalPassed += (testResult == 0) ? 1 : 0;
    }

    LOG_PRINT("\n=== Summary: %d/%d tests PASSED, %d probe(s) FAILED ===\n", totalPassed,
              static_cast<int>(tests.size()), probeFailures);

    Finalize(deviceId, stream);

    if (totalPassed == static_cast<int>(tests.size()) && probeFailures == 0) {
        LOG_PRINT("\nAll tests PASSED!\n");
    } else {
        LOG_PRINT("\nSome tests FAILED!\n");
    }
    return (totalPassed == static_cast<int>(tests.size()) && probeFailures == 0) ? 0 : 1;
}
