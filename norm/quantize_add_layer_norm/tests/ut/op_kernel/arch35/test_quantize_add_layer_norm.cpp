/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "../quantize_add_layer_norm_tiling_def.h"
#include "arch35/quantize_add_layer_norm.cpp"

namespace {
constexpr uint32_t TEST_VL_FP32 = 64;
constexpr float TEST_EPSILON = 1e-5F;
constexpr const char* TEST_DATA_DIR = "./quantize_add_layer_norm_data";

class GmBuffer {
public:
    explicit GmBuffer(size_t size) : data_(static_cast<uint8_t*>(AscendC::GmAlloc(size))) {}
    ~GmBuffer()
    {
        if (data_ != nullptr) {
            AscendC::GmFree(data_);
        }
    }

    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;
    uint8_t* Get() const { return data_; }

private:
    uint8_t* data_;
};

struct KernelCase {
    int32_t rows;
    int32_t cols;
    int32_t colsPerLoop;
    int32_t colsLoopCount;
    int32_t rowsPerLoop;
    uint32_t blockDim;
    int64_t binaryAddNum;
    int64_t binaryAddK;
    int64_t binaryAddLastNum;
    std::array<uint64_t, 3> tilingKeys;
};

bool ReadExact(const std::string& path, size_t expectedSize, void* buffer)
{
    size_t actualSize = 0;
    return ReadFile(path, actualSize, buffer, expectedSize) && actualSize == expectedSize;
}

std::string FormatFloat(float value)
{
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    return stream.str();
}

int RunGoldenCompare(const std::string& mode, float epsilon, bool hasZeroPoints, bool outputX, bool biasBroadcast)
{
    std::string command = "cd " + std::string(TEST_DATA_DIR) + " && python3 compare_data.py float32 " + mode + " " +
                          FormatFloat(epsilon) + " " + (hasZeroPoints ? "1" : "0") + " " + (outputX ? "1" : "0") + " " +
                          (biasBroadcast ? "1" : "0");
    return std::system(command.c_str());
}

template <bool isWelford, uint32_t quantMode, bool isBiasBroadcast>
void RunModeAndCompare(uint64_t tilingKey, const std::string& mode, uint32_t blockDim, size_t elementCount,
                       const GmBuffer& x1, const GmBuffer& x2, const GmBuffer& gamma, const GmBuffer& beta,
                       const GmBuffer& bias, const GmBuffer& scales, const GmBuffer& zeroPoints,
                       const GmBuffer& scalarScale, const GmBuffer& scalarZeroPoint, GmBuffer& y, GmBuffer& x,
                       GmBuffer& workspace, GmBuffer& tiling, float epsilon, bool hasZeroPoints = true,
                       bool outputX = true)
{
    auto* xData = reinterpret_cast<float*>(x.Get());
    std::fill_n(xData, elementCount, std::numeric_limits<float>::quiet_NaN());
    std::fill_n(y.Get(), elementCount * sizeof(int8_t), static_cast<uint8_t>(0xA5));
    uint8_t* scaleInput = mode == "per_tensor" ? scalarScale.Get() : scales.Get();
    uint8_t* zeroPointInput = nullptr;
    if (hasZeroPoints) {
        zeroPointInput = mode == "per_tensor" ? scalarZeroPoint.Get() : zeroPoints.Get();
    }
    auto* tilingData = reinterpret_cast<QuantizeAddLayerNormRegbaseTilingData*>(tiling.Get());
    tilingData->outputX = outputX ? 1 : 0;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF((quantize_add_layer_norm<isWelford, isBiasBroadcast, quantMode, false, false>), blockDim, x1.Get(),
                x2.Get(), gamma.Get(), beta.Get(), bias.Get(), scaleInput, zeroPointInput, y.Get(), x.Get(),
                workspace.Get(), tiling.Get());

    if (outputX) {
        bool completed = true;
        for (size_t i = 0; i < elementCount; ++i) {
            if (!std::isfinite(xData[i])) {
                completed = false;
                break;
            }
        }
        EXPECT_TRUE(completed) << "kernel did not produce a complete residual output for mode " << mode;
        EXPECT_TRUE(
            WriteFile(std::string(TEST_DATA_DIR) + "/float32_output_x.bin", x.Get(), elementCount * sizeof(float)));
    }
    EXPECT_TRUE(
        WriteFile(std::string(TEST_DATA_DIR) + "/float32_output_y.bin", y.Get(), elementCount * sizeof(int8_t)));
    EXPECT_EQ(RunGoldenCompare(mode, epsilon, hasZeroPoints, outputX, isBiasBroadcast), 0)
        << "golden comparison failed for mode " << mode;
}

template <bool isWelford, bool isBiasBroadcast = false>
void RunKernelCase(const KernelCase& testCase, float epsilon = TEST_EPSILON)
{
    const size_t elementCount = static_cast<size_t>(testCase.rows) * testCase.cols;
    const size_t inputBytes = elementCount * sizeof(float);
    const size_t weightBytes = static_cast<size_t>(testCase.cols) * sizeof(float);
    const size_t yBytes = elementCount * sizeof(int8_t);

    GmBuffer x1(inputBytes);
    GmBuffer x2(inputBytes);
    GmBuffer gamma(weightBytes);
    GmBuffer beta(weightBytes);
    GmBuffer bias(inputBytes);
    GmBuffer scales(weightBytes);
    GmBuffer zeroPoints(weightBytes);
    GmBuffer scalarScale(sizeof(float));
    GmBuffer scalarZeroPoint(sizeof(float));
    GmBuffer y(yBytes);
    GmBuffer x(inputBytes);
    GmBuffer workspace(64);
    GmBuffer tiling(sizeof(QuantizeAddLayerNormRegbaseTilingData));

    ASSERT_NE(x1.Get(), nullptr);
    ASSERT_NE(x2.Get(), nullptr);
    ASSERT_NE(gamma.Get(), nullptr);
    ASSERT_NE(beta.Get(), nullptr);
    ASSERT_NE(bias.Get(), nullptr);
    ASSERT_NE(scales.Get(), nullptr);
    ASSERT_NE(zeroPoints.Get(), nullptr);
    ASSERT_NE(scalarScale.Get(), nullptr);
    ASSERT_NE(scalarZeroPoint.Get(), nullptr);
    ASSERT_NE(y.Get(), nullptr);
    ASSERT_NE(x.Get(), nullptr);
    ASSERT_NE(workspace.Get(), nullptr);
    ASSERT_NE(tiling.Get(), nullptr);

    ASSERT_EQ(std::system("cp -rf ../../../../norm/quantize_add_layer_norm/tests/ut/op_kernel/"
                          "quantize_add_layer_norm_data ./"),
              0);
    const std::string generate = "cd " + std::string(TEST_DATA_DIR) + " && python3 gen_data.py '(" +
                                 std::to_string(testCase.rows) + ", " + std::to_string(testCase.cols) + ")' '(" +
                                 std::to_string(testCase.cols) + ")' float32";
    ASSERT_EQ(std::system(generate.c_str()), 0);

    const std::string prefix = std::string(TEST_DATA_DIR) + "/float32_input_";
    ASSERT_TRUE(ReadExact(prefix + "x1.bin", inputBytes, x1.Get()));
    ASSERT_TRUE(ReadExact(prefix + "x2.bin", inputBytes, x2.Get()));
    ASSERT_TRUE(ReadExact(prefix + "gamma.bin", weightBytes, gamma.Get()));
    ASSERT_TRUE(ReadExact(prefix + "beta.bin", weightBytes, beta.Get()));
    ASSERT_TRUE(ReadExact(prefix + "bias.bin", inputBytes, bias.Get()));
    ASSERT_TRUE(ReadExact(prefix + "scales.bin", weightBytes, scales.Get()));
    ASSERT_TRUE(ReadExact(prefix + "zero_points.bin", weightBytes, zeroPoints.Get()));
    std::copy_n(reinterpret_cast<float*>(scales.Get()), 1, reinterpret_cast<float*>(scalarScale.Get()));
    std::copy_n(reinterpret_cast<float*>(zeroPoints.Get()), 1, reinterpret_cast<float*>(scalarZeroPoint.Get()));

    auto* tilingData = reinterpret_cast<QuantizeAddLayerNormRegbaseTilingData*>(tiling.Get());
    tilingData->rowsPerCore = testCase.rows / static_cast<int32_t>(testCase.blockDim);
    tilingData->rowsPerTailCore = tilingData->rowsPerCore;
    tilingData->rowsPerLoop = testCase.rowsPerLoop;
    tilingData->cols = testCase.cols;
    tilingData->colsPerLoop = testCase.colsPerLoop;
    tilingData->colsLoopCount = testCase.colsLoopCount;
    tilingData->colsTail = testCase.cols % testCase.colsPerLoop;
    if (tilingData->colsTail == 0) {
        tilingData->colsTail = testCase.colsPerLoop;
    }
    tilingData->binaryAddNum = testCase.binaryAddNum;
    tilingData->binaryAddK = testCase.binaryAddK;
    tilingData->binaryAddLastNum = testCase.binaryAddLastNum;
    tilingData->eps = epsilon;
    tilingData->outputX = 1;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    RunModeAndCompare<isWelford, QALN_QUANT_MODE_MUL, isBiasBroadcast>(
        testCase.tilingKeys[0], "mul", testCase.blockDim, elementCount, x1, x2, gamma, beta, bias, scales, zeroPoints,
        scalarScale, scalarZeroPoint, y, x, workspace, tiling, epsilon);
    RunModeAndCompare<isWelford, QALN_QUANT_MODE_DIV, isBiasBroadcast>(
        testCase.tilingKeys[1], "div", testCase.blockDim, elementCount, x1, x2, gamma, beta, bias, scales, zeroPoints,
        scalarScale, scalarZeroPoint, y, x, workspace, tiling, epsilon);
    RunModeAndCompare<isWelford, QALN_QUANT_MODE_PER_TENSOR, isBiasBroadcast>(
        testCase.tilingKeys[2], "per_tensor", testCase.blockDim, elementCount, x1, x2, gamma, beta, bias, scales,
        zeroPoints, scalarScale, scalarZeroPoint, y, x, workspace, tiling, epsilon);
    RunModeAndCompare<isWelford, QALN_QUANT_MODE_DIV, isBiasBroadcast>(
        testCase.tilingKeys[1], "div", testCase.blockDim, elementCount, x1, x2, gamma, beta, bias, scales, zeroPoints,
        scalarScale, scalarZeroPoint, y, x, workspace, tiling, epsilon, false, false);
}
} // namespace

class QuantizeAddLayerNormArch35Test : public testing::Test {};

TEST_F(QuantizeAddLayerNormArch35Test, FullLoad) { RunKernelCase<false>({8, 800, 800, 1, 4, 2, 512, 0, 8, {0, 4, 8}}); }

TEST_F(QuantizeAddLayerNormArch35Test, FullLoadSmallCols)
{
    RunKernelCase<false>({4, 32, 32, 1, 4, 1, TEST_VL_FP32, 0, 1, {0, 4, 8}});
}

TEST_F(QuantizeAddLayerNormArch35Test, FullLoadVectorBoundary)
{
    RunKernelCase<false>({4, 64, 64, 1, 4, 1, TEST_VL_FP32, 0, 1, {0, 4, 8}});
}

TEST_F(QuantizeAddLayerNormArch35Test, FullLoadBroadcastBias)
{
    RunKernelCase<false, true>({4, 64, 64, 1, 4, 1, TEST_VL_FP32, 0, 1, {2, 6, 10}});
}

TEST_F(QuantizeAddLayerNormArch35Test, FullLoadH1MinSubnormalEpsilon)
{
    RunKernelCase<false>({1, 1, 1, 1, 1, 1, TEST_VL_FP32, 0, 1, {0, 4, 8}}, std::numeric_limits<float>::denorm_min());
}

TEST_F(QuantizeAddLayerNormArch35Test, Welford)
{
    RunKernelCase<true>({8, 6400, 3200, 2, 1, 2, 2048, 0, 32, {1, 5, 9}});
}

TEST_F(QuantizeAddLayerNormArch35Test, WelfordMinSubnormalEpsilon)
{
    RunKernelCase<true>({8, 6400, 3200, 2, 1, 2, 2048, 0, 32, {1, 5, 9}}, std::numeric_limits<float>::denorm_min());
}

TEST_F(QuantizeAddLayerNormArch35Test, WelfordNonAlignedTail)
{
    RunKernelCase<true>({8, 6417, 5632, 2, 1, 2, 4096, 0, 64, {1, 5, 9}});
}
