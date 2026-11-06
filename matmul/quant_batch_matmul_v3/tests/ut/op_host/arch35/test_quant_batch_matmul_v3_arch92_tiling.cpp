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
 * \file test_quant_batch_matmul_v3_arch92_tiling.cpp
 * \brief Ascend960 tiling internals and CSV regression registration.
 */

#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"

#define private public
#define protected public
#include "../test_quant_batch_matmul_v3_tiling.h"
#include "../../../../op_host/op_tiling/arch35/qbmm_streamk_tiling_960.h"
#include "../../../../op_host/op_tiling/arch35/quant_batch_matmul_v3_checker_arch_9201.h"
#include "../../../../op_host/op_tiling/arch35/tiling_cost_evaluator.h"
#include "../../../../op_host/op_tiling/arch35/l1_tiling_data_calculator.h"
#include "../../../../op_host/op_tiling/arch35/adaptive_sliding_window_perblock_basic_api_tiling.h"
#undef protected
#undef private

namespace optiling {
namespace {
TEST(QuantBatchMatmulV3Perblock960, KeepsFixedBlockAndTailWindowForBothArchitectures)
{
    for (auto arch : {NpuArch::DAV_9201, NpuArch::DAV_9202}) {
        AdaptiveSlidingWindowPerblockBasicAPITiling tiling(nullptr);
        tiling.compileInfo_.npuArch = arch;
        tiling.adaptiveWin_.baseM = tiling.adaptiveWin_.baseN = tiling.adaptiveWin_.baseK = 256;
        tiling.adaptiveWin_.useTailWinLogic = false;
        ASSERT_TRUE(tiling.CalcBasicBlock());
        EXPECT_EQ(tiling.adaptiveWin_.baseM, 128U);
        EXPECT_EQ(tiling.adaptiveWin_.baseN, 128U);
        EXPECT_EQ(tiling.adaptiveWin_.baseK, 128U);
        EXPECT_TRUE(tiling.adaptiveWin_.useTailWinLogic);
    }
}

class QuantBatchMatmulV3LoadBalance : public testing::Test {
protected:
    void SetUp() override
    {
        params.Reset();
        params.opName = "QuantBatchMatmulV3LoadBalance";
        params.mSize = 768;
        params.nSize = 1024;
        params.kSize = 4096;
        params.batchA = params.batchB = params.batchC = 1;
        params.aDtype = params.bDtype = ge::DT_INT8;
        params.cDtype = ge::DT_BF16;
        params.scaleDtype = ge::DT_FLOAT;
        info.npuArch = NpuArch::DAV_9201;
        info.aicNum = info.aivNum = 32;
        info.l0aSize = info.l0bSize = 64 * 1024;
        info.l0cSize = 256 * 1024;
        info.l1Size = 512 * 1024;
        platform.Init();
    }

    void SetSpecs(const std::string& cores, const std::string& ddr, const std::string& l2, const std::string& freq)
    {
        std::map<std::string, std::string> soc{{"ai_core_cnt", cores}};
        std::map<std::string, std::string> rates{{"ddr_rate", ddr}, {"l2_rate", l2}};
        std::map<std::string, std::string> core{{"cube_freq", freq}};
        platform.SetPlatformRes("SoCInfo", soc);
        platform.SetPlatformRes("AICoreMemoryRates", rates);
        platform.SetPlatformRes("AICoreSpec", core);
    }

    QuantBatchMatmulInfo params;
    QuantBatchMatmulV3CompileInfo info{};
    fe::PlatFormInfos platform;
};

TEST_F(QuantBatchMatmulV3LoadBalance, MissingPlatformDoesNotOverwriteSpecs)
{
    params.mSize = 2048; // One full round avoids the small-shape tile adjustment.
    BaseBlockCalculator calculator(params, info);
    SocSpecifyParams specs;
    specs.freq = 7.0;
    EXPECT_FALSE(calculator.GetSocSpecAdjustBalanceInfo(specs));
    EXPECT_DOUBLE_EQ(specs.freq, 7.0);
    ASSERT_TRUE(calculator.Compute(BaseBlockMode::LOAD_BALANCE));
    EXPECT_EQ(calculator.GetOutput().baseM, 256);
    EXPECT_EQ(calculator.GetOutput().baseN, 256);
    EXPECT_EQ(calculator.GetOutput().baseK, 128);
}

TEST_F(QuantBatchMatmulV3LoadBalance, ReadsBandwidthAndComputePowerInExpectedUnits)
{
    SetSpecs("24", "40", "120", "2000");
    BaseBlockCalculator calculator(params, info, 1, &platform);
    SocSpecifyParams specs;
    ASSERT_TRUE(calculator.GetSocSpecAdjustBalanceInfo(specs));
    EXPECT_DOUBLE_EQ(specs.freq, 2.0);
    EXPECT_DOUBLE_EQ(specs.hbmBw, 1.875);
    EXPECT_DOUBLE_EQ(specs.l2Bw, 5.625);
    EXPECT_DOUBLE_EQ(specs.singleCoreComputePower, 16.0);
}

TEST_F(QuantBatchMatmulV3LoadBalance, InvalidPlatformValuesUsePositiveDefaults)
{
    // Include missing, zero, negative, trailing garbage, invalid and overflowing values.
    for (const std::string value : {"", "0", "-1", "12MHz", "bad", "999999999999999999999999"}) {
        SCOPED_TRACE(value);
        SetSpecs(value, value, value, value);
        BaseBlockCalculator calculator(params, info, 1, &platform);
        SocSpecifyParams specs;
        ASSERT_TRUE(calculator.GetSocSpecAdjustBalanceInfo(specs));
        EXPECT_DOUBLE_EQ(specs.freq, 1.65);
        EXPECT_DOUBLE_EQ(specs.hbmBw, 1.65 * 32 * 31 / 1024);
        EXPECT_DOUBLE_EQ(specs.l2Bw, 1.65 * 32 * 100 / 1024);
        EXPECT_DOUBLE_EQ(specs.singleCoreComputePower, 13.2);
        EXPECT_TRUE(std::isfinite(specs.l2Bw / specs.singleCoreComputePower));
    }
}

TEST_F(QuantBatchMatmulV3LoadBalance, Only9202FourBitGroupsDoubleMmadN)
{
    for (const auto arch : {NpuArch::DAV_9201, NpuArch::DAV_9202}) {
        for (int mode = 0; mode < 3; ++mode) {
            SCOPED_TRACE(mode);
            SCOPED_TRACE(static_cast<int>(arch));
            info.npuArch = arch;
            params.isMxPerGroup = mode == 1;
            params.aDtype = mode == 1 ? ge::DT_FLOAT4_E2M1 : ge::DT_INT8;
            params.scaleDtype = mode == 2 ? Ops::NN::DT_HIFLOAT4_SCALE : ge::DT_FLOAT8_E8M0;
            BaseBlockCalculator calculator(params, info, 1, &platform);
            SocSpecifyParams specs;
            ASSERT_TRUE(calculator.GetSocSpecAdjustBalanceInfo(specs));
            const bool doubled = arch == NpuArch::DAV_9202 && mode != 0;
            EXPECT_EQ(specs.mmadFracN, doubled ? 32 : 16);
            EXPECT_DOUBLE_EQ(specs.singleCoreComputePower, doubled ? 26.4 : 13.2);
        }
    }
}

TEST_F(QuantBatchMatmulV3LoadBalance, PlainTileRespectsSplitL0Buffers)
{
    info.l0cSize = 512 * 1024;
    params.kSize = 128;
    for (bool tall : {false, true}) {
        params.mSize = tall ? 1024 : 768;
        params.nSize = tall ? 768 : 1024;
        BaseBlockCalculator calculator(params, info);
        params.transA = false;
        auto split = calculator.ComputeBaseBlockPlain();
        EXPECT_EQ(split.baseM, 256);
        EXPECT_EQ(split.baseN, 256);
        params.transA = true;
        params.transB = false;
        auto outerK = calculator.ComputeBaseBlockPlain();
        EXPECT_EQ(outerK.baseM, tall ? 512 : 256);
        EXPECT_EQ(outerK.baseN, tall ? 256 : 512);
    }
}

TEST_F(QuantBatchMatmulV3LoadBalance, PertokenUsesOneVectorTileWithoutSmallShapeExpansion)
{
    params.mSize = params.nSize = 256;
    params.isPertoken = true;
    BaseBlockCalculator calculator(params, info);
    ASSERT_TRUE(calculator.Compute(BaseBlockMode::LOAD_BALANCE));
    EXPECT_EQ(calculator.GetOutput().baseM, 128);
    EXPECT_EQ(calculator.GetOutput().baseN, 256);
    EXPECT_LE(calculator.GetOutput().baseM * calculator.GetOutput().baseN, 32768);
}

TEST_F(QuantBatchMatmulV3LoadBalance, ManyRoundsKeepPlainTile)
{
    params.mSize = params.nSize = 8192;
    BaseBlockCalculator calculator(params, info, 1, &platform);
    const auto plain = calculator.ComputeBaseBlockPlain();
    ASSERT_TRUE(calculator.Compute(BaseBlockMode::LOAD_BALANCE));
    EXPECT_EQ(calculator.GetOutput().baseM, plain.baseM);
    EXPECT_EQ(calculator.GetOutput().baseN, plain.baseN);
    EXPECT_EQ(calculator.GetOutput().baseK, plain.baseK);
}

TEST_F(QuantBatchMatmulV3LoadBalance, RejectsUnsupportedArchitectureAndEmptySearchBounds)
{
    BaseBlockCalculator calculator(params, info, 1, &platform);
    info.npuArch = NpuArch::DAV_3510;
    EXPECT_FALSE(calculator.Compute(BaseBlockMode::LOAD_BALANCE));
    info.npuArch = NpuArch::DAV_9201;
    info.l0aSize = 1024;
    EXPECT_FALSE(calculator.Compute(BaseBlockMode::LOAD_BALANCE));
    info.l0aSize = 64 * 1024;
    info.l0bSize = 1024;
    EXPECT_FALSE(calculator.Compute(BaseBlockMode::LOAD_BALANCE));
}

TEST_F(QuantBatchMatmulV3LoadBalance, SearchPreservesAlignmentAndL0CForBothArchitectures)
{
    for (const auto arch : {NpuArch::DAV_9201, NpuArch::DAV_9202}) {
        for (bool transA : {false, true}) {
            for (bool transB : {false, true}) {
                SCOPED_TRACE(static_cast<int>(arch));
                SCOPED_TRACE(transA);
                SCOPED_TRACE(transB);
                info.npuArch = arch;
                params.transA = transA;
                params.transB = transB;
                BaseBlockCalculator calculator(params, info, 1, &platform);
                ASSERT_TRUE(calculator.Compute(BaseBlockMode::LOAD_BALANCE));
                const auto& tile = calculator.GetOutput();
                EXPECT_GT(tile.baseM, 0);
                EXPECT_GT(tile.baseN, 0);
                EXPECT_GT(tile.baseK, 0);
                EXPECT_EQ(tile.baseM % calculator.GetBaseMAlignSize(), 0);
                EXPECT_EQ(tile.baseN % calculator.GetBaseNAlignSize(32), 0);
                EXPECT_EQ(tile.baseK % calculator.GetBaseKAlignSize(), 0);
                EXPECT_LE(tile.baseM * tile.baseN * 4, info.l0cSize);
            }
        }
    }
}

TEST_F(QuantBatchMatmulV3LoadBalance, AFullLoadRequiresReusableBatchEvenMxKAndWindow)
{
    params.mSize = 256;
    params.nSize = 4096;
    params.kSize = 512;
    BaseBlockCalculator calculator(params, info);
    EXPECT_TRUE(calculator.IsBalanceAFullLoad(128, 256));
    params.batchA = 2;
    EXPECT_FALSE(calculator.IsBalanceAFullLoad(128, 256));
    params.batchA = 1;
    params.scaleDtype = ge::DT_FLOAT8_E8M0;
    params.kSize = 32;
    EXPECT_TRUE(calculator.IsMxKOdd());
    EXPECT_FALSE(calculator.IsBalanceAFullLoad(128, 256));
    params.kSize = 64;
    EXPECT_FALSE(calculator.IsMxKOdd());
    params.scaleDtype = ge::DT_FLOAT;
    params.kSize = 8192;
    EXPECT_FALSE(calculator.IsBalanceAFullLoad(128, 256));
    params.kSize = 512;
    params.mSize = 768; // six M blocks cannot divide a 32-core ASW window.
    EXPECT_FALSE(calculator.IsBalanceAFullLoad(128, 256));
    params.mSize = 4096;
    params.nSize = 256;
    EXPECT_FALSE(calculator.IsBalanceAFullLoad(128, 256));
}

TEST_F(QuantBatchMatmulV3LoadBalance, MaxBlockCombinesShapeL0L1AndSplitLimits)
{
    BaseBlockCalculator calculator(params, info);
    EXPECT_EQ(calculator.CalMaxBasicBlockLimited(33, 16, info.l0aSize, false), 48);
    EXPECT_EQ(calculator.CalMaxBasicBlockLimited(1024, 16, info.l0aSize, false), 256);
    info.l1Size = 128 * 1024;
    EXPECT_EQ(calculator.CalMaxBasicBlockLimited(1024, 16, info.l0aSize, false), 128);
    params.transA = true;
    params.transB = false;
    EXPECT_EQ(calculator.CalMaxBasicBlockLimited(1024, 16, info.l0aSize, false), 256);
    info.l0cSize = 512 * 1024;
    EXPECT_EQ(calculator.CalMaxBasicBlockLimited(1024, 16, info.l0aSize, true), 512);
}

TEST_F(QuantBatchMatmulV3LoadBalance, BaseKUsesAlignedShapeOrLargestLegalCandidate)
{
    BaseBlockCalculator calculator(params, info);
    BaseBlockRes tile;
    tile.baseM = tile.baseN = 256;
    params.kSize = 33;
    calculator.GetBaseK(tile);
    EXPECT_EQ(tile.baseK, 64);
    params.kSize = 4096;
    calculator.GetBaseK(tile);
    EXPECT_EQ(tile.baseK, 128);
    tile.baseM = 512;
    calculator.GetBaseK(tile);
    EXPECT_EQ(tile.baseK, 64);
    tile.baseM = 1024;
    calculator.GetBaseK(tile);
    EXPECT_EQ(tile.baseK, 32);
    tile.baseM = 128;
    tile.baseN = 128;
    calculator.GetBaseK(tile);
    EXPECT_EQ(tile.baseK, 256);
    info.l0cSize = 512 * 1024;
    tile.baseM = 512;
    calculator.GetBaseK(tile);
    EXPECT_EQ(tile.baseK, 128);
}

TEST_F(QuantBatchMatmulV3LoadBalance, BaseKHalvesOversizedShapeAndOuterKCapacity)
{
    params.kSize = 8192;
    info.l0aSize = 256 * 1024;
    BaseBlockCalculator calculator(params, info);
    BaseBlockRes tile;
    tile.baseM = tile.baseN = 16;
    calculator.GetBaseK(tile);
    EXPECT_EQ(tile.baseK, 4096);
    params.kSize = 16384;
    calculator.GetBaseK(tile);
    EXPECT_EQ(tile.baseK, 4096);
    params.transA = true;
    calculator.GetBaseK(tile);
    EXPECT_EQ(tile.baseK, 4096);
}

TEST_F(QuantBatchMatmulV3LoadBalance, BaseKRejects32ForMxAndFourBitInputs)
{
    BaseBlockCalculator calculator(params, info);
    for (auto dtype : {ge::DT_INT8, ge::DT_INT4, Ops::NN::DT_HIFLOAT4}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        params.aDtype = dtype;
        params.scaleDtype = ge::DT_FLOAT8_E8M0;
        params.isMxPerGroup = true;
        BaseBlockRes tile;
        tile.baseM = dtype == ge::DT_INT8 ? 1024 : 2048;
        tile.baseN = 16;
        tile.baseK = 64;
        calculator.GetBaseK(tile);
        EXPECT_EQ(tile.baseK, 64); // no legal candidate: preserve the previously valid K.
    }
    params.scaleDtype = ge::DT_FLOAT;
    EXPECT_TRUE(calculator.IsBaseK64Aligned(ge::DT_INT4));
    EXPECT_TRUE(calculator.IsBaseK64Aligned(Ops::NN::DT_HIFLOAT4));
    EXPECT_FALSE(calculator.IsBaseK64Aligned(ge::DT_INT8));
    EXPECT_TRUE(calculator.IsDoubleEqual(1.0, 1.0 + 0.000001));
    EXPECT_FALSE(calculator.IsDoubleEqual(1.0, 1.1));
}

class QuantBatchMatmulV3L1Tiling : public QuantBatchMatmulV3LoadBalance {};

TEST_F(QuantBatchMatmulV3L1Tiling, RejectsZeroBaseAndInsufficientBiasOrScaleReservation)
{
    for (const auto& block : {std::array<uint64_t, 3>{0, 128, 64}, {128, 0, 64}, {128, 128, 0}}) {
        L1TilingDataCalculator calculator(params, info, block[0], block[1], block[2]);
        EXPECT_FALSE(calculator.Compute(L1TilingMode::DEFAULT));
        EXPECT_EQ(calculator.GetOutput().stepKa_, 0);
        EXPECT_EQ(calculator.GetOutput().stepKb_, 0);
    }
    params.hasBias = true;
    params.biasDtype = ge::DT_INT32;
    params.isPerChannel = true;
    for (uint64_t l1Size : {1023UL, 2047UL}) {
        info.l1Size = l1Size;
        L1TilingDataCalculator calculator(params, info, 128, 128, 64);
        EXPECT_FALSE(calculator.Compute(L1TilingMode::DEFAULT));
    }
}

TEST_F(QuantBatchMatmulV3L1Tiling, MxDefaultRejectsSpaceBelowOneDoubleBufferedBlock)
{
    params.aDtype = params.bDtype = ge::DT_FLOAT8_E4M3FN;
    params.scaleDtype = params.perTokenScaleDtype = ge::DT_FLOAT8_E8M0;
    params.isMxPerGroup = true;
    info.l1Size = 16 * 1024;
    L1TilingDataCalculator calculator(params, info, 128, 128, 64);
    EXPECT_FALSE(calculator.Compute(L1TilingMode::DEFAULT));
}

TEST_F(QuantBatchMatmulV3L1Tiling, StreamKRelaxesTailAlignmentButNeverExceedsL1)
{
    params.kSize = 96;
    for (bool transA : {false, true}) {
        SCOPED_TRACE(transA);
        params.transA = transA;
        params.transB = false;
        info.l1Size = 32 * 1024;
        L1TilingDataCalculator calculator(params, info, 128, 128, 64, 96);
        ASSERT_TRUE(calculator.Compute(L1TilingMode::STREAMK));
        const auto& result = calculator.GetOutput();
        EXPECT_EQ(result.stepKa_, 1);
        EXPECT_EQ(result.stepKb_, 1);
        EXPECT_EQ(result.depthKa_, 2);
        EXPECT_EQ(result.depthKb_, 2);
        EXPECT_LE((result.depthKa_ + result.depthKb_) * 128 * 64, info.l1Size);
        --info.l1Size;
        L1TilingDataCalculator insufficient(params, info, 128, 128, 64, 96);
        EXPECT_FALSE(insufficient.Compute(L1TilingMode::STREAMK));
    }
}

TEST_F(QuantBatchMatmulV3L1Tiling, StreamKScaleInitializationRejectsZeroDivisors)
{
    L1TilingDataCalculator calculator(params, info, 128, 128, 64);
    auto& data = calculator.l1TilingData_;
    data.stepKa_ = data.stepKb_ = 1;
    EXPECT_FALSE(calculator.InitStreamKScaleFactors(0, 256, 4096));
    EXPECT_FALSE(calculator.InitStreamKScaleFactors(256, 0, 4096));
    EXPECT_FALSE(calculator.InitStreamKScaleFactors(256, 256, 0));
    data.stepKa_ = 0;
    EXPECT_FALSE(calculator.InitStreamKScaleFactors(256, 256, 4096));
    data.stepKa_ = 1;
    data.stepKb_ = 0;
    EXPECT_FALSE(calculator.InitStreamKScaleFactors(256, 256, 4096));
}

TEST_F(QuantBatchMatmulV3L1Tiling, ScaleSpaceChecksEachSubtractionBeforeSubtracting)
{
    L1TilingDataCalculator calculator(params, info, 128, 128, 64);
    calculator.leftL1Size_ = 1024;
    calculator.l1TilingData_.depthKa_ = calculator.l1TilingData_.depthKb_ = 2;
    uint64_t left = 0;
    EXPECT_FALSE(calculator.GetStreamKScaleLeftL1Size(600, 100, left));
    EXPECT_FALSE(calculator.GetStreamKScaleLeftL1Size(400, 200, left));
    ASSERT_TRUE(calculator.GetStreamKScaleLeftL1Size(100, 200, left));
    EXPECT_EQ(left, 424);
}

TEST_F(QuantBatchMatmulV3L1Tiling, ScaleLimitRejectsZeroAndOverflowWithoutChangingFactors)
{
    L1TilingDataCalculator calculator(params, info, 128, 128, 64);
    auto& data = calculator.l1TilingData_;
    data.scaleFactorA_ = 3;
    data.scaleFactorB_ = 7;
    const auto max = std::numeric_limits<uint64_t>::max();
    for (const auto& divisor : {std::array<uint64_t, 2>{0, 64}, {64, 0}, {max, 1}, {max, 2}}) {
        EXPECT_FALSE(calculator.ApplyStreamKScaleFactorL1Limit(1024, divisor[0], divisor[1]));
        EXPECT_EQ(data.scaleFactorA_, 3);
        EXPECT_EQ(data.scaleFactorB_, 7);
    }
}

TEST_F(QuantBatchMatmulV3L1Tiling, ScaleLimitReusesUnusedSpaceWithoutGrowingRequestedFactors)
{
    L1TilingDataCalculator calculator(params, info, 128, 128, 64);
    // A scale factor costs 32 bytes, B costs 64 bytes; total capacity is 640 bytes.
    for (const auto& factors : {std::array<uint64_t, 4>{2, 12, 2, 9}, {20, 2, 16, 2}, {20, 20, 8, 6}, {2, 3, 2, 3}}) {
        auto& data = calculator.l1TilingData_;
        data.scaleFactorA_ = factors[0];
        data.scaleFactorB_ = factors[1];
        ASSERT_TRUE(calculator.ApplyStreamKScaleFactorL1Limit(640, 32, 64));
        EXPECT_EQ(data.scaleFactorA_, factors[2]);
        EXPECT_EQ(data.scaleFactorB_, factors[3]);
        EXPECT_LE(data.scaleFactorA_, factors[0]);
        EXPECT_LE(data.scaleFactorB_, factors[1]);
        EXPECT_LE(data.scaleFactorA_ * 32 + data.scaleFactorB_ * 64, 640);
    }
}

TEST_F(QuantBatchMatmulV3L1Tiling, FullLoadDepthUsesRemainingSpaceAndRoundsGroupedK)
{
    L1TilingDataCalculator calculator(params, info, 128, 128, 64);
    EXPECT_EQ(calculator.GetDepthB1AfullLoad(32 * 1024), 4);
    EXPECT_EQ(calculator.GetDepthA1BfullLoad(32 * 1024), 4);
    params.kSize = 65;
    params.isMxPerGroup = true;
    params.aDtype = params.bDtype = ge::DT_FLOAT4_E2M1;
    // Full-load buffers must hold two complete 64-element groups of four-bit data.
    EXPECT_EQ(calculator.GetSingleCoreAFullLoadSize(), 128 * 128 / 2);
    EXPECT_EQ(calculator.GetSingleCoreBFullLoadSize(), 128 * 128 / 2);
}

TEST_F(QuantBatchMatmulV3L1Tiling, StepAdjustmentHonorsOtherOperandReservation)
{
    params.kSize = 128;
    for (bool adjustA : {false, true}) {
        for (bool fullLoad : {false, true}) {
            SCOPED_TRACE(adjustA);
            SCOPED_TRACE(fullLoad);
            L1TilingDataCalculator calculator(params, info, 64, 128, 64);
            calculator.isAFullLoad_ = !adjustA && fullLoad;
            calculator.isBFullLoad_ = adjustA && fullLoad;
            uint64_t stepA = 2;
            uint64_t stepB = 2;
            calculator.AdjustStepK(0, stepA, stepB, adjustA);
            EXPECT_EQ(stepA, 2);
            EXPECT_EQ(stepB, 2);
            calculator.AdjustStepK(64 * 1024, stepA, stepB, adjustA);
            EXPECT_TRUE(calculator.CheckL1Size(64 * 1024, stepA, stepB));
            EXPECT_FALSE(calculator.CheckL1Size(64 * 1024, stepA + (adjustA ? 1 : 0), stepB + (adjustA ? 0 : 1)));
        }
    }
}

TEST_F(QuantBatchMatmulV3L1Tiling, CarryPassKeepsExistingStepWhenCapacityWouldShrinkIt)
{
    L1TilingDataCalculator calculator(params, info, 64, 64, 64);
    calculator.l1TilingData_.stepKa_ = 4;
    calculator.l1TilingData_.stepKb_ = 1;
    calculator.CarryDataSizePass(24 * 1024, 16);
    EXPECT_EQ(calculator.GetOutput().stepKa_, 4);
    EXPECT_EQ(calculator.GetOutput().stepKb_, 1);
}

TEST_F(QuantBatchMatmulV3L1Tiling, BalancePassImprovesDivisibleStepsWithinCapacity)
{
    for (bool largerA : {false, true}) {
        L1TilingDataCalculator calculator(params, info, 64, 64, 64);
        calculator.l1TilingData_.stepKa_ = largerA ? 8 : 2;
        calculator.l1TilingData_.stepKb_ = largerA ? 2 : 8;
        calculator.BalanceStepKPass(12 * 8192);
        EXPECT_EQ(calculator.GetOutput().stepKa_, largerA ? 8 : 4);
        EXPECT_EQ(calculator.GetOutput().stepKb_, largerA ? 4 : 8);
        EXPECT_TRUE(calculator.CheckL1Size(12 * 8192, calculator.GetOutput().stepKa_, calculator.GetOutput().stepKb_));
    }
}

TEST_F(QuantBatchMatmulV3L1Tiling, CacheLinePassRejectsZeroAndPreservesFullLoadedOperand)
{
    L1TilingDataCalculator calculator(params, info, 64, 64, 64);
    uint64_t stepA = 1;
    uint64_t stepB = 2;
    for (const auto& cacheLine : {std::array<uint64_t, 2>{0, 128}, {64, 0}}) {
        calculator.L1FullLoadCacheLinePass(stepA, stepB, cacheLine[0], cacheLine[1]);
        calculator.NONL1FullLoadCacheLinePass(stepA, stepB, cacheLine[0], cacheLine[1]);
        EXPECT_EQ(stepA, 1);
        EXPECT_EQ(stepB, 2);
    }
    params.transA = false;
    calculator.isBFullLoad_ = true;
    calculator.L1FullLoadCacheLinePass(stepA, stepB, 64, 128);
    EXPECT_EQ(stepA, 8);
    EXPECT_EQ(stepB, 2);
    calculator.isBFullLoad_ = false;
    params.transB = true;
    stepA = 1;
    stepB = 2;
    calculator.NONL1FullLoadCacheLinePass(stepA, stepB, 64, 128);
    EXPECT_EQ(stepA, 4);
    EXPECT_EQ(stepB, 8);
}

class QuantBatchMatmulV3CostEvaluator : public QuantBatchMatmulV3LoadBalance {};

TEST_F(QuantBatchMatmulV3CostEvaluator, BandwidthScalesWithCoresAndPenalizesOuterKMisalignment)
{
    BaseBlockCalculator calculator(params, info, 1, &platform);
    SocSpecifyParams specs;
    ASSERT_TRUE(calculator.GetSocSpecAdjustBalanceInfo(specs));
    TilingCostEvaluator evaluator(specs);
    CaseInfo tile{256, 256, 4096, 128, 128, 128, 32, 512, 2};
    params.transA = true;
    params.transB = false;
    EXPECT_DOUBLE_EQ(evaluator.GetRealL2Bw(tile, params, info), specs.l2Bw);
    EXPECT_DOUBLE_EQ(evaluator.GetRealHbmBw(tile, params, info), specs.hbmBw);
    tile.usedCoreNum = 16;
    EXPECT_DOUBLE_EQ(evaluator.GetRealL2Bw(tile, params, info), specs.l2Bw / 2);
    EXPECT_DOUBLE_EQ(evaluator.GetRealHbmBw(tile, params, info), specs.hbmBw / 2);
    tile.baseM = 96; // An outer-K input now needs split 128-byte memory bursts.
    EXPECT_DOUBLE_EQ(evaluator.GetRealL2Bw(tile, params, info), specs.l2Bw / 2 * 0.85);
    EXPECT_DOUBLE_EQ(evaluator.GetRealHbmBw(tile, params, info), specs.hbmBw / 2 * 0.85);
}

TEST_F(QuantBatchMatmulV3CostEvaluator, Hif4ScaleTrafficIncreasesCostWithoutChangingMmadWork)
{
    BaseBlockCalculator calculator(params, info, 1, &platform);
    SocSpecifyParams specs;
    ASSERT_TRUE(calculator.GetSocSpecAdjustBalanceInfo(specs));
    TilingCostEvaluator evaluator(specs);
    // A partial K block and 63 M/N tiles exercise both the full and final rounds.
    CaseInfo tile{769, 1025, 514, 128, 128, 64, 32, 256, 2};
    params.mSize = tile.mSize;
    params.nSize = tile.nSize;
    params.kSize = tile.kSize;
    params.aDtype = params.bDtype = Ops::NN::DT_HIFLOAT4;
    params.transA = true;
    params.transB = false;
    AicTimeCost withoutScale{};
    evaluator.EvaluateAicTime(tile, params, info, withoutScale);
    params.isHif4PerGroup = true;
    AicTimeCost withScale{};
    evaluator.EvaluateAicTime(tile, params, info, withScale);
    EXPECT_DOUBLE_EQ(withScale.mmadTotalTime, withoutScale.mmadTotalTime);
    EXPECT_GT(withScale.mte1TotalTime, withoutScale.mte1TotalTime);
    EXPECT_GT(withScale.mte2TotalTime, withoutScale.mte2TotalTime);
    EXPECT_GT(withScale.mmadTotalTime, withScale.mmadLastRoundMaxtime);
    EXPECT_GT(withScale.mte1TotalTime, withScale.mte1LastRoundMaxtime);
    EXPECT_GT(withScale.mte2TotalTime, withScale.mte2LastRoundMaxtime);
    EXPECT_GT(withScale.fixpipeTotalTime, withScale.fixpipeLastRoundMaxtime);
    EXPECT_TRUE(std::isfinite(withScale.totalTime));
    EXPECT_GE(withScale.totalTime, withoutScale.totalTime);

    DequantBmm::QuantBatchMatmulV3BasicAPITilingData data{};
    data.matmulTiling.m = tile.mSize;
    data.matmulTiling.n = tile.nSize;
    data.matmulTiling.k = tile.kSize;
    data.matmulTiling.baseM = tile.baseM;
    data.matmulTiling.baseN = tile.baseN;
    data.matmulTiling.baseK = tile.baseK;
    data.matmulTiling.scaleKL1 = tile.kL1;
    const auto fromTiling = evaluator.Evaluate(data, params, info);
    EXPECT_DOUBLE_EQ(fromTiling.totalTime, withScale.totalTime);
    EXPECT_DOUBLE_EQ(fromTiling.mte2TotalTime, withScale.mte2TotalTime);
}

TEST_F(QuantBatchMatmulV3CostEvaluator, StreamKSeparatesDataParallelWorkAndReductionCost)
{
    BaseBlockCalculator calculator(params, info, 1, &platform);
    SocSpecifyParams specs;
    ASSERT_TRUE(calculator.GetSocSpecAdjustBalanceInfo(specs));
    TilingCostEvaluator evaluator(specs);
    for (bool hasDataParallel : {false, true}) {
        SCOPED_TRACE(hasDataParallel);
        // 33 tiles leave one reduction tile after a full 32-core DP round.
        CaseInfo tile{127, hasDataParallel ? 4223UL : 127UL, 8192, 128, 128, 128, 32, 512, 2};
        params.mSize = tile.mSize;
        params.nSize = tile.nSize;
        params.kSize = tile.kSize;
        const auto cost = evaluator.EvaluateStreamkCost(tile, 1024, params, info);
        EXPECT_TRUE(std::isfinite(cost.totalTime));
        EXPECT_GT(cost.skAicCost.totalTime, 0);
        EXPECT_GT(cost.aivTotalTime, 0);
        EXPECT_GT(cost.workspace2UBCost, 0);
        EXPECT_GT(cost.ub2GMCost, 0);
        EXPECT_DOUBLE_EQ(cost.aicCost, cost.dpAicCost.totalTime + cost.skAicCost.totalTime);
        EXPECT_GE(cost.totalTime, cost.aicCost);
        EXPECT_LE(cost.totalTime, cost.aicCost + cost.aivTotalTime);
        if (hasDataParallel) {
            EXPECT_GT(cost.dpAicCost.totalTime, 0);
        } else {
            EXPECT_DOUBLE_EQ(cost.dpAicCost.totalTime, 0);
            EXPECT_DOUBLE_EQ(cost.totalTime, cost.aicCost + cost.aivTotalTime);
        }
    }
}

gert::StorageShape MakeStorageShape(const gert::Shape& shape)
{
    gert::StorageShape storage;
    storage.MutableOriginShape() = shape;
    storage.MutableStorageShape() = shape;
    return storage;
}

QuantBatchMatmulInfo MakeHif4Info()
{
    QuantBatchMatmulInfo input;
    input.Reset();
    input.opName = "QuantBatchMatmulV3Hif4Checker";
    input.aDtype = input.bDtype = Ops::NN::DT_HIFLOAT4;
    input.scaleDtype = input.perTokenScaleDtype = Ops::NN::DT_HIFLOAT4_SCALE;
    input.mSize = 16;
    input.nSize = 48;
    input.kSize = 130;
    input.groupSizeM = input.groupSizeN = 1;
    input.groupSizeK = 64;
    input.isHif4PerGroup = true;
    return input;
}

TEST(QuantBatchMatmulV3Checker960, Hif4ScaleShapesRespectTransposeAndBroadcast)
{
    auto input = MakeHif4Info();
    for (bool transpose : {false, true}) {
        SCOPED_TRACE(transpose);
        input.transA = input.transB = transpose;
        QuantBatchMatmulV3CheckerArch9201 checker(nullptr, input);
        gert::Shape scale = transpose ? gert::Shape({48, 3, 1}) : gert::Shape({3, 48, 1});
        gert::Shape token = transpose ? gert::Shape({3, 16, 1}) : gert::Shape({16, 3, 1});
        auto tokenStorage = MakeStorageShape(token);
        EXPECT_TRUE(checker.Hif4ScaleShapeCheck(scale));
        EXPECT_TRUE(checker.Hif4PertokenScaleShapeCheck(&tokenStorage));
        for (const auto& broadcast : {gert::Shape({1, 3, 1}), gert::Shape({3, 1, 1})}) {
            auto storage = MakeStorageShape(broadcast);
            EXPECT_TRUE(checker.Hif4ScaleShapeCheck(broadcast));
            EXPECT_TRUE(checker.Hif4PertokenScaleShapeCheck(&storage));
        }
        for (const auto& broadcast : {gert::Shape({1, 48, 1}), gert::Shape({48, 1, 1})}) {
            EXPECT_TRUE(checker.Hif4ScaleShapeCheck(broadcast));
        }
        for (const auto& broadcast : {gert::Shape({1, 16, 1}), gert::Shape({16, 1, 1})}) {
            auto storage = MakeStorageShape(broadcast);
            EXPECT_TRUE(checker.Hif4PertokenScaleShapeCheck(&storage));
        }
        scale.SetDim(transpose ? 1 : 0, 2); // ceil(130/64) is three, not two.
        EXPECT_FALSE(checker.Hif4ScaleShapeCheck(scale));
        token.SetDim(transpose ? 0 : 1, 2);
        tokenStorage = MakeStorageShape(token);
        EXPECT_FALSE(checker.Hif4PertokenScaleShapeCheck(&tokenStorage));
    }
}

TEST(QuantBatchMatmulV3Checker960, Hif4ScaleShapesRejectMalformedRankTailAndMn)
{
    QuantBatchMatmulV3CheckerArch9201 checker(nullptr, MakeHif4Info());
    for (const auto& shape : {gert::Shape({3, 48}), gert::Shape({3, 48, 2}), gert::Shape({3, 47, 1})}) {
        EXPECT_FALSE(checker.Hif4ScaleShapeCheck(shape));
    }
    for (const auto& shape : {gert::Shape({16, 3}), gert::Shape({16, 3, 2}), gert::Shape({15, 3, 1})}) {
        auto storage = MakeStorageShape(shape);
        EXPECT_FALSE(checker.Hif4PertokenScaleShapeCheck(&storage));
    }
}

TEST(QuantBatchMatmulV3Checker960, Hif4GroupValidationRejectsMissingScaleAndWrongGroupAxes)
{
    auto input = MakeHif4Info();
    const gert::Shape scale({3, 48, 1});
    const gert::StorageShape token({16, 3, 1}, {16, 3, 1});
    QuantBatchMatmulV3CheckerArch9201 valid(nullptr, input);
    EXPECT_TRUE(valid.CheckInputValidInHif4PerGroupMode(scale, &token, {130, 16, 130, 48}));
    EXPECT_FALSE(valid.CheckInputValidInHif4PerGroupMode(scale, nullptr, {130, 16, 130, 48}));
    for (int axis = 0; axis < 3; ++axis) {
        auto invalid = input;
        if (axis == 0) {
            invalid.groupSizeM = 2;
        }
        if (axis == 1) {
            invalid.groupSizeN = 2;
        }
        if (axis == 2) {
            invalid.groupSizeK = 32;
        }
        QuantBatchMatmulV3CheckerArch9201 checker(nullptr, invalid);
        EXPECT_FALSE(checker.CheckInputValidInHif4PerGroupMode(scale, &token, {130, 16, 130, 48}));
    }
    EXPECT_FALSE(valid.CheckInputValidInHif4PerGroupMode(scale, &token, {129, 16, 130, 48}));
    EXPECT_FALSE(valid.CheckInputValidInHif4PerGroupMode(scale, &token, {130, 16, 129, 48}));
}

class QuantBatchMatmulV3StreamK960 : public testing::Test {
protected:
    void SetUp() override
    {
        platform.Init();
        raw = gert::TilingData::CreateCap(1024);
        workspace = gert::ContinuousVector::Create<size_t>(1);
        gert::StorageShape shape({16, 16}, {16, 16});
        holder = gert::TilingContextFaker()
                     .SetOpType("QuantBatchMatmulV3")
                     .NodeIoNum(1, 1)
                     .InputShapes({&shape})
                     .OutputShapes({&shape})
                     .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .PlatformInfo(&platform)
                     .CompileInfo(&compileInfo)
                     .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_BF16)},
                                 {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                 {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                 {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                     .TilingData(raw.get())
                     .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
                     .Build();
        ASSERT_NE(holder.GetContext<gert::TilingContext>(), nullptr);
        tiling = std::make_unique<QBMMV3StreamKTiling960>(holder.GetContext<gert::TilingContext>());
        auto& input = tiling->inputParams_;
        input.opName = "QuantBatchMatmulV3StreamK960";
        input.aFormat = input.bFormat = input.cFormat = ge::FORMAT_ND;
        input.aDtype = input.bDtype = ge::DT_INT8;
        input.cDtype = ge::DT_BF16;
        input.scaleDtype = ge::DT_UINT64;
        input.mSize = 256;
        input.nSize = 256;
        input.kSize = 8192;
        input.batchA = input.batchB = input.batchC = 1;
        input.isPerTensor = true;
        auto& info = tiling->compileInfo_;
        info.npuArch = NpuArch::DAV_9201;
        info.aicNum = info.aivNum = 32;
        info.l0aSize = info.l0bSize = 64 * 1024;
        info.l0cSize = 256 * 1024;
        info.l1Size = 512 * 1024;
        tiling->aicoreParams_.aicNum = 32;
        tiling->aicoreParams_.l0cSize = info.l0cSize;
    }

    std::unique_ptr<uint8_t[]> raw;
    std::unique_ptr<uint8_t[]> workspace;
    fe::PlatFormInfos platform;
    QuantBatchMatmulV3CompileInfo compileInfo{};
    gert::KernelRunContextHolder holder;
    std::unique_ptr<QBMMV3StreamKTiling960> tiling;
};

TEST_F(QuantBatchMatmulV3StreamK960, CapabilityRequiresMatchingCoresAndSingleBatch)
{
    EXPECT_EQ(tiling->IsCapable(), IsTensorapiCapable());
    tiling->compileInfo_.aivNum = 64;
    EXPECT_FALSE(tiling->IsCapable());
    tiling->compileInfo_.aivNum = 32;
    tiling->inputParams_.batchC = 2;
    EXPECT_FALSE(tiling->IsCapable());
    tiling->inputParams_.batchC = 1;
    tiling->inputParams_.isPertoken = true;
    EXPECT_FALSE(tiling->IsCapable());
}

TEST_F(QuantBatchMatmulV3StreamK960, FloatingBiasMustMatchScaleAndAccumulationDomain)
{
    auto& input = tiling->inputParams_;
    input.hasBias = true;
    for (auto dtype : {ge::DT_FLOAT, ge::DT_BF16}) {
        input.scaleDtype = input.biasDtype = dtype;
        EXPECT_TRUE(tiling->IsPostDequantBiasInput());
        EXPECT_TRUE(tiling->IsPertensorStreamKInput());
        input.biasDtype = ge::DT_FLOAT16;
        EXPECT_FALSE(tiling->IsPostDequantBiasInput());
        EXPECT_FALSE(tiling->IsPertensorStreamKInput());
    }
    input.aDtype = ge::DT_FLOAT8_E4M3FN;
    input.bDtype = ge::DT_HIFLOAT8;
    EXPECT_FALSE(tiling->IsPertensorStreamKInput());
    input.bDtype = ge::DT_FLOAT8_E5M2;
    input.scaleDtype = input.perTokenScaleDtype = input.biasDtype = ge::DT_FLOAT;
    input.isDoubleScale = true;
    EXPECT_TRUE(tiling->IsPostDequantBiasInput());
    EXPECT_TRUE(tiling->IsPertensorStreamKInput());
    input.aFormat = ge::FORMAT_FRACTAL_NZ;
    EXPECT_FALSE(tiling->IsPertensorStreamKInput());
}

TEST_F(QuantBatchMatmulV3StreamK960, SplitKPartitionsCoverKAndStayWithinCoreCount)
{
    ASSERT_TRUE(tiling->CalcBasicBlockV2());
    const auto& sk = tiling->tilingData_.streamKTiling;
    EXPECT_GT(sk.kCnt, 1);
    EXPECT_GT(sk.singleCoreK, 0);
    EXPECT_EQ(sk.singleCoreK % 256, 0);
    EXPECT_GE(sk.singleCoreK * sk.kCnt, tiling->inputParams_.kSize);
    EXPECT_LT(sk.singleCoreK * (sk.kCnt - 1), tiling->inputParams_.kSize);
    EXPECT_LE(sk.usedCoreNum, tiling->compileInfo_.aicNum);
    tiling->CalL1Tiling();
    EXPECT_GT(sk.kL1, 0);
    EXPECT_EQ(sk.kL1 % tiling->tilingData_.matmulTiling.baseK, 0);
    // At this shape the default bandwidth model favors the ASW schedule.
    EXPECT_FALSE(tiling->IsProfilingBetter());
    tiling->tilingData_.streamKTiling.kCnt = 1;
    EXPECT_FALSE(tiling->IsProfilingBetter());
}

TEST_F(QuantBatchMatmulV3StreamK960, WorkspaceAndSerializedDataMatchKernelContract)
{
    tiling->inputParams_.hasBias = true;
    tiling->inputParams_.biasDtype = ge::DT_INT32;
    ASSERT_EQ(tiling->DoLibApiTiling(), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tiling->GetWorkspaceSize(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling->workspaceSize_, 24 * 1024 * 1024); // 16 MiB overhead + 32 FP32 256x256 tiles.
    tiling->tilingData_.streamKTiling.usedCoreNum = 24;
    tiling->tilingDataSize_ = sizeof(tiling->tilingData_);
    tiling->SetTilingData();
    ASSERT_EQ(tiling->PostTiling(), ge::GRAPH_SUCCESS);
    auto* context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(context->GetBlockDim(), 24);
    EXPECT_EQ(context->GetRawTilingData()->GetDataSize(), sizeof(tiling->tilingData_));
    EXPECT_EQ(context->GetWorkspaceSizes(1)[0], tiling->workspaceSize_);
    const auto* data = reinterpret_cast<const DequantBmm::QuantBatchMatmulV3StreamKBasicAPITilingData*>(
        context->GetRawTilingData()->GetData());
    EXPECT_EQ(data->matmulTiling.m, 256);
    EXPECT_EQ(data->matmulTiling.n, 256);
    EXPECT_EQ(data->matmulTiling.k, 8192);
    EXPECT_EQ(data->matmulTiling.isBias, 1);
    EXPECT_EQ(data->params.biasDtype, static_cast<uint32_t>(ge::DT_INT32));
    EXPECT_EQ(data->adaptiveSlidingWin.windowLen, 4);
    EXPECT_EQ(data->params.x2QuantMode, static_cast<uint32_t>(BasicQuantMode::PERTENSOR_MODE));
}

TEST_F(QuantBatchMatmulV3StreamK960, PostTilingRejectsUnalignedAndOversizedPayload)
{
    tiling->tilingDataSize_ = 1;
    EXPECT_EQ(tiling->PostTiling(), ge::GRAPH_PARAM_INVALID);
    tiling->tilingDataSize_ = 2048;
    EXPECT_EQ(tiling->PostTiling(), ge::GRAPH_PARAM_INVALID);
    EXPECT_EQ(holder.GetContext<gert::TilingContext>()->GetRawTilingData()->GetDataSize(), 0);
}
} // namespace
} // namespace optiling

static const auto kCasesDav9201 = GetParams("Dav9201");
static const auto kCasesDav9202 = GetParams("Dav9202");

INSTANTIATE_TEST_CASE_P(QUANTMM_DAV_9201, TestQuantBatchMatmulV3Tiling, testing::ValuesIn(kCasesDav9201));
INSTANTIATE_TEST_CASE_P(QUANTMM_DAV_9202, TestQuantBatchMatmulV3Tiling, testing::ValuesIn(kCasesDav9202));
