/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <array>
#include <utility>
#include "op_common/op_host/util/op_const_def.h"
#include "../../../../op_host/op_tiling/arch35/adaptive_sliding_window_mix_basic_api_tiling.h"
#include "../../../../op_host/op_tiling/arch35/adaptive_sliding_window_mix_tiling.h"
#include "../../../../op_host/op_tiling/arch35/quant_batch_matmul_v3_iterbatch_tiling.h"
#define private public
#include "../../../../op_host/op_tiling/arch35/adaptive_sliding_window_mx_basic_api_tiling.h"
#undef private
#include "../test_quant_batch_matmul_v3_tiling.h"
#include "../../../../op_host/mx_scale_nz_shape.h"
#include "../../../../op_host/op_tiling/quant_batch_matmul_v3_tiling_cache_key.h"

static const std::vector<QuantBatchMatmulV3TilingTestParam> kCasesParamsDav3510 = GetParams("Dav3510");

INSTANTIATE_TEST_CASE_P(QUANTMM_DAV_3510, TestQuantBatchMatmulV3Tiling, testing::ValuesIn(kCasesParamsDav3510));

TEST_F(TestQuantBatchMatmulV3Tiling, multiThreadDav3510)
{
    TestMultiThread(kCasesParamsDav3510.data(), kCasesParamsDav3510.size(), 3);
}

TEST(QuantBatchMatmulScaleNzShape, CanonicalAxesAndPadding)
{
    const gert::Shape logical({3, 47, 2}); // K=130 -> Kg=3, N has a 16-element tail.
    const gert::Shape packed({3, 3, 16, 2});
    EXPECT_TRUE(QuantBatchMatmul::CheckMxScaleNzShape(logical, packed, 47, 130));
    EXPECT_FALSE(QuantBatchMatmul::CheckMxScaleNzShape(gert::Shape({47, 3, 2}), packed, 47, 130));
    EXPECT_FALSE(QuantBatchMatmul::CheckMxScaleNzShape(logical, gert::Shape({3, 3, 16, 1}), 47, 130));
    EXPECT_FALSE(QuantBatchMatmul::CheckMxScaleNzShape(logical, gert::Shape({2, 3, 16, 2}), 47, 130));
    EXPECT_FALSE(QuantBatchMatmul::CheckMxScaleNzShape(gert::Shape({2, 47, 2}), packed, 47, 130));
}

TEST(QuantBatchMatmulScaleNzShape, BatchRankAndEqualAxes)
{
    EXPECT_TRUE(
        QuantBatchMatmul::CheckMxScaleNzShape(gert::Shape({2, 3, 47, 2}), gert::Shape({2, 3, 3, 16, 2}), 47, 130));
    EXPECT_FALSE(
        QuantBatchMatmul::CheckMxScaleNzShape(gert::Shape({2, 3, 47, 2}), gert::Shape({1, 3, 3, 16, 2}), 47, 130));
    EXPECT_FALSE(QuantBatchMatmul::CheckMxScaleNzShape(gert::Shape({3, 47}), gert::Shape({3, 3, 16}), 47, 130));
    EXPECT_FALSE(QuantBatchMatmul::CheckMxScaleNzShape(gert::Shape({3, 47, 2}), gert::Shape({3, 3, 16, 2}), 0, 130));
    // Shape checks cannot infer the meaning of equal axes or detect an earlier wrong data permutation.
    EXPECT_TRUE(QuantBatchMatmul::CheckMxScaleNzShape(gert::Shape({3, 3, 2}), gert::Shape({1, 3, 16, 2}), 3, 130));
}

TEST(QuantBatchMatmulScaleNzShape, ScaleFormatSeparatesTilingCacheEntries)
{
    optiling::QuantBatchMatmulInfo params;
    params.Reset();
    Ops::NN::Optiling::AiCoreParams cores{};
    params.scaleFormat = ge::FORMAT_ND;
    const optiling::QuantBatchMatmulV3HashInput nd(params, cores);
    params.scaleFormat = ge::FORMAT_FRACTAL_NZ;
    const optiling::QuantBatchMatmulV3HashInput nz(params, cores);
    EXPECT_FALSE(nd == nz);
}

namespace {
class MixCapabilityProbe : public optiling::AdaptiveSlidingWindowMixBasicAPITiling {
public:
    MixCapabilityProbe() : AdaptiveSlidingWindowMixBasicAPITiling(nullptr) {}
    bool Check(const optiling::QuantBatchMatmulInfo& info)
    {
        const auto saved = inputParams_;
        inputParams_ = info;
        const bool result = IsCapable();
        inputParams_ = saved;
        return result;
    }
};

class MixStateProbe : public optiling::AdaptiveSlidingWindowMixTiling {
public:
    MixStateProbe() : AdaptiveSlidingWindowMixTiling(nullptr)
    {
        inputParams_.Reset();
        inputParams_.opName = "MixStateProbe";
        inputParams_.mSize = inputParams_.nSize = 256;
        inputParams_.kSize = 4096;
        inputParams_.aDtype = inputParams_.bDtype = ge::DT_INT8;
        inputParams_.cDtype = ge::DT_BF16;
        basicTiling_.baseM = basicTiling_.baseN = 256;
        adaptiveWin_.mBlockCnt = adaptiveWin_.nBlockCnt = 1;
        enableUncache_ = true;
    }
    using AdaptiveSlidingWindowMixTiling::AnalyseFullLoadInfo;
    using AdaptiveSlidingWindowMixTiling::CheckCoreNum;
    using AdaptiveSlidingWindowMixTiling::GetWorkspaceSize;
    using AdaptiveSlidingWindowMixTiling::IsCapable;
    using AdaptiveSlidingWindowTiling::adaptiveWin_;
    using AdaptiveSlidingWindowTiling::basicTiling_;
    using AdaptiveSlidingWindowTiling::enableUncache_;
    using AdaptiveSlidingWindowTiling::isAFullLoad_;
    using AdaptiveSlidingWindowTiling::SetDisableL2cache;
    using QuantBatchMatmulV3TilingBase::aicoreParams_;
    using QuantBatchMatmulV3TilingBase::compileInfo_;
    using QuantBatchMatmulV3TilingBase::inputParams_;
    using QuantBatchMatmulV3TilingBase::workspaceSize_;
};

class IterBatchProbe : public optiling::QuantBatchMatmulV3IterbatchTiling {
public:
    IterBatchProbe() : QuantBatchMatmulV3IterbatchTiling(nullptr)
    {
        inputParams_.Reset();
        inputParams_.opName = "IterBatchProbe";
        inputParams_.mSize = 16;
        inputParams_.nSize = inputParams_.kSize = 32;
        inputParams_.batchA = inputParams_.batchB = inputParams_.batchC = 64;
        inputParams_.aDtype = inputParams_.bDtype = ge::DT_INT8;
        inputParams_.scaleDtype = ge::DT_UINT64;
        inputParams_.biasDtype = ge::DT_INT32;
        inputParams_.isPerChannel = true;
        aicoreParams_.aicNum = 8;
        aicoreParams_.l1Size = 64 * (16 * 32 + 32 * 32) + 32 * sizeof(uint64_t);
    }
    using QuantBatchMatmulV3IterbatchTiling::CalcIterBatch;
    using QuantBatchMatmulV3IterbatchTiling::GetBroadCastInfo;
    using QuantBatchMatmulV3IterbatchTiling::IsCapable;
    using QuantBatchMatmulV3TilingBase::aicoreParams_;
    using QuantBatchMatmulV3TilingBase::inputParams_;
};

class MxStateProbe : public optiling::AdaptiveSlidingWindowMXBasicAPITiling {
public:
    MxStateProbe() : AdaptiveSlidingWindowMXBasicAPITiling(nullptr)
    {
        inputParams_.Reset();
        inputParams_.opName = "MxStateProbe";
        inputParams_.mSize = inputParams_.nSize = 256;
        inputParams_.kSize = 4096;
        inputParams_.aDtype = inputParams_.bDtype = ge::DT_FLOAT8_E4M3FN;
        inputParams_.cDtype = ge::DT_BF16;
        inputParams_.scaleDtype = inputParams_.perTokenScaleDtype = ge::DT_FLOAT8_E8M0;
        inputParams_.isMxPerGroup = true;
        compileInfo_.npuArch = Ops::Base::DAV_3510;
    }
    using AdaptiveSlidingWindowTiling::adaptiveWin_;
    using QuantBatchMatmulV3TilingBase::aicoreParams_;
    using QuantBatchMatmulV3TilingBase::compileInfo_;
    using QuantBatchMatmulV3TilingBase::inputParams_;
};

optiling::QuantBatchMatmulInfo MakeNdMixInfo()
{
    optiling::QuantBatchMatmulInfo info{};
    info.bFormat = ge::FORMAT_ND;
    info.aDtype = ge::DT_INT8;
    info.bDtype = ge::DT_INT8;
    info.cDtype = ge::DT_BF16;
    info.scaleDtype = ge::DT_FLOAT;
    info.hasBias = true;
    info.biasDtype = ge::DT_INT32;
    info.isPerChannel = true;
    return info;
}
} // namespace

TEST(QuantBatchMatmulAswState, L2CapacityAndWindowReuseDetermineCacheMode)
{
    using DequantBmm::L2CacheMode;
    MixStateProbe tiling;
    const uint64_t totalBytes = 256 * 256 * 2 + 2 * 256 * 4096;
    tiling.compileInfo_.l2Size = totalBytes + 1;
    EXPECT_EQ(tiling.SetDisableL2cache(256, 128, 128, 256), L2CacheMode::B_L2_CACHE_DISABLE);
    tiling.compileInfo_.l2Size = totalBytes;
    EXPECT_EQ(tiling.SetDisableL2cache(256, 128, 128, 256), L2CacheMode::ALL_L2_CACHE_DISABLE);
    tiling.adaptiveWin_.mBlockCnt = 2;
    EXPECT_EQ(tiling.SetDisableL2cache(256, 128, 128, 256), L2CacheMode::A_L2_CACHE_DISABLE);
    tiling.adaptiveWin_.nBlockCnt = 2;
    EXPECT_EQ(tiling.SetDisableL2cache(256, 128, 128, 256), L2CacheMode::L2_CACHE_DEFAULT);
    tiling.adaptiveWin_.mBlockCnt = 1;
    EXPECT_EQ(tiling.SetDisableL2cache(256, 128, 128, 256), L2CacheMode::B_L2_CACHE_DISABLE);
    tiling.enableUncache_ = false;
    EXPECT_EQ(tiling.SetDisableL2cache(256, 128, 128, 256), L2CacheMode::L2_CACHE_DEFAULT);
}

TEST(QuantBatchMatmulAswState, UncacheRequiresAlignedInputAndL1InnerAxes)
{
    using DequantBmm::L2CacheMode;
    MixStateProbe tiling;
    tiling.compileInfo_.l2Size = 0;
    for (bool transA : {false, true}) {
        for (bool transB : {false, true}) {
            tiling.inputParams_.transA = transA;
            tiling.inputParams_.transB = transB;
            EXPECT_EQ(tiling.SetDisableL2cache(256, 128, 128, 256), L2CacheMode::ALL_L2_CACHE_DISABLE);
            EXPECT_EQ(tiling.SetDisableL2cache(64, 64, 128, 256), L2CacheMode::B_L2_CACHE_DISABLE);
            EXPECT_EQ(tiling.SetDisableL2cache(256, 128, 64, 64), L2CacheMode::A_L2_CACHE_DISABLE);
        }
    }
    tiling.inputParams_.transA = true;
    tiling.inputParams_.transB = false;
    tiling.inputParams_.mSize = tiling.inputParams_.nSize = 255;
    EXPECT_EQ(tiling.SetDisableL2cache(256, 128, 128, 256), L2CacheMode::L2_CACHE_DEFAULT);
}

TEST(QuantBatchMatmulAswState, RejectedInt4ModeRestoresTypesAndAcceptedModeReservesUnpackSpace)
{
    MixStateProbe tiling;
    tiling.compileInfo_.supportMmadS8S4 = false;
    tiling.inputParams_.aDtype = tiling.inputParams_.bDtype = ge::DT_INT4;
    tiling.inputParams_.cDtype = ge::DT_INT32;
    EXPECT_FALSE(tiling.IsCapable());
    EXPECT_EQ(tiling.inputParams_.aDtype, ge::DT_INT4);
    EXPECT_EQ(tiling.inputParams_.bDtype, ge::DT_INT4);
    tiling.inputParams_.cDtype = ge::DT_BF16;
    ASSERT_TRUE(tiling.IsCapable());
    EXPECT_EQ(tiling.inputParams_.aDtype, ge::DT_INT8);
    EXPECT_EQ(tiling.inputParams_.bDtype, ge::DT_INT8);
    tiling.inputParams_.libApiWorkSpaceSize = 4096;
    tiling.inputParams_.batchC = 2;
    EXPECT_EQ(tiling.GetWorkspaceSize(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.workspaceSize_, 4096U + 3U * 256U * 4096U);
}

TEST(QuantBatchMatmulAswState, FullLoadCollapsesTailSplitAndBatchDisablesFullLoad)
{
    MixStateProbe tiling;
    tiling.inputParams_.kSize = 1024;
    tiling.inputParams_.batchA = 1;
    tiling.adaptiveWin_.baseM = 128;
    tiling.adaptiveWin_.mBaseTailSplitCnt = 2;
    tiling.adaptiveWin_.mTailMain = 64;
    tiling.adaptiveWin_.nBlockCnt = 2;
    tiling.aicoreParams_.aicNum = 32;
    tiling.AnalyseFullLoadInfo();
    EXPECT_TRUE(tiling.isAFullLoad_);
    EXPECT_EQ(tiling.adaptiveWin_.baseM, 64U);
    EXPECT_EQ(tiling.adaptiveWin_.mBaseTailSplitCnt, 1U);
    EXPECT_EQ(tiling.adaptiveWin_.mTailMain, 0U);
    tiling.inputParams_.batchA = 2;
    tiling.AnalyseFullLoadInfo();
    EXPECT_FALSE(tiling.isAFullLoad_);
}

TEST(QuantBatchMatmulAswState, MixCoreRatioDependsOnArchitecture)
{
    MixStateProbe tiling;
    tiling.compileInfo_.aicNum = 32;
    tiling.compileInfo_.npuArch = Ops::Base::DAV_3510;
    tiling.compileInfo_.aivNum = 64;
    EXPECT_TRUE(tiling.CheckCoreNum());
    tiling.compileInfo_.aivNum = 32;
    EXPECT_FALSE(tiling.CheckCoreNum());
    for (auto arch : {NpuArch::DAV_9201, NpuArch::DAV_9202}) {
        tiling.compileInfo_.npuArch = arch;
        tiling.compileInfo_.aivNum = 32;
        EXPECT_TRUE(tiling.CheckCoreNum());
        tiling.compileInfo_.aivNum = 64;
        EXPECT_FALSE(tiling.CheckCoreNum());
    }
}

TEST(QuantBatchMatmulIterBatchState, BroadcastSpanIncludesOnlyInnerBatchAxes)
{
    for (size_t axis = 0; axis < 4; ++axis) {
        IterBatchProbe tiling;
        auto& info = tiling.inputParams_;
        info.batchC4 = 2;
        info.batchC3 = 3;
        info.batchC2 = 5;
        std::array<uint64_t*, 4> batchB{&info.batchB1, &info.batchB2, &info.batchB3, &info.batchB4};
        *batchB[axis] = 2;
        uint64_t count = 0;
        uint64_t inner = 0;
        bool broadcastA = false;
        bool broadcastB = false;
        tiling.GetBroadCastInfo(count, inner, broadcastA, broadcastB);
        EXPECT_EQ(count, 1U);
        EXPECT_EQ(inner, (std::array<uint64_t, 4>{30, 6, 2, 1})[axis]);
        EXPECT_TRUE(broadcastA);
        EXPECT_FALSE(broadcastB);
    }
}

TEST(QuantBatchMatmulIterBatchState, MixedAndPartialBroadcastAxesRejectIterBatch)
{
    IterBatchProbe tiling;
    auto& info = tiling.inputParams_;
    info.batchA4 = 2;
    info.batchB3 = 2;
    EXPECT_FALSE(tiling.IsCapable());
    info.batchA4 = 1;
    info.batchB4 = 2;
    EXPECT_FALSE(tiling.IsCapable());
    info.batchB2 = 2;
    EXPECT_FALSE(tiling.IsCapable());
    info.batchB1 = 2;
    EXPECT_TRUE(tiling.IsCapable());
}

TEST(QuantBatchMatmulIterBatchState, CapacityReservesBiasAndScaleBeforeCountingBatches)
{
    IterBatchProbe tiling;
    EXPECT_EQ(tiling.CalcIterBatch(), 64U);
    tiling.inputParams_.hasBias = true;
    EXPECT_EQ(tiling.CalcIterBatch(), 63U);
    tiling.aicoreParams_.l1Size = 16 * 32 + 32 * 32 + 32 * (sizeof(uint64_t) + sizeof(int32_t));
    EXPECT_EQ(tiling.CalcIterBatch(), 1U);
    EXPECT_FALSE(tiling.IsCapable());
}

TEST(QuantBatchMatmulIterBatchState, UnsupportedQuantizationAndDtypeRejectIterBatch)
{
    IterBatchProbe tiling;
    tiling.inputParams_.isPerChannel = false;
    tiling.inputParams_.isPertoken = true;
    EXPECT_FALSE(tiling.IsCapable());
    tiling.inputParams_.isPertoken = false;
    tiling.inputParams_.isPerTensor = true;
    tiling.inputParams_.bDtype = ge::DT_INT4;
    EXPECT_FALSE(tiling.IsCapable());
}

TEST(QuantBatchMatmulMxState, PingpongScaleFactorNeverGrowsAndHandlesZeroStep)
{
    MxStateProbe tiling;
    uint32_t factor = 64;
    tiling.AdjustScaleFactorForL0CPingpong(factor, 0, 128);
    EXPECT_EQ(factor, 64U);
    tiling.AdjustScaleFactorForL0CPingpong(factor, 1, 0);
    EXPECT_EQ(factor, 64U);
    tiling.AdjustScaleFactorForL0CPingpong(factor, 2, 128);
    EXPECT_EQ(factor, 8U); // At most 2048 scale elements per L1 window.
    factor = 2;
    tiling.AdjustScaleFactorForL0CPingpong(factor, 2, 128);
    EXPECT_EQ(factor, 2U);
    tiling.AdjustScaleFactorForL0CPingpong(factor, 32, 128);
    EXPECT_EQ(factor, 1U);
}

TEST(QuantBatchMatmulMxState, PingpongWritesAdjustedFactorsBackOnlyOnArch35)
{
    MxStateProbe tiling;
    tiling.inputParams_.mSize = tiling.inputParams_.nSize = 4096;
    tiling.inputParams_.kSize = 8192;
    uint32_t factorA = 64;
    uint32_t factorB = 32;
    const bool pingpong = optiling::IsMxL0CPingpong(tiling.inputParams_);
    EXPECT_EQ(tiling.DeriveScaleKL1(factorA, 1, factorB, 2, 128), pingpong ? 2048U : 8192U);
    EXPECT_EQ(factorA, pingpong ? 16U : 64U);
    EXPECT_EQ(factorB, pingpong ? 8U : 32U);
    tiling.compileInfo_.npuArch = NpuArch::DAV_9201;
    factorA = 64;
    factorB = 32;
    EXPECT_EQ(tiling.DeriveScaleKL1(factorA, 1, factorB, 2, 128), 8192U);
    EXPECT_EQ(factorA, 64U);
    EXPECT_EQ(factorB, 32U);
}

TEST(QuantBatchMatmulMxState, RepeatLoadRatioAccountsForBothOperandWindows)
{
    MxStateProbe tiling;
    tiling.adaptiveWin_.mBlockCnt = 1;
    tiling.adaptiveWin_.nBlockCnt = 1;
    EXPECT_FALSE(tiling.ShouldKeepAFullLoadByRepeatLoadRatio());
    tiling.adaptiveWin_.nBlockCnt = 2;
    EXPECT_TRUE(tiling.ShouldKeepAFullLoadByRepeatLoadRatio()); // Repeated A traffic is 1/3 of the total.
    tiling.adaptiveWin_.mBlockCnt = 8;
    EXPECT_FALSE(tiling.ShouldKeepAFullLoadByRepeatLoadRatio()); // Repeated A traffic is 1/10 of the total.
    tiling.adaptiveWin_.mBlockCnt = 0;
    EXPECT_FALSE(tiling.ShouldKeepAFullLoadByRepeatLoadRatio());
    tiling.adaptiveWin_.mBlockCnt = 1;
    tiling.inputParams_.mSize = tiling.inputParams_.nSize = 0;
    EXPECT_FALSE(tiling.ShouldKeepAFullLoadByRepeatLoadRatio());
}

TEST(QuantBatchMatmulMxState, L1PlanWidensScaleOnlyWhenDataAndScaleBuffersFit)
{
    MxStateProbe tiling;
    const MxStateProbe::MxL1EstimateParams params{256, 512, 64, 128, false};
    // Four FP8 data buffers plus two E8M0 scale buffers (one scale per 32 K elements).
    constexpr uint64_t usedBytes = 4 * (64 + 128) * 256 + 2 * (64 + 128) * 16;
    EXPECT_EQ(tiling.CalcUsedL1Size(params, 4), usedBytes);
    tiling.aicoreParams_.l1Size = usedBytes - 1;
    EXPECT_FALSE(tiling.CanFitL1BufferNum(params, 4));
    tiling.aicoreParams_.l1Size = usedBytes;
    EXPECT_TRUE(tiling.CanFitL1BufferNum(params, 4));
    auto plan = tiling.MakeMxL1Plan(params, 4);
    EXPECT_EQ(plan.kL1, 256U);
    EXPECT_EQ(plan.scaleKL1, 512U);
    tiling.aicoreParams_.l1Size = 4 * (64 + 128) * 256 + 2 * (64 + 128) * 128;
    plan = tiling.MakeMxL1Plan(params, 4);
    EXPECT_EQ(plan.scaleKL1, 4096U);
    EXPECT_EQ(plan.l1BufferNum, 4U);
    plan = tiling.MakeMxL1Plan(params, 2);
    EXPECT_EQ(plan.scaleKL1, 512U);
    EXPECT_EQ(plan.l1BufferNum, 2U);
}

TEST(QbmmMixNdCapability, OrdinaryMixUsesBlazeForEveryTranspose)
{
    MixCapabilityProbe tiling;
    const bool supported = optiling::IsTensorapiCapable();
    for (auto dtype : {ge::DT_INT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8}) {
        for (auto output : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
            if (dtype == ge::DT_INT8 && output == ge::DT_FLOAT) {
                continue;
            }
            for (int transA : {0, 1}) {
                for (int transB : {0, 1}) {
                    auto info = MakeNdMixInfo();
                    info.aDtype = info.bDtype = dtype;
                    info.cDtype = output;
                    info.biasDtype = ge::DT_FLOAT;
                    info.transA = transA;
                    info.transB = transB;
                    EXPECT_EQ(tiling.Check(info), supported);
                    info.isPerChannel = false;
                    info.isPertoken = true;
                    EXPECT_EQ(tiling.Check(info), supported);
                }
            }
        }
    }
}

TEST(QbmmMixNdCapability, Int8BiasAndFp8MixedPairs)
{
    MixCapabilityProbe tiling;
    const bool supported = optiling::IsTensorapiCapable();
    auto info = MakeNdMixInfo();
    for (auto bias : {ge::DT_INT32, ge::DT_FLOAT, ge::DT_BF16}) {
        info.biasDtype = bias;
        EXPECT_EQ(tiling.Check(info), supported);
    }
    info.aDtype = ge::DT_FLOAT8_E4M3FN;
    info.bDtype = ge::DT_FLOAT8_E5M2;
    info.biasDtype = ge::DT_FLOAT;
    EXPECT_EQ(tiling.Check(info), supported);
    std::swap(info.aDtype, info.bDtype);
    EXPECT_EQ(tiling.Check(info), supported);
    info.isPerChannel = false;
    info.isPerTensor = info.isDoubleScale = true;
    info.perTokenScaleDtype = ge::DT_FLOAT;
    EXPECT_EQ(tiling.Check(info), supported);
}

TEST(QbmmMixNdCapability, CubeAndExcludedModesDoNotEnterOrdinaryMix)
{
    MixCapabilityProbe tiling;
    auto info = MakeNdMixInfo();
    info.isPerBlock = true;
    EXPECT_FALSE(tiling.Check(info));
    info = MakeNdMixInfo();
    info.isMxPerGroup = true;
    EXPECT_FALSE(tiling.Check(info));
    info = MakeNdMixInfo();
    info.aDtype = info.bDtype = ge::DT_INT4;
    EXPECT_FALSE(tiling.Check(info));
    info = MakeNdMixInfo();
    info.cDtype = ge::DT_INT32;
    EXPECT_FALSE(tiling.Check(info));
    info = MakeNdMixInfo();
    info.scaleDtype = ge::DT_UINT64;
    EXPECT_FALSE(tiling.Check(info));
    info = MakeNdMixInfo();
    info.isPerChannel = false;
    info.isPerTensor = true; // INT32 bias stays in the Cube per-tensor route.
    EXPECT_FALSE(tiling.Check(info));
    info.biasDtype = ge::DT_FLOAT;
    EXPECT_EQ(tiling.Check(info), optiling::IsTensorapiCapable());
}

TEST(QbmmMixNdCapability, Int8NzMixUsesBlazeForEveryTranspose)
{
    MixCapabilityProbe tiling;
    const bool supported = optiling::IsTensorapiCapable();
    for (auto output : {ge::DT_FLOAT16, ge::DT_BF16}) {
        for (auto bias : {ge::DT_INT32, ge::DT_FLOAT, output}) {
            for (int transA : {0, 1}) {
                for (int transB : {0, 1}) {
                    auto info = MakeNdMixInfo();
                    info.bFormat = ge::FORMAT_FRACTAL_NZ;
                    info.cDtype = output;
                    info.biasDtype = bias;
                    info.transA = transA;
                    info.transB = transB;
                    EXPECT_EQ(tiling.Check(info), supported);
                    info.isPerChannel = false;
                    info.isPertoken = true;
                    EXPECT_EQ(tiling.Check(info), supported);
                }
            }
        }
    }
}
