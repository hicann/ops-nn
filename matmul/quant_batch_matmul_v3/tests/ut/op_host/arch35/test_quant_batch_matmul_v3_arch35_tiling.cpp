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
#include "../../../../op_host/op_tiling/arch35/base_block_calculator.h"
#include "../../../../op_host/op_tiling/arch35/quant_batch_matmul_v3_iterbatch_tiling.h"
#define private public
#include "../../../../op_host/op_tiling/arch35/adaptive_sliding_window_mx_basic_api_tiling.h"
#undef private
#include "../test_quant_batch_matmul_v3_tiling.h"
#include "../../../../op_host/mx_scale_nz_shape.h"
#include "../../../../op_host/op_tiling/quant_batch_matmul_v3_tiling_cache_key.h"
#include "tiling_context_faker.h"
// master 的 OnlyNewMixRoutesUseCommonBaseBlockForLongK 用例直接构造 fe::PlatFormInfos；
// SocSpec 下沉后 base_block_calculator.h 不再传递包含该头，UT 自行补齐。
#include "platform/platform_infos_def.h"

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
    MixCapabilityProbe(gert::TilingContext* context) : AdaptiveSlidingWindowMixBasicAPITiling(context) {}
    using AdaptiveSlidingWindowMixBasicAPITiling::CalcBasicBlock;
    using AdaptiveSlidingWindowTiling::adaptiveWin_;
    using QuantBatchMatmulV3TilingBase::aicoreParams_;
    using QuantBatchMatmulV3TilingBase::compileInfo_;
    using QuantBatchMatmulV3TilingBase::inputParams_;

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
    using AdaptiveSlidingWindowMXBasicAPITiling::GetKernelType;
    using AdaptiveSlidingWindowMXBasicAPITiling::tilingData_;
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
    tiling.mxL0CPingpongEligibility_ = optiling::GetMxL0CPingpongEligibility(tiling.inputParams_,
                                                                             tiling.compileInfo_.npuArch, true);
    ASSERT_TRUE(tiling.mxL0CPingpongEligibility_.legacyTail);
    EXPECT_EQ(tiling.DeriveScaleKL1(factorA, 1, factorB, 2, 128), 2048U);
    EXPECT_EQ(factorA, 16U);
    EXPECT_EQ(factorB, 8U);
    tiling.compileInfo_.npuArch = NpuArch::DAV_9201;
    tiling.mxL0CPingpongEligibility_ = optiling::GetMxL0CPingpongEligibility(tiling.inputParams_,
                                                                             tiling.compileInfo_.npuArch, true);
    EXPECT_FALSE(tiling.mxL0CPingpongEligibility_.explicitPlan);
    factorA = 64;
    factorB = 32;
    EXPECT_EQ(tiling.DeriveScaleKL1(factorA, 1, factorB, 2, 128), 8192U);
    EXPECT_EQ(factorA, 64U);
    EXPECT_EQ(factorB, 32U);
}

TEST(QuantBatchMatmulMxState, UsesLegacyPingpongOnlyWhenExplicitPlanCannotFit)
{
    MxStateProbe tiling;
    auto& cube = tiling.tilingData_.matmulTiling;
    cube.baseM = cube.baseN = 128;
    cube.baseK = 64;
    cube.kBL1 = 256;
    cube.dbL0C = 1;
    tiling.compileInfo_.l0aSize = tiling.compileInfo_.l0bSize = 64 * 1024;
    tiling.compileInfo_.l0cSize = 256 * 1024;
    tiling.inputParams_.mSize = tiling.inputParams_.nSize = 4096;
    tiling.inputParams_.kSize = 8192;
    tiling.inputParams_.cDtype = ge::DT_FLOAT;
    tiling.mxL0CPingpongEligibility_ = optiling::GetMxL0CPingpongEligibility(tiling.inputParams_,
                                                                             tiling.compileInfo_.npuArch, true);
    ASSERT_TRUE(tiling.mxL0CPingpongEligibility_.explicitPlan);
    ASSERT_FALSE(tiling.mxL0CPingpongEligibility_.legacyTail);
    const auto pingpong = static_cast<uint64_t>(optiling::QMMKernelType::NO_VEC_EPILOGUE_WITH_MMAPI_MX_L0C_PINGPONG);
    const auto basic = static_cast<uint64_t>(optiling::QMMKernelType::NO_VEC_EPILOGUE_WITH_MMAPI);
    tiling.ApplyMxL0CPingpongPlan();
    EXPECT_EQ(tiling.GetKernelType(), pingpong);
    EXPECT_EQ(tiling.tilingData_.adaptiveSlidingWin.mL0, 128U);

    // A K-L1 window that cannot produce an explicit plan retains the old route only for legacy-eligible inputs.
    cube.kBL1 = 65;
    tiling.ApplyMxL0CPingpongPlan();
    EXPECT_EQ(tiling.GetKernelType(), basic);
    EXPECT_EQ(tiling.tilingData_.adaptiveSlidingWin.mL0, 0U);
    tiling.inputParams_.cDtype = ge::DT_BF16;
    tiling.mxL0CPingpongEligibility_ = optiling::GetMxL0CPingpongEligibility(tiling.inputParams_,
                                                                             tiling.compileInfo_.npuArch, true);
    ASSERT_TRUE(tiling.mxL0CPingpongEligibility_.legacyTail);
    tiling.ApplyMxL0CPingpongPlan();
    EXPECT_EQ(tiling.GetKernelType(), pingpong);
    EXPECT_EQ(tiling.tilingData_.adaptiveSlidingWin.mL0, 0U);

    tiling.mxL0CPingpongEligibility_ = optiling::GetMxL0CPingpongEligibility(tiling.inputParams_,
                                                                             tiling.compileInfo_.npuArch, false);
    tiling.ApplyMxL0CPingpongPlan();
    EXPECT_EQ(tiling.GetKernelType(), basic);
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
    MxStateProbe::MxL1Plan plan{};
    ASSERT_TRUE(tiling.MakeMxL1Plan(params, 4, plan));
    EXPECT_EQ(plan.kL1, 256U);
    EXPECT_EQ(plan.scaleKL1, 512U);
    tiling.aicoreParams_.l1Size = 4 * (64 + 128) * 256 + 2 * (64 + 128) * 128;
    ASSERT_TRUE(tiling.MakeMxL1Plan(params, 4, plan));
    EXPECT_EQ(plan.scaleKL1, 4096U);
    EXPECT_EQ(plan.l1BufferNum, 4U);
    ASSERT_TRUE(tiling.MakeMxL1Plan(params, 2, plan));
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

TEST(QbmmMixNdCapability, OnlyNewMixRoutesUseCommonBaseBlockForLongK)
{
    using optiling::BaseBlockCalculator;
    using optiling::BaseBlockMode;
    // CalcBasicBlock() 经 context_->GetPlatformInfo() 读平台信息。tiling 上下文的读取端按固定下标取槽：
    // [io]=compileInfo、[io+1]=platformInfo、[io+3]=deterministic，而 faker 按 setter 调用顺序从 [io]
    // 起写入扩展槽。因此必须先挂 CompileInfo 再挂 PlatformInfo，否则 platformInfo 落位到 [io]、
    // GetPlatformInfo() 从 [io+1] 读到未初始化内存并解引用，导致 SEGV。占位 compileInfo 不会被读取
    // （本用例直接设置 tiling.compileInfo_ 成员）。
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::QuantBatchMatmulV3CompileInfo contextCompileInfo{};
    gert::StorageShape shape({16, 16}, {16, 16});
    auto holder = gert::TilingContextFaker()
                      .SetOpType("QuantBatchMatmulV3")
                      .NodeIoNum(1, 1)
                      .InputShapes({&shape})
                      .OutputShapes({&shape})
                      .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .CompileInfo(&contextCompileInfo)
                      .PlatformInfo(&platformInfo)
                      .Build();
    MixCapabilityProbe tiling(holder.GetContext<gert::TilingContext>());
    auto& info = tiling.inputParams_;
    info = MakeNdMixInfo();
    info.opName = "QuantBatchMatmulV3";
    info.mSize = 160;
    info.nSize = 384;
    info.kSize = 16384;
    info.transB = true;
    info.batchA = info.batchB = info.batchC = 1;
    auto& compileInfo = tiling.compileInfo_;
    compileInfo.npuArch = Ops::Base::DAV_3510;
    compileInfo.l0aSize = compileInfo.l0bSize = 64 * 1024;
    compileInfo.l0cSize = tiling.aicoreParams_.l0cSize = 256 * 1024;
    struct TestCase {
        ge::Format format;
        bool transA;
        ge::DataType output;
        ge::DataType bias;
        bool wasSupported;
    };
    const TestCase cases[] = {
        {ge::FORMAT_ND, false, ge::DT_BF16, ge::DT_UNDEFINED, false},
        {ge::FORMAT_FRACTAL_NZ, false, ge::DT_BF16, ge::DT_UNDEFINED, true},
        {ge::FORMAT_FRACTAL_NZ, false, ge::DT_BF16, ge::DT_BF16, true},
        {ge::FORMAT_FRACTAL_NZ, false, ge::DT_BF16, ge::DT_FLOAT, true},
        {ge::FORMAT_FRACTAL_NZ, true, ge::DT_BF16, ge::DT_UNDEFINED, false},
        {ge::FORMAT_FRACTAL_NZ, false, ge::DT_FLOAT16, ge::DT_UNDEFINED, false},
        {ge::FORMAT_FRACTAL_NZ, false, ge::DT_BF16, ge::DT_INT32, false},
    };
    for (auto aicNum : {28U, 32U}) {
        compileInfo.aicNum = aicNum;
        for (const auto& test : cases) {
            SCOPED_TRACE(::testing::Message() << "aicNum=" << aicNum << ", weight format=" << test.format << ", transA="
                                              << test.transA << ", output=" << test.output << ", bias=" << test.bias);
            info.bFormat = test.format;
            info.transA = test.transA;
            info.cDtype = test.output;
            info.hasBias = test.bias != ge::DT_UNDEFINED;
            info.biasDtype = test.bias;
            BaseBlockCalculator calculator(info, compileInfo, info.batchC);
            ASSERT_TRUE(calculator.Compute(BaseBlockMode::DEFAULT));
            ASSERT_TRUE(tiling.CalcBasicBlock());
            const auto& expected = calculator.GetOutput();
            const bool useOptimizedBlock = test.wasSupported && aicNum == 28U;
            EXPECT_EQ(tiling.adaptiveWin_.baseM, useOptimizedBlock ? 160UL : expected.baseM);
            EXPECT_EQ(tiling.adaptiveWin_.baseN, useOptimizedBlock ? 16UL : expected.baseN);
            EXPECT_EQ(tiling.adaptiveWin_.baseK, useOptimizedBlock ? 128UL : expected.baseK);
            EXPECT_EQ(tiling.adaptiveWin_.useTailWinLogic, expected.useTailWinLogic);
        }
    }
}

namespace {
class MxTiling : public optiling::AdaptiveSlidingWindowMXBasicAPITiling {
public:
    explicit MxTiling(ge::DataType dtype = ge::DT_FLOAT8_E4M3FN) : AdaptiveSlidingWindowMXBasicAPITiling(nullptr)
    {
        inputParams_.Reset();
        inputParams_.aDtype = inputParams_.bDtype = dtype;
    }
};
constexpr uint64_t L0_AB_BYTES = 65536;
constexpr uint64_t L0_C_BYTES = 262144;
} // namespace

TEST(MxL0TilePlan, SelectsMOrNAndPreservesFallback)
{
    MxTiling tiling;
    const auto choose = [&tiling](uint64_t m, uint64_t n, uint64_t k) {
        return tiling.SelectMxL0TilePlan(m, n, k, L0_AB_BYTES, L0_AB_BYTES, L0_C_BYTES, false, true);
    };
    auto plan = choose(128, 128, 256);
    EXPECT_EQ(plan.mL0, 128U);
    EXPECT_EQ(plan.nL0, 128U);
    EXPECT_EQ(plan.baseK, 128U);
    plan = choose(256, 256, 256);
    EXPECT_EQ(plan.mL0, 256U);
    EXPECT_EQ(plan.nL0, 128U);
    EXPECT_EQ(plan.baseK, 128U);
    plan = choose(256, 144, 256);
    EXPECT_EQ(plan.mL0, 128U);
    EXPECT_EQ(plan.nL0, 144U);
    EXPECT_EQ(plan.baseK, 128U);
    EXPECT_EQ(choose(256, 256, 64).baseK, 64U);
    EXPECT_EQ(choose(0, 128, 256).mL0, 0U);
    EXPECT_EQ(choose(512, 512, 256).mL0, 0U);
    EXPECT_EQ(choose(256, 256, 63).mL0, 0U);
    EXPECT_EQ(tiling.SelectMxL0TilePlan(256, 256, 256, 1024, 1024, L0_C_BYTES, false, true).mL0, 0U);
}

TEST(MxL0TilePlan, UsesFp4CapacityAndTransposeAlignment)
{
    MxTiling fp8;
    const auto fp8Plan = fp8.SelectMxL0TilePlan(256, 256, 512, L0_AB_BYTES, L0_AB_BYTES, L0_C_BYTES, false, true);
    EXPECT_EQ(fp8Plan.baseK, 128U);
    EXPECT_NE(fp8.SelectMxL0TilePlan(32, 128, 256, L0_AB_BYTES, L0_AB_BYTES, L0_C_BYTES, true, true).mL0, 0U);
    EXPECT_NE(fp8.SelectMxL0TilePlan(128, 32, 256, L0_AB_BYTES, L0_AB_BYTES, L0_C_BYTES, false, false).mL0, 0U);
    MxTiling fp4(ge::DT_FLOAT4_E2M1);
    const auto fp4Plan = fp4.SelectMxL0TilePlan(256, 256, 512, L0_AB_BYTES, L0_AB_BYTES, L0_C_BYTES, false, true);
    EXPECT_EQ(fp4Plan.baseK, 256U);
    EXPECT_EQ(fp4Plan.mL0, 256U);
    EXPECT_EQ(fp4Plan.nL0, 128U);
    const auto splitM = fp4.SelectMxL0TilePlan(256, 144, 512, L0_AB_BYTES, L0_AB_BYTES, L0_C_BYTES, false, true);
    EXPECT_EQ(splitM.mL0, 128U);
    EXPECT_EQ(splitM.nL0, 144U);
    EXPECT_EQ(fp4.SelectMxL0TilePlan(32, 128, 256, L0_AB_BYTES, L0_AB_BYTES, L0_C_BYTES, true, true).mL0, 0U);
    EXPECT_EQ(fp4.SelectMxL0TilePlan(128, 32, 256, L0_AB_BYTES, L0_AB_BYTES, L0_C_BYTES, false, false).mL0, 0U);
}

TEST(MxL0TilePlan, BoundsComputeKBeforeAlignmentAndDivisibility)
{
    // The 10/20/66352 regression uses a 16x16 tile and kL1=8192. FP4 capacity permits
    // K=4096, but MMAD cannot encode it. Legal values above 2048 must remain available.
    const std::pair<uint64_t, uint32_t> cases[] = {
        {4096, 2048}, {8064, 4032}, {8192, 2048}, {8256, 2752}, {16384, 2048}};
    MxTiling tiling(ge::DT_FLOAT4_E2M1);
    for (const auto& test : cases) {
        SCOPED_TRACE(::testing::Message() << "kL1=" << test.first);
        const auto plan = tiling.SelectMxL0TilePlan(16, 16, test.first, L0_AB_BYTES, L0_AB_BYTES, L0_C_BYTES, false,
                                                    true);
        ASSERT_EQ(plan.mL0, 16U);
        ASSERT_EQ(plan.nL0, 16U);
        ASSERT_EQ(plan.baseK, test.second);
        EXPECT_LT(plan.baseK, 4096U);
        EXPECT_EQ(plan.baseK % 64U, 0U);
        EXPECT_EQ(test.first % plan.baseK, 0U);
    }
}

TEST(MxL0TilePlan, BoundsComputeMAndNIndependentlyOfCapacity)
{
    // Deliberately enlarge capacity so it cannot hide an MMAD dimension overflow.
    constexpr uint64_t LARGE_L0_BYTES = 16UL * 1024UL * 1024UL;
    MxTiling tiling;
    const auto choose = [&tiling](uint64_t m, uint64_t n) {
        return tiling.SelectMxL0TilePlan(m, n, 128, LARGE_L0_BYTES, LARGE_L0_BYTES, LARGE_L0_BYTES, false, true);
    };
    EXPECT_EQ(choose(4080, 16).mL0, 4080U);
    EXPECT_EQ(choose(16, 4080).nL0, 4080U);
    // The L1 dimension may exceed the instruction range if a legal compute split exists.
    EXPECT_EQ(choose(4096, 16).mL0, 2048U);
    EXPECT_EQ(choose(16, 4096).nL0, 2048U);
    EXPECT_EQ(choose(8192, 16).mL0, 0U);
    EXPECT_EQ(choose(16, 8192).nL0, 0U);
}

TEST(QuantBatchMatmulMxState, ChecksBiasTableUsingL1StrideBeforeSelectingL0Plan)
{
    for (auto dtype : {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E4M3FN}) {
        for (bool explicitPlan : {false, true}) {
            for (uint8_t dbL0C : {1U, 2U}) {
                for (uint32_t baseN : {512U, 528U, 1024U}) {
                    MxStateProbe tiling;
                    tiling.inputParams_.aDtype = tiling.inputParams_.bDtype = dtype;
                    tiling.inputParams_.transA = false;
                    tiling.inputParams_.transB = true;
                    tiling.inputParams_.hasBias = true;
                    tiling.inputParams_.biasDtype = ge::DT_FLOAT;
                    tiling.compileInfo_.l0aSize = tiling.compileInfo_.l0bSize = L0_AB_BYTES;
                    tiling.compileInfo_.l0cSize = L0_C_BYTES;
                    tiling.mxL0CPingpongEligibility_ = {explicitPlan, !explicitPlan};
                    auto& cube = tiling.tilingData_.matmulTiling;
                    cube.baseM = 16;
                    cube.baseN = baseN;
                    cube.baseK = 64;
                    cube.kAL1 = cube.kBL1 = 128;
                    cube.dbL0C = dbL0C;
                    SCOPED_TRACE(::testing::Message() << "dtype=" << dtype << ", baseN=" << baseN
                                                      << ", explicit=" << explicitPlan << ", dbL0C=" << +dbL0C);
                    EXPECT_EQ(tiling.ApplyMxL0CPingpongPlan(), baseN == 512U);
                    if (baseN > 512U) {
                        EXPECT_EQ(cube.dbL0C, dbL0C);
                        EXPECT_EQ(tiling.tilingData_.adaptiveSlidingWin.mL0, 0U);
                    }
                    // No-bias callers, including InplaceAdd, do not consume BT.
                    tiling.inputParams_.hasBias = false;
                    EXPECT_TRUE(tiling.ApplyMxL0CPingpongPlan());
                    // Paths outside MX ping-pong eligibility retain their existing validation.
                    tiling.inputParams_.hasBias = true;
                    tiling.mxL0CPingpongEligibility_ = {};
                    EXPECT_TRUE(tiling.ApplyMxL0CPingpongPlan());
                }
            }
        }
    }
}

TEST(MxL0TilePlan, KeepsComputeDimensionsLegalAcrossMxTypesAndTransposes)
{
    // Match the dtypes admitted by IsMxBasicApiCapable; FP4 E1M2 is not an enabled route.
    for (auto dtype : {ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2}) {
        MxTiling tiling(dtype);
        const uint64_t elementBits = dtype == ge::DT_FLOAT4_E2M1 ? 4 : 8;
        for (bool transA : {false, true}) {
            for (bool transB : {false, true}) {
                const uint64_t m = transA ? 64 : 16;
                const uint64_t n = transB ? 16 : 64;
                for (uint64_t kL1 : {64UL, 256UL, 4096UL, 8064UL, 8192UL, 16384UL}) {
                    SCOPED_TRACE(::testing::Message() << "dtype=" << dtype << ", transA=" << transA
                                                      << ", transB=" << transB << ", kL1=" << kL1);
                    const auto plan = tiling.SelectMxL0TilePlan(m, n, kL1, L0_AB_BYTES, L0_AB_BYTES, L0_C_BYTES, transA,
                                                                transB);
                    ASSERT_NE(plan.baseK, 0U);
                    EXPECT_LT(plan.baseK, 4096U);
                    EXPECT_LT(plan.mL0, 4096U);
                    EXPECT_LT(plan.nL0, 4096U);
                    EXPECT_EQ(plan.baseK % 64U, 0U);
                    EXPECT_EQ(kL1 % plan.baseK, 0U);
                    EXPECT_LE(plan.mL0 * plan.baseK * elementBits / 8, L0_AB_BYTES / 2);
                    EXPECT_LE(plan.nL0 * plan.baseK * elementBits / 8, L0_AB_BYTES / 2);
                    EXPECT_LE(plan.mL0 * plan.nL0 * sizeof(float), L0_C_BYTES / 2);
                }
            }
        }
    }
}

namespace {
class MxL1Probe : public optiling::AdaptiveSlidingWindowMXBasicAPITiling {
public:
    MxL1Probe() : AdaptiveSlidingWindowMXBasicAPITiling(nullptr)
    {
        inputParams_.Reset();
        inputParams_.kSize = 6144;
        inputParams_.aDtype = inputParams_.bDtype = ge::DT_FLOAT8_E4M3FN;
        inputParams_.scaleDtype = inputParams_.perTokenScaleDtype = ge::DT_FLOAT8_E8M0;
        inputParams_.isMxPerGroup = true;
        inputParams_.batchC = 1;
        compileInfo_.npuArch = NpuArch::DAV_3510;
        inputParams_.transA = false;
        inputParams_.transB = true;
        aicoreParams_.l1Size = 524288;
    }
    bool IsTensorApiEnabled() const override { return tensorApiEnabled; }
    bool tensorApiEnabled = true;
    using AdaptiveSlidingWindowMXBasicAPITiling::aicoreParams_;
    using AdaptiveSlidingWindowMXBasicAPITiling::basicTiling_;
    using AdaptiveSlidingWindowMXBasicAPITiling::compileInfo_;
    using AdaptiveSlidingWindowMXBasicAPITiling::inputParams_;
    using AdaptiveSlidingWindowMXBasicAPITiling::tilingData_;
};
} // namespace

TEST(MxL1ThreeBufferPlan, PreservesFeasibleWindowsAndSupportsFp4)
{
    MxL1Probe tiling;
    for (uint64_t window : {2048UL, 4096UL}) {
        MxL1Probe::MxL1EstimateParams params{256, window, 256, 256, false};
        const auto plan = tiling.SelectMxL1Plan(params, 128, window);
        EXPECT_EQ(plan.l1BufferNum, 3U);
        EXPECT_EQ(plan.scaleKL1, window);
    }
    MxL1Probe::MxL1EstimateParams params{256, 6144, 256, 256, false};
    EXPECT_TRUE(tiling.TryShrinkMxThreeBufferScaleKL1(params));
    EXPECT_EQ(params.scaleKL1, 4096UL);
    tiling.inputParams_.aDtype = tiling.inputParams_.bDtype = ge::DT_FLOAT4_E2M1;
    params = {512, 6144, 256, 256, false};
    EXPECT_TRUE(tiling.TryShrinkMxThreeBufferScaleKL1(params));
    EXPECT_EQ(params.scaleKL1, 4096UL);
    params = {256, 6144, 256, 256, false};
    const auto plan = tiling.SelectMxL1Plan(params, 128, params.scaleKL1);
    EXPECT_EQ(plan.l1BufferNum, 4U);
    EXPECT_EQ(plan.scaleKL1, 6144UL);
}

TEST(MxL1ThreeBufferPlan, PreservesTensorApiEligibility)
{
    MxL1Probe tiling;
    const MxL1Probe::MxL1EstimateParams params{256, 6144, 256, 256, false};
    tiling.tensorApiEnabled = false;
    EXPECT_FALSE(tiling.CanFitL1BufferNum(params, 3));
    const auto plan = tiling.SelectMxL1Plan(params, 128, params.scaleKL1);
    EXPECT_EQ(plan.l1BufferNum, 2U);
    EXPECT_EQ(plan.scaleKL1, params.scaleKL1);
}

TEST(MxL1ThreeBufferPlan, KeepsOriginalFourBufferCandidates)
{
    MxL1Probe tiling;
    tiling.inputParams_.kSize = 8192;
    auto plan = tiling.SelectMxL1Plan({512, 8192, 80, 160, false}, 128, 8192);
    EXPECT_EQ(plan.l1BufferNum, 4U);
    EXPECT_EQ(plan.kL1, 256UL);
    EXPECT_EQ(plan.scaleKL1, 8192UL);
    tiling.inputParams_.kSize = 5120;
    tiling.inputParams_.aDtype = tiling.inputParams_.bDtype = ge::DT_FLOAT4_E2M1;
    plan = tiling.SelectMxL1Plan({1024, 5120, 80, 144, false}, 256, 5120);
    EXPECT_EQ(plan.l1BufferNum, 4U);
    EXPECT_EQ(plan.kL1, 512UL);
    EXPECT_EQ(plan.scaleKL1, 5120UL);
}

TEST(MxL1ThreeBufferPlan, PreservesWindowOnFailure)
{
    MxL1Probe tiling;
    MxL1Probe::MxL1EstimateParams params{256, 6144, 256, 256, false};
    tiling.aicoreParams_.l1Size = 410000;
    EXPECT_FALSE(tiling.TryShrinkMxThreeBufferScaleKL1(params));
    EXPECT_EQ(params.scaleKL1, 6144UL);
    params.kL1 = 0;
    EXPECT_FALSE(tiling.TryShrinkMxThreeBufferScaleKL1(params));
    params.kL1 = 255;
    EXPECT_FALSE(tiling.TryShrinkMxThreeBufferScaleKL1(params));
    params.kL1 = 256;
    tiling.inputParams_.kSize = 0;
    EXPECT_FALSE(tiling.TryShrinkMxThreeBufferScaleKL1(params));
    EXPECT_EQ(params.scaleKL1, 6144UL);
}
