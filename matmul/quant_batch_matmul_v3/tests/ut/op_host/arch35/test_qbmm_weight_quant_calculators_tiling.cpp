/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "../../../../op_host/op_tiling/arch35/base_block_calculator.h"
#include "../../../../op_host/op_tiling/arch35/l1_tiling_data_calculator.h"
#include "../../../../op_host/op_tiling/arch35/quant_batch_matmul_v3_tiling_util.h"

namespace {
using namespace optiling;

QuantBatchMatmulInfo MakeMxA8W4Input(bool hif4, uint64_t k)
{
    QuantBatchMatmulInfo input;
    input.Reset();
    input.opName = "QuantBatchMatmulV4";
    input.mSize = 128;
    input.nSize = 1024;
    input.kSize = k;
    input.batchA = input.batchB = input.batchC = 1;
    input.transB = true;
    input.aDtype = ge::DT_FLOAT8_E4M3FN;
    input.bDtype = hif4 ? Ops::NN::DT_HIFLOAT4 : ge::DT_FLOAT4_E2M1;
    input.cDtype = input.biasDtype = ge::DT_FLOAT16;
    input.perTokenScaleDtype = ge::DT_FLOAT8_E8M0;
    input.scaleDtype = hif4 ? Ops::NN::DT_HIFLOAT4_SCALE : ge::DT_FLOAT8_E8M0;
    input.aFormat = ge::FORMAT_ND;
    input.bFormat = ge::FORMAT_FRACTAL_NZ;
    input.isMxA8W4 = true;
    input.isMxPerGroup = !hif4;
    input.groupSizeM = input.groupSizeN = 1;
    input.groupSizeK = 32;
    return input;
}

QuantBatchMatmulV3CompileInfo MakePlatform(NpuArch arch)
{
    QuantBatchMatmulV3CompileInfo info{};
    info.npuArch = arch;
    info.aicNum = 64;
    info.l1Size = 512 * 1024;
    info.l0aSize = info.l0bSize = 64 * 1024;
    info.l0cSize = 512 * 1024;
    return info;
}

TEST(QuantBatchMatmulMxA8W4Calculators, ScaleCountsUseEachOperandType)
{
    // K=72 has two physical 64-K groups: A has four E8M0 elements, HIF4 B has two 4-byte elements.
    EXPECT_EQ(GetScaleBaseK(ge::DT_FLOAT8_E8M0, 72), 4UL);
    EXPECT_EQ(GetScaleBaseK(Ops::NN::DT_HIFLOAT4_SCALE, 72), 2UL);
    EXPECT_EQ(GetSizeWithDataType(GetScaleBaseK(ge::DT_FLOAT8_E8M0, 72), ge::DT_FLOAT8_E8M0), 4UL);
    EXPECT_EQ(GetSizeWithDataType(GetScaleBaseK(Ops::NN::DT_HIFLOAT4_SCALE, 72), Ops::NN::DT_HIFLOAT4_SCALE), 8UL);
    EXPECT_EQ(GetPergroupQuantMode(ge::DT_FLOAT8_E8M0), BasicQuantMode::MX_PERGROUP_MODE);
    EXPECT_EQ(GetPergroupQuantMode(Ops::NN::DT_HIFLOAT4_SCALE), BasicQuantMode::HIF4_PERGROUP_MODE);
    EXPECT_DOUBLE_EQ(GetDtypeBytes(ge::DT_FLOAT8_E4M3FN), 1.0);
    EXPECT_DOUBLE_EQ(GetDtypeBytes(ge::DT_FLOAT4_E2M1), 0.5);
    EXPECT_DOUBLE_EQ(GetDtypeBytes(Ops::NN::DT_HIFLOAT4), 0.5);
    auto input = MakeMxA8W4Input(true, 72);
    EXPECT_TRUE(HasPerGroupScale(input));
    input.Reset();
    EXPECT_FALSE(input.isMxA8W4);
    EXPECT_FALSE(HasPerGroupScale(input));
}

TEST(QuantBatchMatmulMxA8W4Calculators, MxA8W4Hif4DoesNotDouble9202ComputePower)
{
    auto input = MakeMxA8W4Input(true, 128);
    const auto info = MakePlatform(NpuArch::DAV_9202);
    fe::PlatFormInfos platform;
    platform.Init();
    BaseBlockCalculator mxA8W4(input, info, 1, &platform);
    SocSpecifyParams mxA8W4Spec;
    ASSERT_TRUE(mxA8W4.GetSocSpecAdjustBalanceInfo(mxA8W4Spec));
    EXPECT_EQ(mxA8W4Spec.mmadFracN, 16UL);
    EXPECT_EQ(mxA8W4Spec.mmadFracK, 32UL);

    input.aDtype = Ops::NN::DT_HIFLOAT4;
    input.perTokenScaleDtype = Ops::NN::DT_HIFLOAT4_SCALE;
    input.isMxA8W4 = false;
    input.isHif4PerGroup = true;
    BaseBlockCalculator sameType(input, info, 1, &platform);
    SocSpecifyParams sameTypeSpec;
    ASSERT_TRUE(sameType.GetSocSpecAdjustBalanceInfo(sameTypeSpec));
    EXPECT_EQ(sameTypeSpec.mmadFracN, 32UL);
    EXPECT_EQ(sameTypeSpec.mmadFracK, 32UL);
    EXPECT_DOUBLE_EQ(sameTypeSpec.singleCoreComputePower, 4.0 * mxA8W4Spec.singleCoreComputePower);
}

TEST(QuantBatchMatmulMxA8W4Calculators, BaseKStays64AlignedAndFitsUnsplitL0)
{
    for (auto arch : {NpuArch::DAV_9201, NpuArch::DAV_9202}) {
        const auto info = MakePlatform(arch);
        fe::PlatFormInfos platform;
        platform.Init();
        for (bool hif4 : {false, true}) {
            for (uint64_t k : {8UL, 64UL, 96UL, 192UL, 4096UL}) {
                auto input = MakeMxA8W4Input(hif4, k);
                BaseBlockCalculator calc(input, info, 1, &platform);
                ASSERT_TRUE(calc.Compute(BaseBlockMode::LOAD_BALANCE));
                const auto& block = calc.GetOutput();
                EXPECT_GT(block.baseK, 0UL);
                EXPECT_EQ(block.baseK % 64UL, 0UL);
                EXPECT_LE(block.baseM * block.baseK * 2UL, info.l0aSize);
                EXPECT_LE(block.baseN * block.baseK, info.l0bSize); // B is packed 4-bit, double buffered.
            }
        }
    }
}

TEST(QuantBatchMatmulMxA8W4Calculators, L1WindowsFitWithIndependentScaleBudgets)
{
    constexpr uint64_t baseM = 64;
    constexpr uint64_t baseN = 128;
    constexpr uint64_t baseK = 64;
    const auto info = MakePlatform(NpuArch::DAV_9202);
    for (bool hif4 : {false, true}) {
        for (uint64_t k : {8UL, 64UL, 96UL, 192UL, 4096UL}) {
            for (auto mode : {L1TilingMode::DEFAULT, L1TilingMode::A_L1_FULL_LOAD}) {
                auto input = MakeMxA8W4Input(hif4, k);
                L1TilingDataCalculator calc(input, info, baseM, baseN, baseK);
                ASSERT_TRUE(calc.Compute(mode));
                const auto& l1 = calc.GetOutput();
                ASSERT_GT(l1.stepKa_, 0UL);
                ASSERT_GT(l1.stepKb_, 0UL);
                ASSERT_GT(l1.scaleFactorB_, 0UL);
                const uint64_t scaleBBytesPer64K = hif4 ? 4UL : 2UL;
                uint64_t used = l1.depthKb_ * baseN * baseK / 2UL +
                                2UL * baseN * l1.stepKb_ * l1.scaleFactorB_ * scaleBBytesPer64K;
                if (mode == L1TilingMode::A_L1_FULL_LOAD) {
                    const uint64_t groups = (k + 63UL) / 64UL;
                    used += baseM * groups * 64UL + baseM * groups * 2UL;
                } else {
                    EXPECT_LE(l1.stepKa_, 4UL);
                    EXPECT_LE(l1.stepKb_, 4UL);
                    used += l1.depthKa_ * baseM * baseK + 2UL * baseM * l1.stepKa_ * l1.scaleFactorA_ * 2UL;
                }
                EXPECT_LE(used, info.l1Size);
            }
        }
    }
}

TEST(QuantBatchMatmulMxA8W4Calculators, AFullLoadReservesPhysicalKTailBeforeSolvingB)
{
    constexpr uint64_t baseM = 256UL;
    constexpr uint64_t baseN = 16UL;
    constexpr uint64_t baseK = 64UL;
    // Four B base blocks reach the existing 128-byte K-axis transfer target.
    constexpr uint64_t stepKb = 4UL;
    for (auto arch : {NpuArch::DAV_9201, NpuArch::DAV_9202}) {
        for (bool hif4 : {false, true}) {
            for (uint64_t k : {8UL, 64UL, 96UL, 192UL}) {
                SCOPED_TRACE(::testing::Message()
                             << "arch=" << static_cast<int>(arch) << ", hif4=" << hif4 << ", k=" << k);
                auto input = MakeMxA8W4Input(hif4, k);
                input.mSize = baseM;
                const uint64_t groups = (k + 63UL) / 64UL;
                const uint64_t residentABytes = baseM * groups * 64UL + baseM * groups * 2UL;
                const uint64_t oneBDataBytes = baseN * baseK / 2UL;
                const uint64_t oneBScaleBytes = baseN * (hif4 ? 4UL : 2UL);
                auto info = MakePlatform(arch);
                info.l1Size = residentABytes + 2UL * stepKb * (oneBDataBytes + oneBScaleBytes);
                L1TilingDataCalculator calc(input, info, baseM, baseN, baseK);
                ASSERT_TRUE(calc.Compute(L1TilingMode::A_L1_FULL_LOAD));
                const auto& l1 = calc.GetOutput();
                EXPECT_EQ(l1.stepKa_, groups);
                EXPECT_EQ(l1.depthKa_, groups);
                EXPECT_EQ(l1.stepKb_, stepKb);
                EXPECT_EQ(l1.depthKb_, 2UL * stepKb);
                EXPECT_EQ(l1.scaleFactorA_, 1UL);
                EXPECT_EQ(l1.scaleFactorB_, 1UL);
                const uint64_t usedBytes = residentABytes + l1.depthKb_ * oneBDataBytes +
                                           2UL * l1.stepKb_ * l1.scaleFactorB_ * oneBScaleBytes;
                EXPECT_EQ(usedBytes, info.l1Size);
                EXPECT_EQ(input.kSize, k); // Physical padding must not replace logical K.
            }
        }
    }
}
} // namespace
