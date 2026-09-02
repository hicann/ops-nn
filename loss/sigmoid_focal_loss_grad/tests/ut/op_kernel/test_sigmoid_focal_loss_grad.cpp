/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "adv_api/math/power.h"
#include "tikicpulib.h"

template <auto Func, typename... Args>
__aicore__ inline void SigmoidFocalLossGradUtVfCall(Args&&... args);

namespace AscendC {
template <typename T>
__aicore__ inline void SigmoidFocalLossGradUtDataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src,
                                                         const DataCopyExtParams& params,
                                                         const DataCopyPadExtParams<T>&)
{
    std::memcpy(dst.GetPhyAddr(0), src.GetPhyAddr(), params.blockLen);
}

template <typename T>
__aicore__ inline void SigmoidFocalLossGradUtDataCopyPad(GlobalTensor<T> dst, const LocalTensor<T>& src,
                                                         const DataCopyExtParams& params)
{
    std::memcpy(dst.GetPhyAddr(0), src.GetPhyAddr(), params.blockLen);
}

template <typename T, bool IsReuseSource, const PowerConfig& Config>
__aicore__ inline void SigmoidFocalLossGradUtPower(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                                   const T& exponent, const LocalTensor<uint8_t>&, uint32_t count)
{
    auto* dstPtr = reinterpret_cast<T*>(dst.GetPhyAddr());
    const auto* srcPtr = reinterpret_cast<const T*>(src.GetPhyAddr());
    for (uint32_t i = 0; i < count; ++i) {
        dstPtr[i] = static_cast<T>(std::pow(static_cast<float>(srcPtr[i]), static_cast<float>(exponent)));
    }
}
} // namespace AscendC

// The CANN 9.2 tikicpulib used by kernel UT cannot execute arch35 RegBase
// instructions or the advanced Power implementation, and its queued data-copy
// model cannot interleave with scalar runners. Replace only this test
// translation unit's memory and compute dispatch points with scalar equivalents.
// The real kernel source is still included and all dtype/profile/control-flow
// templates are instantiated by these tests.
#define asc_vf_call SigmoidFocalLossGradUtVfCall
#define DataCopyPad SigmoidFocalLossGradUtDataCopyPad
#define Power SigmoidFocalLossGradUtPower
#ifdef DTYPE_PRED
#undef DTYPE_PRED
#endif
#ifdef DTYPE_DOUT
#undef DTYPE_DOUT
#endif
#ifdef DTYPE_WEIGHT
#undef DTYPE_WEIGHT
#endif
#define DTYPE_PRED float
#define DTYPE_DOUT float
#define DTYPE_WEIGHT float
#include "../../../op_kernel/arch35/sigmoid_focal_loss_grad.cpp"
#undef DTYPE_WEIGHT
#undef DTYPE_DOUT
#undef DTYPE_PRED
#undef Power
#undef DataCopyPad
#undef asc_vf_call

template <uint32_t PROFILE>
struct TestDtypeProfile {
    static_assert(PROFILE <= 11, "invalid SigmoidFocalLossGrad test profile");
    using PredT = std::conditional_t<(PROFILE < 4 || PROFILE == 8 || PROFILE == 9), half, float>;
    using DoutT = std::conditional_t<
        (PROFILE == 0 || PROFILE == 1 || PROFILE == 4 || PROFILE == 5 || PROFILE == 8 || PROFILE == 10), half, float>;
    static constexpr bool kHasWeight = PROFILE < 8;
    using WeightT = std::conditional_t<(PROFILE == 0 || PROFILE == 2 || PROFILE == 4 || PROFILE == 6), half, float>;
};

template <auto Func>
struct SigmoidFocalLossGradUtVfRunner;

template <typename PredT, typename DoutT, typename WeightT, bool HasWeight>
struct SigmoidFocalLossGradUtPrepareInputsRunner {
    static void Run(float* predDst, float* targetDst, float* doutDst, float* weightDst, PredT* predSrc,
                    int32_t* targetSrc, DoutT* doutSrc, WeightT* weightSrc, uint32_t count, uint16_t)
    {
        for (uint32_t i = 0; i < count; ++i) {
            predDst[i] = static_cast<float>(predSrc[i]);
            targetDst[i] = static_cast<float>(targetSrc[i]);
            doutDst[i] = static_cast<float>(doutSrc[i]);
            if constexpr (HasWeight) {
                weightDst[i] = static_cast<float>(weightSrc[i]);
            }
        }
    }
};

template <>
struct SigmoidFocalLossGradUtVfRunner<&PrepareInputsVf<half, half, half, true>>
    : SigmoidFocalLossGradUtPrepareInputsRunner<half, half, half, true> {};
template <>
struct SigmoidFocalLossGradUtVfRunner<&PrepareInputsVf<half, half, float, true>>
    : SigmoidFocalLossGradUtPrepareInputsRunner<half, half, float, true> {};
template <>
struct SigmoidFocalLossGradUtVfRunner<&PrepareInputsVf<half, float, half, true>>
    : SigmoidFocalLossGradUtPrepareInputsRunner<half, float, half, true> {};
template <>
struct SigmoidFocalLossGradUtVfRunner<&PrepareInputsVf<half, float, float, true>>
    : SigmoidFocalLossGradUtPrepareInputsRunner<half, float, float, true> {};
template <>
struct SigmoidFocalLossGradUtVfRunner<&PrepareInputsVf<float, half, half, true>>
    : SigmoidFocalLossGradUtPrepareInputsRunner<float, half, half, true> {};
template <>
struct SigmoidFocalLossGradUtVfRunner<&PrepareInputsVf<float, half, float, true>>
    : SigmoidFocalLossGradUtPrepareInputsRunner<float, half, float, true> {};
template <>
struct SigmoidFocalLossGradUtVfRunner<&PrepareInputsVf<float, float, half, true>>
    : SigmoidFocalLossGradUtPrepareInputsRunner<float, float, half, true> {};
template <>
struct SigmoidFocalLossGradUtVfRunner<&PrepareInputsVf<float, float, float, true>>
    : SigmoidFocalLossGradUtPrepareInputsRunner<float, float, float, true> {};
template <>
struct SigmoidFocalLossGradUtVfRunner<&PrepareInputsVf<half, half, float, false>>
    : SigmoidFocalLossGradUtPrepareInputsRunner<half, half, float, false> {};
template <>
struct SigmoidFocalLossGradUtVfRunner<&PrepareInputsVf<half, float, float, false>>
    : SigmoidFocalLossGradUtPrepareInputsRunner<half, float, float, false> {};
template <>
struct SigmoidFocalLossGradUtVfRunner<&PrepareInputsVf<float, half, float, false>>
    : SigmoidFocalLossGradUtPrepareInputsRunner<float, half, float, false> {};
template <>
struct SigmoidFocalLossGradUtVfRunner<&PrepareInputsVf<float, float, float, false>>
    : SigmoidFocalLossGradUtPrepareInputsRunner<float, float, float, false> {};

template <>
struct SigmoidFocalLossGradUtVfRunner<&StableSigmoidExpVf> {
    static void Run(float* positiveDst, float* expDst, float* predSrc, uint32_t count, uint16_t)
    {
        for (uint32_t i = 0; i < count; ++i) {
            const float e = std::exp(-std::abs(predSrc[i]));
            positiveDst[i] = 1.0F / (1.0F + e);
            expDst[i] = e;
        }
    }
};

template <>
struct SigmoidFocalLossGradUtVfRunner<&StableSigmoidVf> {
    static void Run(float* pDst, float* pFloorDst, float* qDst, float* qFloorDst, float* predSrc, uint32_t count,
                    uint16_t)
    {
        constexpr float kLogEps = 1.17549435e-38F;
        for (uint32_t i = 0; i < count; ++i) {
            const float p = predSrc[i] >= 0.0F ? pDst[i] : 1.0F / (1.0F + std::exp(std::abs(predSrc[i])));
            const float q = 1.0F - p;
            pDst[i] = p;
            qDst[i] = q;
            pFloorDst[i] = std::max(p, kLogEps);
            qFloorDst[i] = std::max(q, kLogEps);
        }
    }
};

template <>
struct SigmoidFocalLossGradUtVfRunner<&DetectClampedPowerBaseVf> {
    static void Run(float* minBaseDst, float* pBaseSrc, float* qBaseSrc, uint32_t count, uint16_t)
    {
        float minBase = std::numeric_limits<float>::max();
        for (uint32_t i = 0; i < count; ++i) {
            minBase = std::min(minBase, std::min(pBaseSrc[i], qBaseSrc[i]));
        }
        minBaseDst[0] = minBase;
    }
};

template <>
struct SigmoidFocalLossGradUtVfRunner<&CorrectClampedPowerVf> {
    static void Run(float* powGammaDst, float* powGammaPlusOneDst, float* baseSrc, float gamma, uint32_t count,
                    uint16_t)
    {
        constexpr float kLogEps = 1.17549435e-38F;
        for (uint32_t i = 0; i < count; ++i) {
            const float logBase = std::log(baseSrc[i]);
            if (baseSrc[i] <= kLogEps || powGammaDst[i] <= kLogEps) {
                powGammaDst[i] = std::exp(logBase * gamma);
            }
            if (baseSrc[i] <= kLogEps || powGammaPlusOneDst[i] <= kLogEps) {
                powGammaPlusOneDst[i] = std::exp(logBase * (gamma + 1.0F));
            }
        }
    }
};

template <>
struct SigmoidFocalLossGradUtVfRunner<&DPosVf> {
    static void Run(float* dst, float* pSrc, float* pFloorSrc, float* qGammaSrc, float* qGamma1Src, float alpha,
                    float gamma, uint32_t count, uint16_t)
    {
        for (uint32_t i = 0; i < count; ++i) {
            dst[i] = alpha * gamma * qGammaSrc[i] * std::log(pFloorSrc[i]) * pSrc[i] - alpha * qGamma1Src[i];
        }
    }
};

template <>
struct SigmoidFocalLossGradUtVfRunner<&DNegVf> {
    static void Run(float* dst, float* qSrc, float* qFloorSrc, float* pGammaSrc, float* pGamma1Src, float alpha,
                    float gamma, uint32_t count, uint16_t)
    {
        for (uint32_t i = 0; i < count; ++i) {
            dst[i] = gamma * (alpha - 1.0F) * pGammaSrc[i] * qSrc[i] * std::log(qFloorSrc[i]) +
                     (1.0F - alpha) * pGamma1Src[i];
        }
    }
};

template <>
struct SigmoidFocalLossGradUtVfRunner<&ComposeRawVf> {
    static void Run(float* rawDst, float* dposSrc, float* dnegSrc, float* targetSrc, uint32_t count, uint16_t)
    {
        for (uint32_t i = 0; i < count; ++i) {
            rawDst[i] = dposSrc[i] * (1.0F - targetSrc[i]) + dnegSrc[i] * targetSrc[i];
        }
    }
};

template <typename PredT, bool HasWeight>
struct SigmoidFocalLossGradUtScaleStoreRunner {
    static void Run(PredT* gradDst, float* weightedDst, float* scaledDst, float* rawSrc, float* weightSrc,
                    float* doutSrc, float reduceMeanCoef, uint32_t count, uint16_t)
    {
        for (uint32_t i = 0; i < count; ++i) {
            const float weighted = HasWeight ? rawSrc[i] * weightSrc[i] : rawSrc[i];
            const float scaled = weighted * doutSrc[i] * reduceMeanCoef;
            weightedDst[i] = weighted;
            scaledDst[i] = scaled;
            gradDst[i] = static_cast<PredT>(scaled);
        }
    }
};

template <>
struct SigmoidFocalLossGradUtVfRunner<&ScaleStoreVf<half, true>> : SigmoidFocalLossGradUtScaleStoreRunner<half, true> {
};
template <>
struct SigmoidFocalLossGradUtVfRunner<&ScaleStoreVf<float, true>>
    : SigmoidFocalLossGradUtScaleStoreRunner<float, true> {};
template <>
struct SigmoidFocalLossGradUtVfRunner<&ScaleStoreVf<half, false>>
    : SigmoidFocalLossGradUtScaleStoreRunner<half, false> {};
template <>
struct SigmoidFocalLossGradUtVfRunner<&ScaleStoreVf<float, false>>
    : SigmoidFocalLossGradUtScaleStoreRunner<float, false> {};

template <auto Func, typename... Args>
__aicore__ inline void SigmoidFocalLossGradUtVfCall(Args&&... args)
{
    SigmoidFocalLossGradUtVfRunner<Func>::Run(std::forward<Args>(args)...);
}

namespace {

constexpr size_t kSystemWorkspaceBytes = 16UL * 1024UL * 1024UL;

template <typename T>
std::vector<T> CastInput(const std::vector<float>& input)
{
    std::vector<T> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<T>(input[i]);
    }
    return output;
}

template <typename T>
float ToFloat(T value)
{
    return static_cast<float>(value);
}

float StableSigmoid(float x)
{
    if (x >= 0.0F) {
        return 1.0F / (1.0F + std::exp(-x));
    }
    const float expX = std::exp(x);
    return expX / (1.0F + expX);
}

float GoldenElement(float pred, int32_t target, float dout, float weight, float alpha, float gamma,
                    float reduceMeanCoef)
{
    constexpr float kMinNormal = 1.17549435e-38F;
    const float p = StableSigmoid(pred);
    const float q = 1.0F - p;
    const float pFloor = std::max(p, kMinNormal);
    const float qFloor = std::max(q, kMinNormal);
    const float dpos = alpha * gamma * p * std::pow(qFloor, gamma) * std::log(pFloor) -
                       alpha * std::pow(qFloor, gamma + 1.0F);
    const float dneg = gamma * (alpha - 1.0F) * std::pow(pFloor, gamma) * q * std::log(qFloor) +
                       (1.0F - alpha) * std::pow(pFloor, gamma + 1.0F);
    const float raw = dpos * (1.0F - static_cast<float>(target)) + dneg * static_cast<float>(target);
    return raw * weight * dout * reduceMeanCoef;
}

SigmoidFocalLossGradTilingData MakeTiling(int64_t dim0, int64_t blockFormer, int64_t blockNum, int64_t ubFormer,
                                          float alpha, float gamma, float reduceMeanCoef)
{
    const int64_t blockTail = dim0 - (blockNum - 1) * blockFormer;
    SigmoidFocalLossGradTilingData td{};
    td.dim0 = dim0;
    td.coreNum = static_cast<int32_t>(blockNum);
    td.blockFormer = blockFormer;
    td.blockNum = blockNum;
    td.ubFormer = ubFormer;
    td.ubLoopOfFormerBlock = (blockFormer + ubFormer - 1) / ubFormer;
    td.ubTailOfFormerBlock = blockFormer - (td.ubLoopOfFormerBlock - 1) * ubFormer;
    td.ubLoopOfTailBlock = (blockTail + ubFormer - 1) / ubFormer;
    td.ubTailOfTailBlock = blockTail - (td.ubLoopOfTailBlock - 1) * ubFormer;
    td.alpha = alpha;
    td.gamma = gamma;
    td.reduceMeanCoef = reduceMeanCoef;
    return td;
}

template <uint32_t MODE>
void RunAndVerify(const SigmoidFocalLossGradTilingData& td, bool useExtremeValues = false)
{
    using Traits = TestDtypeProfile<MODE>;
    using PredT = typename Traits::PredT;
    using DoutT = typename Traits::DoutT;
    using WeightT = typename Traits::WeightT;
    constexpr bool kHasWeight = Traits::kHasWeight;

    std::vector<float> predSource(static_cast<size_t>(td.dim0));
    std::vector<float> doutSource(static_cast<size_t>(td.dim0));
    std::vector<float> weightSource(static_cast<size_t>(td.dim0));
    std::vector<int32_t> targetSource(static_cast<size_t>(td.dim0));
    for (int64_t i = 0; i < td.dim0; ++i) {
        predSource[i] = static_cast<float>((i % 17) - 8) * 0.625F;
        doutSource[i] = 0.25F + static_cast<float>(i % 7) * 0.125F;
        weightSource[i] = -0.75F + static_cast<float>(i % 11) * 0.2F;
        targetSource[i] = static_cast<int32_t>(i & 1);
    }
    if (useExtremeValues && td.dim0 >= 6) {
        predSource[0] = -80.0F;
        predSource[1] = 80.0F;
        predSource[2] = -20.0F;
        predSource[3] = 20.0F;
        predSource[4] = -0.0F;
        predSource[5] = 0.0F;
    }

    const auto predHost = CastInput<PredT>(predSource);
    const auto doutHost = CastInput<DoutT>(doutSource);
    const auto weightHost = CastInput<WeightT>(weightSource);

    const size_t predBytes = predHost.size() * sizeof(PredT);
    const size_t targetBytes = targetSource.size() * sizeof(int32_t);
    const size_t doutBytes = doutHost.size() * sizeof(DoutT);
    const size_t weightBytes = weightHost.size() * sizeof(WeightT);

    auto* pred = static_cast<uint8_t*>(AscendC::GmAlloc(predBytes));
    auto* target = static_cast<uint8_t*>(AscendC::GmAlloc(targetBytes));
    auto* dout = static_cast<uint8_t*>(AscendC::GmAlloc(doutBytes));
    auto* weight = kHasWeight ? static_cast<uint8_t*>(AscendC::GmAlloc(weightBytes)) : nullptr;
    auto* grad = static_cast<uint8_t*>(AscendC::GmAlloc(predBytes));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(kSystemWorkspaceBytes));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(td)));
    ASSERT_NE(pred, nullptr);
    ASSERT_NE(target, nullptr);
    ASSERT_NE(dout, nullptr);
    ASSERT_NE(grad, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);
    if constexpr (kHasWeight) {
        ASSERT_NE(weight, nullptr);
    }

    std::memcpy(pred, predHost.data(), predBytes);
    std::memcpy(target, targetSource.data(), targetBytes);
    std::memcpy(dout, doutHost.data(), doutBytes);
    if constexpr (kHasWeight) {
        std::memcpy(weight, weightHost.data(), weightBytes);
    }
    std::memset(grad, 0, predBytes);
    std::memcpy(tiling, &td, sizeof(td));

    auto kernel = [](GM_ADDR predAddr, GM_ADDR targetAddr, GM_ADDR doutAddr, GM_ADDR weightAddr, GM_ADDR gradAddr,
                     GM_ADDR, GM_ADDR tilingAddr) {
        SigmoidFocalLossGradKernel<PredT, DoutT, WeightT, kHasWeight> kernel;
        kernel.Init(predAddr, targetAddr, doutAddr, weightAddr, gradAddr,
                    reinterpret_cast<const SigmoidFocalLossGradTilingData*>(tilingAddr));
        kernel.Process();
    };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(GET_TPL_TILING_KEY(static_cast<uint64_t>(kHasWeight)));
    ICPU_RUN_KF(kernel, static_cast<uint32_t>(td.blockNum), pred, target, dout, weight, grad, workspace, tiling);

    const auto* actual = reinterpret_cast<const PredT*>(grad);
    const float atol = std::is_same_v<PredT, half> ? 2.0e-2F : 8.0e-4F;
    const float rtol = std::is_same_v<PredT, half> ? 2.0e-2F : 2.0e-3F;
    for (int64_t i = 0; i < td.dim0; ++i) {
        const float predValue = ToFloat(predHost[static_cast<size_t>(i)]);
        const float doutValue = ToFloat(doutHost[static_cast<size_t>(i)]);
        const float weightValue = kHasWeight ? ToFloat(weightHost[static_cast<size_t>(i)]) : 1.0F;
        const float expectedFp32 = GoldenElement(predValue, targetSource[static_cast<size_t>(i)], doutValue,
                                                 weightValue, td.alpha, td.gamma, td.reduceMeanCoef);
        const float expected = ToFloat(static_cast<PredT>(expectedFp32));
        const float actualValue = ToFloat(actual[i]);
        ASSERT_TRUE(std::isfinite(actualValue)) << "profile=" << MODE << " index=" << i;
        EXPECT_NEAR(actualValue, expected, atol + rtol * std::abs(expected))
            << "profile=" << MODE << " index=" << i << " pred=" << predValue
            << " target=" << targetSource[static_cast<size_t>(i)];
    }

    AscendC::GmFree(pred);
    AscendC::GmFree(target);
    AscendC::GmFree(dout);
    if constexpr (kHasWeight) {
        AscendC::GmFree(weight);
    }
    AscendC::GmFree(grad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

#define SIGMOID_FOCAL_LOSS_GRAD_MODE_TEST(mode, ubFormer)                     \
    TEST(SigmoidFocalLossGradKernelModeTest, Mode##mode)                      \
    {                                                                         \
        const auto td = MakeTiling(129, 129, 1, ubFormer, 0.25F, 2.0F, 1.0F); \
        RunAndVerify<mode>(td);                                               \
    }

SIGMOID_FOCAL_LOSS_GRAD_MODE_TEST(0, 128)
SIGMOID_FOCAL_LOSS_GRAD_MODE_TEST(1, 128)
SIGMOID_FOCAL_LOSS_GRAD_MODE_TEST(2, 128)
SIGMOID_FOCAL_LOSS_GRAD_MODE_TEST(3, 128)
SIGMOID_FOCAL_LOSS_GRAD_MODE_TEST(4, 128)
SIGMOID_FOCAL_LOSS_GRAD_MODE_TEST(5, 128)
SIGMOID_FOCAL_LOSS_GRAD_MODE_TEST(6, 128)
SIGMOID_FOCAL_LOSS_GRAD_MODE_TEST(7, 64)
SIGMOID_FOCAL_LOSS_GRAD_MODE_TEST(8, 128)
SIGMOID_FOCAL_LOSS_GRAD_MODE_TEST(9, 128)
SIGMOID_FOCAL_LOSS_GRAD_MODE_TEST(10, 128)
SIGMOID_FOCAL_LOSS_GRAD_MODE_TEST(11, 64)

#undef SIGMOID_FOCAL_LOSS_GRAD_MODE_TEST

TEST(SigmoidFocalLossGradKernelTest, HandlesExtremeLogitsAndMeanReduction)
{
    const int64_t dim0 = 257;
    const auto td = MakeTiling(dim0, dim0, 1, 64, -0.125F, 3.25F, 1.0F / static_cast<float>(dim0));
    RunAndVerify<7>(td, true);
}

TEST(SigmoidFocalLossGradKernelTest, HandlesMultipleBlocksAndUbTiles)
{
    const auto td = MakeTiling(2050, 1024, 3, 64, 0.4F, 1.5F, 1.0F);
    RunAndVerify<11>(td);
}

TEST(SigmoidFocalLossGradKernelTest, EntryGuardsLeaveOutputUntouched)
{
    const auto td = MakeTiling(64, 64, 1, 64, 0.25F, 2.0F, 1.0F);
    auto* input = static_cast<uint8_t*>(AscendC::GmAlloc(64 * sizeof(float)));
    auto* target = static_cast<uint8_t*>(AscendC::GmAlloc(64 * sizeof(int32_t)));
    auto* output = static_cast<uint8_t*>(AscendC::GmAlloc(64 * sizeof(float)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(td)));
    ASSERT_NE(input, nullptr);
    ASSERT_NE(target, nullptr);
    ASSERT_NE(output, nullptr);
    ASSERT_NE(tiling, nullptr);
    std::memset(output, 0x5A, 64 * sizeof(float));
    std::memcpy(tiling, &td, sizeof(td));
    std::vector<uint8_t> expected(64 * sizeof(float), 0x5A);

    auto weightedKernel = [](GM_ADDR pred, GM_ADDR targetAddr, GM_ADDR dout, GM_ADDR weight, GM_ADDR grad,
                             GM_ADDR workspace, GM_ADDR tilingAddr) {
        ::sigmoid_focal_loss_grad<1>(pred, targetAddr, dout, weight, grad, workspace, tilingAddr);
    };
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(GET_TPL_TILING_KEY(1));
    ICPU_RUN_KF(weightedKernel, 1, input, target, input, nullptr, output, nullptr, tiling);
    EXPECT_EQ(std::memcmp(output, expected.data(), expected.size()), 0);

    auto invalidTiling = td;
    invalidTiling.dim0 = 0;
    std::memcpy(tiling, &invalidTiling, sizeof(invalidTiling));
    ICPU_RUN_KF(weightedKernel, 1, input, target, input, input, output, nullptr, tiling);
    EXPECT_EQ(std::memcmp(output, expected.data(), expected.size()), 0);

    AscendC::GmFree(input);
    AscendC::GmFree(target);
    AscendC::GmFree(output);
    AscendC::GmFree(tiling);
}

} // namespace
