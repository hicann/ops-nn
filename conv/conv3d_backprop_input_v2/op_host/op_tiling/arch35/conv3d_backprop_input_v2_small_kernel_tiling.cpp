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
 * \file conv3d_backprop_input_v2_small_kernel_tiling.cpp
 * \brief small kernel tiling template: N-axis full load, M-axis split, maximize core utilization
 */

#include <map>
#include <numeric>
#include <log/log.h>
#include "error_util.h"
#include <util/math_util.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "op_host/tiling_templates_registry.h"
#include "conv/common/op_host/op_tiling/conv_platform_util.h"
#include "conv3d_backprop_input_v2_small_kernel_tiling.h"

namespace {
constexpr uint8_t ENABLE_SMALL_KERNEL = 4;
constexpr uint8_t REVERSE_ONLY = 2;
constexpr uint8_t NO_SPLIT_KERNEL = 0;
constexpr uint64_t SMALL_KERNEL_COMPUTE_THRESHOLD = 144 * 2048 * 2048;
} // namespace

namespace Ops {
namespace NN {
namespace Conv {

bool Conv3DDXV2SmallKernelTiling::IsCapable()
{
    if (!CheckSmallKernelEnable()) {
        return false;
    }

    uint64_t cinAlign = Ops::Base::CeilAlign(static_cast<uint64_t>(runInfo_.dedx_cin_g),
                                             static_cast<uint64_t>(tilingRunInfo_.n0));
    if (cinAlign > MAX_BASE_MN) {
        return false;
    }

    if (Conv3DDXV2InnerProductTiling::GetTilingFromRepo()) {
        isGetTilingFromRepo = true;
    }
    return true;
}

ge::graphStatus Conv3DDXV2SmallKernelTiling::DoLibApiTiling()
{
    OP_LOGD(opName_, "Enable small kernel tiling");
    tilingRunInfo_.enableSmallKernel = true;

    if (isGetTilingFromRepo) {
        OP_LOGD(context_->GetNodeName(),
                "Conv3DBackpropInputV2 AscendC: SmallKernel get tiling from knowledge_tiling success.");
        PrintTilingSummary();
        return ge::GRAPH_SUCCESS;
    }

    CoreTilingParams coreParams;
    L0TilingParams l0Params;
    SetSmallKernelCoreInfo(coreParams, l0Params);

    InitBaseMNK(l0Params);

    L1TilingParams l1Params;
    Conv3DDXV2InnerProductTiling::InitL1Params(l1Params, l0Params);

    CalStepK(l1Params, l0Params);

    SetTilingCondition(coreParams, l1Params, l0Params);
    Conv3DDXV2InnerProductTiling::SetTilingData(coreParams, l1Params, l0Params);
    Conv3DDXV2InnerProductTiling::PrintTilingSummary();
    return ge::GRAPH_SUCCESS;
}

void Conv3DDXV2SmallKernelTiling::InitBaseMNK(L0TilingParams& l0Params)
{
    l0Params.al0Pbuffer = DB_OFF;
    l0Params.bl0Pbuffer = DB_OFF;
    l0Params.cl0Pbuffer = DB_OFF;

    uint64_t coutAlign = Ops::Base::CeilAlign(static_cast<uint64_t>(runInfo_.dedy_cout_g),
                                              static_cast<uint64_t>(tilingRunInfo_.k0));
    uint64_t kTotal = coutAlign * runInfo_.kernel_h * runInfo_.kernel_w;

    auto calcMaxBaseK = [this, &l0Params, kTotal]() -> uint32_t {
        uint32_t maxBaseKByL0a = static_cast<uint32_t>(platformInfo_.l0_ab_size / l0Params.al0Pbuffer / dtypeByteL0a_ /
                                                       l0Params.baseM);
        uint32_t maxBaseKByL0b = static_cast<uint32_t>(platformInfo_.l0_ab_size / l0Params.bl0Pbuffer / dtypeByteL0b_ /
                                                       l0Params.baseN);
        uint32_t maxBaseK = std::min({maxBaseKByL0a, maxBaseKByL0b, static_cast<uint32_t>(kTotal)});
        maxBaseK = std::max(maxBaseK / tilingRunInfo_.k0, ONE_U32) * tilingRunInfo_.k0;
        maxBaseK = std::min(maxBaseK, static_cast<uint32_t>(kTotal));
        return maxBaseK == 0 ? tilingRunInfo_.k0 : maxBaseK;
    };

    l0Params.baseK = calcMaxBaseK();

    uint32_t kIter = static_cast<uint32_t>(Ops::Base::CeilDiv(kTotal, static_cast<uint64_t>(l0Params.baseK)));
    if (kIter >= TWO_U32) {
        l0Params.al0Pbuffer = DB_ON;
        l0Params.bl0Pbuffer = DB_ON;
        l0Params.baseK = calcMaxBaseK();
    }
}

void Conv3DDXV2SmallKernelTiling::SetSmallKernelCoreInfo(CoreTilingParams& coreParams, L0TilingParams& l0Params)
{
    enableA1Db_ = false;
    coreParams.singleCoreDin = ONE_U32;
    coreParams.singleCoreCout = static_cast<uint32_t>(runInfo_.dedy_cout_g);

    uint64_t cinAlign = Ops::Base::CeilAlign(static_cast<uint64_t>(runInfo_.dedx_cin_g),
                                             static_cast<uint64_t>(tilingRunInfo_.n0));
    if (cinAlign == 0) {
        coreParams.singleCoreM = 0;
        l0Params.baseM = 0;
        return;
    }
    coreParams.singleCoreCin = cinAlign;
    l0Params.baseN = static_cast<uint32_t>(cinAlign);

    uint64_t hwI = static_cast<uint64_t>(runInfo_.dedx_h) * runInfo_.dedx_w;
    uint64_t batchDepth = static_cast<uint64_t>(runInfo_.batch_n) * runInfo_.dedx_d;
    uint64_t coreNum = static_cast<uint64_t>(coreNum_);
    uint64_t m0 = static_cast<uint64_t>(tilingRunInfo_.m0);
    if (batchDepth == 0 || coreNum == 0 || m0 == 0) {
        coreParams.singleCoreM = 0;
        l0Params.baseM = 0;
        return;
    }

    uint64_t maxSingleCoreMByL0C = CalcSmallKernelMaxMByL0C(cinAlign, l0Params.cl0Pbuffer);
    uint64_t maxM = std::min(hwI, static_cast<uint64_t>(MAX_BASE_MN));
    uint64_t maxSingleCoreM = std::min(maxSingleCoreMByL0C, CalcMaxSingleCoreMByL1(maxM, DB_OFF));

    if (maxSingleCoreM < m0) {
        coreParams.singleCoreM = 0;
        l0Params.baseM = 0;
        return;
    }

    // 基本块分块决策: 负载均衡 > 核利用率 > 单轮核数
    // A1 DB 决策: A1 DB_OFF 优先，存在单核多轮次计算则 A1 DB_ON，若超出buffer约束则回退 A1 DB_OFF
    uint64_t bestSingleCoreM = SelectSmallKernelCoreMWithBuffering(hwI, batchDepth, coreNum, m0, maxM,
                                                                   maxSingleCoreMByL0C);

    coreParams.singleCoreM = bestSingleCoreM;
    l0Params.baseM = static_cast<uint32_t>(Ops::Base::CeilAlign(bestSingleCoreM, m0));
}

uint64_t Conv3DDXV2SmallKernelTiling::SelectSmallKernelCoreMWithBuffering(uint64_t hwI, uint64_t batchDepth,
                                                                          uint64_t coreNum, uint64_t m0, uint64_t maxM,
                                                                          uint64_t maxSingleCoreMByL0C)
{
    uint64_t bestSingleCoreM = SelectSmallKernelCoreM(
        hwI, batchDepth, coreNum, m0, std::min(maxSingleCoreMByL0C, CalcMaxSingleCoreMByL1(maxM, DB_OFF)));
    uint64_t baseMCnt = Ops::Base::CeilDiv(hwI, bestSingleCoreM);
    uint64_t baseTotalCnt = batchDepth * baseMCnt;
    uint64_t baseUsedCoreNum = std::min(baseTotalCnt, coreNum);
    uint64_t baseCalRound = baseUsedCoreNum == 0 ? 0 : baseTotalCnt / baseUsedCoreNum;
    uint64_t baseTailCnt = baseUsedCoreNum == 0 ? 0 : baseTotalCnt - baseCalRound * baseUsedCoreNum;
    enableA1Db_ = baseCalRound > ONE_U64 || baseTailCnt > 0;
    if (!enableA1Db_) {
        return bestSingleCoreM;
    }
    uint64_t maxSingleCoreMByDb = std::min(maxSingleCoreMByL0C, CalcMaxSingleCoreMByL1(maxM, DB_ON));
    if (maxSingleCoreMByDb < m0) {
        enableA1Db_ = false;
        return bestSingleCoreM;
    }
    return SelectSmallKernelCoreM(hwI, batchDepth, coreNum, m0, maxSingleCoreMByDb);
}

uint64_t Conv3DDXV2SmallKernelTiling::CalcSmallKernelMaxMByL0C(uint64_t cinAlign, uint32_t cl0Pbuffer) const
{
    const uint64_t m0 = tilingRunInfo_.m0;
    const uint64_t floatSize = ge::GetSizeByDataType(ge::DT_FLOAT);
    if (cinAlign == 0 || m0 == 0 || cl0Pbuffer == 0 || floatSize == 0) {
        return 0;
    }
    uint64_t l0cElementCount = platformInfo_.l0_c_size / cl0Pbuffer / floatSize;
    return (l0cElementCount / cinAlign / m0) * m0;
}

uint64_t Conv3DDXV2SmallKernelTiling::CalcSmallKernelCandidateM(uint64_t hwI, uint64_t mCnt, uint64_t maxMByBuffer,
                                                                uint64_t m0) const
{
    uint64_t candidate = Ops::Base::CeilAlign(Ops::Base::CeilDiv(hwI, mCnt), m0);
    candidate = std::min({candidate, hwI, static_cast<uint64_t>(MAX_BASE_MN), maxMByBuffer});
    uint64_t alignedWi = std::max(candidate / runInfo_.dedx_w, ONE_U64) * runInfo_.dedx_w;
    if (Ops::Base::CeilDiv(hwI, alignedWi) == Ops::Base::CeilDiv(hwI, candidate)) {
        candidate = alignedWi;
    }
    return Ops::Base::FloorAlign(std::min(candidate, maxMByBuffer), m0);
}

uint64_t Conv3DDXV2SmallKernelTiling::CalcSmallKernelCoreScore(uint64_t hwI, uint64_t batchDepth, uint64_t coreNum,
                                                               uint64_t singleCoreM) const
{
    uint64_t totalCnt = batchDepth * Ops::Base::CeilDiv(hwI, singleCoreM);
    uint64_t usedCoreNum = std::min(totalCnt, coreNum);
    if (usedCoreNum == 0) {
        return UINT64_MAX;
    }
    uint64_t calRound = totalCnt / usedCoreNum;
    uint64_t tailCnt = totalCnt - calRound * usedCoreNum;
    return tailCnt * 1000 + (coreNum - usedCoreNum) * 10 + calRound;
}

uint64_t Conv3DDXV2SmallKernelTiling::SelectSmallKernelCoreM(uint64_t hwI, uint64_t batchDepth, uint64_t coreNum,
                                                             uint64_t m0, uint64_t maxMByBuffer) const
{
    uint64_t idealMCnt = std::max(coreNum / batchDepth, ONE_U64);
    uint64_t minMCnt = std::max(idealMCnt / 2, ONE_U64);
    uint64_t maxMCnt = std::min(Ops::Base::CeilDiv(hwI, static_cast<uint64_t>(BASIC_BLOCK_SIZE_64)), idealMCnt * 2);
    uint64_t bestSingleCoreM = CalcSmallKernelCandidateM(hwI, idealMCnt, maxMByBuffer, m0);
    uint64_t bestScore = CalcSmallKernelCoreScore(hwI, batchDepth, coreNum, bestSingleCoreM);
    for (uint64_t mCnt = minMCnt; mCnt <= maxMCnt; ++mCnt) {
        uint64_t singleCoreM = CalcSmallKernelCandidateM(hwI, mCnt, maxMByBuffer, m0);
        if (singleCoreM < BASIC_BLOCK_SIZE_64) {
            break;
        }
        uint64_t score = CalcSmallKernelCoreScore(hwI, batchDepth, coreNum, singleCoreM);
        if (score < bestScore) {
            bestScore = score;
            bestSingleCoreM = singleCoreM;
        }
    }
    return bestSingleCoreM;
}

void Conv3DDXV2SmallKernelTiling::CalStepK(L1TilingParams& l1Params, const L0TilingParams& l0Params)
{
    (void)l0Params;
    l1Params.al1Pbuffer = enableA1Db_ ? DB_ON : DB_OFF;
    l1Params.bl1Pbuffer = DB_OFF;
    l1Params.stepKa = ONE_U32;
    l1Params.stepKb = ONE_U32;
}

void Conv3DDXV2SmallKernelTiling::SetTilingCondition(const CoreTilingParams& coreParams, const L1TilingParams& l1Params,
                                                     const L0TilingParams& l0Params)
{
    loadB1Condition_ = ENABLE_SMALL_KERNEL;
    loadB2Condition_ = REVERSE_ONLY;
    kernelSplitMode_ = NO_SPLIT_KERNEL;
    groupConvMode_ = TILING_GROUP_MODE_ORIGIN;
    tilingRunInfo_.enableVecTransFlag = false;
}

uint64_t Conv3DDXV2SmallKernelTiling::CalSmallKernelLocalHo(uint64_t maxM, uint64_t wi, uint64_t hk, uint64_t dilationH,
                                                            uint64_t hoExpand)
{
    uint64_t hiCount = Ops::Base::CeilDiv(maxM + wi - 1, wi);
    uint64_t receptiveHo = hiCount + (hk - 1) * dilationH;
    return std::min(receptiveHo, hoExpand);
}

uint64_t Conv3DDXV2SmallKernelTiling::CalcSmallKernelA1Size(uint64_t baseM) const
{
    uint64_t hoExpand = (static_cast<uint64_t>(runInfo_.dedy_h) - 1) * runInfo_.stride_h + 1;
    uint64_t woExpand = (static_cast<uint64_t>(runInfo_.dedy_w) - 1) * runInfo_.stride_w + 1;
    uint64_t coutAlign = Ops::Base::CeilAlign(static_cast<uint64_t>(runInfo_.dedy_cout_g),
                                              static_cast<uint64_t>(tilingRunInfo_.k0));
    uint64_t localHo = Ops::Base::CeilDiv(baseM + runInfo_.dedx_w - 1, static_cast<uint64_t>(runInfo_.dedx_w)) +
                       (runInfo_.kernel_h - 1) * runInfo_.dilation_h;
    localHo = std::min(localHo, hoExpand);
    return localHo * woExpand * coutAlign * dtypeByteL0a_;
}

uint64_t Conv3DDXV2SmallKernelTiling::CalcSmallKernelL1FixedSize() const
{
    uint64_t coutAlign = Ops::Base::CeilAlign(static_cast<uint64_t>(runInfo_.dedy_cout_g),
                                              static_cast<uint64_t>(tilingRunInfo_.k0));
    uint64_t cinAlign = Ops::Base::CeilAlign(static_cast<uint64_t>(runInfo_.dedx_cin_g),
                                             static_cast<uint64_t>(tilingRunInfo_.n0));
    uint64_t b1Size = static_cast<uint64_t>(runInfo_.kernel_h) * runInfo_.kernel_w * coutAlign * cinAlign *
                      dtypeByteL0b_;
    uint64_t biasSize = 0;
    if (hasBiasFlag_) {
        uint64_t dtypeByteBtBuffer = (runInfo_.a_dtype_bytes == ge::GetSizeByDataType(ge::DT_INT8)) ?
                                         ge::GetSizeByDataType(ge::DT_INT32) :
                                         ge::GetSizeByDataType(ge::DT_FLOAT16);
        biasSize = Ops::Base::CeilAlign(cinAlign * dtypeByteBtBuffer, static_cast<uint64_t>(BYTE_BLOCK));
    }
    uint64_t scaleSize = 0;
    if (hasScaleFlag_ && runInfo_.quantMode == static_cast<uint8_t>(QuantMode::VECTOR_QUANT)) {
        scaleSize = cinAlign * ge::GetSizeByDataType(ge::DT_INT64);
    }
    return b1Size + biasSize + scaleSize;
}

uint64_t Conv3DDXV2SmallKernelTiling::CalcMaxSingleCoreMByL1(uint64_t maxM, uint32_t a1Pbuffer) const
{
    const uint64_t m0 = tilingRunInfo_.m0;
    if (a1Pbuffer == 0 || m0 == 0 || maxM < m0 || platformInfo_.l1_size <= CalcSmallKernelL1FixedSize()) {
        return 0;
    }
    const uint64_t a1BankBudget = (platformInfo_.l1_size - CalcSmallKernelL1FixedSize()) / a1Pbuffer;
    uint64_t low = 1;
    uint64_t high = maxM / m0;
    uint64_t best = 0;
    while (low <= high) {
        uint64_t mid = low + (high - low) / 2;
        uint64_t candidateM = mid * m0;
        if (CalcSmallKernelA1Size(candidateM) <= a1BankBudget) {
            best = candidateM;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return best;
}

bool Conv3DDXV2SmallKernelTiling::HasSupportedSmallKernelDimensions() const
{
    return runInfo_.kernel_d == 1 && runInfo_.dedx_d == 1 && runInfo_.dedy_d == 1 && runInfo_.groups == 1;
}

bool Conv3DDXV2SmallKernelTiling::HasSupportedSmallKernelFormats() const
{
    return runInfo_.outBackpropFormat == ge::FORMAT_NCDHW && runInfo_.filterFormat == ge::FORMAT_NDHWC &&
           runInfo_.yFormat == ge::FORMAT_NCDHW;
}

bool Conv3DDXV2SmallKernelTiling::HasSupportedSmallKernelPadding() const
{
    return runInfo_.backprop_pad_l >= 0 && runInfo_.backprop_pad_r >= 0 && runInfo_.backprop_pad_u >= 0 &&
           runInfo_.backprop_pad_d >= 0 && runInfo_.backprop_pad_l <= PAD_DIM_UP &&
           runInfo_.backprop_pad_r <= PAD_DIM_UP && runInfo_.backprop_pad_u <= PAD_DIM_UP &&
           runInfo_.backprop_pad_d <= PAD_DIM_UP;
}

bool Conv3DDXV2SmallKernelTiling::HasSmallKernelComputationBudget() const
{
    if (static_cast<int32_t>(dtypeByteL0a_) != ge::GetSizeByDataType(ge::DT_FLOAT16) ||
        static_cast<int32_t>(dtypeByteL0b_) != ge::GetSizeByDataType(ge::DT_FLOAT16)) {
        return true;
    }
    uint64_t computation = static_cast<uint64_t>(runInfo_.dedx_h) * runInfo_.dedx_w * runInfo_.kernel_h *
                           runInfo_.kernel_w * runInfo_.dedy_cout_g * runInfo_.dedx_cin_g;
    return computation < SMALL_KERNEL_COMPUTE_THRESHOLD;
}

bool Conv3DDXV2SmallKernelTiling::HasSmallKernelBufferBudget() const
{
    if (tilingRunInfo_.n0 == 0 || tilingRunInfo_.m0 == 0) {
        return false;
    }
    uint64_t cinAlign = Ops::Base::CeilAlign(static_cast<uint64_t>(runInfo_.dedx_cin_g),
                                             static_cast<uint64_t>(tilingRunInfo_.n0));
    uint64_t maxMByL0C = CalcSmallKernelMaxMByL0C(cinAlign, DB_OFF);
    uint64_t hwI = static_cast<uint64_t>(runInfo_.dedx_h) * runInfo_.dedx_w;
    uint64_t maxSingleCoreM = CalcMaxSingleCoreMByL1(std::min(hwI, static_cast<uint64_t>(MAX_BASE_MN)), DB_OFF);
    uint64_t l1UsedSize = CalcSmallKernelL1FixedSize() + CalcSmallKernelA1Size(maxSingleCoreM);
    return maxMByL0C >= tilingRunInfo_.m0 && maxSingleCoreM >= tilingRunInfo_.m0 && l1UsedSize <= platformInfo_.l1_size;
}

bool Conv3DDXV2SmallKernelTiling::CheckSmallKernelEnable()
{
    if (!IsSocVersionFuse(context_)) {
        return false;
    }
    // 维度要求: D=1, group=1
    // format要求: outBackprop/y=NCDHW, filter=NDHWC
    if (!HasSupportedSmallKernelDimensions() || !HasSupportedSmallKernelFormats() ||
        !HasSupportedSmallKernelPadding() || !HasSmallKernelComputationBudget()) {
        return false;
    }
    return HasSmallKernelBufferBudget();
}

REGISTER_TILING_TEMPLATE("Conv3DBackpropInputV2", Conv3DDXV2SmallKernelTiling, 96);

} // namespace Conv
} // namespace NN
} // namespace Ops
