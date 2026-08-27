/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_dense_grad_base.h
 * \brief embedding_dense_grad_base
 */

#ifndef EMBEDDING_DENSE_GRAD_BASE_H
#define EMBEDDING_DENSE_GRAD_BASE_H

#include <cstdint>
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "../inc/load_store_utils.h"

using namespace AscendC;
using namespace ops;

namespace EmbeddingDenseGradBase {
constexpr uint32_t HELP_FRE = 2;
constexpr uint32_t SORT_STAT_PADDING = 64;
constexpr uint32_t PROCESS_GROUP = 5;
constexpr uint32_t BLOCK_ALIGN_B32 = platform::GetUbBlockSize() / sizeof(float);

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

constexpr AscendC::Reg::CastTrait castTraitB162B32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::Reg::CastTrait castTraitI32F32 = {AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::UNKNOWN,
                                                     AscendC::Reg::MaskMergeMode::ZEROING,
                                                     AscendC::RoundMode::CAST_RINT};

typedef struct {
    uint32_t segCount;
    uint16_t loopSegCount;
    uint16_t loopSegCoutTail;
    uint32_t gradWeightGmindex;
    uint32_t gradPerRowNum;
    uint32_t ubOutOffset;
    __ubuf__ uint32_t* sortedIndiceIndexAddr;
} GroupAddParams;

template <typename U>
class EmbeddingDenseGradProcessIndices {
public:
    static constexpr uint32_t vfIndicesNum_ = platform::GetVRegSize() / sizeof(U);
    static constexpr uint32_t shiftOffset_ = platform::GetUbBlockSize() / sizeof(U);
    static constexpr uint32_t perVfIndicesIdx_ = platform::GetVRegSize() / sizeof(int32_t);

    TQue<QuePosition::VECIN, 1> inQueueIndice_;
    TBuf<QuePosition::VECCALC> noDupBuf_;
    TBuf<QuePosition::VECCALC> noDupProcessLoopBuf_;
    TBuf<QuePosition::VECCALC> sortedIndiceBuf_;
    TBuf<QuePosition::VECCALC> sortedIndiceIndexBuf_;
    TBuf<QuePosition::VECCALC> sharedTmpBuf_;

    GlobalTensor<U> indicesGm_;

    __aicore__ inline void IndicesInitBuffer(TPipe& pipe, uint32_t indicesNumber, uint32_t sortSharedBufSize,
                                             GM_ADDR indices)
    {
        indicesGm_.SetGlobalBuffer((__gm__ U*)(indices));

        uint32_t indicesAlignB32 = Aligned(static_cast<uint32_t>(indicesNumber * sizeof(uint32_t)),
                                           platform::GetUbBlockSize());
        uint32_t indicesAlign = Aligned(static_cast<uint32_t>(indicesNumber * sizeof(U)), platform::GetUbBlockSize());

        pipe.InitBuffer(inQueueIndice_, 1, indicesAlign);
        pipe.InitBuffer(noDupBuf_, indicesAlignB32 + SORT_STAT_PADDING);
        pipe.InitBuffer(noDupProcessLoopBuf_, indicesAlignB32);
        pipe.InitBuffer(sortedIndiceBuf_, indicesAlign + SORT_STAT_PADDING);
        pipe.InitBuffer(sortedIndiceIndexBuf_, indicesAlignB32);
        pipe.InitBuffer(sharedTmpBuf_, sortSharedBufSize);
    }

    __aicore__ inline void CopyIndicesFromGm(int64_t indicesOffset, uint32_t indicesNumber)
    {
        LocalTensor<U> indicesLocalIn = inQueueIndice_.AllocTensor<U>();

        DataCopyPadExtParams<U> padParams{false, 0, 0, 0};
        DataCopyExtParams copyParam{1, static_cast<uint32_t>(indicesNumber * sizeof(U)), 0, 0, 0};
        DataCopyPad(indicesLocalIn, indicesGm_[indicesOffset], copyParam, padParams);

        inQueueIndice_.EnQue(indicesLocalIn);
    }

    __aicore__ inline void UniqueGetElm(const LocalTensor<U>& sortedIndice, LocalTensor<int32_t>& noDupRes,
                                        uint32_t indicesFactorReal, int64_t& arNum)
    {
        __ubuf__ U* sortedIndicesAddr = (__ubuf__ U*)sortedIndice.GetPhyAddr();
        __ubuf__ int32_t* noDupResAddr = (__ubuf__ int32_t*)noDupRes.GetPhyAddr();

        uint16_t loopCnt = (uint16_t)((indicesFactorReal + vfIndicesNum_) / vfIndicesNum_);

        int32_t scalar = 0;
        uint32_t counter = indicesFactorReal + 1;
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<int32_t> orderReg;
            AscendC::Reg::RegTensor<int32_t> selReg;
            AscendC::Reg::RegTensor<U> indicesReg;
            AscendC::Reg::RegTensor<U> indicesShiftOneReg;

            AscendC::Reg::MaskReg cmpMask;
            AscendC::Reg::MaskReg maskRegUpdate;
            AscendC::Reg::UnalignRegForLoad u0;
            Reg::UnalignRegForStore ureg;
            AscendC::Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();

            for (uint16_t i = 0; i < loopCnt; ++i) {
                scalar = i * vfIndicesNum_;
                auto sortedIndicesAddrUpdate = sortedIndicesAddr + shiftOffset_ + i * vfIndicesNum_;
                AscendC::Reg::Arange(orderReg, scalar);
                maskRegUpdate = AscendC::Reg::UpdateMask<U>(counter);
                AscendC::Reg::LoadAlign(indicesReg, sortedIndicesAddrUpdate);
                AscendC::Reg::LoadUnAlignPre(u0, sortedIndicesAddrUpdate - 1);
                AscendC::Reg::LoadUnAlign<U>(indicesShiftOneReg, u0, sortedIndicesAddrUpdate - 1);
                AscendC::Reg::Compare<U, CMPMODE::NE>(cmpMask, indicesReg, indicesShiftOneReg, maskRegUpdate);
                if constexpr (IsSameType<U, int64_t>::value) {
                    AscendC::Reg::MaskReg maskHalf;
                    AscendC::Reg::Pack<AscendC::Reg::HighLowPart::LOWEST>(maskHalf, cmpMask);
                    // vSQZ
                    AscendC::Reg::Squeeze<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(selReg, orderReg, maskHalf);
                } else {
                    // vSQZ
                    AscendC::Reg::Squeeze<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(selReg, orderReg, cmpMask);
                }
                AscendC::Reg::StoreUnAlign<int32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(noDupResAddr, selReg,
                                                                                                 ureg);
                AscendC::Reg::StoreUnAlignPost(noDupResAddr, ureg);
            }
        }
    }

    __aicore__ inline void UniqueStat(LocalTensor<int32_t>& noDupRes, int64_t& arNum)
    {
        LocalTensor<int32_t> noDupResProcessLoop = noDupProcessLoopBuf_.Get<int32_t>();

        __ubuf__ int32_t* noDupResAddr = (__ubuf__ int32_t*)noDupRes.GetPhyAddr();
        __ubuf__ int32_t* noDupResProcessLoopAddr = (__ubuf__ int32_t*)noDupResProcessLoop.GetPhyAddr();

        uint16_t loopCntStatFre = (arNum + perVfIndicesIdx_ - 1) / perVfIndicesIdx_;
        uint32_t counterStatFre = static_cast<uint32_t>(arNum);
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<int32_t> beginReg;
            AscendC::Reg::RegTensor<int32_t> endReg;
            AscendC::Reg::RegTensor<int32_t> subReg;
            AscendC::Reg::RegTensor<int32_t> divReg, allFiveReg;
            AscendC::Reg::MaskReg maskRegUpdate;
            AscendC::Reg::UnalignRegForLoad u0;
            AscendC::Reg::MaskReg maskRegFull = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();

            Duplicate(allFiveReg, static_cast<int32_t>(PROCESS_GROUP),
                      maskRegFull); // Abs before reducemax, scaleReg >= 0
            for (uint16_t i = 0; i < loopCntStatFre; i++) {
                auto noDupResAddrUpdate = noDupResAddr + i * perVfIndicesIdx_ + 1;
                maskRegUpdate = AscendC::Reg::UpdateMask<int32_t>(counterStatFre);
                AscendC::Reg::LoadAlign(beginReg, noDupResAddr + i * perVfIndicesIdx_);
                AscendC::Reg::LoadUnAlignPre(u0, noDupResAddrUpdate);
                AscendC::Reg::LoadUnAlign<int32_t>(endReg, u0, noDupResAddrUpdate);
                AscendC::Reg::Sub(subReg, endReg, beginReg, maskRegUpdate);
                AscendC::Reg::Div(divReg, subReg, allFiveReg, maskRegUpdate);
                AscendC::Reg::StoreAlign(noDupResAddr + i * perVfIndicesIdx_, subReg, maskRegUpdate);
                AscendC::Reg::StoreAlign(noDupResProcessLoopAddr + i * perVfIndicesIdx_, divReg, maskRegUpdate);
            }
        }

        event_t eventV_S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventV_S);
        WaitFlag<HardEvent::V_S>(eventV_S);
    }

    __aicore__ inline void ProcessIndices(const LocalTensor<U>& sortedIndice, uint32_t indicesFactorReal,
                                          int64_t& arNum)
    {
        LocalTensor<U> indicesLocal = inQueueIndice_.DeQue<U>();
        LocalTensor<int32_t> noDupRes = noDupBuf_.Get<int32_t>();
        LocalTensor<uint32_t> sortedIndiceIndex = sortedIndiceIndexBuf_.Get<uint32_t>();
        LocalTensor<uint8_t> sharedTmpBuffer = sharedTmpBuf_.Get<uint8_t>();

        Duplicate(sortedIndice, static_cast<U>(-1), shiftOffset_ * HELP_FRE + indicesFactorReal);

        LocalTensor<U> sortedDstLocal = sortedIndice[shiftOffset_];
        LocalTensor<uint32_t> sortedIndexLocal = sortedIndiceIndex[0];
        static constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};
        AscendC::Sort<U, false, sortConfig>(sortedDstLocal, sortedIndexLocal, indicesLocal, sharedTmpBuffer,
                                            static_cast<uint32_t>(indicesFactorReal));

        Duplicate(noDupRes, 0, indicesFactorReal);
        UniqueGetElm(sortedIndice, noDupRes, indicesFactorReal, arNum);
        arNum = ((AscendC::Reg::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t)) - 1;
        UniqueStat(noDupRes, arNum);

        inQueueIndice_.FreeTensor(indicesLocal);
    }

    template <typename srcDtype, typename resDtype, typename VGatherIndexDType, bool isFullLoad = false,
              bool isScale = false>
    __aicore__ inline void ProcessPerGradGroup(__ubuf__ srcDtype* gradLocalAddr, __ubuf__ resDtype* resBufAddr,
                                               Reg::MaskReg& maskRegUpdate, Reg::RegTensor<int32_t>& serReg,
                                               GroupAddParams& params)
    {
        Reg::RegTensor<VGatherIndexDType> initIdsReg, idsReg;
        Reg::RegTensor<srcDtype> gatherOut;
        Reg::RegTensor<float> gatherOutB32, weightReg, tmpWeightReg;
        Reg::MaskReg maskReg = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

        Reg::Duplicate(weightReg, (float)0, maskReg);
        // Main
        for (uint16_t pGIdx = 0; pGIdx < params.loopSegCount; ++pGIdx) {
            Reg::Duplicate(tmpWeightReg, (float)0, maskReg);
            for (uint16_t pIdx = 0; pIdx < (uint16_t)PROCESS_GROUP; ++pIdx) {
                // vlds
                Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>(
                    (Reg::RegTensor<uint32_t>&)initIdsReg,
                    params.sortedIndiceIndexAddr + (params.gradWeightGmindex + PROCESS_GROUP * pGIdx + pIdx));
                Reg::Muls(idsReg, initIdsReg, params.gradPerRowNum, maskRegUpdate);
                Reg::Add(idsReg, (Reg::RegTensor<VGatherIndexDType>&)serReg, idsReg, maskRegUpdate);
                Reg::Gather(gatherOut, gradLocalAddr, idsReg, maskRegUpdate);
                if constexpr (is_same<half, srcDtype>::value) {
                    Cast<float, half, EmbeddingDenseGradBase::castTraitB162B32>(gatherOutB32, gatherOut, maskRegUpdate);
                    Reg::Add(tmpWeightReg, tmpWeightReg, gatherOutB32, maskRegUpdate);
                } else if constexpr (is_same<bfloat16_t, srcDtype>::value) {
                    Cast<float, bfloat16_t, EmbeddingDenseGradBase::castTraitB162B32>(gatherOutB32, gatherOut,
                                                                                      maskRegUpdate);
                    Reg::Add(tmpWeightReg, tmpWeightReg, gatherOutB32, maskRegUpdate);
                } else {
                    Reg::Add(tmpWeightReg, tmpWeightReg, gatherOut, maskRegUpdate);
                }
            }
            Reg::Add(weightReg, weightReg, tmpWeightReg, maskRegUpdate);
        }

        Reg::Duplicate(tmpWeightReg, (float)0, maskReg);
        __ubuf__ uint32_t* sortedIndicesIndexTailAddr = params.sortedIndiceIndexAddr + params.gradWeightGmindex +
                                                        params.loopSegCount * PROCESS_GROUP;
        // Tail
        for (uint16_t pIdxTail = 0; pIdxTail < params.loopSegCoutTail; ++pIdxTail) {
            // vlds
            Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_BRC_B32>((Reg::RegTensor<uint32_t>&)initIdsReg,
                                                                  sortedIndicesIndexTailAddr + pIdxTail);
            Reg::Muls(initIdsReg, initIdsReg, params.gradPerRowNum, maskRegUpdate);
            Reg::Add(initIdsReg, (Reg::RegTensor<VGatherIndexDType>&)serReg, initIdsReg, maskRegUpdate);
            Reg::Gather(gatherOut, gradLocalAddr, initIdsReg, maskRegUpdate);
            if constexpr (is_same<half, srcDtype>::value) {
                Cast<float, half, EmbeddingDenseGradBase::castTraitB162B32>(gatherOutB32, gatherOut, maskRegUpdate);
                Reg::Add(tmpWeightReg, tmpWeightReg, gatherOutB32, maskRegUpdate);
            } else if constexpr (is_same<bfloat16_t, srcDtype>::value) {
                Cast<float, bfloat16_t, EmbeddingDenseGradBase::castTraitB162B32>(gatherOutB32, gatherOut,
                                                                                  maskRegUpdate);
                Reg::Add(tmpWeightReg, tmpWeightReg, gatherOutB32, maskRegUpdate);
            } else {
                Reg::Add(tmpWeightReg, tmpWeightReg, gatherOut, maskRegUpdate);
            }
        }

        Reg::Add(weightReg, weightReg, tmpWeightReg, maskRegUpdate);

        if constexpr (isFullLoad) {
            if constexpr (isScale) {
                Reg::RegTensor<int32_t> scaleFreqInt;
                Reg::RegTensor<float> scaleFreqReg;
                Reg::Duplicate(scaleFreqInt, static_cast<int32_t>(params.segCount), maskReg);
                Reg::Cast<float, int32_t, EmbeddingDenseGradBase::castTraitI32F32>(scaleFreqReg, scaleFreqInt, maskReg);
                Reg::Div(weightReg, weightReg, scaleFreqReg, maskRegUpdate);
            }
            ops::StoreOneTensorForDtypeT<resDtype>(resBufAddr, weightReg, maskRegUpdate, params.ubOutOffset);
        } else {
            Reg::StoreAlign(((__ubuf__ float*)resBufAddr + params.ubOutOffset), weightReg, maskRegUpdate);
        }
    }
};
} // namespace EmbeddingDenseGradBase

#endif
