/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV3D_BACKPROP_INPUT_SMALL_KERNEL_ADVANCE_H
#define CONV3D_BACKPROP_INPUT_SMALL_KERNEL_ADVANCE_H

namespace AscendC {

constexpr uint64_t CONV3D_DX_SMALL_DQ_SCALAR_QF_ONE = 0x37800000;
constexpr uint8_t EVENT_ID_FIX_M = 4;
constexpr uint8_t EVENT_ID_MTE2_FIX = 3;
constexpr uint8_t EVENT_ID_M_FIX = 5;
constexpr uint8_t EVENT_ID_M_MTE1_0 = 0;
constexpr uint8_t EVENT_ID_M_MTE1_1 = 1;
constexpr event_t EVENT_ID_A1_PING = static_cast<event_t>(0);
constexpr event_t EVENT_ID_A1_PONG = static_cast<event_t>(1);
constexpr uint32_t SMALL_KERNEL_L0_AB_BUF_SIZE = 64 * 1024;
constexpr uint8_t SHIFT_VALUE_LEN = 58;

template <typename filterType, int filterFormat, typename dedyType, int dedyFormat, typename yType, int yFormat,
          typename biasType, int biasFormat, uint8_t b2Condition, uint8_t kernelSplitMode, uint8_t groupMode,
          uint8_t b1Condition = TPL_GM_TO_L1, bool enableC04Flag = false, typename scaleType = uint64_t,
          int scaleFormat = FORMAT_MAX>
class Conv3dDxSmallKernel {
public:
    using L0cT = typename Convolution3DBackprop::GetDstType<dedyType>::Type;

    __aicore__ inline Conv3dDxSmallKernel() {}

    __aicore__ inline void Init(GM_ADDR filter, GM_ADDR dedy, GM_ADDR y, GM_ADDR workSpace,
                                const conv_bp_v2_kernel::Conv3DBackpropInputV2TilingData* tilingData,
                                GM_ADDR bias = nullptr, GM_ADDR scale = nullptr)
    {
        (void)workSpace;
        hasBias_ = bias != nullptr;
        tiling_ = &tilingData->conv3DDxTiling;
        filterGm_.SetGlobalBuffer((__gm__ filterType*)filter);
        dedyGm_.SetGlobalBuffer((__gm__ dedyType*)dedy);
        yGm_.SetGlobalBuffer((__gm__ yType*)y);
        if (unlikely(hasBias_)) {
            biasGm_.SetGlobalBuffer((__gm__ biasType*)bias);
        }
        if constexpr (GetScaleFormat(scaleFormat) != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
            scaleGm_.SetGlobalBuffer((__gm__ scaleType*)scale);
        }
        ComputeScalarTiling();
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV_SHOULD_RETURN {
            return;
        }

        if (GetAicBlockIdx() >= usedCoreNum_) {
            return;
        }

        CalSmallBlock();
    }

private:
#include "../convolution_3d_backprop/conv3d_bp_small_kernel_func.h"

    const conv_bp_v2_kernel::TConv3DInputV2Tiling* tiling_ = nullptr;
    GlobalTensor<filterType> filterGm_;
    GlobalTensor<dedyType> dedyGm_;
    GlobalTensor<yType> yGm_;
    GlobalTensor<biasType> biasGm_;
    GlobalTensor<scaleType> scaleGm_;

    bool hasBias_ = false;
    uint64_t hiWi_ = 0;
    uint64_t diHiWi_ = 0;
    uint64_t doHoWo_ = 0;
    uint64_t hoExpand_ = 0;
    uint64_t woExpand_ = 0;
    uint64_t hkWk_ = 0;
    uint64_t mCnt_ = 0;
    uint64_t nCnt_ = 0;
    uint64_t usedCoreNum_ = 0;
    uint64_t calRound_ = 0;
    uint64_t tailCnt_ = 0;
    uint32_t coutAlign_ = 0;
    uint32_t kTotal_ = 0;
    uint32_t kIter_ = 0;
    uint32_t a1ElemCount_ = 0;
    uint32_t a1BufBytes_ = 0;
    uint32_t a1Pbuffer_ = 1;
    uint32_t b1OffBytes_ = 0;
    uint32_t b1ElemCount_ = 0;
    uint32_t al0BufBytes_ = 0;
    uint32_t bl0BufBytes_ = 0;
    uint32_t l0Pbuffer_ = 1;

    __aicore__ inline void ComputeScalarTiling()
    {
        hiWi_ = static_cast<uint64_t>(tiling_->hi) * tiling_->wi;
        diHiWi_ = static_cast<uint64_t>(tiling_->di) * hiWi_;
        uint64_t hoWo = static_cast<uint64_t>(tiling_->ho) * tiling_->wo;
        doHoWo_ = static_cast<uint64_t>(tiling_->dout) * hoWo;
        hoExpand_ = (static_cast<uint64_t>(tiling_->ho) - 1) * tiling_->strideH + 1;
        woExpand_ = (static_cast<uint64_t>(tiling_->wo) - 1) * tiling_->strideW + 1;
        hkWk_ = static_cast<uint64_t>(tiling_->hk) * tiling_->wk;
        coutAlign_ = static_cast<uint32_t>(
            DivCeil(static_cast<uint64_t>(tiling_->cout), static_cast<uint64_t>(tiling_->c0)) * tiling_->c0);
        uint32_t smallCinAlign = Convolution3DBackpropFunc::AlignUp16(tiling_->singleCoreCin);

        kTotal_ = static_cast<uint32_t>(static_cast<uint64_t>(coutAlign_) * tiling_->hk * tiling_->wk);
        kIter_ = static_cast<uint32_t>(DivCeil(static_cast<uint64_t>(kTotal_), static_cast<uint64_t>(tiling_->baseK)));
        mCnt_ = DivCeil(hiWi_, tiling_->singleCoreM);
        nCnt_ = DivCeil(static_cast<uint64_t>(tiling_->cin), static_cast<uint64_t>(tiling_->singleCoreCin));
        uint64_t smallTotalCnt = static_cast<uint64_t>(tiling_->batch) * mCnt_ * nCnt_;
        usedCoreNum_ = min(smallTotalCnt, tiling_->coreNum);
        calRound_ = usedCoreNum_ == 0 ? 0 : smallTotalCnt / usedCoreNum_;
        tailCnt_ = usedCoreNum_ == 0 ? 0 : smallTotalCnt - calRound_ * usedCoreNum_;
        uint64_t localHoExpand = CalSmallKernelLocalHo(tiling_->baseM, tiling_->wi, tiling_->hk, tiling_->dilationH,
                                                       hoExpand_);
        a1ElemCount_ = static_cast<uint32_t>(localHoExpand * woExpand_ * coutAlign_);
        a1Pbuffer_ = tiling_->al1Pbuffer == 0 ? 1 : tiling_->al1Pbuffer;
        a1BufBytes_ = a1ElemCount_ * sizeof(dedyType);
        b1OffBytes_ = a1Pbuffer_ * a1BufBytes_;
        b1ElemCount_ = static_cast<uint32_t>(hkWk_ * coutAlign_ * smallCinAlign);
        l0Pbuffer_ = min(tiling_->al0Pbuffer, tiling_->bl0Pbuffer);
        if (l0Pbuffer_ == 0) {
            l0Pbuffer_ = 1;
        }
        al0BufBytes_ = SMALL_KERNEL_L0_AB_BUF_SIZE / l0Pbuffer_;
        bl0BufBytes_ = SMALL_KERNEL_L0_AB_BUF_SIZE / l0Pbuffer_;
    }

    __aicore__ inline static uint64_t CalSmallKernelLocalHo(uint64_t maxM, uint64_t wi, uint64_t hk, uint64_t dilationH,
                                                            uint64_t hoExpand)
    {
        uint64_t hiCount = DivCeil(maxM + wi - 1, wi);
        uint64_t receptiveHo = hiCount + (hk - 1) * dilationH;
        return min(receptiveHo, hoExpand);
    }

    __aicore__ inline uint32_t GetBiasL1OffBytes() const
    {
        uint32_t afterB1 = b1OffBytes_ + b1ElemCount_ * sizeof(filterType);
        return (afterB1 + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }

    __aicore__ inline uint32_t GetScaleL1OffBytes() const
    {
        uint32_t biasL1OffBytes = GetBiasL1OffBytes();
        uint32_t afterBias = hasBias_ ? biasL1OffBytes + tiling_->singleCoreCin * sizeof(biasType) : biasL1OffBytes;
        return (afterBias + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
    }

    __aicore__ inline void InitStaticL1(uint32_t nIdx)
    {
        uint32_t nStart = nIdx * tiling_->singleCoreCin;
        uint32_t curN = tiling_->singleCoreCin;
        if (nStart + curN > tiling_->cin) {
            curN = tiling_->cin - nStart;
        }
        uint32_t curNAlign = Convolution3DBackpropFunc::AlignUp16(curN);
        LocalTensor<filterType> b1(TPosition::B1, b1OffBytes_, b1ElemCount_);
        LoadWeightL1(b1, nStart, curN, curNAlign);
        LoadBiasScaleL1(nStart, curN);
    }

    __aicore__ inline void PreloadSmallBlockA1(uint32_t batchIdx, uint32_t mIdx, uint32_t a1Buf)
    {
        uint32_t mStart = mIdx * tiling_->singleCoreM;
        uint32_t curM = tiling_->singleCoreM;
        if (mStart + curM > hiWi_) {
            curM = hiWi_ - mStart;
        }
        uint32_t curMAlign = Convolution3DBackpropFunc::AlignUp16(curM);
        uint32_t localHoStart = 0;
        uint32_t localHoSize = 0;
        uint32_t localPadUp = 0;
        uint32_t localMStart = 0;
        CalcLocalA1Params(mStart, curMAlign, localHoStart, localHoSize, localPadUp, localMStart);
        LocalTensor<dedyType> a1(TPosition::A1, a1Buf * a1BufBytes_, a1ElemCount_);
        LoadDedyL1(a1, batchIdx, localHoStart, localHoSize);
    }

    __aicore__ inline event_t GetA1EventId(uint32_t a1Buf) const
    {
        return a1Buf == 0 ? EVENT_ID_A1_PING : EVENT_ID_A1_PONG;
    }

    __aicore__ inline void CalSmallBlockSingleBuffer(uint64_t curRound, uint64_t basicIdx, uint32_t batchIdx,
                                                     uint32_t mIdx, uint32_t nIdx)
    {
        for (uint64_t roundIdx = 0; roundIdx < curRound; ++roundIdx) {
            bool needReuseA1 = roundIdx + 1 < curRound;
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_A1_PING);
            PreloadSmallBlockA1(batchIdx, mIdx, 0);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID_A1_PING);
            CalSmallBlockCore(batchIdx, mIdx, nIdx, 0, needReuseA1);
            if (needReuseA1) {
                basicIdx += usedCoreNum_;
                CalSmallBlockIdx(basicIdx, batchIdx, mIdx, nIdx);
            }
        }
    }

    __aicore__ inline void CalSmallBlockDoubleBuffer(uint64_t curRound, uint64_t basicIdx, uint32_t batchIdx,
                                                     uint32_t mIdx, uint32_t nIdx)
    {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID_A1_PING);
        PreloadSmallBlockA1(batchIdx, mIdx, 0);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID_A1_PING);
        for (uint64_t roundIdx = 0; roundIdx < curRound; ++roundIdx) {
            uint32_t curA1Buf = static_cast<uint32_t>(roundIdx & 1U);
            bool hasNext = roundIdx + 1 < curRound;
            bool needReuseA1 = roundIdx + 2 < curRound;
            uint32_t nextBatchIdx = 0;
            uint32_t nextMIdx = 0;
            uint32_t nextNIdx = 0;
            if (hasNext) {
                basicIdx += usedCoreNum_;
                CalSmallBlockIdx(basicIdx, nextBatchIdx, nextMIdx, nextNIdx);
                event_t nextA1EventId = GetA1EventId(curA1Buf ^ 1U);
                WaitFlag<HardEvent::MTE1_MTE2>(nextA1EventId);
                PreloadSmallBlockA1(nextBatchIdx, nextMIdx, curA1Buf ^ 1U);
                SetFlag<HardEvent::MTE2_MTE1>(nextA1EventId);
            }
            CalSmallBlockCore(batchIdx, mIdx, nIdx, curA1Buf, needReuseA1);
            if (hasNext) {
                batchIdx = nextBatchIdx;
                mIdx = nextMIdx;
                nIdx = nextNIdx;
            }
        }
    }

    __aicore__ inline void CalSmallBlock()
    {
        uint64_t blockIdx = GetAicBlockIdx();
        uint64_t curRound = calRound_ + (blockIdx < tailCnt_ ? 1 : 0);
        uint32_t batchIdx = 0;
        uint32_t mIdx = 0;
        uint32_t nIdx = 0;
        CalSmallBlockIdx(blockIdx, batchIdx, mIdx, nIdx);
        SetFlag<HardEvent::FIX_M>(EVENT_ID_FIX_M);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_A1_PING);
        if (a1Pbuffer_ > 1 && curRound > 1) {
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID_A1_PONG);
        }
        InitStaticL1(nIdx);
        if (a1Pbuffer_ == 1) {
            CalSmallBlockSingleBuffer(curRound, blockIdx, batchIdx, mIdx, nIdx);
        } else {
            CalSmallBlockDoubleBuffer(curRound, blockIdx, batchIdx, mIdx, nIdx);
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID_FIX_M);
    }

    __aicore__ inline void CalSmallBlockIdx(uint64_t basicBlockIdx, uint32_t& batchIdx, uint32_t& mIdx, uint32_t& nIdx)
    {
        uint64_t mnCnt = mCnt_ * nCnt_;
        batchIdx = static_cast<uint32_t>(basicBlockIdx / mnCnt);
        basicBlockIdx -= static_cast<uint64_t>(batchIdx) * mnCnt;
        mIdx = basicBlockIdx / nCnt_;
        basicBlockIdx -= static_cast<uint64_t>(mIdx) * nCnt_;
        nIdx = basicBlockIdx;
    }

    __aicore__ inline bool GetSmallBlockRange(uint32_t mIdx, uint32_t nIdx, uint32_t& mStart, uint32_t& nStart,
                                              uint32_t& curM, uint32_t& curN) const
    {
        mStart = mIdx * tiling_->singleCoreM;
        nStart = nIdx * tiling_->singleCoreCin;
        if (mStart >= hiWi_ || nStart >= tiling_->cin) {
            return false;
        }
        curM = static_cast<uint32_t>(tiling_->singleCoreM);
        if (mStart + curM > hiWi_) {
            curM = static_cast<uint32_t>(hiWi_ - mStart);
        }
        curN = static_cast<uint32_t>(tiling_->singleCoreCin);
        if (nStart + curN > tiling_->cin) {
            curN = tiling_->cin - nStart;
        }
        return true;
    }

    __aicore__ inline void PrepareSmallBlockChannelWise(uint32_t curN)
    {
        if (hasBias_) {
            LoadBiasToBT(curN);
        }
        if constexpr (GetScaleFormat(scaleFormat) != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
            if (tiling_->quantMode == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
                SetFlag<HardEvent::MTE2_FIX>(EVENT_ID_MTE2_FIX);
                WaitFlag<HardEvent::MTE2_FIX>(EVENT_ID_MTE2_FIX);
            }
        }
    }

    __aicore__ inline void CalSmallBlockMmad(const LocalTensor<dedyType>& a1, const LocalTensor<filterType>& b1,
                                             LocalTensor<L0cT>& c0, uint32_t localHoSize, uint32_t localPadUp,
                                             uint32_t localMStart, uint32_t curMAlign, uint32_t curNAlign,
                                             uint32_t curN, event_t a1EventId, bool needReuseA1)
    {
        MmadParams mmCommand;
        mmCommand.m = curMAlign;
        mmCommand.n = curNAlign;
        bool firstMmad = true;
#if (__NPU_ARCH__ == 5102)
        if constexpr (std::is_same<dedyType, half>::value) {
            mmCommand.fixShiftVal = tiling_->fixedShiftVal;
        }
#endif
        for (uint32_t kIter = 0; kIter < kIter_; ++kIter) {
            uint32_t kOff = kIter * tiling_->baseK;
            uint32_t curK = min(tiling_->baseK, kTotal_ - kOff);
            uint32_t buf = l0Pbuffer_ == 1 ? 0 : (kIter & 1);
            event_t eventId = static_cast<event_t>(buf);
            LocalTensor<dedyType> a0(TPosition::A2, buf * al0BufBytes_, al0BufBytes_ / sizeof(dedyType));
            LocalTensor<filterType> b0(TPosition::B2, buf * bl0BufBytes_, bl0BufBytes_ / sizeof(filterType));
            WaitFlag<HardEvent::M_MTE1>(eventId);
            LoadAL0(a0, a1, localHoSize, localPadUp, localMStart, curMAlign, kOff, curK);
            LoadBL0(b0, b1, kOff, curK, curNAlign);
            SetFlag<HardEvent::MTE1_M>(eventId);
            WaitFlag<HardEvent::MTE1_M>(eventId);
            if (kIter + 1 == kIter_ && needReuseA1) {
                SetFlag<HardEvent::MTE1_MTE2>(a1EventId);
            }
            mmCommand.k = curK;
            if (firstMmad && hasBias_) {
                LocalTensor<L0cT> biasBT(TPosition::C2, 0, Convolution3DBackpropFunc::AlignUp16(curN));
                mmCommand.cmatrixInitVal = false;
                mmCommand.cmatrixSource = true;
                mmCommand.isBias = true;
                Mmad(c0, a0, b0, biasBT, mmCommand);
            } else {
                mmCommand.cmatrixInitVal = firstMmad;
                mmCommand.cmatrixSource = false;
                mmCommand.isBias = false;
                Mmad(c0, a0, b0, mmCommand);
            }
            firstMmad = false;
            SetFlag<HardEvent::M_MTE1>(eventId);
        }
    }

    __aicore__ inline void CalSmallBlockCore(uint32_t batchIdx, uint32_t mIdx, uint32_t nIdx, uint32_t a1Buf,
                                             bool needReuseA1)
    {
        uint32_t mStart = 0;
        uint32_t nStart = 0;
        uint32_t curM = 0;
        uint32_t curN = 0;
        if (!GetSmallBlockRange(mIdx, nIdx, mStart, nStart, curM, curN)) {
            return;
        }

        LocalTensor<dedyType> a1(TPosition::A1, a1Buf * a1BufBytes_, a1ElemCount_);
        LocalTensor<filterType> b1(TPosition::B1, b1OffBytes_, b1ElemCount_);
        LocalTensor<L0cT> c0(TPosition::CO1, 0, (256 * 1024) / sizeof(L0cT));
        uint32_t curMAlign = Convolution3DBackpropFunc::AlignUp16(curM);
        uint32_t curNAlign = Convolution3DBackpropFunc::AlignUp16(curN);
        event_t a1EventId = GetA1EventId(a1Buf);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_MTE1_0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID_M_MTE1_1);
        uint32_t localHoStart = 0;
        uint32_t localHoSize = 0;
        uint32_t localPadUp = 0;
        uint32_t localMStart = 0;
        CalcLocalA1Params(mStart, curMAlign, localHoStart, localHoSize, localPadUp, localMStart);
        WaitFlag<HardEvent::MTE2_MTE1>(a1EventId);
        PrepareSmallBlockChannelWise(curN);

        WaitFlag<HardEvent::FIX_M>(EVENT_ID_FIX_M);

        CalSmallBlockMmad(a1, b1, c0, localHoSize, localPadUp, localMStart, curMAlign, curNAlign, curN, a1EventId,
                          needReuseA1);

        WaitFlag<HardEvent::M_MTE1>(EVENT_ID_M_MTE1_0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID_M_MTE1_1);

        SetFlag<HardEvent::M_FIX>(EVENT_ID_M_FIX);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID_M_FIX);

        CopyOut(c0, batchIdx, mStart, nStart, curM, curMAlign, curN);

        SetFlag<HardEvent::FIX_M>(EVENT_ID_FIX_M);
    }
};

} // namespace AscendC

#endif // CONV3D_BACKPROP_INPUT_SMALL_KERNEL_ADVANCE_H
