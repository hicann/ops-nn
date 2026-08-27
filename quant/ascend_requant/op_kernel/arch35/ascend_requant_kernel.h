/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "kernel_operator.h"
#include "ascend_requant_tiling_data.h"
#include "ascend_requant_struct.h"

constexpr AscendC::Reg::CastTrait kCastF32FromS32 = {AscendC::Reg::RegLayout::UNKNOWN, AscendC::Reg::SatMode::NO_SAT,
                                                     AscendC::Reg::MaskMergeMode::ZEROING,
                                                     AscendC::RoundMode::CAST_NONE};
constexpr AscendC::Reg::CastTrait kCastF32FromS64 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                     AscendC::Reg::MaskMergeMode::ZEROING,
                                                     AscendC::RoundMode::CAST_RINT};
constexpr AscendC::Reg::CastTrait kCastH16FromF32RndSat = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                           AscendC::Reg::MaskMergeMode::ZEROING,
                                                           AscendC::RoundMode::CAST_RINT};
constexpr AscendC::Reg::CastTrait kCastS8FromH16RndSat = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING,
                                                          AscendC::RoundMode::CAST_RINT};

template <typename XT, typename ST, typename YT, bool DO_RELU>
__simd_vf__ inline void RequantVF(__ubuf__ YT* dstAddr, __ubuf__ XT* src0Addr, __ubuf__ ST* src1Addr, uint32_t count,
                                  uint32_t oneRepeatSize, uint16_t repeatTimes);

__aicore__ inline void GetCoreRange(int64_t coreId, int64_t tilesMain, int64_t coresTail, int64_t& start, int64_t& end)
{
    if (coreId < coresTail) {
        start = coreId * (tilesMain + 1);
        end = start + tilesMain + 1;
    } else {
        start = coresTail * (tilesMain + 1) + (coreId - coresTail) * tilesMain;
        end = start + tilesMain;
    }
}

__aicore__ inline int64_t GetUBSplitRange(int64_t aOOff, int64_t aO, int64_t aI, int64_t aITail)
{
    return (aOOff == aO - 1) ? aITail : aI;
}

__aicore__ inline bool FlatToEffectiveCoord(int64_t flat, const int64_t* maxBroShape, int64_t rank, int64_t splitAxis,
                                            int64_t aI, int64_t aO, int64_t* effCoord)
{
    for (int64_t d = 0; d < rank; d++)
        effCoord[d] = 0;
    if (aO <= 0)
        return false;
    int64_t aOOff = flat % aO;
    int64_t outer = flat / aO;
    for (int64_t d = splitAxis - 1; d >= 0; d--) {
        effCoord[d] = outer % maxBroShape[d];
        outer /= maxBroShape[d];
    }
    effCoord[splitAxis] = aOOff * aI;
    return true;
}

__aicore__ inline int64_t CalcInputOffset(const int64_t* effCoord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++)
        offset += effCoord[d] * strides[d];
    return offset;
}

__aicore__ inline int64_t CalcOutputOffset(const int64_t* effCoord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++)
        offset += effCoord[d] * strides[d];
    return offset;
}

template <int64_t RANK, bool DO_RELU>
class AscendRequantKernel {
    static constexpr int64_t ND = (RANK <= 5) ? RANK : 5;
    static constexpr uint32_t VL = AscendC::GetVecLen() / sizeof(int32_t);

    using XT = int32_t;
    using ST = uint64_t;
    using YT = int8_t;

    AscendC::TPipe pipe_;
    const AscendRequantTilingData<RANK>* td_;
    AscendC::GlobalTensor<XT> gmX_;
    AscendC::GlobalTensor<ST> gmS_;
    AscendC::GlobalTensor<YT> gmY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[PHYS_NODES];
    AscendC::MultiCopyParams<XT, ND> nddmaX_;
    AscendC::MultiCopyParams<ST, ND> nddmaS_;
    int64_t nddmaOuterItersX_;
    int64_t nddmaOuterItersS_;
    int64_t nddmaDims_;

public:
    __aicore__ inline void Init(GM_ADDR inputs[MAX_INPUT_SLOTS], GM_ADDR outputs[MAX_OUTPUT_SLOTS],
                                const AscendRequantTilingData<RANK>* td)
    {
        td_ = td;
        gmX_.SetGlobalBuffer((__gm__ XT*)inputs[0]);
        gmS_.SetGlobalBuffer((__gm__ ST*)inputs[1]);
        gmY_.SetGlobalBuffer((__gm__ YT*)outputs[0]);

        for (int i = 0; i < PHYS_NODES; i++)
            pipe_.InitBuffer(buf_[i], td_->perBufBytes);

        const int64_t* dstShape = td_->maxBroShape;
        int64_t k = td_->split.axis;
        nddmaDims_ = (RANK - k <= ND) ? (RANK - k) : ND;

        InitNddmaParams<XT>(nddmaX_, nddmaOuterItersX_, dstShape, k);
        InitNddmaParams<ST>(nddmaS_, nddmaOuterItersS_, dstShape, k);
    }

    __aicore__ inline void Process()
    {
        int32_t evMTE2toV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        int32_t evVtoMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        int32_t evMTE3toMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));

        int64_t start, end;
        GetCoreRange(AscendC::GetBlockIdx(), td_->multicore.tilesMain, td_->multicore.coresTail, start, end);

        int64_t innerCount = 1;
        for (int64_t d = td_->split.axis + 1; d < RANK; d++)
            innerCount *= td_->maxBroShape[d];

        int64_t coord[8] = {};
        for (int64_t flat = start; flat < end; flat++) {
            int64_t aISeg = GetUBSplitRange(flat % td_->split.aO, td_->split.aO, td_->split.aI, td_->split.aITail);
            int64_t count = aISeg * innerCount;
            FlatToEffectiveCoord(flat, td_->maxBroShape, RANK, td_->split.axis, td_->split.aI, td_->split.aO, coord);

            if (flat != start)
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);

            CopyInBrcX(coord, 0, aISeg);
            CopyInBrcS(coord, 1, aISeg);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);

            uint16_t rep = AscendC::CeilDivision(static_cast<uint32_t>(count), VL);
            asc_vf_call<RequantVF<XT, ST, YT, DO_RELU>>(
                (__ubuf__ YT*)buf_[2].Get<YT>().GetPhyAddr(), (__ubuf__ XT*)buf_[0].Get<XT>().GetPhyAddr(),
                (__ubuf__ ST*)buf_[1].Get<ST>().GetPhyAddr(), static_cast<uint32_t>(count), VL, rep);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);

            CopyOutOne(coord, 0, 2, aISeg);

            if (flat != end - 1)
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
        }
    }

private:
    template <typename T>
    __aicore__ inline void InitNddmaParams(AscendC::MultiCopyParams<T, ND>& params, int64_t& outerIters,
                                           const int64_t* dstShape, int64_t k)
    {
        int64_t inner = 1;
        int64_t nd = 0;
        for (int64_t d = RANK - 1; d >= k && nd < ND; d--) {
            params.loopInfo.loopSize[nd] = (d == k) ? 0 : dstShape[d];
            params.loopInfo.loopSrcStride[nd] = td_->inputStrides[(std::is_same_v<T, XT>) ? 0 : 1][d];
            params.loopInfo.loopDstStride[nd] = inner;
            params.loopInfo.loopLpSize[nd] = 0;
            params.loopInfo.loopRpSize[nd] = 0;
            inner *= (d == k) ? td_->split.aI : dstShape[d];
            nd++;
        }
        for (; nd < ND; nd++) {
            params.loopInfo.loopSize[nd] = 1;
            params.loopInfo.loopSrcStride[nd] = 0;
            params.loopInfo.loopDstStride[nd] = inner;
            params.loopInfo.loopLpSize[nd] = 0;
            params.loopInfo.loopRpSize[nd] = 0;
        }
        outerIters = 1;
        for (int64_t d = k; d < RANK - nddmaDims_; d++)
            outerIters *= (d == k) ? td_->split.aI : dstShape[d];
    }

    __aicore__ inline void CopyInBrcX(const int64_t* coord, int slot, int64_t aISeg)
    {
        int64_t k = td_->split.axis;
        int64_t off = CalcInputOffset(coord, td_->inputStrides[0], RANK);
        const int64_t* dstShape = td_->maxBroShape;

        auto params = nddmaX_;
        int64_t kNd = RANK - 1 - k;
        int64_t inner = 1;
        for (int64_t nd = 0; nd < ND; nd++) {
            if (nd == kNd)
                params.loopInfo.loopSize[nd] = aISeg;
            params.loopInfo.loopDstStride[nd] = inner;
            inner *= params.loopInfo.loopSize[nd];
        }

        static constexpr AscendC::NdDmaConfig cfg = {false, AscendC::NdDmaConfig::unsetPad,
                                                     AscendC::NdDmaConfig::unsetPad, false};
        if constexpr (RANK <= 5) {
            AscendC::DataCopy<XT, ND, cfg>(buf_[slot].Get<XT>(), gmX_[off], params);
        } else {
            AscendC::LocalTensor<XT> buf = buf_[slot].Get<XT>();
            for (int64_t oi = 0; oi < nddmaOuterItersX_; oi++) {
                int64_t elemAdj = 0, tmp = oi;
                for (int64_t d = RANK - nddmaDims_ - 1; d >= k; d--) {
                    int64_t sz = (d == k) ? aISeg : dstShape[d];
                    elemAdj += (tmp % sz) * td_->inputStrides[0][d];
                    tmp /= sz;
                }
                AscendC::DataCopy<XT, ND, cfg>(buf[oi * inner], gmX_[off + elemAdj], params);
            }
        }
    }

    __aicore__ inline void CopyInBrcS(const int64_t* coord, int slot, int64_t aISeg)
    {
        int64_t k = td_->split.axis;
        int64_t off = CalcInputOffset(coord, td_->inputStrides[1], RANK);
        const int64_t* dstShape = td_->maxBroShape;

        auto params = nddmaS_;
        int64_t kNd = RANK - 1 - k;
        int64_t inner = 1;
        for (int64_t nd = 0; nd < ND; nd++) {
            if (nd == kNd)
                params.loopInfo.loopSize[nd] = aISeg;
            params.loopInfo.loopDstStride[nd] = inner;
            inner *= params.loopInfo.loopSize[nd];
        }

        static constexpr AscendC::NdDmaConfig cfg = {false, AscendC::NdDmaConfig::unsetPad,
                                                     AscendC::NdDmaConfig::unsetPad, false};
        if constexpr (RANK <= 5) {
            AscendC::DataCopy<ST, ND, cfg>(buf_[slot].Get<ST>(), gmS_[off], params);
        } else {
            AscendC::LocalTensor<ST> buf = buf_[slot].Get<ST>();
            for (int64_t oi = 0; oi < nddmaOuterItersS_; oi++) {
                int64_t elemAdj = 0, tmp = oi;
                for (int64_t d = RANK - nddmaDims_ - 1; d >= k; d--) {
                    int64_t sz = (d == k) ? aISeg : dstShape[d];
                    elemAdj += (tmp % sz) * td_->inputStrides[1][d];
                    tmp /= sz;
                }
                AscendC::DataCopy<ST, ND, cfg>(buf[oi * inner], gmS_[off + elemAdj], params);
            }
        }
    }

    __aicore__ inline void CopyOutOne(const int64_t* coord, int outputIdx, int slot, int64_t aISeg)
    {
        int64_t off = CalcOutputOffset(coord, td_->outputStrides[outputIdx], RANK);
        int64_t splitElems = aISeg;
        int64_t innerElems = 1;
        for (int64_t d = td_->split.axis + 1; d < RANK; d++)
            innerElems *= td_->outputShapes[outputIdx][d];
        int64_t cnt = splitElems * innerElems;

        AscendC::DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = cnt * sizeof(YT);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        AscendC::DataCopyPad(gmY_[off], buf_[slot].Get<YT>(), extParams);
    }
};

template <typename XT, typename ST, typename YT, bool DO_RELU>
__simd_vf__ inline void RequantVF(__ubuf__ YT* dstAddr, __ubuf__ XT* src0Addr, __ubuf__ ST* src1Addr, uint32_t count,
                                  uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<XT> xReg;
    AscendC::Reg::RegTensor<int64_t, AscendC::Reg::RegTraitNumTwo> i64Reg;
    AscendC::Reg::RegTensor<float> fp32XReg;
    AscendC::Reg::RegTensor<float> fp32SReg;
    AscendC::Reg::RegTensor<float> correctedSReg;
    AscendC::Reg::RegTensor<float> scaledReg;
    AscendC::Reg::RegTensor<half> f16Reg;
    AscendC::Reg::RegTensor<YT> yReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::MaskReg negMask;

    static constexpr float kUint64AsFloat = static_cast<float>(uint64_t(1) << 63) * 2.0f;

    uint32_t sreg = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::Reg::UpdateMask<uint32_t>(sreg);

        AscendC::Reg::LoadAlign(xReg, src0Addr + i * oneRepeatSize);
        AscendC::Reg::LoadAlign(i64Reg, (__ubuf__ int64_t*)(src1Addr + i * oneRepeatSize));

        AscendC::Reg::Cast<float, XT, kCastF32FromS32>(fp32XReg, xReg, mask);
        AscendC::Reg::Cast<float, int64_t, kCastF32FromS64>(fp32SReg, i64Reg, mask);
        AscendC::Reg::Compares<int64_t, AscendC::CMPMODE::LT>(negMask, i64Reg, int64_t(0), mask);
        AscendC::Reg::Adds(correctedSReg, fp32SReg, kUint64AsFloat, negMask);
        AscendC::Reg::Select(fp32SReg, correctedSReg, fp32SReg, negMask);
        AscendC::Reg::Mul(scaledReg, fp32XReg, fp32SReg, mask);
        AscendC::Reg::Cast<half, float, kCastH16FromF32RndSat>(f16Reg, scaledReg, mask);
        AscendC::Reg::Cast<YT, half, kCastS8FromH16RndSat>(yReg, f16Reg, mask);
        constexpr bool kApplyMaxZero = DO_RELU;
        if constexpr (kApplyMaxZero) {
            AscendC::Reg::Maxs(yReg, yReg, YT(0), mask);
        }
        AscendC::Reg::StoreAlign<YT, AscendC::Reg::StoreDist::DIST_PACK4_B32>(dstAddr + i * oneRepeatSize, yReg, mask);
    }
}
