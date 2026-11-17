/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_v2_weight_5hd_impl.h
 * \brief
 */

#ifndef CONV3D_V2_WEIGHT_5HD_IMPL_H
#define CONV3D_V2_WEIGHT_5HD_IMPL_H

#include "conv3d_v2_config.h"
#include "conv3d_v2_util.h"
#include "conv3d_v2_instr.h"

namespace Conv3dFunc {
using namespace ConvFunc;
using namespace conv;

constexpr uint16_t WEIGHT_5HD_SYNC_MODE = 2;
constexpr uint16_t WEIGHT_5HD_SYNC_ID_BASE = 9;
constexpr uint16_t WEIGHT_5HD_BARRIER_ID = 8;
constexpr uint64_t WEIGHT_5HD_TRANS_CHN = 16;
constexpr uint64_t WEIGHT_5HD_DB_BYTES = 32;

__aicore__ inline uint64_t CalcTailSize(uint64_t step, uint64_t total, uint64_t offset)
{
    if (offset >= total) {
        return 0;
    }
    return step < (total - offset) ? step : (total - offset);
}

template <class Intf>
class Weight5HDConvertTools {
public:
    __aicore__ inline Weight5HDConvertTools() {}

    __aicore__ inline void SetParams(Intf* self, uint64_t chunk)
    {
        self_ = self;
        const Ops::NN::Conv3dV2::Conv3DV2TilingDataV2* td = self_->ctx.convTilingData;
        const uint64_t hw = td->kh * td->kw;
        kdTotal_ = td->kd;
        const uint64_t k0 = Intf::k0;
        const uint64_t dtype = Intf::sizeOfWeight;
        const uint64_t alignHwByte = AlignB(hw * dtype, 32U);
        rowElems_ = alignHwByte / dtype;

        const uint64_t cinAlign = AlignB(static_cast<uint64_t>(td->cin), static_cast<uint64_t>(k0));
        const uint64_t rowsPerSlice = cinAlign * hw;
        if (static_cast<uint64_t>(td->kBL1) >= rowsPerSlice) {
            dkPerChunk_ = static_cast<uint64_t>(td->kBL1) / rowsPerSlice;
            dkStart_ = chunk * dkPerChunk_;
            dkChunk_ = CalcTailSize(dkPerChunk_, kdTotal_, dkStart_);
            cinStart_ = 0;
            cinSpan_ = cinAlign;
            cinUbChunks_ = CeilDiv(cinSpan_, cinPerUbChunk_);
        } else {
            dkStart_ = (chunk * static_cast<uint64_t>(td->kBL1)) / rowsPerSlice;
            cinStart_ = ((chunk * static_cast<uint64_t>(td->kBL1)) % rowsPerSlice) / hw;
            const uint64_t cinSpanFull = static_cast<uint64_t>(td->kBL1) / hw;
            cinSpan_ = (cinStart_ + cinSpanFull > cinAlign) ? (cinAlign - cinStart_) : cinSpanFull; // 尾 chunk 收窄
            dkPerChunk_ = 1;
            dkChunk_ = 1;
            cinUbChunks_ = CeilDiv(cinSpan_, cinPerUbChunk_);
        }
        dkUbChunks_ = CeilDiv(dkChunk_, dkPerUbChunk_);

        const uint64_t aivIdx = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t aivNum = static_cast<uint64_t>(GetBlockNum());
        const uint64_t cout = static_cast<uint64_t>(td->cout);
        const uint64_t nStep = CeilDiv(cout, aivNum);
        nStart_ = aivIdx * nStep;
        nCnt_ = (nStart_ >= cout) ? 0 : (nStep < (cout - nStart_) ? nStep : (cout - nStart_));
        bUbNStep_ = td->bUbNStep;

        inTensor_ = self_->ctx.ndUbBuf.template Get<typename Intf::WeightT>();
        outTensor_ = self_->ctx.nzUbBuf.template Get<typename Intf::WeightT>();

        copyInParams_.blockLen = static_cast<uint32_t>(hw * dtype);
        copyInParams_.srcStride = static_cast<int64_t>((kdTotal_ - 1U) * hw * dtype);
        copyInParams_.dstStride = 0;
        padParams_.isPad = true;
        padParams_.rightPadding = static_cast<uint8_t>((alignHwByte - hw * dtype) / dtype);
        padParams_.paddingValue = 0;

        copyOutParams_.blockLen = static_cast<uint32_t>(hw * k0 * dtype);
        copyOutParams_.srcStride = static_cast<int64_t>((rowElems_ - hw) * k0 * dtype / WEIGHT_5HD_DB_BYTES);
        copyOutParams_.dstStride = 0;

        evVToMte2_ = static_cast<event_t>(self_->ctx.pipe.FetchEventID(HardEvent::V_MTE2));
        evMte2ToV_ = static_cast<event_t>(self_->ctx.pipe.FetchEventID(HardEvent::MTE2_V));
        evVToMte3_ = static_cast<event_t>(self_->ctx.pipe.FetchEventID(HardEvent::V_MTE3));
    }

    __aicore__ inline void CopyIn(uint64_t nBase, uint64_t curCoutChunk, uint64_t cinBase, uint64_t cinRead,
                                  uint64_t cinCoverAligned, uint64_t curDkChunk, uint64_t kdBase)
    {
        const Ops::NN::Conv3dV2::Conv3DV2TilingDataV2* td = self_->ctx.convTilingData;
        const uint64_t hw = td->kh * td->kw;
        copyInParams_.blockCount = static_cast<uint16_t>(cinRead);
        for (uint64_t nLocal = 0; nLocal < curCoutChunk; ++nLocal) {
            const uint64_t n = nBase + nLocal;
            for (uint64_t dkLocal = 0; dkLocal < curDkChunk; ++dkLocal) {
                const uint64_t dk = kdBase + dkLocal;
                DataCopyPad(inTensor_[((nLocal * curDkChunk + dkLocal) * cinCoverAligned) * rowElems_],
                            self_->ctx.bgm[((n * td->cin + cinBase) * kdTotal_ + dk) * hw], copyInParams_, padParams_);
            }
        }
    }

    __aicore__ inline void ZeroUbChunk(uint64_t curCoutChunk, uint64_t cinCoverAligned, uint64_t curDkChunk)
    {
        const uint64_t elems = curCoutChunk * curDkChunk * cinCoverAligned * rowElems_;
        if constexpr (Intf::sizeOfWeight == 1) {
            LocalTensor<uint8_t> inBytes = self_->ctx.ndUbBuf.template Get<uint8_t>();
            Duplicate<uint8_t>(inBytes, 0, elems);
        } else {
            Duplicate<typename Intf::WeightT>(inTensor_, 0, elems);
        }
    }

    __aicore__ inline void TransTo5HD(uint64_t curCoutChunk, uint64_t cinCoverAligned, uint64_t curDkChunk)
    {
        const uint64_t k0 = Intf::k0;
        const uint64_t repeatTimes = rowElems_ / (32U / Intf::sizeOfWeight); // m = 每行 DB 数
        const uint64_t transCntPerNd = cinCoverAligned / WEIGHT_5HD_TRANS_CHN;
        const bool singleRepeat = (repeatTimes == 1U);
        const uint32_t srcStride = singleRepeat ? 0U : 1U;
        TransDataTo5HDParams transParams;
        transParams.repeatTimes = static_cast<uint8_t>(repeatTimes);
        transParams.srcRepStride = srcStride;
        transParams.srcHighHalf = false;
        transParams.dstHighHalf = false;
        uint64_t srcList[WEIGHT_5HD_TRANS_CHN];
        uint64_t dstList[WEIGHT_5HD_TRANS_CHN];
        for (uint64_t nLocal = 0; nLocal < curCoutChunk; ++nLocal) {
            for (uint64_t dkLocal = 0; dkLocal < curDkChunk; ++dkLocal) {
                const uint64_t ndBase = (nLocal * curDkChunk + dkLocal) * cinCoverAligned * rowElems_;
                for (uint64_t g = 0; g < transCntPerNd; ++g) {
                    const uint64_t srcBase = ndBase +
                                             g * WEIGHT_5HD_TRANS_CHN * rowElems_; // 本组 16 通道行首（in-half）
                    for (uint64_t i = 0; i < WEIGHT_5HD_TRANS_CHN; ++i) {
                        srcList[i] = static_cast<uint64_t>(inTensor_[srcBase + i * rowElems_].GetPhyAddr());
                    }
                    if constexpr (Intf::sizeOfWeight == 4) {
                        transParams.dstRepStride = singleRepeat ? 0 : 8;
                        const uint64_t c1aBase = ndBase + g * 2U * rowElems_ * k0;
                        const uint64_t c1bBase = c1aBase + rowElems_ * k0;
                        for (uint64_t i = 0; i < 8U; ++i) {
                            dstList[i * 2] = static_cast<uint64_t>(outTensor_[c1aBase + i * k0].GetPhyAddr());
                            dstList[i * 2 + 1] = static_cast<uint64_t>(outTensor_[c1bBase + i * k0].GetPhyAddr());
                        }
                        TransDataTo5HD<typename Intf::WeightT>(dstList, srcList, transParams);
                    } else if constexpr (Intf::sizeOfWeight == 1) {
                        transParams.dstRepStride = singleRepeat ? 0 : 32;
                        transParams.dstHighHalf = ((g & 1U) == 1U);
                        const uint64_t c1Base = ndBase + (g / 2U) * rowElems_ * k0;
                        for (uint64_t posHalf = 0; posHalf < 2U; ++posHalf) {
                            transParams.srcHighHalf = (posHalf == 1U);
                            for (uint64_t i = 0; i < WEIGHT_5HD_TRANS_CHN; ++i) {
                                dstList[i] = static_cast<uint64_t>(
                                    outTensor_[c1Base + (posHalf * WEIGHT_5HD_TRANS_CHN + i) * k0].GetPhyAddr());
                            }
                            TransDataTo5HD<uint8_t>(dstList, srcList, transParams);
                        }
                    } else {
                        transParams.dstRepStride = singleRepeat ? 0 : 16;
                        const uint64_t c1Base = ndBase + g * rowElems_ * k0;
                        for (uint64_t i = 0; i < WEIGHT_5HD_TRANS_CHN; ++i) {
                            dstList[i] = static_cast<uint64_t>(outTensor_[c1Base + i * k0].GetPhyAddr());
                        }
                        TransDataTo5HD<typename Intf::WeightT>(dstList, srcList, transParams);
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyOut(uint64_t nBase, uint64_t curCoutChunk, uint64_t cinBase, uint64_t curC1Cnt,
                                   uint64_t curDkChunk, uint64_t kdBase, uint64_t cinCoverAligned)
    {
        const Ops::NN::Conv3dV2::Conv3DV2TilingDataV2* td = self_->ctx.convTilingData;
        const uint64_t hw = td->kh * td->kw;
        const uint64_t k0 = Intf::k0;
        const uint64_t c1Total = CeilDiv(static_cast<uint64_t>(td->cin), k0); // 尾 c1 的补零通道在 GM 无位置
        copyOutParams_.blockCount = static_cast<uint16_t>(curC1Cnt);
        for (uint64_t nLocal = 0; nLocal < curCoutChunk; ++nLocal) {
            const uint64_t n = nBase + nLocal;
            for (uint64_t dkLocal = 0; dkLocal < curDkChunk; ++dkLocal) {
                const uint64_t dk = kdBase + dkLocal;
                DataCopyPad(self_->ctx.weight5HDGM[((n * kdTotal_ + dk) * c1Total + (cinBase / k0)) * hw * k0],
                            outTensor_[((nLocal * curDkChunk + dkLocal) * cinCoverAligned) * rowElems_],
                            copyOutParams_);
            }
        }
    }

    uint64_t kdTotal_ = 0;
    uint64_t dkPerChunk_ = 0; // 一个 chunk 的深度片数
    uint64_t dkStart_ = 0;    // 本 chunk 深度起点
    uint64_t dkChunk_ = 0;
    uint64_t transdata5hdCinAlign_ = 0;
    uint64_t dkPerUbChunk_ = 0;
    uint64_t cinPerUbChunk_ = 0; // 一个 UB chunk 的 cin 通道数（同上）
    uint64_t cinUbChunks_ = 0;
    uint64_t dkUbChunks_ = 0; // chunk 内 dk 方向 UB chunk 数
    uint64_t cinStart_ = 0;
    uint64_t cinSpan_ = 0;
    uint64_t nStart_ = 0; // 本 AIV 的 cout 窗口起点
    uint64_t nCnt_ = 0;   // 本 AIV 的 cout 窗口大小
    uint64_t bUbNStep_ = 0;
    event_t evVToMte2_ = static_cast<event_t>(0);
    event_t evMte2ToV_ = static_cast<event_t>(0);
    event_t evVToMte3_ = static_cast<event_t>(0);

private:
    Intf* self_ = nullptr;
    uint64_t rowElems_ = 0;
    LocalTensor<typename Intf::WeightT> inTensor_;
    LocalTensor<typename Intf::WeightT> outTensor_;
    DataCopyExtParams copyInParams_;
    DataCopyPadExtParams<typename Intf::WeightT> padParams_;
    DataCopyExtParams copyOutParams_;
};

template <class Intf>
class Weight5HDProcessTools {
public:
    __aicore__ inline Weight5HDProcessTools() {}

    __aicore__ inline void Weight5HDVecInit(Intf* self)
    {
        self->ctx.singleCoreCi = self->ctx.convTilingData->singleCoreCi;
        self->ctx.singleCoreCo = self->ctx.convTilingData->singleCoreCo;

        Weight5HDInitKValue(self);

        const Ops::NN::Conv3dV2::Conv3DV2TilingDataV2* td = self->ctx.convTilingData;
        const uint64_t hw = td->kh * td->kw;
        const uint64_t alignHwByte = AlignB(hw * Intf::sizeOfWeight, 32U);
        const uint64_t rowElems = alignHwByte / Intf::sizeOfWeight;
        convertTools_.transdata5hdCinAlign_ = AlignB(static_cast<uint64_t>(td->cin),
                                                     Intf::k0 > WEIGHT_5HD_TRANS_CHN ? Intf::k0 : WEIGHT_5HD_TRANS_CHN);
        convertTools_.dkPerUbChunk_ = CeilDiv(static_cast<uint64_t>(td->bUbKStep), convertTools_.transdata5hdCinAlign_);
        convertTools_.cinPerUbChunk_ = static_cast<uint64_t>(td->bUbKStep) / convertTools_.dkPerUbChunk_;
        const uint64_t ubElems = td->bUbNStep * convertTools_.cinPerUbChunk_ * convertTools_.dkPerUbChunk_ * rowElems;
        self->ctx.pipe.InitBuffer(self->ctx.ndUbBuf, ubElems * Intf::sizeOfWeight);
        self->ctx.pipe.InitBuffer(self->ctx.nzUbBuf, ubElems * Intf::sizeOfWeight);
    }

    __aicore__ inline void BindGm(const GlobalTensor<typename Intf::WeightT>& srcGm,
                                  const GlobalTensor<typename Intf::WeightT>& dstGm)
    {
        (void)srcGm;
        (void)dstGm;
        return;
    }

    __aicore__ inline bool Weight5HDVecImpl(Intf* self)
    {
        const Ops::NN::Conv3dV2::Conv3DV2TilingDataV2* td = self->ctx.convTilingData;
        const uint64_t numBlocks = static_cast<uint64_t>(td->batchDim) * td->doDim * td->hoDim * td->nDim *
                                   td->groupDim;
        uint64_t aivBlock = static_cast<uint64_t>(GetBlockIdx());
        const bool isRealCore = aivBlock < numBlocks;
        for (uint64_t chunk = 0; chunk < self->ctx.weight5hdChunkNum; ++chunk) {
            if (chunk >= 2U && isRealCore) {
                CrossCoreWaitFlag<WEIGHT_5HD_SYNC_MODE>(WEIGHT_5HD_SYNC_ID_BASE + (chunk & 1U));
            }
            Weight5HDConvertChunk(self, chunk);
            CrossCoreSetFlag<0, PIPE_MTE3>(WEIGHT_5HD_BARRIER_ID);
            CrossCoreWaitFlag<0>(WEIGHT_5HD_BARRIER_ID);
            if (isRealCore) {
                CrossCoreSetFlag<WEIGHT_5HD_SYNC_MODE, PIPE_MTE3>(WEIGHT_5HD_SYNC_ID_BASE + (chunk & 1U));
            }
        }
        return false;
    }

    __aicore__ inline void BL1WaitReady(Intf* self)
    {
        if (self->ctx.weight5hdLoadCount < self->ctx.weight5hdChunkNum) {
            CrossCoreWaitFlag<WEIGHT_5HD_SYNC_MODE, PIPE_MTE2>(WEIGHT_5HD_SYNC_ID_BASE +
                                                               (self->ctx.weight5hdLoadCount & 1U));
        }
    }

    __aicore__ inline void BL1ReadDone(Intf* self)
    {
        if (self->ctx.weight5hdLoadCount + 2U < self->ctx.weight5hdChunkNum) {
            CrossCoreSetFlag<WEIGHT_5HD_SYNC_MODE, PIPE_MTE2>(WEIGHT_5HD_SYNC_ID_BASE +
                                                              (self->ctx.weight5hdLoadCount & 1U));
        }
        self->ctx.weight5hdLoadCount++;
    }

    __aicore__ inline void BL1SyncLoad(Intf* self, uint64_t kBL1Iter, uint64_t nBL1Iter)
    {
        self->ctx.bl1 = self->ctx.queueBL1.template AllocTensor<typename Intf::WeightT>();
        BL1WaitReady(self);
        self->ctx.loadBL1Ins.LoadBL1From5HD(kBL1Iter, nBL1Iter);
        self->ctx.queueBL1.EnQue(self->ctx.bl1);
        BL1ReadDone(self);
    }

    __aicore__ inline void Weight5HDInitKValue(Intf* self)
    {
        const Ops::NN::Conv3dV2::Conv3DV2TilingDataV2* td = self->ctx.convTilingData;
        const uint64_t cinAlign = AlignB(static_cast<uint64_t>(td->cin), Intf::k0);
        const uint64_t totalK = cinAlign * td->kd * td->kh * td->kw;
        self->ctx.weight5hdChunkNum = CeilDiv(totalK, static_cast<uint64_t>(td->kBL1));
    }

private:
    Weight5HDConvertTools<Intf> convertTools_;

    __aicore__ inline void Weight5HDConvertChunk(Intf* self, uint64_t chunk)
    {
        convertTools_.SetParams(self, chunk);
        if (convertTools_.nCnt_ == 0) {
            return;
        }
        const uint64_t k0 = Intf::k0;
        const uint64_t cin = self->ctx.convTilingData->cin;

        for (uint64_t nLocal = 0; nLocal < convertTools_.nCnt_; nLocal += convertTools_.bUbNStep_) {
            const uint64_t nBase = convertTools_.nStart_ + nLocal;
            const uint64_t curCoutChunk = CalcTailSize(convertTools_.bUbNStep_, convertTools_.nCnt_, nLocal);

            for (uint64_t kdLocal = 0; kdLocal < convertTools_.dkUbChunks_; ++kdLocal) {
                const uint64_t kdBase = convertTools_.dkStart_ + kdLocal * convertTools_.dkPerUbChunk_;
                const uint64_t curDkChunk = CalcTailSize(convertTools_.dkPerUbChunk_, convertTools_.dkChunk_,
                                                         kdLocal * convertTools_.dkPerUbChunk_);

                for (uint64_t cinLocal = 0; cinLocal < convertTools_.cinUbChunks_; ++cinLocal) {
                    const uint64_t cinBase = convertTools_.cinStart_ + cinLocal * convertTools_.cinPerUbChunk_;
                    const uint64_t cinCover = CalcTailSize(convertTools_.cinPerUbChunk_, convertTools_.cinSpan_,
                                                           cinLocal * convertTools_.cinPerUbChunk_);
                    const uint64_t cinRead = CalcTailSize(convertTools_.cinPerUbChunk_, cin, cinBase);
                    const uint64_t cinCoverAligned = (cinCover + 15U) / 16U * 16U;
                    const uint64_t curC1Cnt = cinCover / k0; // 真实 c1 数（cinCover 是 k0 倍数）

                    convertTools_.ZeroUbChunk(curCoutChunk, cinCoverAligned, curDkChunk);
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(convertTools_.evVToMte2_);
                    WaitFlag<HardEvent::V_MTE2>(convertTools_.evVToMte2_);
                    convertTools_.CopyIn(nBase, curCoutChunk, cinBase, cinRead, cinCoverAligned, curDkChunk, kdBase);

                    SetFlag<HardEvent::MTE2_V>(convertTools_.evMte2ToV_);
                    WaitFlag<HardEvent::MTE2_V>(convertTools_.evMte2ToV_);
                    convertTools_.TransTo5HD(curCoutChunk, cinCoverAligned, curDkChunk);

                    SetFlag<HardEvent::V_MTE3>(convertTools_.evVToMte3_);
                    WaitFlag<HardEvent::V_MTE3>(convertTools_.evVToMte3_);
                    convertTools_.CopyOut(nBase, curCoutChunk, cinBase, curC1Cnt, curDkChunk, kdBase, cinCoverAligned);
                }
            }
        }
    }
};

} // namespace Conv3dFunc

#endif
