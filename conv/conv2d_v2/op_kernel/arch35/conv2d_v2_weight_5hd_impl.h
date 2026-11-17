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
 * \file conv2d_v2_weight_5hd_impl.h
 * \brief
 */

#ifndef CONV2D_V2_WEIGHT_5HD_IMPL_H
#define CONV2D_V2_WEIGHT_5HD_IMPL_H

#include "conv2d_v2_config.h"
#include "conv2d_v2_util.h"
#include "conv2d_v2_instr.h"

namespace Conv2dFunc {
using namespace AscendC;
using namespace conv;

constexpr uint16_t WEIGHT_5HD_SYNC_ID_BASE = 9;
constexpr uint16_t WEIGHT_5HD_BARRIER_ID = 8;
constexpr uint16_t WEIGHT_5HD_SYNC_MODE_0 = 0;
constexpr uint16_t WEIGHT_5HD_SYNC_MODE_2 = 2;
constexpr uint16_t WEIGHT_5HD_EVENT_ID_NUM = 2;
constexpr uint16_t WEIGHT_5HD_NUM_2 = 2;
constexpr uint64_t WEIGHT_5HD_ADDR_LIST_SIZE = 16;
constexpr uint16_t WEIGHT_5HD_DST_REP_STRIDE_8 = 8;
constexpr uint16_t WEIGHT_5HD_DST_REP_STRIDE_16 = 16;
constexpr uint16_t WEIGHT_5HD_DST_REP_STRIDE_32 = 32;

template <typename WeightT>
struct Weight5HDCfg {
    uint64_t cin = 0;
    uint64_t coutTotal = 0;
    uint64_t hw = 0;
    uint64_t kBL1 = 0;
    uint64_t singleCoreCi = 0;
    uint32_t bUbNStep = 0;
    uint32_t bUbKStep = 0;
    uint64_t aivNum = 0;
    uint64_t blockDim = 0;
    uint64_t isRealCore = 1;
};

template <typename WeightT>
class Weight5HDVecTools {
public:
    static constexpr uint32_t K0 = static_cast<uint32_t>(C0_SIZE) / static_cast<uint32_t>(sizeof(WeightT));
    static constexpr bool IS_FP32 = AscendC::IsSameType<WeightT, float>::value;
    static constexpr bool IS_HIF8 = AscendC::IsSameType<WeightT, hifloat8_t>::value;
    using Weight5HDT = typename Conditional<IS_HIF8, uint8_t, WeightT>::type;

    __aicore__ inline Weight5HDVecTools() {}

    static __aicore__ inline uint64_t CalcKB(const Weight5HDCfg<WeightT>& cfg)
    {
        const uint64_t totalK = AlignB(cfg.singleCoreCi, static_cast<uint64_t>(K0)) * cfg.hw;
        return CeilDiv(totalK, cfg.kBL1);
    }

    __aicore__ inline void Init(const Weight5HDCfg<WeightT>& cfg, TPipe& pipe, TBuf<TPosition::VECIN>& ndUbBuf,
                                TBuf<TPosition::VECOUT>& nzUbBuf)
    {
        cfg_ = cfg;
        kb_ = CalcKB(cfg);
        hw_ = cfg.hw;
        alignedHW_ = AlignB(cfg.hw, static_cast<uint64_t>(K0));
        segTileNum_ = CeilDiv(static_cast<uint64_t>(cfg.bUbKStep), static_cast<uint64_t>(K0));
        ndSlotSize_ = static_cast<uint64_t>(cfg.bUbKStep) * alignedHW_;
        nzTileElems_ = alignedHW_ * K0;
        nzSlotSize_ = segTileNum_ * nzTileElems_;

        aivIdx_ = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t coutTotal = cfg.coutTotal;
        singleN_ = CeilDiv(coutTotal, cfg.aivNum);
        nBegin_ = aivIdx_ * singleN_;
        nOwn_ = (nBegin_ >= coutTotal) ? 0 : (singleN_ < (coutTotal - nBegin_) ? singleN_ : (coutTotal - nBegin_));

        pipe.InitBuffer(ndUbBuf, static_cast<uint32_t>(cfg.bUbNStep * ndSlotSize_ * sizeof(WeightT)));
        pipe.InitBuffer(nzUbBuf, static_cast<uint32_t>(cfg.bUbNStep * nzSlotSize_ * sizeof(WeightT)));
        ndUbBuf_ = &ndUbBuf;
        nzUbBuf_ = &nzUbBuf;

        evV2M_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE2));
        evM2V_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
        evV2M3_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
        evM32M_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void BindGm(const GlobalTensor<WeightT>& srcGm, const GlobalTensor<WeightT>& dstGm)
    {
        srcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Weight5HDT*>(srcGm.GetPhyAddr(0)), srcGm.GetSize());
        dstGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Weight5HDT*>(dstGm.GetPhyAddr(0)), dstGm.GetSize());
    }

    __aicore__ inline void RunVec()
    {
        const bool isRealCore = cfg_.isRealCore != 0U;
        for (uint64_t c = 0; c < kb_; c++) {
            if (c >= WEIGHT_5HD_EVENT_ID_NUM && isRealCore) {
                CrossCoreWaitFlag<WEIGHT_5HD_SYNC_MODE_2>(WEIGHT_5HD_SYNC_ID_BASE + (c & 1));
            }
            if (nOwn_ > 0U) {
                ConvertChunk(c);
            }
            CrossCoreSetFlag<WEIGHT_5HD_SYNC_MODE_0, PIPE_MTE3>(WEIGHT_5HD_BARRIER_ID);
            CrossCoreWaitFlag<WEIGHT_5HD_SYNC_MODE_0, PIPE_MTE3>(WEIGHT_5HD_BARRIER_ID);
            if (isRealCore) {
                CrossCoreSetFlag<WEIGHT_5HD_SYNC_MODE_2, PIPE_MTE3>(WEIGHT_5HD_SYNC_ID_BASE + (c & 1));
            }
        }
    }

private:
    __aicore__ inline void ConvertChunk(uint64_t chunk)
    {
        const uint64_t cinPerChunk = cfg_.kBL1 / hw_;
        const uint64_t c1PerChunk = cinPerChunk / K0;
        const uint64_t alignCin = AlignB(cfg_.singleCoreCi, static_cast<uint64_t>(K0));
        const uint64_t chunkCin = cinPerChunk < (alignCin - chunk * cinPerChunk) ? cinPerChunk :
                                                                                   (alignCin - chunk * cinPerChunk);
        const uint64_t c1Base = chunk * c1PerChunk;
        const uint64_t ciLoops = CeilDiv(chunkCin, static_cast<uint64_t>(cfg_.bUbKStep));
        const uint64_t nLoops = CeilDiv(nOwn_, static_cast<uint64_t>(cfg_.bUbNStep));
        for (uint64_t nl = 0; nl < nLoops; nl++) {
            const uint64_t curN = (cfg_.bUbNStep < (nOwn_ - nl * cfg_.bUbNStep)) ? cfg_.bUbNStep :
                                                                                   (nOwn_ - nl * cfg_.bUbNStep);
            const uint64_t nBase = nBegin_ + nl * cfg_.bUbNStep;
            for (uint64_t ci = 0; ci < ciLoops; ci++) {
                const uint64_t curCin = (cfg_.bUbKStep < (chunkCin - ci * cfg_.bUbKStep)) ?
                                            cfg_.bUbKStep :
                                            (chunkCin - ci * cfg_.bUbKStep);
                const uint64_t kLocal = (c1Base + ci * (static_cast<uint64_t>(cfg_.bUbKStep) / K0)) * hw_ * K0;
                ConvertBatch(nBase, curN, kLocal, curCin);
            }
        }
    }

    __aicore__ inline void ConvertBatch(uint64_t nBase, uint64_t nCnt, uint64_t kLocal, uint64_t cinCnt)
    {
        LocalTensor<Weight5HDT> nd = ndUbBuf_->template Get<Weight5HDT>();
        LocalTensor<Weight5HDT> nz = nzUbBuf_->template Get<Weight5HDT>();
        uint64_t tileCnt = CeilDiv(cinCnt, static_cast<uint64_t>(K0));

        PipeBarrier<PIPE_V>();
        Duplicate(nd, static_cast<Weight5HDT>(0), static_cast<int32_t>(cfg_.bUbNStep * ndSlotSize_));
        SetFlag<HardEvent::V_MTE2>(evV2M_);
        WaitFlag<HardEvent::V_MTE2>(evV2M_);
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = static_cast<uint16_t>(cinCnt);
        copyInParams.blockLen = static_cast<uint32_t>(hw_ * sizeof(Weight5HDT));
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPadExtParams<Weight5HDT> padParams{false, 0, 0, 0};
        for (uint64_t i = 0; i < nCnt; i++) {
            uint64_t srcBase = (nBase + i) * cfg_.cin * hw_ + kLocal;
            DataCopyPad(nd[i * ndSlotSize_], srcGm_[srcBase], copyInParams, padParams);
        }

        SetFlag<HardEvent::MTE2_V>(evM2V_);
        WaitFlag<HardEvent::MTE2_V>(evM2V_);
        for (uint64_t i = 0; i < nCnt; i++) {
            TransTile(nd[i * ndSlotSize_], nz[i * nzSlotSize_], tileCnt);
        }

        SetFlag<HardEvent::V_MTE3>(evV2M3_);
        WaitFlag<HardEvent::V_MTE3>(evV2M3_);
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = static_cast<uint16_t>(tileCnt);
        copyOutParams.blockLen = static_cast<uint32_t>(hw_ * K0 * sizeof(Weight5HDT));
        copyOutParams.srcStride = (alignedHW_ - hw_) * K0 * sizeof(Weight5HDT) / C0_SIZE;
        copyOutParams.dstStride = 0;
        for (uint64_t i = 0; i < nCnt; i++) {
            const uint64_t dstBase = (nBase + i) * CeilDiv(cfg_.cin, static_cast<uint64_t>(K0)) * hw_ * K0 + kLocal;
            DataCopyPad(dstGm_[dstBase], nz[i * nzSlotSize_], copyOutParams);
        }
        SetFlag<HardEvent::MTE3_MTE2>(evM32M_);
        WaitFlag<HardEvent::MTE3_MTE2>(evM32M_);
    }

    __aicore__ inline void TransTile(LocalTensor<Weight5HDT> nd, LocalTensor<Weight5HDT> nz, uint64_t tileCnt)
    {
        TransDataTo5HDParams transParams;
        transParams.srcHighHalf = false;
        transParams.dstHighHalf = false;
        transParams.repeatTimes = static_cast<uint16_t>(alignedHW_ / K0);
        const bool singleRepeat = (transParams.repeatTimes == 1U);
        transParams.srcRepStride = singleRepeat ? 0U : 1U;

        uint64_t srcList[WEIGHT_5HD_ADDR_LIST_SIZE];
        uint64_t dstList[WEIGHT_5HD_ADDR_LIST_SIZE];
        if constexpr (IS_HIF8) {
            transParams.dstRepStride = singleRepeat ? 0U : WEIGHT_5HD_DST_REP_STRIDE_32;
            for (uint64_t c1 = 0; c1 < tileCnt; c1++) {
                const uint64_t dstTile = c1 * nzTileElems_;
                for (uint64_t chHalf = 0; chHalf < WEIGHT_5HD_NUM_2; chHalf++) {
                    for (uint64_t i = 0; i < WEIGHT_5HD_ADDR_LIST_SIZE; i++) {
                        srcList[i] = static_cast<uint64_t>(
                            nd[(c1 * K0 + chHalf * WEIGHT_5HD_ADDR_LIST_SIZE + i) * alignedHW_].GetPhyAddr());
                    }
                    for (uint64_t posHalf = 0; posHalf < WEIGHT_5HD_NUM_2; posHalf++) {
                        transParams.srcHighHalf = (posHalf == 1);
                        transParams.dstHighHalf = (chHalf == 1);
                        for (uint64_t i = 0; i < WEIGHT_5HD_ADDR_LIST_SIZE; i++) {
                            dstList[i] = static_cast<uint64_t>(
                                nz[dstTile + (posHalf * WEIGHT_5HD_ADDR_LIST_SIZE + i) * K0].GetPhyAddr());
                        }
                        TransDataTo5HD<Weight5HDT>(dstList, srcList, transParams);
                    }
                }
            }
        } else if constexpr (IS_FP32) {
            transParams.dstRepStride = singleRepeat ? 0U : WEIGHT_5HD_DST_REP_STRIDE_8;
            const uint64_t c1Pairs = CeilDiv(tileCnt, WEIGHT_5HD_NUM_2);
            for (uint64_t p = 0; p < c1Pairs; p++) {
                const uint64_t c1A = p * WEIGHT_5HD_NUM_2;
                const uint64_t c1B = c1A + 1;
                for (uint64_t j = 0; j < WEIGHT_5HD_ADDR_LIST_SIZE / WEIGHT_5HD_NUM_2; j++) {
                    dstList[WEIGHT_5HD_NUM_2 * j] = static_cast<uint64_t>(nz[c1A * nzTileElems_ + j * K0].GetPhyAddr());
                    dstList[WEIGHT_5HD_NUM_2 * j + 1] = static_cast<uint64_t>(
                        nz[c1B * nzTileElems_ + j * K0].GetPhyAddr());
                }
                for (uint64_t i = 0; i < WEIGHT_5HD_ADDR_LIST_SIZE; i++) {
                    srcList[i] = static_cast<uint64_t>(
                        nd[(p * WEIGHT_5HD_ADDR_LIST_SIZE + i) * alignedHW_].GetPhyAddr());
                }
                TransDataTo5HD<Weight5HDT>(dstList, srcList, transParams);
            }
        } else {
            transParams.dstRepStride = singleRepeat ? 0U : WEIGHT_5HD_DST_REP_STRIDE_16;
            for (uint64_t c1 = 0; c1 < tileCnt; c1++) {
                const uint64_t dstTile = c1 * nzTileElems_;
                for (uint64_t i = 0; i < WEIGHT_5HD_ADDR_LIST_SIZE; i++) {
                    srcList[i] = static_cast<uint64_t>(nd[(c1 * K0 + i) * alignedHW_].GetPhyAddr());
                    dstList[i] = static_cast<uint64_t>(nz[dstTile + i * K0].GetPhyAddr());
                }
                TransDataTo5HD<Weight5HDT>(dstList, srcList, transParams);
            }
        }
    }

    Weight5HDCfg<WeightT> cfg_;
    TBuf<TPosition::VECIN>* ndUbBuf_ = nullptr;
    TBuf<TPosition::VECOUT>* nzUbBuf_ = nullptr;
    GlobalTensor<Weight5HDT> srcGm_;
    GlobalTensor<Weight5HDT> dstGm_;

    uint64_t aivIdx_ = 0;
    uint64_t singleN_ = 0;
    uint64_t nBegin_ = 0;
    uint64_t nOwn_ = 0;
    uint64_t kb_ = 0;
    uint64_t hw_ = 0;
    uint64_t alignedHW_ = 0;
    uint64_t segTileNum_ = 0;
    uint64_t ndSlotSize_ = 0;
    uint64_t nzTileElems_ = 0;
    uint64_t nzSlotSize_ = 0;

    event_t evV2M_ = static_cast<event_t>(0);  // V_MTE2
    event_t evM2V_ = static_cast<event_t>(0);  // MTE2_V
    event_t evV2M3_ = static_cast<event_t>(0); // V_MTE3
    event_t evM32M_ = static_cast<event_t>(0); // MTE3_MTE2
};

template <class Intf>
class Weight5HDProcessTools {
public:
    __aicore__ inline Weight5HDProcessTools() {}

    __aicore__ inline void Weight5HDAicInit(Intf* self)
    {
        kb_ = MakeCfg(self).kb_;
        loadedChunkCnt_ = 0;
    }

    __aicore__ inline void BindGm(const GlobalTensor<typename Intf::WeightT>& srcGm,
                                  const GlobalTensor<typename Intf::WeightT>& dstGm)
    {
        core_.BindGm(srcGm, dstGm);
    }

    __aicore__ inline void Weight5HDVecInit(Intf* self)
    {
        core_.Init(MakeCfg(self), self->ctx.pipe, self->ctx.ndUbBuf, self->ctx.nzUbBuf);
    }

    __aicore__ inline bool Weight5HDVecImpl(Intf* self)
    {
        core_.RunVec();
        return false;
    }

    __aicore__ inline void BL1SyncLoad(Intf* self, uint64_t kBL1Iter, uint64_t nBL1Iter)
    {
        self->ctx.bl1 = self->ctx.queueBL1.template AllocTensor<typename Intf::WeightT>();
        if (kBL1Iter >= loadedChunkCnt_) {
            CrossCoreWaitFlag<WEIGHT_5HD_SYNC_MODE_2, PIPE_MTE2>(WEIGHT_5HD_SYNC_ID_BASE + (kBL1Iter & 1));
            self->ctx.loadBL1Ins.LoadBL1From5HD(kBL1Iter, nBL1Iter);
            if (kBL1Iter + WEIGHT_5HD_EVENT_ID_NUM < kb_) {
                CrossCoreSetFlag<WEIGHT_5HD_SYNC_MODE_2, PIPE_MTE2>(WEIGHT_5HD_SYNC_ID_BASE + (kBL1Iter & 1));
            }
            loadedChunkCnt_++;
        } else {
            self->ctx.loadBL1Ins.LoadBL1From5HD(kBL1Iter, nBL1Iter);
        }
        self->ctx.queueBL1.template EnQue(self->ctx.bl1);
    }

private:
    struct Weight5HDCfgWrapper : public Weight5HDCfg<typename Intf::WeightT> {
        uint64_t kb_ = 0;
    };

    __aicore__ inline Weight5HDCfgWrapper MakeCfg(const Intf* self)
    {
        Weight5HDCfgWrapper cfg;
        cfg.cin = self->ctx.convTilingData->cin;
        cfg.coutTotal = self->ctx.convTilingData->orgCo;
        cfg.hw = self->ctx.convTilingData->kernelHxkernelW;
        cfg.kBL1 = self->ctx.convTilingData->kBL1;
        cfg.singleCoreCi = self->ctx.convTilingData->singleCoreCi;
        cfg.bUbNStep = self->ctx.convTilingData->bUbNStep;
        cfg.bUbKStep = self->ctx.convTilingData->bUbKStep;
        cfg.aivNum = static_cast<uint64_t>(GetBlockNum());
        cfg.blockDim = static_cast<uint64_t>(self->ctx.convTilingData->batchDim * self->ctx.convTilingData->hoDim *
                                             self->ctx.convTilingData->nDim);
        cfg.isRealCore = static_cast<uint64_t>(GetBlockIdx() < cfg.blockDim);
        cfg.kb_ = Weight5HDVecTools<typename Intf::WeightT>::CalcKB(cfg);
        return cfg;
    }

    Weight5HDVecTools<typename Intf::WeightT> core_;
    uint64_t kb_ = 0;
    uint64_t loadedChunkCnt_ = 0;
};

template <typename WeightT, typename TilingData>
__aicore__ inline void RunWeight5HDVec(const TilingData& tiling, GM_ADDR filter, GM_ADDR workspace)
{
    TPipe pipe;
    TBuf<TPosition::VECIN> ndUbBuf;
    TBuf<TPosition::VECOUT> nzUbBuf;
    GlobalTensor<WeightT> srcGm;
    srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ WeightT*>(filter));
    GlobalTensor<WeightT> dstGm;
    dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ WeightT*>(workspace));

    Weight5HDCfg<WeightT> cfg;
    cfg.cin = tiling.cin;
    cfg.coutTotal = tiling.cout;
    cfg.hw = static_cast<uint64_t>(tiling.kh) * tiling.kw;
    cfg.kBL1 = tiling.kBL1;
    cfg.singleCoreCi = tiling.singleCoreCi;
    cfg.bUbNStep = tiling.bUbNStep;
    cfg.bUbKStep = tiling.bUbKStep;
    cfg.aivNum = static_cast<uint64_t>(GetBlockNum());
    cfg.blockDim = static_cast<uint64_t>(tiling.batchDim * tiling.nDim * tiling.hoDim);
    const uint64_t aivIdx = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t nxm = static_cast<uint64_t>(tiling.nDim) * tiling.hoDim;
    const uint64_t nIdx = (aivIdx % nxm) / static_cast<uint64_t>(tiling.hoDim);
    const uint64_t mIdx = (aivIdx % nxm) % static_cast<uint64_t>(tiling.hoDim);
    const bool isRealCore = (nIdx * static_cast<uint64_t>(tiling.singleCoreCo) < static_cast<uint64_t>(tiling.cout)) &&
                            (mIdx * static_cast<uint64_t>(tiling.singleCoreHo) <
                             static_cast<uint64_t>(tiling.hout) * tiling.wout);
    cfg.isRealCore = isRealCore ? 1UL : 0UL;

    Weight5HDVecTools<WeightT> tools;
    tools.Init(cfg, pipe, ndUbBuf, nzUbBuf);
    tools.BindGm(srcGm, dstGm);
    tools.RunVec();
}

} // namespace Conv2dFunc

#endif // CONV2D_V2_WEIGHT_5HD_IMPL_H
