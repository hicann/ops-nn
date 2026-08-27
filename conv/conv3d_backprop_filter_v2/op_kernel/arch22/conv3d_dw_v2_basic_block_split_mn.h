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
 * \file conv3d_dw_v2_basic_block_split_mn.h
 * \brief
 */

#ifndef CONV3D_BACKPROP_FILTER_BASIC_BLOCK_SPLIT_MN_H
#define CONV3D_BACKPROP_FILTER_BASIC_BLOCK_SPLIT_MN_H

#include "conv3d_dw_v2_basic_block.h"

namespace AscendC {
template <typename xType, int xFormat, typename dedyType, int dedyFormat, typename yType, int yFormat>
class Conv3dDwBasicBlockSplitMN : public Conv3dDwBasicBlock<xType, xFormat, dedyType, dedyFormat, yType, yFormat> {
public:
    __aicore__ inline Conv3dDwBasicBlockSplitMN(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dedy, GM_ADDR y, GM_ADDR workSpace,
                                const Conv3DBackpropFilterV2TilingData* tilingData)
    {
        this->InitCommonTilingData(tilingData);
        InitSplitTilingData(tilingData);
        // init global buffer
        this->xGm_.SetGlobalBuffer((__gm__ xType*)x);
        this->dedyGm_.SetGlobalBuffer((__gm__ dedyType*)dedy);
        this->yGm_.SetGlobalBuffer((__gm__ yType*)y);
        this->dw_.Init(&(tilingData->dwTiling));
        this->workspaceXGm_.SetGlobalBuffer((__gm__ xType*)workSpace);
    }

    __aicore__ inline void Process()
    {
        if (block_idx >= this->usedCoreNum_) {
            return;
        }
        if ASCEND_IS_AIV {
            if constexpr (xFormat != FORMAT_NCDHW) {
                return;
            }
            this->InitTransdataBuffer();
        }
        CalBasicBlock();
        this->dw_.End();
    }

protected:
    static constexpr uint8_t SYNC_MODE2 = 2;

    __aicore__ inline void InitSplitTilingData(const Conv3DBackpropFilterV2TilingData* tilingData)
    {
        this->singleShapeM_ = tilingData->basicBlockTiling.singleCoreM;
        this->singleShapeK_ = this->k_;
        this->singleShapeN_ = tilingData->basicBlockTiling.singleCoreN;
    }

    __aicore__ inline void CalBasicBlock()
    {
        uint32_t mCnt = DivCeil(this->m_, this->singleShapeM_);
        uint32_t mCoreTail = this->m_ - (mCnt - 1) * this->singleShapeM_;

        uint64_t nCnt = DivCeil(this->n_, this->singleShapeN_);
        uint64_t nCoreTail = this->n_ - (nCnt - 1) * this->singleShapeN_;

        // 记录基本块的位置
        uint64_t batchDout = static_cast<uint64_t>(this->batch_) * this->dout_;
        uint64_t mnCnt = mCnt * nCnt;
        uint64_t totalCnt = batchDout * mnCnt;
        uint64_t calRound = totalCnt / this->usedCoreNum_;
        uint64_t tailCnt = totalCnt - calRound * this->usedCoreNum_;
        uint64_t basicBlockIdx = 0;

        // 拖尾的部分依次分配到前面的核计算，这些核会多算一轮
        if (block_idx < tailCnt) {
            basicBlockIdx = block_idx * calRound + block_idx;
            ++calRound;
        } else {
            basicBlockIdx = block_idx * calRound + tailCnt;
        }

        // 1:M*K 行优先绑核，2:M*K 列优先绑核
        uint64_t batchDoutIndex = 0;
        uint64_t batchDoutNcnt = batchDout * nCnt;
        this->kCoreIndx_ = 0; // 默认不分核
        uint64_t syncTimes = 0;
        for (uint64_t j = 0; j < calRound; ++j) {
            if (this->coreBindOrder_ == ROW_FIRST) {
                // 行优先, NDC1HWC0的行方向是C0即N方向
                this->mCoreIndx_ = basicBlockIdx / batchDoutNcnt;
                batchDoutIndex = (basicBlockIdx - this->mCoreIndx_ * batchDoutNcnt) / nCnt;
                this->nCoreIndx_ = basicBlockIdx - this->mCoreIndx_ * batchDoutNcnt - batchDoutIndex * nCnt;
            } else {
                // 列优先, NDC1HWC0的列方向是Cout方向
                uint64_t batchDoutNIndex = basicBlockIdx / mCnt;
                this->mCoreIndx_ = basicBlockIdx - batchDoutNIndex * mCnt;
                this->nCoreIndx_ = batchDoutNIndex % nCnt;
                batchDoutIndex = basicBlockIdx / mnCnt;
            }
            uint64_t batchIdx = batchDoutIndex / this->dout_;
            uint64_t doutIdx = batchDoutIndex - batchIdx * this->dout_;

            basicBlockIdx++;

            // 不可用totalCnt - 1作为尾块, totalCnt包含batch*dout
            uint64_t mCoreUse = (this->mCoreIndx_ == (mCnt - 1)) ? mCoreTail : this->singleShapeM_;
            uint64_t nCoreUse = (this->nCoreIndx_ == (nCnt - 1)) ? nCoreTail : this->singleShapeN_;

            this->CalcOffset(batchIdx, doutIdx, 0, 0, true);
            this->ReCalDkCinSingleCoreShape(batchIdx, doutIdx, 0, 0);
            if (this->singleShapeNInCurrentHo_ == 0 || this->singleShapeMInCurrentHo_ == 0) {
                continue;
            }
            nCoreUse = nCoreUse > this->singleShapeNInCurrentHo_ ? this->singleShapeNInCurrentHo_ : nCoreUse;
            this->dw_.SetOutBackprop(this->dedyGm_[this->offsetA_]);
            this->hoStartIdx_ = this->kCoreIndx_ * this->singleCoreHo_;
            this->dw_.SetSingleShape(mCoreUse, nCoreUse, this->singleShapeK_, this->hoStartIdx_);
            this->dw_.SetFmap(this->xGm_[this->offsetB_]);
#if __CCE_AICORE__ == 220
            if constexpr (xFormat == FORMAT_NCDHW) { // Transdata Merge
                uint64_t wsOffset = block_idx * this->DOUBLE_BUFFER * this->gmPongOffset +
                                    (this->gmPingPongEventId_ ? this->gmPingOffset : this->gmPongOffset);
                this->dw_.SetFmap(this->workspaceXGm_[wsOffset]);
                if ASCEND_IS_AIV {
                    if (syncTimes > 1) {
                        CrossCoreWaitFlag(this->SYNC_AIC_AIV_FLAG + this->gmPingPongEventId_);
                    }
                    this->TransDataTo6HD(batchIdx, doutIdx);
                    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(this->SYNC_AIV_AIC_FLAG + this->gmPingPongEventId_);
                    this->gmPingPongEventId_ &= 1;
                    this->gmPingPongEventId_ ^= 1;
                    syncTimes += 1;
                }
                if ASCEND_IS_AIC {
                    if constexpr (xFormat == FORMAT_NCDHW) { // Transdata Merge
                        CrossCoreWaitFlag(this->SYNC_AIV_AIC_FLAG + this->gmPingPongEventId_);
                        this->dw_.IterateAll(this->yGm_[this->offsetC_], 1);
                        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(this->SYNC_AIC_AIV_FLAG + this->gmPingPongEventId_);
                        this->gmPingPongEventId_ &= 1;
                        this->gmPingPongEventId_ ^= 1;
                    }
                }
            } else {
                this->dw_.IterateAll(this->yGm_[this->offsetC_], 1);
            }
#else
            this->dw_.IterateAll(this->yGm_[this->offsetC_], 1);
#endif
        }
    }
};
} // namespace AscendC

#endif
