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
 * \file block_scheduler_wqbmm_asw.h
 * \brief wqbmmv2 ASW 滑窗调度器：m/n 尾块拆分、4 行窗口 S 形扫描，tile 按"先 MN 后 batch"线性展开。
 *        tiling 数据经模板 Params 以 duck typing 传入（字段名与 wqbmmv2_tiling::WqbmmV2AswBasicTilingData 一致），
 *        本头不依赖具体算子头文件。
 */
#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "cmct/block/block_scheduler_policy.h"
#include "cmct/block/block_scheduler_utils.h"
#include "cmct/utils/common_utils.h"

namespace Cmct::Gemm::Block {

template <class ProblemShape_, class L1TileShape_, class L0TileShape_>
class WqbmmBlockSchedulerAswt {
public:
    int64_t mTileNum_{0};
    int64_t nTileNum_{0};
    int64_t blockIdx_{0};
    int64_t perCoreBlockNum_{0};
    int64_t blockNum_{0};
    int64_t batch_{0};
    int64_t k_{0};
    int64_t tailL1M_{0};
    int64_t tailL1N_{0};
    int64_t mTailCnt_{1};
    int64_t nTailCnt_{1};
    int64_t tailCnt_{1};
    int64_t tileNum_{1};
    int64_t mainWindow_{1};
    int64_t mainRow_{1};
    int64_t tailWindow_{1};
    int64_t mTileIdx_{1};
    int64_t nTileIdx_{1};
    int64_t lastTileIdx_{-1};
    int64_t nSplitOffset_{0};
    int64_t mSplitOffset_{0};
    int64_t blkK_{0};
    int64_t mL1_{0};
    int64_t nL1_{0};
    int64_t kL1_{0};
    int64_t baseM_{0};
    int64_t baseN_{0};
    int64_t baseK_{0};
    bool aL2CacheDisable_{false};
    bool bL2CacheDisable_{false};
    int64_t mL1NormCnt_{0};
    int64_t mL1TailSplitCnt_{1};
    int64_t mL1TailMain_{0};
    int64_t mL1TailLast_{0};
    int64_t nL1NormCnt_{0};
    int64_t nL1TailSplitCnt_{1};
    int64_t nL1TailMain_{0};
    int64_t nL1TailLast_{0};

    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockL1L0Shape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;

    // tiling 数据指针 + L2 cache 关闭标记（枚举到 bool 的映射由调用方完成，保持本头与算子解耦）
    template <class TilingData>
    struct Params {
        const TilingData* tilingData = nullptr;
        bool aL2CacheDisable{false};
        bool bL2CacheDisable{false};
    };

public:
    template <class TilingData>
    __aicore__ inline WqbmmBlockSchedulerAswt(const ProblemShape& shape, int64_t blockIdx, int64_t blockNum,
                                              const Params<TilingData>& params)
        : blockIdx_(blockIdx), blockNum_(blockNum)
    {
        const TilingData* tilingData = params.tilingData;
        aL2CacheDisable_ = params.aL2CacheDisable;
        bL2CacheDisable_ = params.bL2CacheDisable;
        k_ = shape.k;
        batch_ = Max(shape.b, 1L);
        mL1_ = tilingData->mL1;
        nL1_ = tilingData->nL1;
        kL1_ = tilingData->kL1;
        baseM_ = tilingData->baseM;
        baseN_ = tilingData->baseN;
        baseK_ = tilingData->baseK;
        mTileNum_ = CeilDiv(shape.m, mL1_);
        nTileNum_ = CeilDiv(shape.n, nL1_);
        perCoreBlockNum_ = CeilDiv(mTileNum_ * nTileNum_ * batch_, blockNum_);
        tileNum_ = mTileNum_ * nTileNum_;
        int64_t tailTileNum = tileNum_ % blockNum_;
        mL1TailSplitCnt_ = tilingData->mBaseTailSplitCnt;
        nL1TailSplitCnt_ = tilingData->nBaseTailSplitCnt;
        mL1NormCnt_ = mTileNum_ - mL1TailSplitCnt_;
        nL1NormCnt_ = nTileNum_ - nL1TailSplitCnt_;
        tailL1M_ = shape.m - mL1NormCnt_ * mL1_;
        tailL1N_ = shape.n - nL1NormCnt_ * nL1_;
        mL1TailMain_ = mL1TailSplitCnt_ == 1 ? tailL1M_ : tilingData->mTailMain;
        mL1TailLast_ = tailL1M_ - (mL1TailSplitCnt_ - 1) * mL1TailMain_;
        nL1TailMain_ = nL1TailSplitCnt_ == 1 ? tailL1N_ : tilingData->nTailMain;
        nL1TailLast_ = tailL1N_ - (nL1TailSplitCnt_ - 1) * nL1TailMain_;
        blkK_ = k_;
        if (batch_ == 1) {
            mTailCnt_ = tilingData->mTailCnt;
            nTailCnt_ = tilingData->nTailCnt;
            int64_t mTailSplit = CeilDiv(mL1TailLast_, mTailCnt_);
            int64_t nTailSplit = CeilDiv(nL1TailLast_, nTailCnt_);
            mTailCnt_ = CeilDiv(mL1TailLast_, mTailSplit);
            nTailCnt_ = CeilDiv(nL1TailLast_, nTailSplit);
            tailCnt_ = mTailCnt_ * nTailCnt_;
            tileNum_ += (tailCnt_ - 1) * tailTileNum;
        }
        mainWindow_ = WINDOW_LEN < mTileNum_ ? WINDOW_LEN : mTileNum_;
        mainRow_ = mTileNum_ / mainWindow_ - 1;
        tailWindow_ = mTileNum_ - mainRow_ * mainWindow_;
    }

    __aicore__ inline int64_t GetTileNum() { return tileNum_ * batch_; }

    __aicore__ inline bool GetAL2CacheDisable() { return aL2CacheDisable_; }

    __aicore__ inline bool GetBL2CacheDisable() { return bL2CacheDisable_; }

    __aicore__ inline AscendC::Shape<int64_t, int64_t, int64_t, int64_t> GetTileL1Shape()
    {
        return {mL1_, nL1_, kL1_, 1};
    }

    __aicore__ inline AscendC::Shape<int64_t, int64_t, int64_t, int64_t> GetTileL0Shape()
    {
        return {baseM_, baseN_, baseK_, 1};
    }

    __aicore__ inline int64_t GetBlockNum(int64_t blockNum)
    {
        if (tileNum_ * batch_ < blockNum) {
            return tileNum_ * batch_;
        }
        return blockNum;
    }

    // 返回 {blkM, blkN, blkK, batch, mL0, nL0}
    __aicore__ inline BlockL1L0Shape GetBlockShape(int64_t tileIdx, int64_t mOffset, int64_t nOffset)
    {
        UpdateMNTileIdx(tileIdx);
        int64_t blkM = mL1_;
        int64_t blkN = nL1_;
        if (nTileIdx_ >= nL1NormCnt_) {
            blkN = nTileIdx_ == (nTileNum_ - 1) ? nL1TailLast_ : nL1TailMain_;
        }
        if (mTileIdx_ >= mL1NormCnt_) {
            blkM = mTileIdx_ == (mTileNum_ - 1) ? mL1TailLast_ : mL1TailMain_;
        }
        int64_t mL0 = blkM;
        int64_t nL0 = blkN;
        if (tileIdx / blockNum_ != (perCoreBlockNum_ - 1) || tailCnt_ == 1) {
            // mL1, nL1, k, batch, mL0, nL0
            mL0 = Min(Min(baseM_, blkM), blkM - mOffset);
            nL0 = Min(Min(baseN_, blkN), blkN - nOffset);
            return {blkM, blkN, blkK_, batch_, mL0, nL0};
        }
        // 尾块按 mTailCnt/nTailCnt 再拆给多核
        int64_t splitBlkM = CeilDiv(blkM, mTailCnt_);
        int64_t splitBlkN = CeilDiv(blkN, nTailCnt_);
        int64_t mSplitIdx = (blockIdx_ % tailCnt_) % mTailCnt_;
        int64_t nSplitIdx = (blockIdx_ % tailCnt_) / mTailCnt_;
        mSplitOffset_ = mSplitIdx * splitBlkM;
        nSplitOffset_ = nSplitIdx * splitBlkN;
        if (mSplitOffset_ >= blkM || nSplitOffset_ >= blkN) {
            return {0, 0, blkK_, batch_, 0, 0};
        }
        splitBlkM = Min(blkM - mSplitOffset_, splitBlkM);
        splitBlkN = Min(blkN - nSplitOffset_, splitBlkN);
        mL0 = Min(Min(baseM_, splitBlkM), splitBlkM - mOffset);
        nL0 = Min(Min(baseN_, splitBlkN), splitBlkN - nOffset);
        return {splitBlkM, splitBlkN, blkK_, batch_, mL0, nL0};
    }

    // 返回 {mOffset, nOffset, 0(本路径不切k), batchIdx}
    __aicore__ inline BlockCoord GetBlockCoord(int64_t tileIdx)
    {
        UpdateMNTileIdx(tileIdx);
        int64_t batchIdx = 0;
        if (batch_ > 1) {
            batchIdx = tileIdx / tileNum_;
        }
        int64_t mOffset = mTileIdx_ * mL1_ + mSplitOffset_;
        int64_t nOffset = nTileIdx_ * nL1_ + nSplitOffset_;
        if (mTileIdx_ > mL1NormCnt_) {
            mOffset = mL1NormCnt_ * mL1_ + (mTileIdx_ - mL1NormCnt_) * mL1TailMain_ + mSplitOffset_;
        }
        if (nTileIdx_ > nL1NormCnt_) {
            nOffset = nL1NormCnt_ * nL1_ + (nTileIdx_ - nL1NormCnt_) * nL1TailMain_ + nSplitOffset_;
        }
        return {mOffset, nOffset, 0, batchIdx};
    }

private:
    // 4 行窗口 S 形扫描
    __aicore__ inline void UpdateMNTileIdx(int64_t tmpIdx)
    {
        if (lastTileIdx_ == tmpIdx) {
            return;
        }
        lastTileIdx_ = tmpIdx;

        int64_t tileIdx = tmpIdx % tileNum_;
        if (tileIdx / blockNum_ == (perCoreBlockNum_ - 1) && tailCnt_ > 1) {
            tileIdx = (perCoreBlockNum_ - 1) * blockNum_ + blockIdx_ / tailCnt_;
        }
        int64_t rowIdx = tileIdx / nTileNum_ / mainWindow_;
        if (rowIdx < mainRow_) {
            mTileIdx_ = rowIdx * mainWindow_ + tileIdx % mainWindow_;
            nTileIdx_ = (tileIdx / mainWindow_) % nTileNum_;
        } else {
            rowIdx = mainRow_;
            int64_t tailIndex = tileIdx - mainRow_ * mainWindow_ * nTileNum_;
            mTileIdx_ = mainRow_ * mainWindow_ + tailIndex % tailWindow_;
            nTileIdx_ = (tailIndex / tailWindow_) % nTileNum_;
        }
        if ((rowIdx & 1) != 0) { // 奇数窗口行需要反向扫描
            nTileIdx_ = nTileNum_ - 1 - nTileIdx_;
        }
    }
};

// Selector 偏特化：WqbmmAswtScheduler 标签 -> wqbmmv2 ASW 调度器本体
template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, WqbmmAswtScheduler, TransA_, TransB_> {
    using SchedulerOp = WqbmmBlockSchedulerAswt<ProblemShape_, L1TileShape_, L0TileShape_>;
};

} // namespace Cmct::Gemm::Block
