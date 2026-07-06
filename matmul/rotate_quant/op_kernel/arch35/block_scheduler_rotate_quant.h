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
 * \file block_scheduler_rotate_quant.h
 * \brief
 */

#ifndef CMCT_BLOCK_SCHEDULER_ROTATE_QUANT_BUILTIN_H
#define CMCT_BLOCK_SCHEDULER_ROTATE_QUANT_BUILTIN_H

#include "cmct/block/block_scheduler_policy.h"
#include "cmct/block/block_scheduler_utils.h"
#include "cmct/utils/common_utils.h"
#include "rotate_quant_tiling_data.h"
namespace Cmct {
namespace Gemm {
namespace Block {
template <class ProblemShape_, class L1TileShape_, class L0TileShape_, int64_t FullLoadMode_ = 0>
class BlockSchedulerRotateQuantBuiltIn {
public:
    uint32_t k_{0};
    uint32_t m_{0};
    uint32_t n_{0};
    uint32_t b_{0};
    uint32_t mL1_{0};
    uint32_t kL1_{0};
    uint32_t nL1_{0};
    uint32_t baseM_{0};
    uint32_t baseN_{0};
    uint32_t baseK_{0};

    uint32_t nKRatio_{0};
    uint32_t tailML1_{0};

    uint32_t blockNum_{1};
    uint32_t mNum_{0};
    uint32_t totalM_{0};

    float dstTypeMax_{0.0};
    float invDstTypeMax_{0.0};
    int32_t axis_{0};
    int32_t roundMode_{0};
    int32_t scaleAlg_{0};
    uint8_t trans_{0};

    uint32_t loopM_{0};
    uint32_t tailM_{0};

    static constexpr uint64_t BLOCK_SIZE_16 = 16UL;
    static constexpr uint64_t BLOCK_SIZE_32 = 32UL;
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockL1L0Shape = Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;
    struct Params {
        const RotateQuantAptOpt::RotateQuantAptTilingData* tilingData;
    };

public:
    __aicore__ inline BlockSchedulerRotateQuantBuiltIn(const ProblemShape& shape, int64_t blockNum,
                                                       const Params& params)
    {
        m_ = shape.m;
        n_ = shape.n; // x :(m_,n_)
        k_ = shape.k; // rot: (k_,k_)
        b_ = shape.b;
        mL1_ = params.tilingData->mL1; // m_line * nKRatio
        kL1_ = params.tilingData->kL1; // K
        nL1_ = params.tilingData->nL1; // K
        baseM_ = params.tilingData->baseM;
        baseN_ = params.tilingData->baseN;
        baseK_ = params.tilingData->baseK;
        nKRatio_ = params.tilingData->nKRatio;

        dstTypeMax_ = params.tilingData->dstTypeMax;
        invDstTypeMax_ = params.tilingData->invDstTypeMax;
        axis_ = params.tilingData->axis;
        roundMode_ = params.tilingData->roundMode;
        scaleAlg_ = params.tilingData->scaleAlg;
        trans_ = params.tilingData->trans;

        blockNum_ = (blockNum > 0) ? static_cast<uint32_t>(blockNum) : 1;
        if (b_ > 1) {
            mNum_ = (m_ + mL1_ - 1) / mL1_;
            totalM_ = mNum_ * b_;
            loopM_ = (totalM_ + blockNum_ - 1) / blockNum_;
            tailM_ = totalM_ % blockNum_;
            return;
        }
        tailML1_ = params.tilingData->tailML1;
        uint32_t totalDataM = m_ * nKRatio_;
        totalM_ = (totalDataM + mL1_ - 1) / mL1_;
    }

    __aicore__ inline int64_t GetTileNum() { return totalM_; }

    __aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetTileL1Shape() { return {mL1_, nL1_, kL1_, 1}; }

    __aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetTileL0Shape() { return {baseM_, baseN_, baseK_, 1}; }

    __aicore__ inline int64_t GetAxis() { return axis_; }

    __aicore__ inline int64_t GetRoundMode() { return roundMode_; }

    __aicore__ inline int64_t GetScaleAlg() { return scaleAlg_; }

    __aicore__ inline float GetDstTypeMax() { return dstTypeMax_; }

    __aicore__ inline float GetInvDstTypeMax() { return invDstTypeMax_; }

    __aicore__ inline int64_t GetBlockNum(int64_t blockNum)
    {
        int64_t tilingBlockNum = 0;
        if (totalM_ < blockNum) {
            tilingBlockNum = totalM_;
        } else {
            tilingBlockNum = blockNum;
        }
        return tilingBlockNum;
    }

    __aicore__ inline BlockL1L0Shape GetBlockShape(int64_t tileIdx)
    {
        int64_t blkN = nL1_;
        int64_t blkK = kL1_;
        int64_t blkM = mL1_;

        if (b_ > 1) {
            int64_t mIdx = tileIdx / b_;
            if (mIdx == mNum_ - 1) {
                blkM = m_ - (mNum_ - 1) * mL1_;
            }
        } else if (tileIdx == static_cast<int64_t>(totalM_) - 1) {
            blkM = tailML1_;
        }

        return {blkM, blkN, blkK, mL1_, baseM_, baseN_};
    }

    __aicore__ inline int64_t GetOffset(int64_t tileIdx)
    {
        if (b_ > 1) {
            int64_t mIdx = tileIdx / b_;
            int64_t bIdx = tileIdx % b_;
            return mIdx * mL1_ * b_ + bIdx;
        }
        return tileIdx * mL1_;
    }

    __aicore__ inline int64_t GetRotOffset(int64_t tileIdx)
    {
        if (b_ == 1) {
            return 0;
        }
        int64_t bIdx = tileIdx % b_;
        return bIdx * k_ * k_;
    }

    __aicore__ inline BlockCoord GetBlockCoord(int tileIdx) { return {0, 0, 0, 0}; }
};

template <class ProblemShape_, class L1TileShape_, class L0TileShape_, bool TransA_, bool TransB_>
struct BlockSchedulerSelector<ProblemShape_, L1TileShape_, L0TileShape_, Cmct::Gemm::BuiltInRotateQuantScheduler,
                              TransA_, TransB_> {
    using SchedulerOp = BlockSchedulerRotateQuantBuiltIn<ProblemShape_, L1TileShape_, L0TileShape_>;
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif