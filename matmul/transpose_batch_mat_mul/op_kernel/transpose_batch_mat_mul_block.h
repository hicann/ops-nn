/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file transpose_batch_mat_mul_block.h
 * \brief
 */
#ifndef TRANSPOSE_BATCH_MAT_MUL_BLOCK_H
#define TRANSPOSE_BATCH_MAT_MUL_BLOCK_H

#include "../batch_mat_mul_v3/batch_mat_mul_v3_com_base_block_struct.h"
#include "../mat_mul_v3/mat_mul_v3_common.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"


namespace TransposeBatchMatmulSpace {

using namespace AscendC;
using namespace matmul;
enum class TBMM_MODE : int32_t {
    BMM_TRANS = 0,
    TRANS_BMM_TRANS = 1,
    BMM_TRANS_TRANS = 2,
    TRANS_BMM_TRANS_TRANS = 3
};

class TransposeBatchMatMulBlock {
   public:
    __aicore__ inline TransposeBatchMatMulBlock() {}
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
    __aicore__ inline void Init(const void *tilingData);
    template <class C_TYPE>
    __aicore__ inline void InitBatchCount();
    __aicore__ inline void InitL2Cache();
    __aicore__ inline void InitBlockIndex();
    __aicore__ inline void UpdateBlockParams();
    __aicore__ inline void UpdateBlockParams(uint64_t mTileIndex, uint64_t nTileIndex);
    __aicore__ inline void UpdateBlockCnt(uint64_t mTileIndex, uint64_t nTileIndex);
    __aicore__ inline void UpdateBlockIndex();
    __aicore__ inline void UpdateBasicIndex(const uint64_t roundIdx, uint64_t mTileIndex, uint64_t nTileIndex);
    template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, int MODE>
    __aicore__ inline void CalcGMOffset(uint64_t mTileIndex, uint64_t nTileIndex);
    template <class A_TYPE, int MODE>
    __aicore__ inline void CalcAOffset(uint64_t batchAIndex, uint64_t mCntIndex, uint64_t mInnerBatchOffset);
    template <class B_TYPE>
    __aicore__ inline void CalcBOffset(uint64_t batchBIndex, uint64_t nCntIndex);
    template <class C_TYPE, int MODE>
    __aicore__ inline void CalcCOffset(uint64_t batchCIndex, uint64_t mCntIndex, uint64_t nCntIndex,
                                       uint64_t mInnerBatchOffset);

   public:
    static constexpr uint64_t DIAGONAL_THRESHOLD = 5;

    CommonKernelBlockOffset offset_;
    CommonKernelBaseBlockArgs params_;
    const TBMMTilingData *tbmmTilingData_;
    bool indexInit_ = false;
    bool batchLastFlag_ = false;
    uint64_t offsetScales_ = 0;
    int32_t batchSplitFactor_ = 1;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void TransposeBatchMatMulBlock::Init(const void *tilingData)
{
    tbmmTilingData_ = static_cast<const TBMMTilingData *>(tilingData);

    InitBatchCount<C_TYPE>();

    // 默认使用行优先
    params_.rowOrder = ROW_FIRST;
    params_.mCntUse = params_.batchCnt * params_.mCnt;
    params_.nCntUse = params_.nCnt;
    params_.mTileCnt = params_.mCntUse;
    params_.nTileCnt = params_.nCntUse;
    params_.mCntTail = params_.mTileCnt;
    params_.nCntTail = params_.nTileCnt;
    params_.totalTileCnt = params_.mTileCnt * params_.nTileCnt;

    // 使能L2cache
    InitL2Cache();
    if (params_.preCoreNum == 0) {
        params_.preCoreNum = tbmmTilingData_->multiBatchInfo.batchUsedCoreNum;
    }

    params_.c0Size = 0;
    using A_T = typename A_TYPE::T;
    GetSizeC0<A_T>(params_.c0Size);
    params_.alignedOriM = MMV3DivCeil(tbmmTilingData_->matmulTiling.matmulTiling.M, ALIGNED_H) * ALIGNED_H;
    params_.alignedOriN =
        MMV3DivCeil(tbmmTilingData_->matmulTiling.matmulTiling.N, params_.c0Size) * params_.c0Size;
    params_.alignedKaSize =
        MMV3DivCeil(tbmmTilingData_->matmulTiling.matmulTiling.Ka, params_.c0Size) * params_.c0Size;
    params_.alignedKbSize = MMV3DivCeil(tbmmTilingData_->matmulTiling.matmulTiling.Kb, ALIGNED_H) * ALIGNED_H;
    // A B矩阵都是对齐矩阵
    if constexpr (A_TYPE::isTrans) {
        params_.alignedOriM =
            MMV3DivCeil(tbmmTilingData_->matmulTiling.matmulTiling.M, params_.c0Size) * params_.c0Size;
        params_.alignedKaSize =
            MMV3DivCeil(tbmmTilingData_->matmulTiling.matmulTiling.Ka, ALIGNED_H) * ALIGNED_H;
    }
    if constexpr (B_TYPE::isTrans) {
        params_.alignedOriN = MMV3DivCeil(tbmmTilingData_->matmulTiling.matmulTiling.N, ALIGNED_H) * ALIGNED_H;
        params_.alignedKbSize =
            MMV3DivCeil(tbmmTilingData_->matmulTiling.matmulTiling.Kb, params_.c0Size) * params_.c0Size;
    }
    batchSplitFactor_ = tbmmTilingData_->batchSplitFactor <= 0 ? 1 : tbmmTilingData_->batchSplitFactor;
}

template <class C_TYPE>
__aicore__ inline void TransposeBatchMatMulBlock::InitBatchCount()
{
    params_.isHf32 = tbmmTilingData_->matmulTiling.matmulRunInfo.isHf32;
    params_.batchA1 = tbmmTilingData_->multiBatchInfo.aBatchDim0;
    params_.batchA2 = tbmmTilingData_->multiBatchInfo.aBatchDim1;
    params_.batchA3 = tbmmTilingData_->multiBatchInfo.aBatchDim2;
    params_.batchA4 = tbmmTilingData_->multiBatchInfo.aBatchDim3;
    params_.batchB1 = tbmmTilingData_->multiBatchInfo.bBatchDim0;
    params_.batchB2 = tbmmTilingData_->multiBatchInfo.bBatchDim1;
    params_.batchB3 = tbmmTilingData_->multiBatchInfo.bBatchDim2;
    params_.batchB4 = tbmmTilingData_->multiBatchInfo.bBatchDim3;
    params_.batchC1 = tbmmTilingData_->multiBatchInfo.cBatchDim0;
    params_.batchC2 = tbmmTilingData_->multiBatchInfo.cBatchDim1;
    params_.batchC3 = tbmmTilingData_->multiBatchInfo.cBatchDim2;
    params_.batchC4 = tbmmTilingData_->multiBatchInfo.cBatchDim3;
    params_.aBatchDimAll = tbmmTilingData_->multiBatchInfo.aBatchDimAll;
    params_.bBatchDimAll = tbmmTilingData_->multiBatchInfo.bBatchDimAll;
    params_.cBatchDimAll = tbmmTilingData_->multiBatchInfo.cBatchDimAll;
    params_.biasWithBatch = tbmmTilingData_->multiBatchInfo.biasWithBatch;

    params_.batchCnt = params_.batchC1 * params_.batchC2 * params_.batchC3 * params_.batchC4;
    params_.mCnt = MMV3DivCeil(
        tbmmTilingData_->matmulTiling.matmulTiling.M,
        static_cast<uint64_t>(tbmmTilingData_->matmulTiling.matmulTiling.singleCoreM));  // m方向上分多少个base块
    params_.nCnt = MMV3DivCeil(
        tbmmTilingData_->matmulTiling.matmulTiling.N,
        static_cast<uint64_t>(tbmmTilingData_->matmulTiling.matmulTiling.singleCoreN));  // n方向上分多少个base块
    params_.index = 0;  // 当前block_idx的起始基本块Index，这个idex是按照先循环N，再循环M的次序
    params_.nBaseTail =
        tbmmTilingData_->matmulTiling.matmulTiling.N -
        (params_.nCnt - 1) * tbmmTilingData_->matmulTiling.matmulTiling.singleCoreN;  // n方向上的baseN尾块
    params_.mBaseTail =
        tbmmTilingData_->matmulTiling.matmulTiling.M -
        (params_.mCnt - 1) * tbmmTilingData_->matmulTiling.matmulTiling.singleCoreM;  // m方向上的baseM尾块
    params_.totalCnt = params_.batchCnt * params_.mCnt * params_.nCnt;  // 所有的基本块个数
    params_.round = (params_.totalCnt + tbmmTilingData_->multiBatchInfo.batchUsedCoreNum - 1) /
                    tbmmTilingData_->multiBatchInfo.batchUsedCoreNum;  // 每一个core最大做base块计算的次数
    params_.realRound = 0;                                                    // 单核做多少次base块计算
    params_.preCoreNum =
        params_.totalCnt %
        tbmmTilingData_->multiBatchInfo.batchUsedCoreNum;  // 从0core开始有多少个core会多做一次base块
    params_.blockIdxStart = 0;
    params_.blockIdxEnd = 0;
    params_.preTotalBlock = 0;
}

__aicore__ inline void TransposeBatchMatMulBlock::InitL2Cache()
{
    const L2cacheTilePara &tilingL2 = tbmmTilingData_->matmulTiling.tileL2cacheTiling;
    params_.mTileCntL2 = static_cast<uint64_t>(tilingL2.mTileCntL2); // M 方向上Tile份数
    params_.nTileCntL2 = static_cast<uint64_t>(tilingL2.nTileCntL2); // N 方向上Tile份数
    params_.enableL2Cache = ((params_.mTileCntL2 > 1) || (params_.nTileCntL2 > 1));
    if (params_.enableL2Cache) {
        params_.mTileCnt = tilingL2.mTileBlock; // 每份mTile包含的base块个数
        params_.nTileCnt = tilingL2.nTileBlock; // 每份nTile包含的base块个数
        params_.totalTileCnt = params_.mTileCnt * params_.nTileCnt;
        params_.mCntTail = params_.mCnt - (params_.mTileCnt * (DivCeil(params_.mCnt, params_.mTileCnt) - 1)); // M方向上mtile尾块的个数
        params_.nCntTail = params_.nCnt - (params_.nTileCntL2 - 1) * params_.nTileCnt; // N方向nTile尾块base块个数
        params_.mCntUse = params_.mTileCnt;
        params_.nCntUse = params_.nTileCnt;
        params_.rowOrder = 0;
    }
}

__aicore__ inline void TransposeBatchMatMulBlock::InitBlockIndex()
{
    if (indexInit_) {                                 // ND
        params_.blockIdxStart = params_.blockIdxEnd;  // 开始运算时，首核的索引
    } else {
        params_.blockIdxStart = 0;  // 第一次调用初始化为 0
        indexInit_ = true;
    }
    params_.blockIdxEnd = (params_.blockIdxStart + params_.preCoreNum) %
                          tbmmTilingData_->matmulTiling.matmulTiling.usedCoreNum;  // 结束运算时，尾核的索引
    if (params_.rowOrder == 0) {
        uint64_t indexStart = params_.blockIdxStart;
        uint64_t indexEnd = params_.blockIdxEnd;
        if (indexStart < indexEnd) {
            if ((indexStart <= GetBlockIdx()) && (GetBlockIdx() < indexEnd)) {
                params_.realRound = params_.round;
            } else {
                params_.realRound = params_.round - 1;
            }
        } else if (indexEnd < indexStart) {
            if ((indexEnd <= GetBlockIdx()) && (GetBlockIdx() < indexStart)) {
                params_.realRound = params_.round - 1;
            } else {
                params_.realRound = params_.round;
            }
        } else {
            params_.realRound = params_.round;
        }
    } else {
        if (GetBlockIdx() < params_.preCoreNum) {
            if (params_.rowOrder == ROW_FIRST) {
                params_.index = GetBlockIdx() * params_.round;
            } else {
                params_.preTotalBlock = GetBlockIdx() * params_.round;
                params_.index =
                    params_.preTotalBlock / params_.mCnt + params_.preTotalBlock % params_.mCnt * params_.nCnt;
            }
            params_.realRound = params_.round;  // 前面preCoreNum个core会多做一次
        } else {
            if (params_.rowOrder == ROW_FIRST) {
                params_.index = GetBlockIdx() * (params_.round - 1) + params_.preCoreNum;
            } else {
                params_.preTotalBlock = GetBlockIdx() * (params_.round - 1) + params_.preCoreNum;
                params_.index =
                    params_.preTotalBlock / params_.mCnt + params_.preTotalBlock % params_.mCnt * params_.nCnt;
            }
            params_.realRound = params_.round - 1;  // 后面的core会少做一次
        }
    }
}

__aicore__ inline void TransposeBatchMatMulBlock::UpdateBasicIndex(const uint64_t roundIdx, uint64_t mTileIndex,
                                                                   uint64_t nTileIndex)
{
    uint64_t newBlockIdx =
        (GetBlockIdx() + tbmmTilingData_->matmulTiling.matmulTiling.usedCoreNum - params_.blockIdxStart) %
            tbmmTilingData_->matmulTiling.matmulTiling.usedCoreNum +
        roundIdx * tbmmTilingData_->matmulTiling.matmulTiling.usedCoreNum;
    uint64_t mIdx = newBlockIdx % params_.mCntUse;
    uint64_t nIdx = 0;
    if ((params_.mCntUse != 0) && (params_.nCntUse != 0)) {
        nIdx = (newBlockIdx + newBlockIdx / MMLcm(params_.mCntUse, params_.nCntUse)) % params_.nCntUse;
    }
    params_.index = mIdx * params_.nCntUse + nIdx;
}

__aicore__ inline void TransposeBatchMatMulBlock::UpdateBlockParams(uint64_t mTileIndex, uint64_t nTileIndex)
{
    if (!(params_.enableL2Cache)) {
        if ((params_.index + 1) % (params_.mCnt * params_.nCnt) == 0) {
            // 最后一块是尾块
            params_.singleCoreM = params_.mBaseTail;
            params_.singleCoreN = params_.nBaseTail;
        } else if (((params_.index + 1) % (params_.mCnt * params_.nCnt)) > ((params_.mCnt - 1) * params_.nCnt)) {
            // M方向尾块
            params_.singleCoreM = params_.mBaseTail;
            params_.singleCoreN = tbmmTilingData_->matmulTiling.matmulTiling.singleCoreN;
        } else if ((params_.index + 1) % params_.nCnt == 0) {
            // N方向尾块
            params_.singleCoreM = tbmmTilingData_->matmulTiling.matmulTiling.singleCoreM;
            params_.singleCoreN = params_.nBaseTail;
        } else {
            // 对齐整块
            params_.singleCoreM = tbmmTilingData_->matmulTiling.matmulTiling.singleCoreM;
            params_.singleCoreN = tbmmTilingData_->matmulTiling.matmulTiling.singleCoreN;
        }
    } else {
        // 当前mTile在batch内，还原到单batch内index
        // batch内M方向尾Tile块的尾Tile行取baseTail；N按nTileCnt进行计算是否取baseTail
        if (batchLastFlag_ && (nTileIndex == (params_.nTileCntL2 - 1)) &&
            (params_.index == (params_.totalTileCnt - 1))) {
            params_.singleCoreM = params_.mBaseTail;
            params_.singleCoreN = params_.nBaseTail;
        } else if (batchLastFlag_ && (params_.index >= (params_.mCntUse - 1) * params_.nCntUse)) {
            params_.singleCoreM = params_.mBaseTail;
            params_.singleCoreN = tbmmTilingData_->matmulTiling.matmulTiling.singleCoreN;
        } else if ((nTileIndex == (params_.nTileCntL2 - 1)) && ((params_.index + 1) % params_.nCntUse == 0)) {
            params_.singleCoreM = tbmmTilingData_->matmulTiling.matmulTiling.singleCoreM;
            params_.singleCoreN = params_.nBaseTail;
        } else {
            params_.singleCoreM = tbmmTilingData_->matmulTiling.matmulTiling.singleCoreM;
            params_.singleCoreN = tbmmTilingData_->matmulTiling.matmulTiling.singleCoreN;
        }
    }
}

__aicore__ inline void TransposeBatchMatMulBlock::UpdateBlockCnt(uint64_t bmTileIndex, uint64_t nTileIndex)
{
    // 当前Tile切分不跨Batch
    // 计算偏移时每一个Tile块的起始偏移是无需区分尾块场景的，跨batch用M计算偏移，batch内用singcoreM计算
    uint32_t batchTileCnt = CeilDiv(params_.mCnt, params_.mTileCnt);  // mCnt > 0 mtileCnt > 0 无等于0场景
    uint64_t curBatchCnt = bmTileIndex / batchTileCnt;
    uint64_t curBatchTileIdx = bmTileIndex % batchTileCnt;
    params_.mTileAddrOffset =
        curBatchCnt * tbmmTilingData_->matmulTiling.matmulTiling.M +
        curBatchTileIdx * params_.mTileCnt * tbmmTilingData_->matmulTiling.matmulTiling.singleCoreM;
    params_.nTileAddrOffset =
        nTileIndex * params_.nTileCnt * tbmmTilingData_->matmulTiling.matmulTiling.singleCoreN;
    batchLastFlag_ = ((bmTileIndex + 1) % DivCeil(params_.mCnt, params_.mTileCnt) == 0);
    if (batchLastFlag_ && (nTileIndex == (params_.nTileCntL2 - 1))) {
        params_.totalTileCnt = params_.mCntTail * params_.nCntTail;
        params_.mCntUse = params_.mCntTail;
        params_.nCntUse = params_.nCntTail;
    } else if (batchLastFlag_) {
        params_.totalTileCnt = params_.mCntTail * params_.nTileCnt;
        params_.mCntUse = params_.mCntTail;
        params_.nCntUse = params_.nTileCnt;
    } else if (nTileIndex == (params_.nTileCntL2 - 1)) {
        params_.totalTileCnt = params_.mTileCnt * params_.nCntTail;
        params_.mCntUse = params_.mTileCnt;
        params_.nCntUse = params_.nCntTail;
    } else {
        params_.totalTileCnt = params_.mTileCnt * params_.nTileCnt;
        params_.mCntUse = params_.mTileCnt;
        params_.nCntUse = params_.nTileCnt;
    }
    params_.round = DivCeil(params_.totalTileCnt,
                            static_cast<uint64_t>(tbmmTilingData_->matmulTiling.matmulTiling.usedCoreNum));
    params_.preCoreNum = params_.totalTileCnt % tbmmTilingData_->matmulTiling.matmulTiling.usedCoreNum;
    if (params_.preCoreNum == 0) {
        params_.preCoreNum = static_cast<uint64_t>(tbmmTilingData_->matmulTiling.matmulTiling.usedCoreNum);
    }
}

__aicore__ inline void TransposeBatchMatMulBlock::UpdateBlockParams()
{
    if ((params_.index + 1) % (params_.mCnt * params_.nCnt) == 0) {
        // 最后一块是尾块
        params_.singleCoreM = params_.mBaseTail;
        params_.singleCoreN = params_.nBaseTail;
    } else if ((params_.index + 1) % (params_.mCnt * params_.nCnt) > (params_.mCnt - 1) * params_.nCnt) {
        // m方向尾块
        params_.singleCoreM = params_.mBaseTail;
        params_.singleCoreN = tbmmTilingData_->matmulTiling.matmulTiling.singleCoreN;
    } else if ((params_.index + 1) % params_.nCnt == 0) {
        // n方向尾块
        params_.singleCoreM = tbmmTilingData_->matmulTiling.matmulTiling.singleCoreM;
        params_.singleCoreN = params_.nBaseTail;
    } else {
        // 对齐整块
        params_.singleCoreM = tbmmTilingData_->matmulTiling.matmulTiling.singleCoreM;
        params_.singleCoreN = tbmmTilingData_->matmulTiling.matmulTiling.singleCoreN;
    }
}

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, int MODE>
__aicore__ inline void TransposeBatchMatMulBlock::CalcGMOffset(uint64_t mTileIndex, uint64_t nTileIndex)
{
    // batch, M 合轴 处理 index
    uint64_t bmTileStartIndex =
        params_.mTileAddrOffset / static_cast<uint64_t>(tbmmTilingData_->matmulTiling.matmulTiling.M);
    uint64_t tileInnerMIndex = params_.index / (params_.nCntUse * params_.mCnt);
    uint64_t totalBmTileIndex = bmTileStartIndex + tileInnerMIndex;

    // batch维度合并到M轴一起使用，先解析再重新计算
    uint64_t batchC1Index = totalBmTileIndex / (params_.batchC2 * params_.batchC3 * params_.batchC4);
    uint64_t batchC2Index =
        totalBmTileIndex % (params_.batchC2 * params_.batchC3 * params_.batchC4) / (params_.batchC3 * params_.batchC4);
    uint64_t batchC3Index = totalBmTileIndex % (params_.batchC3 * params_.batchC4) / params_.batchC4;
    uint64_t batchC4Index = totalBmTileIndex % params_.batchC4;
    uint64_t batchCIndex = totalBmTileIndex;

    // Tile 内 mIndex
    uint64_t mCntIndex = (params_.index / params_.nCntUse) % params_.mCnt;
    uint64_t nCntIndex = params_.index % params_.nCntUse;
    uint64_t mInnerBatchOffset =
        params_.mTileAddrOffset % static_cast<uint64_t>(tbmmTilingData_->matmulTiling.matmulTiling.M);
    CalcCOffset<C_TYPE, MODE>(batchCIndex, mCntIndex, nCntIndex, mInnerBatchOffset);

    uint64_t batchA1Index = batchC1Index % params_.batchA1;
    uint64_t batchA2Index = batchC2Index % params_.batchA2;
    uint64_t batchA3Index = batchC3Index % params_.batchA3;
    uint64_t batchA4Index = batchC4Index % params_.batchA4;
    uint64_t batchAIndex = batchA1Index * (params_.batchA2 * params_.batchA3 * params_.batchA4) +
                           batchA2Index * (params_.batchA3 * params_.batchA4) + batchA3Index * params_.batchA4 +
                           batchA4Index;
    CalcAOffset<A_TYPE, MODE>(batchAIndex, mCntIndex, mInnerBatchOffset);

    uint64_t batchB1Index = batchC1Index % params_.batchB1;
    uint64_t batchB2Index = batchC2Index % params_.batchB2;
    uint64_t batchB3Index = batchC3Index % params_.batchB3;
    uint64_t batchB4Index = batchC4Index % params_.batchB4;
    uint64_t batchBIndex = batchB1Index * (params_.batchB2 * params_.batchB3 * params_.batchB4) +
                           batchB2Index * (params_.batchB3 * params_.batchB4) + batchB3Index * params_.batchB4 +
                           batchB4Index;
    CalcBOffset<B_TYPE>(batchBIndex, nCntIndex);

    offsetScales_ = batchBIndex * static_cast<uint64_t>(tbmmTilingData_->matmulTiling.matmulTiling.N) +
                    (nCntIndex * tbmmTilingData_->matmulTiling.matmulTiling.singleCoreN) +
                    params_.nTileAddrOffset;

    if (tbmmTilingData_->matmulTiling.matmulTiling.isBias) {
        offset_.offsetBias =
            params_.biasWithBatch
                ? batchCIndex * static_cast<uint64_t>(tbmmTilingData_->matmulTiling.matmulTiling.N) +
                      nCntIndex * static_cast<uint64_t>(tbmmTilingData_->matmulTiling.matmulTiling.singleCoreN) +
                      params_.nTileAddrOffset
                : nCntIndex * static_cast<uint64_t>(tbmmTilingData_->matmulTiling.matmulTiling.singleCoreN) +
                      params_.nTileAddrOffset;
    }
}

template <class A_TYPE, int MODE>
__aicore__ inline void TransposeBatchMatMulBlock::CalcAOffset(uint64_t batchAIndex, uint64_t mCntIndex,
                                                              uint64_t mInnerBatchOffset)
{
    uint64_t offsetAM = mCntIndex * tbmmTilingData_->matmulTiling.matmulTiling.singleCoreM;
    if constexpr (MODE == static_cast<int>(TBMM_MODE::TRANS_BMM_TRANS) || MODE == static_cast<int>(TBMM_MODE::TRANS_BMM_TRANS_TRANS)) {
        offset_.offsetA =
            A_TYPE::isTrans
                ? batchAIndex * tbmmTilingData_->matmulTiling.matmulTiling.M + offsetAM + mInnerBatchOffset
                : offsetAM * params_.batchCnt * tbmmTilingData_->matmulTiling.matmulTiling.Ka +
                    batchAIndex * tbmmTilingData_->matmulTiling.matmulTiling.Ka +
                    mInnerBatchOffset * tbmmTilingData_->matmulTiling.matmulTiling.Ka;
    }
    if constexpr (MODE == static_cast<int>(TBMM_MODE::BMM_TRANS) || MODE == static_cast<int>(TBMM_MODE::BMM_TRANS_TRANS)) {
        offset_.offsetA =
            offsetAM * tbmmTilingData_->matmulTiling.matmulTiling.Ka +
            batchAIndex * tbmmTilingData_->matmulTiling.matmulTiling.Ka * tbmmTilingData_->matmulTiling.matmulTiling.M +
            mInnerBatchOffset * tbmmTilingData_->matmulTiling.matmulTiling.Ka;
    }
}

template <class B_TYPE>
__aicore__ inline void TransposeBatchMatMulBlock::CalcBOffset(uint64_t batchBIndex, uint64_t nCntIndex)
{
    uint64_t offsetBBatch = batchBIndex * static_cast<uint64_t>(tbmmTilingData_->matmulTiling.matmulTiling.N) *
                            tbmmTilingData_->matmulTiling.matmulTiling.Kb;
    offset_.offsetB =
        B_TYPE::isTrans ? offsetBBatch +
                              (nCntIndex * static_cast<uint64_t>(tbmmTilingData_->matmulTiling.matmulTiling.Kb) *
                               tbmmTilingData_->matmulTiling.matmulTiling.singleCoreN) +
                              (params_.nTileAddrOffset * tbmmTilingData_->matmulTiling.matmulTiling.Kb)
                        : offsetBBatch + (nCntIndex * tbmmTilingData_->matmulTiling.matmulTiling.singleCoreN) +
                              params_.nTileAddrOffset;
}

template <class C_TYPE, int MODE>
__aicore__ inline void TransposeBatchMatMulBlock::CalcCOffset(uint64_t batchCIndex, uint64_t mCntIndex,
                                                              uint64_t nCntIndex, uint64_t mInnerBatchOffset)
{
    if constexpr (MODE == static_cast<int>(TBMM_MODE::TRANS_BMM_TRANS) || MODE == static_cast<int>(TBMM_MODE::BMM_TRANS)) {
        offset_.offsetC = mCntIndex * tbmmTilingData_->matmulTiling.matmulTiling.singleCoreM * params_.batchCnt *
                          tbmmTilingData_->matmulTiling.matmulTiling.N +
                          batchCIndex * tbmmTilingData_->matmulTiling.matmulTiling.N +
                          nCntIndex * tbmmTilingData_->matmulTiling.matmulTiling.singleCoreN +
                          mInnerBatchOffset * tbmmTilingData_->matmulTiling.matmulTiling.N + params_.nTileAddrOffset;
    }
    if constexpr (MODE == static_cast<int>(TBMM_MODE::TRANS_BMM_TRANS_TRANS) || MODE == static_cast<int>(TBMM_MODE::BMM_TRANS_TRANS)) {
        uint64_t innerBatch = params_.batchCnt / batchSplitFactor_;
        uint64_t batchIdx1 = batchCIndex % innerBatch;
        uint64_t batchIdx2 = batchCIndex / innerBatch;
        offset_.offsetC = (batchIdx2 * tbmmTilingData_->matmulTiling.matmulTiling.M +
                           mCntIndex * tbmmTilingData_->matmulTiling.matmulTiling.singleCoreM) * innerBatch *
                           tbmmTilingData_->matmulTiling.matmulTiling.N +
                           batchIdx1 * tbmmTilingData_->matmulTiling.matmulTiling.N +
                           nCntIndex * tbmmTilingData_->matmulTiling.matmulTiling.singleCoreN +
                           mInnerBatchOffset * tbmmTilingData_->matmulTiling.matmulTiling.N + params_.nTileAddrOffset;
    }
}

__aicore__ inline void TransposeBatchMatMulBlock::UpdateBlockIndex()
{
    if (params_.rowOrder == ROW_FIRST) {
        params_.index += 1;
    } else if (params_.rowOrder == COL_FIRST) {
        params_.index += params_.nCnt;
        if (params_.index >= params_.totalCnt) {
            params_.index = params_.index % params_.totalCnt + 1;
        }
    }
}
} // namespace TransposeBatchMatmulSpace
#endif // TRANSPOSE_BATCH_MAT_MUL_BLOCK_H