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
 * \file dual_level_quant_batch_matmul_block.h
 * \brief
 */

#ifndef DUAL_LEVEL_QUANT_BATCH_MATMUL_BLOCK_H
#define DUAL_LEVEL_QUANT_BATCH_MATMUL_BLOCK_H
#include "../dual_level_quant_batch_matmul_tiling_data.h"
#include "op_kernel/math_util.h"
#include "tool_arch35.h"

namespace DualLevelQuantBatchMatmul::Arch35 {

struct DualLevelQbmmBasicBlockOffsetParams {
    uint64_t mGmOffset;
    uint64_t kGmOffset;
    uint64_t nGmOffset;
    uint64_t singleCoreM; // 当前轮实际处理的大小
    uint64_t singleCoreN;
    uint64_t l0BaseK;
    uint64_t mSize;
    uint64_t kSize;
    uint64_t nSize;
    uint64_t mUbSize;
    uint64_t level0GroupSize;
    uint64_t level0ScaleKUbSize;
    uint64_t level1ScaleKL1Size;
    uint64_t level0ScaleGmGroupNum; // GM上x1Level0Scale groupNum轴的长度
};

// 用于4:8分核的参数，辅助tiling计算
struct DualLevelQbmmBaseBlockArgs {
    // 确定基本形状
    uint64_t mCnt;     // m方向基本块数量
    uint64_t nCnt;     // n方向基本块数量
    uint64_t totalCnt; // 基本块总数
    uint64_t round;    // 总的轮数

    // 尾块切分参数
    uint64_t mBaseNormCnt; // 使用主轮基本块大小的基本块个数
    uint64_t nBaseNormCnt;
    uint64_t mBaseTailMain; // 尾块重切分后的基本块大小和尾块大小
    uint64_t mBaseTailLast;
    uint64_t nBaseTailMain;
    uint64_t nBaseTailLast;

    // 4:8分核的走位控制和坐标映射
    uint64_t mCoreNum;      // m方向使用的核数，主窗口的m方向边长 singleWinM
    uint64_t mainRow;       // m轴方向round轮数，主窗口行数
    uint64_t mTailCoreNum;  // 尾轮m方向使用的核数
    uint64_t totalTailTile; // 尾块重切分，尾块部分重切后的tile块数量

    // 状态变量
    uint64_t mIndex;           // m方向的基本块索引
    uint64_t nIndex;           // n方向的基本块索引
    uint64_t mSplitAddrOffset; // 尾块重切分偏移量
    uint64_t nSplitAddrOffset;
};

class DualLevelQuantBatchMatmulBaseBlock {
public:
    __aicore__ inline DualLevelQuantBatchMatmulBaseBlock() = default;

    __aicore__ inline void Init(const DualLevelQuantBatchMatmulBasicTilingData* tilingData, uint64_t blockIdx);

    __aicore__ inline void UpdateBasicIndex(uint64_t roundIdx);

    __aicore__ inline void UpdateBlockParams(uint64_t roundIdx);

    __aicore__ inline void CalcGMOffset();

    __aicore__ inline void ResetAddressOffsets();

    uint64_t blockIdx_;
    const DualLevelQuantBatchMatmulBasicTilingData* tiling_;
    DualLevelQbmmBaseBlockArgs params_;
    DualLevelQbmmBasicBlockOffsetParams offset_;

private:
    static constexpr uint64_t CUBE_BLOCK = 16UL;
    static constexpr uint64_t WINDOW_LEN = 4;
    static constexpr uint64_t L0_BASE_K_SIZE = 512;
    static constexpr uint64_t LEVEL1_SCALE_K_L1_SIZE = 4096;
    static constexpr uint64_t LEVEL0_SCALE_K_UB_RATIO = 32;
    static constexpr uint64_t LEVEL0_SCALE_K_UB_SIZE = L0_BASE_K_SIZE * LEVEL0_SCALE_K_UB_RATIO;
};

__aicore__ inline void DualLevelQuantBatchMatmulBaseBlock::Init(
    const DualLevelQuantBatchMatmulBasicTilingData* tilingData, uint64_t blockIdx)
{
    tiling_ = tilingData;
    blockIdx_ = blockIdx;

    offset_.mSize = tilingData->mSize;
    offset_.kSize = tilingData->kSize;
    offset_.nSize = tilingData->nSize;
    offset_.level0GroupSize = tilingData->level0GroupSize;
    offset_.level0ScaleKUbSize = LEVEL0_SCALE_K_UB_SIZE; // 512 * 32，x1_level0_scale一次载入的k方向的大小
    offset_.l0BaseK = L0_BASE_K_SIZE;
    offset_.level1ScaleKL1Size = LEVEL1_SCALE_K_L1_SIZE;
    offset_.level0ScaleGmGroupNum = Ops::Base::CeilDiv(offset_.kSize, offset_.level0GroupSize);

    params_.mCnt = Ops::Base::CeilDiv<uint64_t>(tiling_->mSize, tiling_->mL1Size);
    params_.nCnt = Ops::Base::CeilDiv<uint64_t>(tiling_->nSize, tiling_->nL1Size);
    params_.totalCnt = params_.mCnt * params_.nCnt;
    params_.round = Ops::Base::CeilDiv<uint64_t>(params_.totalCnt, tiling_->usedCoreNum);

    params_.mCoreNum = DualLevelQuantBatchMatmul::Arch35::Min(WINDOW_LEN, params_.mCnt);
    params_.mainRow = params_.mCnt / params_.mCoreNum - 1; // 主轮滑窗行数
    params_.mTailCoreNum =
        params_.mCnt - params_.mCoreNum * params_.mainRow; // m轴尾块分核大小，mCnt % params_.mCoreNum
    params_.totalTailTile = tiling_->mTailTile * tiling_->nTailTile;

    // 尾块重切基本块大小
    params_.mBaseNormCnt = params_.mCnt - tiling_->mBaseTailSplitCnt;
    params_.nBaseNormCnt = params_.nCnt - tiling_->nBaseTailSplitCnt;
    uint64_t mBaseTail = tiling_->mSize - params_.mBaseNormCnt * tiling_->mL1Size;
    uint64_t nBaseTail = tiling_->nSize - params_.nBaseNormCnt * tiling_->nL1Size;
    params_.mBaseTailMain = tiling_->mBaseTailSplitCnt == 1 ? mBaseTail : tiling_->mTailMain;
    params_.mBaseTailLast = mBaseTail - (tiling_->mBaseTailSplitCnt - 1) * params_.mBaseTailMain;
    params_.nBaseTailMain = tiling_->nBaseTailSplitCnt == 1 ? nBaseTail : tiling_->nTailMain;
    params_.nBaseTailLast = nBaseTail - (tiling_->nBaseTailSplitCnt - 1) * params_.nBaseTailMain;

    ResetAddressOffsets();
}

__aicore__ inline void DualLevelQuantBatchMatmulBaseBlock::ResetAddressOffsets()
{
    params_.mSplitAddrOffset = 0UL;
    params_.nSplitAddrOffset = 0UL;
}

__aicore__ inline void DualLevelQuantBatchMatmulBaseBlock::UpdateBasicIndex(uint64_t roundIdx)
{
    // 新一轮分核Id，最后一轮进行尾块重切分，多个核会分到同一个原始基本块上，但实际处理的是重切分后的块
    uint64_t newBlockIdx = (roundIdx == params_.round - 1) ? (blockIdx_ / params_.totalTailTile) : blockIdx_;
    // L1基本块在整个输入中的线性索引，每个round消费usedCoreNum个基本块
    uint64_t index = newBlockIdx + roundIdx * tiling_->usedCoreNum;
    // 轮次index，m轴方向的第几轮
    uint64_t rowIdx = index / params_.nCnt / params_.mCoreNum;

    uint64_t mCoreNum;
    if (rowIdx < params_.mainRow) {
        // 非尾轮 4:8 切分
        mCoreNum = params_.mCoreNum;
    } else {
        // 尾轮
        rowIdx = params_.mainRow;
        // 和非尾轮一样，但index使用尾轮的index和mCoreNum，不能越界
        mCoreNum = params_.mTailCoreNum;
        index = index - params_.mainRow * params_.mCoreNum * params_.nCnt;
    }

    // m轴优先排index
    // mIndex = 轮次index + 当前轮次中m轴index
    params_.mIndex = rowIdx * params_.mCoreNum + index % mCoreNum;
    // nIndex = n轴index取余计算 在原矩阵中的index
    params_.nIndex = (index / mCoreNum) % params_.nCnt;

    // 按奇偶行进行蛇形走位，提高L2命中率
    if (rowIdx & 1) {
        params_.nIndex = params_.nCnt - 1 - params_.nIndex;
    }
}

__aicore__ inline void DualLevelQuantBatchMatmulBaseBlock::UpdateBlockParams(uint64_t roundIdx)
{
    // 非尾块用L1基本块处理，否则按尾块大小处理
    offset_.singleCoreM = tiling_->mL1Size;
    if (params_.mIndex >= params_.mBaseNormCnt) {
        offset_.singleCoreM = params_.mIndex >= (params_.mCnt - 1) ? params_.mBaseTailLast : params_.mBaseTailMain;
    }
    offset_.singleCoreN = tiling_->nL1Size;
    if (params_.nIndex >= params_.nBaseNormCnt) {
        offset_.singleCoreN = params_.nIndex >= (params_.nCnt - 1) ? params_.nBaseTailLast : params_.nBaseTailMain;
    }
    // 如果Tiling配置不需要对Tile进行核间再切分(TailTile为1)，则直接结束
    if (tiling_->mTailTile == 1 && tiling_->nTailTile == 1) {
        return;
    }

    // 最后一轮，按tiling给出的mTailTile/nTailTile重切分
    if (roundIdx == params_.round - 1) {
        // 计算重切分后的子块大小
        uint64_t singleCoreMSplit = Ops::Base::CeilDiv<uint64_t>(offset_.singleCoreM, tiling_->mTailTile);
        uint64_t singleCoreNSplit = Ops::Base::CeilDiv<uint64_t>(offset_.singleCoreN, tiling_->nTailTile);
        singleCoreNSplit = Ops::Base::CeilAlign<uint64_t>(singleCoreNSplit, CUBE_BLOCK);
        // 计算重切后尾块中的分核索引
        uint64_t mSplitIdx = (blockIdx_ % params_.totalTailTile) % tiling_->mTailTile;
        uint64_t nSplitIdx = (blockIdx_ % params_.totalTailTile) / tiling_->mTailTile;
        // 更新尾块重切分、分核后、位于原始tile中的偏移量
        params_.mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
        params_.nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
        // 偏移量已经超出了tile总大小或多余的尾核，不需要参与计算
        if (params_.mSplitAddrOffset >= offset_.singleCoreM || params_.nSplitAddrOffset >= offset_.singleCoreN ||
            blockIdx_ >= (params_.totalTailTile * (params_.totalCnt % tiling_->usedCoreNum))) {
            offset_.singleCoreM = 0;
            offset_.singleCoreN = 0;
            return;
        }

        // 根据是否是重切分后的尾块，计算重切分后的singleCoreM/singleCoreN
        if (params_.mSplitAddrOffset + singleCoreMSplit > offset_.singleCoreM) {
            offset_.singleCoreM = offset_.singleCoreM - singleCoreMSplit * mSplitIdx;
        } else {
            offset_.singleCoreM = singleCoreMSplit;
        }
        if (params_.nSplitAddrOffset + singleCoreNSplit > offset_.singleCoreN) {
            offset_.singleCoreN = offset_.singleCoreN - singleCoreNSplit * nSplitIdx;
        } else {
            offset_.singleCoreN = singleCoreNSplit;
        }
    }
}

__aicore__ inline void DualLevelQuantBatchMatmulBaseBlock::CalcGMOffset()
{
    // 计算当前Tile在矩阵M和N维度上的总逻辑偏移
    // 总偏移 = 块索引 * 标准基本块大小 + 重切分子块偏移
    uint64_t mOffset = params_.mIndex * tiling_->mL1Size + params_.mSplitAddrOffset;
    uint64_t nOffset = params_.nIndex * tiling_->nL1Size + params_.nSplitAddrOffset;
    if (params_.mIndex > params_.mBaseNormCnt) {
        mOffset = mOffset - (params_.mIndex - params_.mBaseNormCnt) * (tiling_->mL1Size - params_.mBaseTailMain);
    }
    if (params_.nIndex > params_.nBaseNormCnt) {
        nOffset = nOffset - (params_.nIndex - params_.nBaseNormCnt) * (tiling_->nL1Size - params_.nBaseTailMain);
    }

    offset_.mGmOffset = mOffset;
    offset_.nGmOffset = nOffset;
}

} // namespace DualLevelQuantBatchMatmul::Arch35
#endif // DUAL_LEVEL_QUANT_BATCH_MATMUL_BLOCK_H