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
 * \file conv_bp_wino_data_blocks.h
 * \brief
 */

#ifndef CONV_BP_DATA_BLOCKS_H
#define CONV_BP_DATA_BLOCKS_H

#include "conv_bp_wino_util.h"
#include "../util/conv_bp_common_data_blocks.h"

namespace WinoDetail {
using namespace AscendC;

// ★蛇形分核走位 4 件套（SwizzleTopology2D / BlockIterDirection / GetBlockFromSwizzle2D /
// BlockIterator）已于第十七轮泛化迁入 BpUtils、第二十一轮改名轮拆分至
// conv_bp_common_data_blocks.h——winograd 侧引用公共版，运行时逻辑零变化（迁移泛化点见
// data_blocks 头注释：SingleShape 改 Create 入参、coreNum/blockNum 可选显式传入，
// 本文件调用均走默认 GetBlockNum 路径）
using BpUtils::BlockIterator;
using BpUtils::BlockIterDirection;
using BpUtils::CIN;
using BpUtils::COUT;
using BpUtils::GetBlockFromSwizzle2D;
using BpUtils::SwizzleTopology2D;

template <typename TilingT>
class BatchTileKIterator {
public:
    // 支持切K,K的维度为[batch,CeilDiv(tileH,SingleShapeTileH)] //当前简单点不切tileW
    __aicore__ inline explicit BatchTileKIterator(uint32_t batch, uint32_t tilesH, uint32_t tilesW, uint32_t kBegin,
                                                  uint32_t kLength)
        : batch_(batch),
          tilesH_(tilesH),
          tilesW_(tilesW),
          hSteps_(Ops::Base::CeilDiv(tilesH, SingleShapeTileH)),
          wSteps_(Ops::Base::CeilDiv(tilesW, SingleShapeTileW)),
          kBegin_(kBegin),
          kLength_(kLength)
    {
        Update(0, 0);
    }

    // 在一单位k里会实际有几次循环
    __aicore__ inline uint32_t StepInSingleK() const { return wSteps_; }

    __aicore__ inline HWBox TileBox() const
    {
        HWBox tile = {};
        tile.hIdx = tileHIdx_;
        tile.wIdx = tileWIdx_;
        tile.hLength = Std::min(SingleShapeTileH, tilesH_ - tileHIdx_);
        tile.wLength = Std::min(SingleShapeTileW, tilesW_ - tileWIdx_);
        tile.elements = tile.hLength * tile.wLength;
        return tile;
    }

    __aicore__ inline void Next()
    {
        if (unlikely(end_)) {
            return;
        }
        // 单位k内W方向按自然顺序递增,完整的块在前,尾块在最后
        tileWIdx_ += SingleShapeTileW;
        if (tileWIdx_ >= tilesW_) {
            processedKStep_++;
            if (processedKStep_ >= kLength_) {
                end_ = true;
            } else {
                Update(processedKStep_, 0);
            }
        } else {
            Update(processedKStep_, tileWIdx_);
        }
    }

    __aicore__ inline bool More() const { return !end_; }

    __aicore__ inline uint32_t TileKIdx() const { return tileKIdx_; }

    __aicore__ inline uint32_t BatchIdx() const { return batchIdx_; }

private:
    __aicore__ inline void Update(uint32_t processedSteps, uint32_t tileWIdx)
    {
        uint32_t kStep = processedSteps + kBegin_;
        batchIdx_ = kStep / hSteps_;
        uint32_t singleShapeTileHIdx = kStep - batchIdx_ * hSteps_;
        tileHIdx_ = singleShapeTileHIdx * SingleShapeTileH;
        tileWIdx_ = tileWIdx;
        uint32_t singleShapeTileWIdx = tileWIdx_ / SingleShapeTileW;
        tileKIdx_ = singleShapeTileHIdx * wSteps_ + singleShapeTileWIdx;
    }

    constexpr static uint32_t SingleShapeTileH = BlockConfig::SingleShapeTileH<TilingT>();
    constexpr static uint32_t SingleShapeTileW = BlockConfig::SingleShapeTileW<TilingT>();
    const uint32_t tilesH_;
    const uint32_t tilesW_;
    const uint32_t batch_;
    const uint32_t hSteps_;
    const uint32_t wSteps_;
    const uint32_t kBegin_;
    const uint32_t kLength_;
    uint32_t tileHIdx_ = 0;
    uint32_t tileWIdx_ = 0;
    uint32_t batchIdx_ = 0;
    uint32_t tileKIdx_ = 0;
    uint32_t processedKStep_ = 0;
    bool end_ = false;
};

template <typename TilingT>
class SegmentTileKIterator {
public:
    __aicore__ explicit inline SegmentTileKIterator(uint32_t segmentsHint, BatchTileKIterator<TilingT>& kIter)
        : kIter_(kIter), segments_(segmentsHint)
    {}

    __aicore__ inline uint32_t StepInSingleK() const { return kIter_.StepInSingleK(); }

    __aicore__ inline HWBox TileBox() const { return kIter_.TileBox(); }

    __aicore__ inline void Next()
    {
        if (likely(More())) {
            count_++;
            kIter_.Next();
        }
    }

    __aicore__ inline bool More() const { return ReachSegmentsLimit() ? false : kIter_.More(); }

    __aicore__ inline uint32_t TileKIdx() const { return kIter_.TileKIdx(); }

    __aicore__ inline uint32_t BatchIdx() const { return kIter_.BatchIdx(); }

    __aicore__ inline void ResetSegmentsLimit() { count_ = 0; }

    __aicore__ inline bool AllSegmentsHasDone() const { return !kIter_.More(); }

private:
    __aicore__ inline bool ReachSegmentsLimit() const { return count_ >= segments_; }

    BatchTileKIterator<TilingT>& kIter_;
    const uint32_t segments_;
    uint32_t count_ = 0;
};

struct SplitKState {
    uint32_t kBegin = 0;
    uint32_t kLength = 0;
    // 单个kGroup最多会被分到几个k
    uint32_t kMaxLength = 0;
    // k轴被切分成几个组
    uint16_t kGroups = 0;
    // 当前核所在的kGroup的索引
    uint16_t kGroupIdx = 0;
    // 当前kGroup的核数
    uint16_t kGroupStartCoreId = 0;
    uint16_t kGroupCoreNum = 0;
    // 当前核负责的尾块索引,用于tailGm寻址
    uint16_t tailBlockId = 0;
};

template <BlockIterDirection MainBlockIterDir, typename TilingT>
class TailBlockSplitKIterator {
public:
    // 由于主轮的走位不是按照固定的矩形方式走的，TailBlocks在整个基本块里的形状不一定能用一个矩形表示，所以构造函数里需要
    // 传入主轮使用的SwizzleTopology2D解算实际坐标
    inline __aicore__ TailBlockSplitKIterator(uint32_t tailBlockCnt, const SwizzleTopology2D& topology, uint32_t totalK,
                                              uint32_t cout, uint32_t cin, uint16_t singleShapeCin,
                                              uint16_t singleShapeCout)
        : topology_(topology),
          tailBlockCnt_(tailBlockCnt),
          topologyTailIter_((topology.TotalCnt() - tailBlockCnt) / GetBlockNum()),
          totalK_(totalK),
          cout_(cout),
          cin_(cin),
          singleShapeCout_(singleShapeCout),
          singleShapeCin_(singleShapeCin)
    {}

    // 将尾块按k轴平分到核上，要求尾轮基本块不超过核数一半
    // 所有核会被拆分成kGroup个组，每个组处理所有尾块的相同k段
    // 这样可以保证kGroup内的逻辑应当和主轮相同，本质是处理TailBlocks个基本块，只是k轴缩短
    // 分不进kGroup的尾核统一塞进最后一个kGroup，这样子外层不用单独适配尾轮的空跑代码，所以最后一个kGroup的核可能会多一些
    //
    //   kGroup = Std::min(totalCoreNum / tailBlockCnt_, totalK_)
    //   kCores = CoreNums/kGroup
    //
    //                                 KGroup(2)
    //                 -  +--------------+---------------+
    //                 |  |core0         |kCore          |
    //    TailBlocks  -|  +--------------+---------------+
    //                 |  |core1         |kCore1         |
    //                 -  +--------------+---------------+
    //                    |core2(idle)   |kCore2(idle)   |
    //                    +--------------+---------------+
    //                    |..............|...............|
    //                    +--------------+---------------+
    //                    |kCore-1(idle) |kCore*2-1(idle)|
    //                    +--------------+---------------+
    //                                   |...............|
    //                                   +---------------+
    //                                   |LastCore(idle) |
    //                                   +---------------+
    //
    inline __aicore__ void GetLocalBlock(CoutCinRange& cRange, SplitKState& k) const
    {
        uint16_t totalCoreNum = GetBlockNum();

        // 最多多少个core去切分同一个基本块的k
        k.kGroups = Std::min(totalCoreNum / tailBlockCnt_, totalK_);
        // 每个kGroup里有几个core
        uint32_t kCores = totalCoreNum / k.kGroups;

        uint32_t coreId = AicCoreId();
        // 不在任意一个kGroup里的尾核统统塞进最后一个kGroup里面，不能完全不管，得参与全核同步
        k.kGroupIdx = Std::min(coreId / kCores, k.kGroups - 1);
        k.kGroupCoreNum = kCores;
        k.kGroupStartCoreId = kCores * k.kGroupIdx;
        if (k.kGroupIdx == k.kGroups - 1) {
            k.kGroupCoreNum = totalCoreNum - k.kGroupStartCoreId;
        }

        RemainderDistributionSpliter splitter(totalK_, k.kGroups);
        splitter.GetSplit(k.kGroupIdx, k.kBegin, k.kLength);
        k.kMaxLength = splitter.GetMaxLength();

        uint16_t tailBlockIdx = coreId - k.kGroupStartCoreId;
        k.tailBlockId = tailBlockIdx;
        GetBlockFromSwizzle2D<MainBlockIterDir>(topology_, topologyTailIter_, tailBlockIdx, singleShapeCout_,
                                                singleShapeCin_, cout_, cin_, cRange);
    }

private:
    const uint32_t singleShapeCout_;
    const uint32_t singleShapeCin_;
    const SwizzleTopology2D topology_;
    const uint16_t tailBlockCnt_;
    const uint32_t topologyTailIter_;
    const uint32_t totalK_;
    const uint32_t cout_;
    const uint32_t cin_;
};
} // namespace WinoDetail

#endif // CONV_BP_DATA_BLOCKS_H
