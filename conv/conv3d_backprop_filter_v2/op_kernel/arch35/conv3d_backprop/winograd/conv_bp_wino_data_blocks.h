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

namespace WinoDetail {
using namespace AscendC;

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
          fullWSteps_(tilesW / SingleShapeTileW),
          kBegin_(kBegin),
          kLength_(kLength),
          hasTailW_(tilesW != fullWSteps_ * SingleShapeTileW),
          wStage_(fullWSteps_ > 0 ? FULL_W_STAGE : TAIL_W_STAGE)
    {
        Update();
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
        // 优先循环完整的shape在循环尾块,这样子测出来VF性能会好一些好点
        // 但是优先循环的HW方向都完整的代码有点复杂
        // 所以当前先循环W方向完整的块
        if (wStage_ == FULL_W_STAGE) {
            NextFullWStage();
        } else {
            NextTailWStage();
        }
    }

    __aicore__ inline bool More() const { return !end_; }

    __aicore__ inline uint32_t TileKIdx() const { return tileKIdx_; }

    __aicore__ inline uint32_t BatchIdx() const { return batchIdx_; }

private:
    __aicore__ inline void NextFullWStage()
    {
        fullWStepIdx_++;
        if (fullWStepIdx_ < fullWSteps_) {
            Update();
            return;
        }
        fullWStepIdx_ = 0;
        processedKStep_++;
        if (processedKStep_ < kLength_) {
            Update();
            return;
        }
        // 所有 K 的 full-W 都处理完了，切到 tail-W。
        if (hasTailW_) {
            wStage_ = TAIL_W_STAGE;
            processedKStep_ = 0;
            fullWStepIdx_ = 0;
            Update();
        } else {
            end_ = true;
        }
    }

    __aicore__ inline void NextTailWStage()
    {
        processedKStep_++;
        if (processedKStep_ >= kLength_) {
            end_ = true;
            return;
        }
        Update();
    }

    __aicore__ inline void Update()
    {
        uint32_t kStep = processedKStep_ + kBegin_;
        batchIdx_ = kStep / hSteps_;
        uint32_t singleShapeTileHIdx = kStep - batchIdx_ * hSteps_;
        tileHIdx_ = singleShapeTileHIdx * SingleShapeTileH;
        if (wStage_ == FULL_W_STAGE) {
            tileWIdx_ = fullWStepIdx_ * SingleShapeTileW;
        } else {
            tileWIdx_ = fullWSteps_ * SingleShapeTileW;
        }
        uint32_t singleShapeTileWIdx = tileWIdx_ / SingleShapeTileW;
        tileKIdx_ = singleShapeTileHIdx * wSteps_ + singleShapeTileWIdx;
    }

    constexpr static uint32_t SingleShapeTileH = BlockConfig::SingleShapeTileH<TilingT>();
    constexpr static uint32_t SingleShapeTileW = BlockConfig::SingleShapeTileW<TilingT>();
    constexpr static uint8_t FULL_W_STAGE = 0;
    constexpr static uint8_t TAIL_W_STAGE = 1;
    const uint32_t tilesH_;
    const uint32_t tilesW_;
    const uint32_t batch_;
    const uint32_t hSteps_;
    const uint32_t wSteps_;
    const uint32_t fullWSteps_;
    const uint32_t kBegin_;
    const uint32_t kLength_;
    uint32_t tileHIdx_ = 0;
    uint32_t tileWIdx_ = 0;
    uint32_t batchIdx_ = 0;
    uint32_t tileKIdx_ = 0;
    uint32_t processedKStep_ = 0;
    uint32_t fullWStepIdx_ = 0;
    const bool hasTailW_;
    uint8_t wStage_;
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

class SwizzleTopology2D {
public:
    // 实现简单的Tile和蛇形走位，所有核构成一个blockHW块进行递进，提升L2cache的命中率
    // 尾轮自适应，仅最后一轮才会产生空转
    //                          blockW(4)
    //                    |-----------------------|
    //                 -  +-----+-----+-----+-----+-----+-----+-----+
    //                 |  |core0|core2|core4|core6|core0|core2|core4|
    //       blockH(2)-|  +-----+-----+-----+-----+-----+-----+-----+
    //                 |  |core1|core3|core5|core7|core1|core3|core5|
    //                 -  +-----+-----+-----+-----+-----+-----+-----+  HCnt
    //                    |core4|core3|core2|core1|core0|core7|core6|
    //                    +-----+-----+-----+-----+-----+-----+-----+
    //                                       WCnt

    __aicore__ static inline void CalBlockGrid(uint32_t h, uint32_t w, uint16_t& outBlockH, uint16_t& outBlockW)
    {
        constexpr uint16_t CORE_NUM_32 = 32;
        constexpr uint16_t CORE_NUM_28 = 28;
        constexpr uint16_t CORE_NUM_36 = 36;
        constexpr uint16_t GRID_H_32C = 4;
        constexpr uint16_t GRID_W_32C = 8;
        constexpr uint16_t GRID_H_28C = 4;
        constexpr uint16_t GRID_W_28C = 7;
        constexpr uint16_t GRID_H_36C = 6;
        constexpr uint16_t GRID_W_36C = 6;
        uint16_t coreNum = GetBlockNum();
        uint16_t bestH = 1;
        uint16_t bestW = coreNum;

        // 常用核数配置直接写死，不用再去跑一遍循环
        if (coreNum == CORE_NUM_32) {
            bestH = GRID_H_32C;
            bestW = GRID_W_32C;
        } else if (coreNum == CORE_NUM_28) {
            bestH = GRID_H_28C;
            bestW = GRID_W_28C;
        } else if (coreNum == CORE_NUM_36) {
            bestH = GRID_H_36C;
            bestW = GRID_W_36C;
        } else {
            for (uint16_t i = 1; i * i <= coreNum; i++) {
                if (coreNum % i == 0) {
                    bestH = i;
                    bestW = coreNum / i;
                }
            }
        }

        // 形状匹配：将较大的维度分配给张量中较大的那个轴，进一步减少跨行/跨列跳跃
        if (h >= w) {
            outBlockH = Std::max(bestH, bestW);
            outBlockW = Std::min(bestH, bestW);
        } else {
            outBlockH = Std::min(bestH, bestW);
            outBlockW = Std::max(bestH, bestW);
        }
    }

    __aicore__ inline SwizzleTopology2D(uint32_t h, uint32_t w, uint16_t blockH, uint16_t blockW)
        : h_(h), w_(w), blockH_(blockH), blockW_(blockW), fullSuperRows_(h / blockH), totalCnt_(h * w)
    {}

    __aicore__ inline bool GetHW(uint32_t loopIdx, uint16_t coreId, uint32_t& outH, uint32_t& outW) const
    {
        uint32_t flattenIdx = loopIdx * GetBlockNum() + coreId;
        // 拦截越界
        if (unlikely(flattenIdx >= totalCnt_)) {
            outH = h_;
            outW = w_;
            return false;
        }

        uint32_t dummy;
        ComputeHW(flattenIdx, outH, outW, dummy);
        return true;
    }

    // 包围盒计算：计算当前轮次在 H 和 W 方向触达的最远逻辑边界
    __aicore__ inline void GetBoundHW(uint32_t loopIdx, uint32_t& boundH, uint32_t& boundW) const
    {
        uint32_t startIdx = loopIdx * GetBlockNum();
        if (unlikely(startIdx >= totalCnt_)) {
            boundH = 0;
            boundW = 0;
            return;
        }

        uint32_t endIdx = Std::min(startIdx + GetBlockNum(), totalCnt_) - 1;

        uint32_t dummyH1, startW, startSuperIdx;
        ComputeHW(startIdx, dummyH1, startW, startSuperIdx);

        uint32_t dummyH2, endW, endSuperIdx;
        ComputeHW(endIdx, dummyH2, endW, endSuperIdx);

        // ================= W 轴边界检测,由于存在蛇形走位，需要按照奇偶额外判断 =================
        if (startSuperIdx == endSuperIdx) {
            // 1. 未换行：直接取最大值
            boundW = Std::max(startW, endW);
        } else if (endSuperIdx - startSuperIdx >= SNAKE_PATTERN_PERIOD) {
            // 2. 跨越多行：中间必然包含一个完整的偶数行，绝对会撞击右侧墙壁
            boundW = w_ - 1;
        } else {
            // 3. 恰好相邻跨越 1 行
            if (startSuperIdx % SNAKE_PATTERN_PERIOD == 0) {
                // 偶切奇：在右侧墙壁折返，必然触碰 w_ - 1
                boundW = w_ - 1;
            } else {
                // 奇切偶：在左侧墙壁(W=0)折返，极值由起点或终点决定
                boundW = Std::max(startW, endW);
            }
        }
        // =========================================================

        boundH = endSuperIdx * blockH_ + Std::min(blockH_, h_ - endSuperIdx * blockH_) - 1;
    }

    __aicore__ inline uint32_t TotalCnt() const { return totalCnt_; }

    static constexpr uint32_t SNAKE_PATTERN_PERIOD = 2;

private:
    __aicore__ inline void ComputeHW(uint32_t flattenIdx, uint32_t& outH, uint32_t& outW, uint32_t& outSuperIdx) const
    {
        const uint32_t superRowElements = blockH_ * w_;
        const uint32_t fullSuperRowElements = fullSuperRows_ * superRowElements;
        const uint32_t superIdx = flattenIdx < fullSuperRowElements ? flattenIdx / superRowElements : fullSuperRows_;
        const uint32_t localIdx = flattenIdx - superIdx * superRowElements;
        const uint32_t superRowH = superIdx * blockH_;
        const uint32_t localBlockH = Std::min(blockH_, h_ - superRowH);
        // 每个BlockHW里面按H方向优先递进,也就是连续核的范围为(H0,W0),(H1,W0),(H2,W0)
        // 列H方向优先按当前实现起来较为简单
        outH = superRowH + localIdx % localBlockH;
        const uint32_t forwardW = localIdx / localBlockH;
        // 蛇形走位，先从头走到尾，在从尾走到头
        outW = (superIdx % SNAKE_PATTERN_PERIOD == 0) ? forwardW : (w_ - 1 - forwardW);
        outSuperIdx = superIdx;
    }

    const uint32_t h_;
    const uint32_t w_;
    const uint16_t blockH_;
    const uint16_t blockW_;
    const uint32_t fullSuperRows_;
    const uint32_t totalCnt_;
};

enum BlockIterDirection {
    COUT,
    CIN,
};

template <BlockIterDirection IterDir>
static __aicore__ inline bool GetBlockFromSwizzle2D(const SwizzleTopology2D& topology, uint32_t loopIdx,
                                                    uint16_t coreId, uint32_t singleShapeCout, uint32_t singleShapeCin,
                                                    uint32_t cout, uint32_t cin, CoutCinRange& cRange)
{
    uint32_t topoH, topoW;
    bool valid = topology.GetHW(loopIdx, coreId, topoH, topoW);

    uint32_t coutBlockIdx = (IterDir == CIN) ? topoH : topoW;
    uint32_t cinBlockIdx = (IterDir == CIN) ? topoW : topoH;

    cRange.coutIdx = coutBlockIdx * singleShapeCout;
    cRange.cinIdx = cinBlockIdx * singleShapeCin;
    cRange.coutLength = valid ? Std::min(singleShapeCout, cout - cRange.coutIdx) : 0;
    cRange.cinLength = valid ? Std::min(singleShapeCin, cin - cRange.cinIdx) : 0;

    return valid;
}

template <BlockIterDirection IterDir, typename TilingT>
class BlockIterator {
public:
    static constexpr uint32_t SingleShapeCout = BlockConfig::SingleShapeCout<TilingT>();
    static constexpr uint32_t SingleShapeCin = BlockConfig::SingleShapeCin<TilingT>();

    inline __aicore__ bool More() const { return loopIdx_ < blocksIterCnt_; }

    // 获取当前aic计算的基本块范围,若当前核无基本块计算则返回false并且将length设置为0
    inline __aicore__ bool GetLocalBlock(CoutCinRange& cRange) const { return GetBlock(AicCoreId(), cRange); }

    inline __aicore__ bool GetBlock(uint16_t coreId, CoutCinRange& cRange) const
    {
        return GetBlockFromSwizzle2D<IterDir>(topology_, loopIdx_, coreId, SingleShapeCout, SingleShapeCin, cout_, cin_,
                                              cRange);
    }

    // 获取本轮全核计算涉及基本块的cout/cin范围最大值
    inline __aicore__ void GetClusterBlockUpperBound(uint32_t& outCoutBound, uint32_t& outCinBound) const
    {
        uint32_t boundH, boundW;
        topology_.GetBoundHW(loopIdx_, boundH, boundW);

        // 边界反向映射
        uint32_t boundCoutBlockIdx = (IterDir == CIN) ? boundH : boundW;
        uint32_t boundCinBlockIdx = (IterDir == CIN) ? boundW : boundH;

        // 转化为实际的空间维度绝对边界
        outCoutBound = Std::min((boundCoutBlockIdx + 1) * SingleShapeCout, cout_);
        outCinBound = Std::min((boundCinBlockIdx + 1) * SingleShapeCin, cin_);
    }

    inline __aicore__ void Next() { loopIdx_++; }

    inline __aicore__ const SwizzleTopology2D& GetSwizzleTopology() const { return topology_; }

    inline __aicore__ uint32_t GetTailBlockCnt() const
    {
        const uint32_t mainBlockNum = blocksIterCnt_ * GetBlockNum();
        return topology_.TotalCnt() > mainBlockNum ? topology_.TotalCnt() - mainBlockNum : 0;
    }

    static inline __aicore__ BlockIterator Create(bool onlyIterMainBlocks, uint32_t cout, uint32_t cin)
    {
        uint32_t coutCnt = Ops::Base::CeilDiv(cout, SingleShapeCout);
        uint32_t cinCnt = Ops::Base::CeilDiv(cin, SingleShapeCin);
        uint32_t topologyH = (IterDir == CIN) ? coutCnt : cinCnt;
        uint32_t topologyW = (IterDir == CIN) ? cinCnt : coutCnt;
        uint16_t blockH, blockW;
        SwizzleTopology2D::CalBlockGrid(topologyH, topologyW, blockH, blockW);
        return BlockIterator(cout, cin, topologyH, topologyW, blockH, blockW, onlyIterMainBlocks);
    }

private:
    inline __aicore__ explicit BlockIterator(uint32_t cout, uint32_t cin, uint32_t topologyH, uint32_t topologyW,
                                             uint16_t topologyBlockH, uint16_t topologyBlockW, bool onlyIterMainBlocks)
        : cout_(cout),
          cin_(cin),
          topology_(topologyH, topologyW, topologyBlockH, topologyBlockW),
          blocksIterCnt_(GetBlockIterCnt(onlyIterMainBlocks, topology_.TotalCnt()))
    {}

    inline __aicore__ static uint32_t GetBlockIterCnt(bool onlyIterMainBlocks, uint32_t totalBlocks)
    {
        uint16_t blockNum = GetBlockNum();
        uint32_t mainIterCnt = totalBlocks / blockNum;
        uint32_t tailBlocks = totalBlocks - mainIterCnt * blockNum;

        if (onlyIterMainBlocks) {
            // 尾轮空闲核超过一半时，将这些block留到后续切k处理
            return tailBlocks > (blockNum / 2) ? mainIterCnt + 1 : mainIterCnt;
        } else {
            return mainIterCnt + (tailBlocks > 0 ? 1 : 0);
        }
    }

    const uint32_t cout_;
    const uint32_t cin_;
    const SwizzleTopology2D topology_;
    const uint32_t blocksIterCnt_;
    uint32_t loopIdx_ = 0;
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
                                              uint32_t cout, uint32_t cin)
        : topology_(topology),
          tailBlockCnt_(tailBlockCnt),
          topologyTailIter_((topology.TotalCnt() - tailBlockCnt) / GetBlockNum()),
          totalK_(totalK),
          cout_(cout),
          cin_(cin)
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
        GetBlockFromSwizzle2D<MainBlockIterDir>(topology_, topologyTailIter_, tailBlockIdx, SingleShapeCout,
                                                SingleShapeCin, cout_, cin_, cRange);
    }

private:
    static constexpr uint32_t SingleShapeCout = BlockConfig::SingleShapeCout<TilingT>();
    static constexpr uint32_t SingleShapeCin = BlockConfig::SingleShapeCin<TilingT>();
    const SwizzleTopology2D topology_;
    const uint16_t tailBlockCnt_;
    const uint32_t topologyTailIter_;
    const uint32_t totalK_;
    const uint32_t cout_;
    const uint32_t cin_;
};
} // namespace WinoDetail

#endif // CONV_BP_DATA_BLOCKS_H
