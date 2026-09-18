/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file conv_bp_common_data_blocks.h
 * \brief 卷积反向公共蛇形分核走位层：SwizzleTopology2D / BlockIterDirection /
 *        GetBlockFromSwizzle2D / BlockIterator。
 *        依赖方向单向：本文件 → conv_bp_common_util.h（CoutCinRange/AicCoreId）；
 *        math_util（CeilDiv）与 utils/std/algorithm（AscendC::Std::min/max）随 util 链可见
 */

#ifndef CONV_BP_COMMON_DATA_BLOCKS_H
#define CONV_BP_COMMON_DATA_BLOCKS_H

#include "conv_bp_common_util.h"

namespace BpUtils {
// ==================== 蛇形分核走位 ====================
// SingleShapeCout/Cin 与 coreNum/blockNum 均为 Create 入参（blockNum 默认 0 =
// GetBlockNum()；直调/UT 场景显式传入更稳）；核号走 BpUtils::AicCoreId()

class SwizzleTopology2D {
public:
    // 所有核构成一个 blockHW 块蛇形递进（提升 L2 命中率），尾轮自适应仅末轮空转：
    //                    |-----------------------| blockW(4)
    //                 -  +-----+-----+-----+-----+-----+-----+-----+
    //                 |  |core0|core2|core4|core6|core0|core2|core4|
    //       blockH(2)-|  +-----+-----+-----+-----+-----+-----+-----+
    //                 |  |core1|core3|core5|core7|core1|core3|core5|
    //                 -  +-----+-----+-----+-----+-----+-----+-----+  HCnt
    //                    |core4|core3|core2|core1|core0|core7|core6|
    //                    +-----+-----+-----+-----+-----+-----+-----+
    //                                       WCnt

    __aicore__ static inline void CalBlockGrid(uint32_t h, uint32_t w, uint16_t& outBlockH, uint16_t& outBlockW,
                                               uint32_t coreNum)
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
        uint16_t realCoreNum = static_cast<uint16_t>(coreNum);
        uint16_t bestH = 1;
        uint16_t bestW = realCoreNum;

        // 常用核数配置直接写死，不用再去跑一遍循环
        if (realCoreNum == CORE_NUM_32) {
            bestH = GRID_H_32C;
            bestW = GRID_W_32C;
        } else if (realCoreNum == CORE_NUM_28) {
            bestH = GRID_H_28C;
            bestW = GRID_W_28C;
        } else if (realCoreNum == CORE_NUM_36) {
            bestH = GRID_H_36C;
            bestW = GRID_W_36C;
        } else {
            for (uint16_t i = 1; i * i <= realCoreNum; i++) {
                if (realCoreNum % i == 0) {
                    bestH = i;
                    bestW = realCoreNum / i;
                }
            }
        }

        // 形状匹配：将较大的维度分配给张量中较大的那个轴，进一步减少跨行/跨列跳跃
        if (h >= w) {
            outBlockH = AscendC::Std::max(bestH, bestW);
            outBlockW = AscendC::Std::min(bestH, bestW);
        } else {
            outBlockH = AscendC::Std::min(bestH, bestW);
            outBlockW = AscendC::Std::max(bestH, bestW);
        }
    }

    __aicore__ inline SwizzleTopology2D(uint32_t h, uint32_t w, uint16_t blockH, uint16_t blockW, uint32_t coreNum)
        : h_(h),
          w_(w),
          blockH_(blockH),
          blockW_(blockW),
          fullSuperRows_(h / blockH),
          totalCnt_(h * w),
          coreNum_(coreNum)
    {}

    __aicore__ inline bool GetHW(uint32_t loopIdx, uint16_t coreId, uint32_t& outH, uint32_t& outW) const
    {
        uint32_t flattenIdx = loopIdx * coreNum_ + coreId;
        // 越界判断
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
        uint32_t startIdx = loopIdx * coreNum_;
        if (unlikely(startIdx >= totalCnt_)) {
            boundH = 0;
            boundW = 0;
            return;
        }

        uint32_t endIdx = AscendC::Std::min(startIdx + coreNum_, totalCnt_) - 1;

        uint32_t dummyH1, startW, startSuperIdx;
        ComputeHW(startIdx, dummyH1, startW, startSuperIdx);

        uint32_t dummyH2, endW, endSuperIdx;
        ComputeHW(endIdx, dummyH2, endW, endSuperIdx);

        // ================= W 轴边界检测,由于存在蛇形走位，需要按照奇偶额外判断 =================
        if (startSuperIdx == endSuperIdx) {
            // 1. 未换行：直接取最大值
            boundW = AscendC::Std::max(startW, endW);
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
                boundW = AscendC::Std::max(startW, endW);
            }
        }
        // =========================================================

        boundH = endSuperIdx * blockH_ + AscendC::Std::min(blockH_, h_ - endSuperIdx * blockH_) - 1;
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
        const uint32_t localBlockH = AscendC::Std::min(blockH_, h_ - superRowH);
        // 每个BlockHW里面按H方向优先递进,也就是连续核的范围为(H0,W0),(H1,W0),(H2,W0)
        // 列内 H 方向优先
        outH = superRowH + localIdx % localBlockH;
        const uint32_t forwardW = localIdx / localBlockH;
        // 蛇形走位：奇偶行反向
        outW = (superIdx % SNAKE_PATTERN_PERIOD == 0) ? forwardW : (w_ - 1 - forwardW);
        outSuperIdx = superIdx;
    }

    const uint32_t h_;
    const uint32_t w_;
    const uint16_t blockH_;
    const uint16_t blockW_;
    const uint32_t fullSuperRows_;
    const uint32_t totalCnt_;
    const uint32_t coreNum_;
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
    cRange.coutLength = valid ? AscendC::Std::min(singleShapeCout, cout - cRange.coutIdx) : 0;
    cRange.cinLength = valid ? AscendC::Std::min(singleShapeCin, cin - cRange.cinIdx) : 0;

    return valid;
}

template <BlockIterDirection IterDir>
class BlockIterator {
public:
    inline __aicore__ bool More() const { return loopIdx_ < blocksIterCnt_; }

    // 获取当前aic计算的基本块范围,若当前核无基本块计算则返回false并且将length设置为0
    inline __aicore__ bool GetLocalBlock(CoutCinRange& cRange) const { return GetBlock(AicCoreId(), cRange); }

    inline __aicore__ bool GetBlock(uint16_t coreId, CoutCinRange& cRange) const
    {
        return GetBlockFromSwizzle2D<IterDir>(topology_, loopIdx_, coreId, singleShapeCout_, singleShapeCin_, cout_,
                                              cin_, cRange);
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
        outCoutBound = AscendC::Std::min((boundCoutBlockIdx + 1) * singleShapeCout_, cout_);
        outCinBound = AscendC::Std::min((boundCinBlockIdx + 1) * singleShapeCin_, cin_);
    }

    inline __aicore__ void Next() { loopIdx_++; }

    inline __aicore__ const SwizzleTopology2D& GetSwizzleTopology() const { return topology_; }

    inline __aicore__ uint32_t GetTailBlockCnt() const
    {
        const uint32_t mainBlockNum = blocksIterCnt_ * coreNum_;
        return topology_.TotalCnt() > mainBlockNum ? topology_.TotalCnt() - mainBlockNum : 0;
    }

    // blockNum：核数来源，0 = GetBlockNum()（Create 内一次兜底），>0 = 显式指定
    static inline __aicore__ BlockIterator Create(bool onlyIterMainBlocks, uint32_t cout, uint32_t cin,
                                                  uint32_t singleShapeCout, uint32_t singleShapeCin,
                                                  uint32_t blockNum = 0)
    {
        if (blockNum == 0) {
            blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        }
        uint32_t coutCnt = Ops::Base::CeilDiv(cout, singleShapeCout);
        uint32_t cinCnt = Ops::Base::CeilDiv(cin, singleShapeCin);
        uint32_t topologyH = (IterDir == CIN) ? coutCnt : cinCnt;
        uint32_t topologyW = (IterDir == CIN) ? cinCnt : coutCnt;
        uint16_t blockH, blockW;
        SwizzleTopology2D::CalBlockGrid(topologyH, topologyW, blockH, blockW, blockNum);
        return BlockIterator(cout, cin, topologyH, topologyW, blockH, blockW, singleShapeCout, singleShapeCin,
                             onlyIterMainBlocks, blockNum);
    }

private:
    inline __aicore__ explicit BlockIterator(uint32_t cout, uint32_t cin, uint32_t topologyH, uint32_t topologyW,
                                             uint16_t topologyBlockH, uint16_t topologyBlockW, uint32_t singleShapeCout,
                                             uint32_t singleShapeCin, bool onlyIterMainBlocks, uint32_t blockNum)
        : cout_(cout),
          cin_(cin),
          singleShapeCout_(singleShapeCout),
          singleShapeCin_(singleShapeCin),
          topology_(topologyH, topologyW, topologyBlockH, topologyBlockW, blockNum),
          coreNum_(blockNum),
          blocksIterCnt_(GetBlockIterCnt(onlyIterMainBlocks, coreNum_, topology_.TotalCnt()))
    {}

    inline __aicore__ static uint32_t GetBlockIterCnt(bool onlyIterMainBlocks, uint32_t coreNum, uint32_t totalBlocks)
    {
        uint32_t mainIterCnt = totalBlocks / coreNum;
        uint32_t tailBlocks = totalBlocks - mainIterCnt * coreNum;

        if (onlyIterMainBlocks) {
            // 尾轮空闲核超过半数时留到后续切 k
            return tailBlocks > (coreNum / 2) ? mainIterCnt + 1 : mainIterCnt;
        } else {
            return mainIterCnt + (tailBlocks > 0 ? 1 : 0);
        }
    }

    const uint32_t cout_;
    const uint32_t cin_;
    const uint32_t singleShapeCout_;
    const uint32_t singleShapeCin_;
    const SwizzleTopology2D topology_;
    const uint32_t coreNum_;
    const uint32_t blocksIterCnt_;
    uint32_t loopIdx_ = 0;
};

} // namespace BpUtils

#endif // CONV_BP_COMMON_DATA_BLOCKS_H
