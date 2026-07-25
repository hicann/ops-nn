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
 * \file conv_bp_wino.h
 * \brief
 */

#ifndef CONV_BP_WINO_H
#define CONV_BP_WINO_H

#include "conv_bp_wino_detail.h"
#include "conv_bp_wino_inv_transform.h"

using namespace AscendC;

template <typename SrcT, typename DstT, typename TilingT>
class ConvBackpropFilterWinograd {
public:
    static constexpr bool ResidentFmap = BlockConfig::ResidentTarget<TilingT>() == BlockConfig::InputTensor::FMAP;

    __aicore__ inline ConvBackpropFilterWinograd(const WinoFmapFwdTransformer<SrcT, TilingT>& fmap,
                                                 const WinoDyFwdTransformer<SrcT, TilingT>& dy,
                                                 __gm__ SrcT* nk1c1k0c0Gm, NK1C1K0C0::Shape<SrcT>& nk1c1k0c0Shape,
                                                 __gm__ DstT* yGm, __gm__ float* tailGm,
                                                 WinoMMAD<SrcT, TilingT>& winoMmad, uint32_t tilesH, uint32_t tilesW,
                                                 uint32_t batch)
        : tilesH_(tilesH),
          tilesW_(tilesW),
          batch_(batch),
          cin_(fmap.SrcC()),
          cout_(dy.SrcC()),
          gm2l1_(nk1c1k0c0Gm, nk1c1k0c0Shape),
          dwFwd_(fmap, dy),
          dwMmad_(winoMmad),
          dwInv_(yGm, tailGm)
    {}

    inline void __aicore__ Init()
    {
        if ASCEND_IS_AIV {
            dwFwd_.Init();
            dwInv_.Init();
        }
        dwMmad_.Init(ub2l1_);
    }

    inline void __aicore__ End()
    {
        if ASCEND_IS_AIV {
            dwFwd_.End();
        }

        gm2l1_.End();
        ub2l1_.End();
        l0c2ubSync_.End();
        dwMmad_.End();
    }

    inline void __aicore__ IterateAll()
    {
        using namespace WinoDetail;
        // 驻留fmap就往cout方向循环,减少执行驻留带来的全局同步影响
        constexpr BlockIterDirection BasicBlockIterDir = ResidentFmap ? COUT : CIN;

        constexpr uint32_t singleShapeTileH = BlockConfig::SingleShapeTileH<TilingT>();
        uint32_t cuttableK = batch_ * Ops::Base::CeilDiv(tilesH_, singleShapeTileH);

        // 可切k的话进行尾轮循环
        auto blockIter = BlockIterator<BasicBlockIterDir, TilingT>::Create(cuttableK > 1, cout_, cin_);

        uint32_t watermarkResidentC = 0;

        while (blockIter.More()) {
            CoutCinRange localBlock;
            blockIter.GetLocalBlock(localBlock);

            uint32_t clusterCoutBound, clusterCinBound;
            blockIter.GetClusterBlockUpperBound(clusterCoutBound, clusterCinBound);
            uint32_t residentCBound = ResidentFmap ? clusterCinBound : clusterCoutBound;

            BatchTileKIterator<TilingT> kIter(batch_, tilesH_, tilesW_, 0, cuttableK);
            // 主轮不切K，cout整个轴在搬出时不做交织切分
            IterateK(localBlock, kIter, residentCBound, watermarkResidentC, {}, false);

            blockIter.Next();
            watermarkResidentC = Std::max(watermarkResidentC, residentCBound);
        }

        if (blockIter.GetTailBlockCnt() == 0) {
            return;
        }

        auto tailIter = TailBlockSplitKIterator<BasicBlockIterDir, TilingT>(
            blockIter.GetTailBlockCnt(), blockIter.GetSwizzleTopology(), cuttableK, cout_, cin_);

        CoutCinRange localBlock;
        SplitKState splitKState;
        tailIter.GetLocalBlock(localBlock, splitKState);
        uint32_t residentCBound = ResidentFmap ? cin_ : cout_;

        BatchTileKIterator<TilingT> kIter(batch_, tilesH_, tilesW_, splitKState.kBegin, splitKState.kLength);

        IterateK<true>(localBlock, kIter, residentCBound, watermarkResidentC, splitKState,
                       // 切k不均衡时要补一轮同步
                       splitKState.kLength < splitKState.kMaxLength);

        if ASCEND_IS_AIV {
            dwInv_.TailInterleaveWrite(localBlock, cin_, splitKState.kGroups, splitKState.kGroupIdx,
                                       splitKState.tailBlockId);
        }
    }

private:
    template <bool IsTailSplitK = false>
    inline __aicore__ void IterateK(const CoutCinRange& localBlock, WinoDetail::BatchTileKIterator<TilingT>& kIter,
                                    uint32_t residentCBound, uint32_t watermarkResidentC,
                                    const WinoDetail::SplitKState& splitKState, bool appendResidentCrossCoreSync)
    {
        constexpr uint32_t singleShapeTileH = BlockConfig::SingleShapeTileH<TilingT>();
        constexpr uint32_t singleShapeTileW = BlockConfig::SingleShapeTileW<TilingT>();
        constexpr uint32_t ReduceKTileSegmentsLimit = 32768 / (singleShapeTileH * singleShapeTileW);
        WinoDetail::SegmentTileKIterator<TilingT> segmentKIter(ReduceKTileSegmentsLimit, kIter);

        bool shouldResidentTransform = residentCBound > watermarkResidentC;
        bool firstIter = true;

        bool appendTailKSync = false;
        if constexpr (IsTailSplitK) {
            appendTailKSync = appendResidentCrossCoreSync && shouldResidentTransform;
        }

        while (!segmentKIter.AllSegmentsHasDone()) {
            if ASCEND_IS_AIC {
                dwMmad_.IterateK(localBlock, segmentKIter, gm2l1_, ub2l1_, shouldResidentTransform);

                if (appendTailKSync && segmentKIter.AllSegmentsHasDone()) {
                    // 尾轮处理时切k不均衡需要额外补一次全核同步
                    // 要是芯片跨核同步支持分组不强制全核一起来就好了
                    for (uint32_t i = 0; i != kIter.StepInSingleK(); i++) {
                        gm2l1_.WaitData();
                        gm2l1_.DeQue();
                    }
                }
            }

            if ASCEND_IS_AIV {
                dwFwd_.IterateK(localBlock, segmentKIter, gm2l1_, ub2l1_, watermarkResidentC, residentCBound,
                                IsTailSplitK ? splitKState.kGroupStartCoreId : 0,
                                IsTailSplitK ? splitKState.kGroupCoreNum : GetBlockNum());

                if (appendTailKSync && segmentKIter.AllSegmentsHasDone()) {
                    for (uint32_t i = 0; i != kIter.StepInSingleK(); i++) {
                        gm2l1_.WaitSlot();
                        gm2l1_.EnQue();
                    }
                }
            }

            TransformOutput<IsTailSplitK>(localBlock, splitKState, !firstIter);

            segmentKIter.ResetSegmentsLimit();
            firstIter = false;
        }
    }

    template <bool IsTailSplitK>
    __aicore__ inline void TransformOutput(const CoutCinRange& localBlock, const WinoDetail::SplitKState& splitKState,
                                           bool atomicAdd)
    {
        // 当前ub很难同时放下正变换和逆变换的速率，所以逆变换需要停掉整个正变换，并空出整个ub来逆变换，
        constexpr uint32_t invBufSize = WinoInvBufUtil::GetInvBufTotalSizeInBytes<TilingT>();
        static_assert(invBufSize < TOTAL_UB_SIZE, "illegal buffer size");
        auto invBuf = LocalTensor<float>(TPosition::VECIN, 0, invBufSize);

        if (likely(localBlock.NotEmpty())) {
            if ASCEND_IS_AIC {
                dwMmad_.Fixpipe2UB(l0c2ubSync_, localBlock, invBuf);
            }
            if ASCEND_IS_AIV {
                dwInv_.template TransformOutput<IsTailSplitK>(l0c2ubSync_, localBlock, cin_, invBuf,
                                                              splitKState.kGroupIdx, splitKState.kGroups,
                                                              splitKState.tailBlockId, atomicAdd);
                dwInv_.BlockMTE2ByMTE3();
            }
        }
    }

    const uint32_t tilesH_;
    const uint32_t tilesW_;
    const uint32_t batch_;
    const uint32_t cin_;
    const uint32_t cout_;

    WinoDetail::FwdTransformGM2L1Queue<SrcT> gm2l1_;
    WinoDetail::FwdTransformUB2L1Queue<SrcT> ub2l1_;
    WinoDetail::InvTransformL0C2UBSyncQueue<TilingT> l0c2ubSync_;
    WinoDetail::AivFwdTransformer<SrcT, TilingT> dwFwd_;
    WinoDetail::AicMmadComputer<SrcT, TilingT> dwMmad_;
    WinoInvTransformer<DstT, TilingT> dwInv_;
};

#endif // CONV_BP_WINO_H
