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

    // D轴配置(di/dout/kd/padF)构造时传入: kd作为基本块维度在IterateAll内按[kd, cout, cin]网格遍历,
    // 不对外暴露D轴设置接口, 本类包含完整运算逻辑
    __aicore__ inline ConvBackpropFilterWinograd(const WinoFmapFwdTransformer<SrcT, TilingT>& fmap,
                                                 const WinoDyFwdTransformer<SrcT, TilingT>& dy,
                                                 __gm__ SrcT* nk1c1k0c0Gm, NK1C1K0C0::Shape<SrcT>& nk1c1k0c0Shape,
                                                 __gm__ DstT* yGm, __gm__ float* tailGm,
                                                 WinoMMAD<SrcT, TilingT>& winoMmad, uint32_t tilesH, uint32_t tilesW,
                                                 uint32_t batch, uint32_t di, uint32_t dout, uint32_t kd, uint32_t padF,
                                                 const BlockConfig::RtTiling& rtTiling)
        : tilesH_(tilesH),
          tilesW_(tilesW),
          batch_(batch),
          di_(di),
          dout_(dout),
          kd_(kd),
          padF_(padF),
          cin_(fmap.SrcC()),
          cout_(dy.SrcC()),
          gm2l1_(nk1c1k0c0Gm, nk1c1k0c0Shape),
          rtTiling_(rtTiling),
          dwFwd_(fmap, dy, rtTiling),
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
        // dout作为归约轴全折叠进k轴(不预切有效区间): k维为[dout, batch, CeilDiv(tileH, SingleShapeTileH)],
        // pad区k步进在IterateK内按WinoIsValidD逐点跳过
        uint32_t cuttableK = dout_ * batch_ * Ops::Base::CeilDiv(tilesH_, singleShapeTileH);

        // kd为最外层基本块维度: 一个kd的cout/cin蛇形循环完成后进入下一个kd的cout/cin,
        // 各kd余下的尾块统一下放给尾轮切k(尾轮按kd串行分子轮处理, 见下方尾轮注释)
        auto blockIter = BlockIterator<BasicBlockIterDir>::Create(cuttableK > 1, cout_, cin_, rtTiling_.singleShapeCout,
                                                                  rtTiling_.singleShapeCin, 0, kd_);

        uint32_t watermarkResidentC = 0;
        uint32_t prevKdIdx = 0;

        while (blockIter.More()) {
            CoutCinRange localBlock;
            blockIter.GetLocalBlock(localBlock);

            uint32_t kdIdx = blockIter.GetLocalKdIdx();
            // kd切换后驻留staging需按新kd重写(fmap侧内容随kdIdx偏移, pad区跳过的槽位也需补齐),
            // 重置水位让新kd的首个覆盖轮重新写入
            if (kdIdx != prevKdIdx) {
                watermarkResidentC = 0;
                prevKdIdx = kdIdx;
            }

            uint32_t clusterCoutBound, clusterCinBound;
            blockIter.GetClusterBlockUpperBound(clusterCoutBound, clusterCinBound);
            uint32_t residentCBound = ResidentFmap ? clusterCinBound : clusterCoutBound;

            BatchTileKIterator<TilingT> kIter(batch_, tilesH_, tilesW_, 0, cuttableK);
            // 主轮不切K，cout整个轴在搬出时不做交织切分
            IterateK(localBlock, kIter, residentCBound, watermarkResidentC, {}, false, kdIdx);

            blockIter.Next();
            watermarkResidentC = Std::max(watermarkResidentC, residentCBound);
        }

        const uint32_t tailBlockCntPerKd = blockIter.GetTailBlockCnt();
        if (tailBlockCntPerKd == 0) {
            return;
        }

        // 尾轮切k按kd串行分子轮: 同一子轮内所有核kdIdx一致, pad区k步进跳过模式与驻留任务
        // taskOffset轮转保持全核同步, 驻留staging按该kd整体重写(watermark清零), 避免各kd尾块
        // 并发混跑时gm2l1全核barrier计数失配死锁与staging槽位互踩; 每个子轮内尾块统一切k,
        // 尾块累加写出仅执行一次。2D(kd=1)为单子轮且保留主轮watermark, 行为与原有尾轮一致
        const uint32_t tailWatermark = kd_ == 1 ? watermarkResidentC : 0;
        for (uint32_t tailKdIdx = 0; tailKdIdx < kd_; tailKdIdx++) {
            auto tailIter = TailBlockSplitKIterator<BasicBlockIterDir, TilingT>(
                tailBlockCntPerKd, blockIter.GetSwizzleTopology(), cuttableK, cout_, cin_, rtTiling_.singleShapeCin,
                rtTiling_.singleShapeCout);

            CoutCinRange localBlock;
            SplitKState splitKState;
            tailIter.GetLocalBlock(localBlock, splitKState);
            uint32_t residentCBound = ResidentFmap ? cin_ : cout_;

            BatchTileKIterator<TilingT> kIter(batch_, tilesH_, tilesW_, splitKState.kBegin, splitKState.kLength);

            IterateK<true>(localBlock, kIter, residentCBound, tailWatermark, splitKState,
                           // 切k不均衡时要补一轮同步
                           splitKState.kLength < splitKState.kMaxLength, tailKdIdx);

            // 尾轮kd子轮间必须全核隔离: 槽位按(blockId,kGroup)寻址跨子轮复用, 且子轮内首段为
            // 非原子覆写; 若无此屏障, 快核提前进入下一子轮的槽位写会污染慢核尚未完成的聚合读,
            // 导致先写子轮的y切片整块错值(实测SwinUnetr hf32用例kd0/kd1切片约12%元素非确定性
            // 精度错误, kd2恒正确)。kd=1为单子轮无跨子轮复用, 不加屏障保持原有行为
            if (kd_ > 1) {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::SyncAll<false>();
            }

            if ASCEND_IS_AIV {
                dwInv_.TailInterleaveWrite(localBlock, cin_, splitKState.kGroups, splitKState.kGroupIdx,
                                           splitKState.tailBlockId, tailKdIdx, kd_);
            }

            // 聚合读+CopyOut全部完成后才允许进入下一kd子轮(其首段会覆写槽位)
            if (kd_ > 1) {
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::SyncAll<false>();
            }
        }
    }

private:
    template <bool IsTailSplitK = false>
    inline __aicore__ void IterateK(const CoutCinRange& localBlock, WinoDetail::BatchTileKIterator<TilingT>& kIter,
                                    uint32_t residentCBound, uint32_t watermarkResidentC,
                                    const WinoDetail::SplitKState& splitKState, bool appendResidentCrossCoreSync,
                                    uint32_t kdIdx)
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
            // 本段是否存在合法(非D轴pad区)k步进: AIC/AIV用同一WinoIsValidD谓词对同一segment
            // 迭代得到一致结论; 整段均为pad区时本段无任何MmadCompute, 需将逆变换输入清零,
            // 避免Fixpipe搬出上一段残留L0C脏写y/重复累加(尾轮则写脏tailGm槽位)
            bool segmentHasValidK = false;
            if ASCEND_IS_AIC {
                segmentHasValidK = dwMmad_.IterateK(localBlock, segmentKIter, gm2l1_, ub2l1_, shouldResidentTransform,
                                                    kdIdx, di_, padF_);

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
                segmentHasValidK = dwFwd_.IterateK(localBlock, segmentKIter, gm2l1_, ub2l1_, watermarkResidentC,
                                                   residentCBound, IsTailSplitK ? splitKState.kGroupStartCoreId : 0,
                                                   IsTailSplitK ? splitKState.kGroupCoreNum : GetBlockNum(), kdIdx, di_,
                                                   padF_);

                if (appendTailKSync && segmentKIter.AllSegmentsHasDone()) {
                    for (uint32_t i = 0; i != kIter.StepInSingleK(); i++) {
                        gm2l1_.WaitSlot();
                        gm2l1_.EnQue();
                    }
                }
            }

            TransformOutput<IsTailSplitK>(localBlock, splitKState, !firstIter, kdIdx, segmentHasValidK);

            segmentKIter.ResetSegmentsLimit();
            firstIter = false;
        }
    }

    template <bool IsTailSplitK>
    __aicore__ inline void TransformOutput(const CoutCinRange& localBlock, const WinoDetail::SplitKState& splitKState,
                                           bool atomicAdd, uint32_t kdIdx, bool segmentHasValidK)
    {
        // 当前ub很难同时放下正变换和逆变换的速率，所以逆变换需要停掉整个正变换，并空出整个ub来逆变换，
        constexpr uint32_t invBufSize = WinoInvBufUtil::GetInvBufTotalSizeInBytes<TilingT>();
        static_assert(invBufSize < TOTAL_UB_SIZE, "illegal buffer size");
        auto invBuf = LocalTensor<float>(TPosition::VECIN, 0, invBufSize);

        if (likely(localBlock.NotEmpty())) {
            if ASCEND_IS_AIC {
                // 整段pad区时AIC仍照常Fixpipe(维持l0c2ubSync配对), 残留内容由AIV侧清零输入覆盖
                dwMmad_.Fixpipe2UB(l0c2ubSync_, localBlock, invBuf);
            }
            if ASCEND_IS_AIV {
                dwInv_.template TransformOutput<IsTailSplitK>(
                    l0c2ubSync_, localBlock, cin_, invBuf, splitKState.kGroupIdx, splitKState.kGroups,
                    splitKState.tailBlockId, atomicAdd, kdIdx, kd_, segmentHasValidK);
                dwInv_.BlockMTE2ByMTE3();
            }
        }
    }

    const uint32_t tilesH_;
    const uint32_t tilesW_;
    const uint32_t batch_;
    const uint32_t di_;
    const uint32_t dout_;
    const uint32_t kd_;
    const uint32_t padF_;
    const uint32_t cin_;
    const uint32_t cout_;

    WinoDetail::FwdTransformGM2L1Queue<SrcT> gm2l1_;
    WinoDetail::FwdTransformUB2L1Queue<SrcT> ub2l1_;
    WinoDetail::InvTransformL0C2UBSyncQueue<TilingT> l0c2ubSync_;
    const BlockConfig::RtTiling& rtTiling_;
    WinoDetail::AivFwdTransformer<SrcT, TilingT> dwFwd_;
    WinoDetail::AicMmadComputer<SrcT, TilingT> dwMmad_;
    WinoInvTransformer<DstT, TilingT> dwInv_;
};

#endif // CONV_BP_WINO_H
