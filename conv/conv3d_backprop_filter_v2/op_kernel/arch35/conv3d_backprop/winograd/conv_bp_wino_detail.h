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
 * \file conv_bp_wino_detail.h
 * \brief
 */

#ifndef CONV_BP_WINO_DETAIL_H
#define CONV_BP_WINO_DETAIL_H

#include "conv_bp_wino_mmad.h"
#include "conv_bp_wino_transform_dy.h"
#include "conv_bp_wino_transform_fmap.h"
#include "conv_bp_wino_data_blocks.h"

using namespace AscendC;

namespace WinoDetail {
static constexpr uint8_t CROSS_CORE_AIC_SYNC_FLAG = 0;
static constexpr uint8_t CROSS_CORE_AIV2AIC_SEND_UB2GM_FLAG = 1;
static constexpr uint8_t CROSS_CORE_AIC2AIV_RECV_GM2L1_FLAG = 2;
static constexpr uint8_t CROSS_CORE_AIV2AIC_SEND_UB2L1_FLAG = 3;
static constexpr uint8_t CROSS_CORE_AIC2AIV_RECV_UB2L1_FLAG = 4;
static constexpr uint8_t CROSS_CORE_AIC2AIV_SEND_MMAD_DATA_FLAG = 5;
static constexpr uint8_t CROSS_CORE_AIC2AIV_RECV_MMAD_DATA_FLAG = 6;

template <typename T>
using FwdTransformGM2L1Queue = GM2L1Queue<T, CROSS_CORE_AIV2AIC_SEND_UB2GM_FLAG, CROSS_CORE_AIC2AIV_RECV_GM2L1_FLAG,
                                          CROSS_CORE_AIC_SYNC_FLAG>;

template <typename T>
using FwdTransformUB2L1Queue = UB2L1Queue<T, CROSS_CORE_AIV2AIC_SEND_UB2L1_FLAG, CROSS_CORE_AIC2AIV_RECV_UB2L1_FLAG>;

template <typename TilingT>
using InvTransformL0C2UBSyncQueue = CVSyncQue<
    CVSyncQueConfig<PIPE_FIX, PIPE_V, PIPE_MTE3, CROSS_CORE_AIC2AIV_SEND_MMAD_DATA_FLAG,
                    CROSS_CORE_AIC2AIV_RECV_MMAD_DATA_FLAG, BlockConfig::InvTransformBufCnt<TilingT>(), true> >;

template <typename T, typename TilingT>
class AivFwdTransformer {
public:
    static constexpr uint8_t BUF_CNT = BlockConfig::SingleTransformBufCnt<TilingT>();

    __aicore__ inline AivFwdTransformer(const WinoFmapFwdTransformer<T, TilingT>& fmapFwd,
                                        const WinoDyFwdTransformer<T, TilingT>& dyFwd)
        : fmapFwd_(fmapFwd), dyFwd_(dyFwd)
    {}

    __aicore__ inline void Init()
    {
        constexpr uint32_t fwdTmpBufSize = GetFwdTmpBufSize() * sizeof(T);
        constexpr uint32_t fwdSrcBufSize = GetFwdSrcBufSize() * sizeof(T) * BUF_CNT;
        constexpr uint32_t fwdOutBufSize = GetFwdOutBufSize() * sizeof(T) * BUF_CNT;

        TBuf<TPosition::VECIN> transformFwdTmpBuf;
        TBuf<TPosition::VECIN> transformFwdSrcBuf;
        TBuf<TPosition::VECIN> transformFwdOutBuf;

        static_assert((fwdTmpBufSize + fwdSrcBufSize + fwdOutBufSize) < TOTAL_UB_SIZE, "illegal buffer size");

        TPipe* pipe = GetTPipePtr();
        pipe->InitBuffer(transformFwdTmpBuf, fwdTmpBufSize);
        pipe->InitBuffer(transformFwdSrcBuf, fwdSrcBufSize);
        pipe->InitBuffer(transformFwdOutBuf, fwdOutBufSize);
        transformFwdTmpVBuf_ = transformFwdTmpBuf.Get<T>();
        transformFwdSrcVBuf_ = transformFwdSrcBuf.Get<T>();
        transformFwdOutVBuf_ = transformFwdOutBuf.Get<T>();

        for (uint8_t i = 0; i < BUF_CNT; i++) {
            transformFwdEventFlags_[i] = TransformVFlag::AllocEventId(pipe);
            SetFlag<HardEvent::V_MTE2>(transformFwdEventFlags_[i].v2mte2);
            SetFlag<HardEvent::MTE3_V>(transformFwdEventFlags_[i].mte32v);
        }
    }

    __aicore__ inline void IterateK(const CoutCinRange& localBlock, SegmentTileKIterator<TilingT>& kIter,
                                    FwdTransformGM2L1Queue<T>& gm2l1Que, FwdTransformUB2L1Queue<T>& ub2l1Que,
                                    uint32_t watermarkResidentC, uint32_t residentCBound,
                                    uint16_t residentKGroupStartCoreIdx, uint16_t residentKGroupCoreNum)
    {
        using BlockConfig::InputTensor;
        constexpr InputTensor ResidentTarget = BlockConfig::ResidentTarget<TilingT>();
        constexpr InputTensor TensorT0 = ResidentTarget != InputTensor::FMAP ? InputTensor::FMAP : InputTensor::DY;
        constexpr InputTensor TensorT1 = ResidentTarget == InputTensor::FMAP ? InputTensor::FMAP : InputTensor::DY;

        StreamTaskInfo streamT0;
        ComputeT0TaskInfo<TensorT0>(localBlock, streamT0);

        StreamTaskInfo streamT1;
        ResidentTaskInfo residentT1;
        ComputeT1TaskInfo<TensorT1>(localBlock, residentCBound, watermarkResidentC, streamT1, residentT1);

        uint32_t residentTaskOffset = 0;
        while (kIter.More()) {
            HWBox tile = kIter.TileBox();

            if (residentCBound > watermarkResidentC) {
                typename TransformFunctions::GM2L1Ctx gm2l1Ctx = {kIter.BatchIdx(), kIter.TileKIdx(), {gm2l1Que}};
                gm2l1Que.WaitSlot();

                ProcessResidentTransform<TensorT1>(tile, gm2l1Ctx, residentT1, residentTaskOffset,
                                                   residentKGroupStartCoreIdx, residentKGroupCoreNum);

                gm2l1Que.EnQue();
            }

            typename TransformFunctions::UB2L1Ctx ub2l1Ctx = {kIter.BatchIdx(), kIter.TileKIdx(), {ub2l1Que, 0}};
            ub2l1Que.WaitSlot();

            ProcessStreamingTransform<TensorT1>(tile, ub2l1Ctx, streamT1);

            ProcessStreamingTransform<TensorT0>(tile, ub2l1Ctx, streamT0);

            ub2l1Que.EnQue();

            kIter.Next();
        }
    }

    __aicore__ inline void End()
    {
        // 不wait看文档说状态会残留?
        for (uint8_t i = 0; i < BUF_CNT; i++) {
            WaitFlag<HardEvent::V_MTE2>(transformFwdEventFlags_[i].v2mte2);
            WaitFlag<HardEvent::MTE3_V>(transformFwdEventFlags_[i].mte32v);
        }
    }

private:
    struct StreamTaskInfo;
    struct ResidentTaskInfo;

    template <BlockConfig::InputTensor TensorT0>
    __aicore__ inline void ComputeT0TaskInfo(const CoutCinRange& localBlock, StreamTaskInfo& stream) const
    {
        // 当前非驻留矩阵区域
        uint32_t localCIdx = localBlock.GetIdx<TensorT0>();
        uint16_t localCLen = localBlock.GetLen<TensorT0>();
        stream.cLocalIdx = localCIdx;
        stream.cIdx = localCIdx;
        stream.cLen = localCLen;
        stream.singleCoreCLen = Ops::Base::CeilDiv(Ops::Base::CeilDiv(stream.cLen, C0<T>()), AivNumInBlock()) * C0<T>();
    }

    template <BlockConfig::InputTensor TensorT1>
    __aicore__ inline void ComputeT1TaskInfo(const CoutCinRange& localBlock, uint32_t residentCBound,
                                             uint32_t watermarkResidentC, StreamTaskInfo& stream,
                                             ResidentTaskInfo& resident) const
    {
        uint32_t localCIdx = localBlock.GetIdx<TensorT1>();
        uint16_t localCLen = localBlock.GetLen<TensorT1>();

        //[resident,stream]
        stream.cLocalIdx = localCIdx;
        stream.cIdx = localCIdx + SingleShapeResidentC;
        stream.cLen = Std::max(localCLen, SingleShapeResidentC) - SingleShapeResidentC;
        stream.singleCoreCLen = Ops::Base::CeilDiv(Ops::Base::CeilDiv(stream.cLen, C0<T>()), AivNumInBlock()) * C0<T>();

        if (residentCBound > watermarkResidentC) {
            constexpr uint16_t singleShapeC = BlockConfig::SingleShapeC<TilingT, TensorT1>();
            uint32_t t1FullCLen = residentCBound - watermarkResidentC;
            uint32_t t1MainCBlk = t1FullCLen / singleShapeC;
            uint16_t t1TailCLen = t1FullCLen % singleShapeC;

            resident.cIdx = watermarkResidentC;
            resident.singleShapeTailC = t1TailCLen;
            resident.tailCTaskCnt = Ops::Base::CeilDiv(Std::min(t1TailCLen, SingleShapeResidentC),
                                                       SingleShapeTransformC);
            resident.cTaskCnt = resident.tailCTaskCnt + t1MainCBlk * TaskPerSingleResidentC;
        }
    }

    struct TransformFunctions {
        struct GM2L1 {
            FwdTransformGM2L1Queue<T>& queue;
        };

        struct UB2L1 {
            FwdTransformUB2L1Queue<T>& queue;
            uint32_t ub2l1Offset = 0;
        };

        template <typename L1Method>
        struct Context {
            uint32_t batchIdx = 0;
            uint32_t kIdx = 0;
            L1Method l1method;

            __aicore__ inline auto& GetL1Queue() { return l1method.queue; }
        };

        using GM2L1Ctx = Context<GM2L1>;
        using UB2L1Ctx = Context<UB2L1>;

        template <typename TransformConfig, typename L1Method>
        __aicore__ inline static void CopyIn(const WinoTransformer<TransformConfig>& transformer, const TileBox& box,
                                             Context<L1Method>& ctx, LocalTensor<T>& transformFwdSrcVBuf)
        {
            transformer.CopyIn(transformFwdSrcVBuf, box, ctx.batchIdx);
        }

        template <typename TransformConfig, typename L1Method>
        __aicore__ inline static void Compute(const WinoTransformer<TransformConfig>& transformer, const TileBox& box,
                                              Context<L1Method>& dummy, LocalTensor<T>& transformFwdSrcVBuf,
                                              LocalTensor<T>& transformFwdOutVBuf, LocalTensor<T>& transformFwdTmpVBuf)
        {
            transformer.Compute(transformFwdSrcVBuf, transformFwdOutVBuf, transformFwdTmpVBuf, box);
        }

        template <typename TransformConfig, typename L1Method>
        __aicore__ inline static void CopyOut(const WinoTransformer<TransformConfig>& transformer, const TileBox& box,
                                              Context<L1Method>& ctx, LocalTensor<T>& transformFwdOutVBuf)
        {
            NK1C1K0C0::CopyK0Params ckp;
            ckp.batchIdx = ctx.batchIdx;
            ckp.k1Idx = ctx.kIdx;
            transformer.SetNK1C1K0C0CopyParams(ckp, box);

            if constexpr (Std::is_same_v<L1Method, GM2L1>) {
                GM2L1& gm2l1 = ctx.l1method;
                gm2l1.queue.Write(ckp, transformFwdOutVBuf);
            } else {
                UB2L1& ub2l1 = ctx.l1method;
                if constexpr (Std::is_same_v<TransformConfig, WinoTransformDetail::DyConfig<T, TilingT> >) {
                    ub2l1.queue.WriteDy(ckp, transformFwdOutVBuf, ub2l1.ub2l1Offset);
                } else {
                    ub2l1.queue.WriteFmap(ckp, transformFwdOutVBuf, ub2l1.ub2l1Offset);
                }
            }
        }
    };

    struct ResidentTaskInfo {
        uint32_t cIdx;
        uint32_t cTaskCnt;
        uint16_t singleShapeTailC;
        uint16_t tailCTaskCnt;
    };

    template <BlockConfig::InputTensor TransformType>
    __aicore__ inline void ProcessResidentTransform(const HWBox& tile, typename TransformFunctions::GM2L1Ctx& ctx,
                                                    const ResidentTaskInfo& task, uint32_t& taskOffset,
                                                    uint16_t residentKGroupStartCoreIdx, uint16_t residentKGroupCoreNum)
    {
        using TransformConfig = Std::conditional_t<TransformType == BlockConfig::InputTensor::FMAP,
                                                   WinoTransformDetail::FmapConfig<T, TilingT>,
                                                   WinoTransformDetail::DyConfig<T, TilingT> >;

        const uint32_t coreId = AivCoreId() - residentKGroupStartCoreIdx * AivNumInBlock();
        const uint32_t stride = residentKGroupCoreNum * AivNumInBlock();

        for (uint32_t taskId = (coreId + stride - taskOffset) % stride; taskId < task.cTaskCnt; taskId += stride) {
            uint32_t cBlockIdx = taskId / TaskPerSingleResidentC;
            uint32_t taskIdxInCBlock = taskId % TaskPerSingleResidentC;

            uint32_t cBlockOffset = cBlockIdx * BlockConfig::SingleShapeC<TilingT, TransformType>();
            uint32_t offsetInCBlock = taskIdxInCBlock * SingleShapeTransformC;
            bool isTailTask = taskId >= task.cTaskCnt - task.tailCTaskCnt;
            uint32_t cLengthInBlock = SingleShapeResidentC;
            if (isTailTask) {
                cLengthInBlock = Std::min(SingleShapeResidentC, task.singleShapeTailC);
            }
            Execute(GetTransformer<TransformType>(), ctx,
                    TransformFunctions::template CopyIn<TransformConfig, typename TransformFunctions::GM2L1>,
                    TransformFunctions::template Compute<TransformConfig, typename TransformFunctions::GM2L1>,
                    TransformFunctions::template CopyOut<TransformConfig, typename TransformFunctions::GM2L1>, tile,
                    task.cIdx + cBlockOffset, offsetInCBlock, cLengthInBlock);
        }

        taskOffset = (taskOffset + task.cTaskCnt) % stride;
    }

    struct StreamTaskInfo {
        uint32_t cLocalIdx;
        uint32_t cIdx;
        uint32_t cLen;
        uint16_t singleCoreCLen;
    };

    template <BlockConfig::InputTensor TransformType>
    __aicore__ inline void ProcessStreamingTransform(const HWBox& tile, typename TransformFunctions::UB2L1Ctx& ctx,
                                                     const StreamTaskInfo& tasks)
    {
        using TransformConfig = Std::conditional_t<TransformType == BlockConfig::InputTensor::FMAP,
                                                   WinoTransformDetail::FmapConfig<T, TilingT>,
                                                   WinoTransformDetail::DyConfig<T, TilingT> >;

        const uint32_t cOffset = GetSubBlockIdx() * tasks.singleCoreCLen;
        uint32_t cIdx = tasks.cIdx + cOffset;
        uint32_t cLength = Std::min(tasks.singleCoreCLen, tasks.cLen - cOffset);

        for (uint32_t c = 0; c < cLength; c += SingleShapeTransformC) {
            // c一定是C0对齐，所以tile元素直接乘上c值就行
            ctx.l1method.ub2l1Offset = tile.elements * F23_TRANSFORM_TILE_ELEMENTS_16 * (cIdx + c - tasks.cLocalIdx);

            Execute(GetTransformer<TransformType>(), ctx,
                    TransformFunctions::template CopyIn<TransformConfig, typename TransformFunctions::UB2L1>,
                    TransformFunctions::template Compute<TransformConfig, typename TransformFunctions::UB2L1>,
                    TransformFunctions::template CopyOut<TransformConfig, typename TransformFunctions::UB2L1>, tile,
                    cIdx, c, cLength);
        }
    }

    template <typename TransformConfig, typename Ctx, typename CopyIn, typename Compute, typename CopyOut>
    __aicore__ inline void Execute(const WinoTransformer<TransformConfig>& transformer, Ctx& ctx, CopyIn copyIn,
                                   Compute compute, CopyOut copyOut, const HWBox& tile, uint32_t cIdx,
                                   uint32_t cStartOffset, uint32_t cLength)
    {
        constexpr uint32_t srcBufLen = GetFwdSrcBufSize();
        constexpr uint32_t outBufLen = GetFwdOutBufSize();
        LocalTensor<T> transformFwdSrcVBuf = transformFwdSrcVBuf_[bufIndex * srcBufLen];
        LocalTensor<T> transformFwdOutVBuf = transformFwdOutVBuf_[bufIndex * outBufLen];
        const TransformVFlag& eventFlag = transformFwdEventFlags_[bufIndex];

        constexpr uint32_t singleShapeTransformC = BlockConfig::SingleTransformC1<TilingT>() * C0<T>();
        uint32_t cStartIdx = cIdx + cStartOffset;
        uint32_t cExeLength = Std::min(singleShapeTransformC, cIdx + cLength - cStartIdx);
        const TileBox box = transformer.CalculateSrcBox(tile, cStartIdx, cExeLength);

        WaitFlag<HardEvent::V_MTE2>(eventFlag.v2mte2);

        copyIn(transformer, box, ctx, transformFwdSrcVBuf);

        SetFlag<HardEvent::MTE2_V>(eventFlag.mte22v);

        WaitFlag<HardEvent::MTE2_V>(eventFlag.mte22v);
        WaitFlag<HardEvent::MTE3_V>(eventFlag.mte32v);

        compute(transformer, box, ctx, transformFwdSrcVBuf, transformFwdOutVBuf, transformFwdTmpVBuf_);

        SetFlag<HardEvent::V_MTE2>(eventFlag.v2mte2);
        SetFlag<HardEvent::V_MTE3>(eventFlag.v2mte3);

        WaitFlag<HardEvent::V_MTE3>(eventFlag.v2mte3);

        copyOut(transformer, box, ctx, transformFwdOutVBuf);

        SetFlag<HardEvent::MTE3_V>(eventFlag.mte32v);
        bufIndex = (bufIndex + 1) % BUF_CNT;
    }

    static constexpr __aicore__ inline uint32_t GetFwdTmpBufSize()
    {
        constexpr uint32_t t0 = WinoTransformDetail::GetTmpBufLength<WinoTransformDetail::FmapConfig<T, TilingT> >();
        constexpr uint32_t t1 = WinoTransformDetail::GetTmpBufLength<WinoTransformDetail::DyConfig<T, TilingT> >();
        return ConstexprMaths::Max(t0, t1);
    }

    static constexpr __aicore__ inline uint32_t GetFwdSrcBufSize()
    {
        constexpr uint32_t t0 = WinoTransformDetail::GetInputBufSize<WinoTransformDetail::FmapConfig<T, TilingT> >();
        constexpr uint32_t t1 = WinoTransformDetail::GetInputBufSize<WinoTransformDetail::DyConfig<T, TilingT> >();
        return ConstexprMaths::Max(t0, t1);
    }

    static constexpr __aicore__ inline uint32_t GetFwdOutBufSize()
    {
        constexpr uint32_t
            t0 = WinoTransformDetail::GetTransformBufSize<WinoTransformDetail::FmapConfig<T, TilingT> >();
        constexpr uint32_t t1 = WinoTransformDetail::GetTransformBufSize<WinoTransformDetail::DyConfig<T, TilingT> >();
        return ConstexprMaths::Max(t0, t1);
    }

    template <BlockConfig::InputTensor t>
    __aicore__ inline auto& GetTransformer() const
    {
        if constexpr (t == BlockConfig::InputTensor::FMAP) {
            return fmapFwd_;
        } else if (t == BlockConfig::InputTensor::DY) {
            return dyFwd_;
        }
    }

    struct TransformVFlag {
        TEventID mte22v;
        TEventID v2mte2;
        TEventID mte32v;
        TEventID v2mte3;

        static __aicore__ inline TransformVFlag AllocEventId(TPipe* pipe)
        {
            return {pipe->AllocEventID<HardEvent::MTE2_V>(), pipe->AllocEventID<HardEvent::V_MTE2>(),
                    pipe->AllocEventID<HardEvent::MTE3_V>(), pipe->AllocEventID<HardEvent::V_MTE3>()};
        }
    };

    static constexpr uint16_t SingleShapeResidentC = BlockConfig::SingleShapeResidentC<TilingT>();
    static constexpr uint16_t SingleShapeTransformC = BlockConfig::SingleTransformC1<TilingT>() * C0<T>();
    static constexpr uint16_t TaskPerSingleResidentC = ConstexprMaths::CeilDiv(SingleShapeResidentC,
                                                                               SingleShapeTransformC);

    const WinoFmapFwdTransformer<T, TilingT>& fmapFwd_;
    const WinoDyFwdTransformer<T, TilingT>& dyFwd_;

    LocalTensor<T> transformFwdTmpVBuf_;
    LocalTensor<T> transformFwdSrcVBuf_;
    LocalTensor<T> transformFwdOutVBuf_;
    TransformVFlag transformFwdEventFlags_[BUF_CNT];

    uint8_t bufIndex = 0;
};

template <typename T, typename TilingT>
class AicMmadComputer {
public:
    static constexpr BlockConfig::InputTensor ResidentTarget = BlockConfig::ResidentTarget<TilingT>();

    __aicore__ inline explicit AicMmadComputer(WinoMMAD<T, TilingT>& winoMmad) : winoMmad_(winoMmad) {}

    inline void __aicore__ Init(FwdTransformUB2L1Queue<T>& ub2l1)
    {
        winoMmad_.Init();
        auto l1BufPing = winoMmad_.GetL1Buf(false);
        auto l1BufPong = winoMmad_.GetL1Buf(true);

        constexpr uint8_t FMAP_BUF_IDX = 1;
        constexpr uint8_t DY_BUF_IDX = 0;
        LocalTensor<T> l1FmapBuf[2] = {Std::get<FMAP_BUF_IDX>(l1BufPing), Std::get<FMAP_BUF_IDX>(l1BufPong)};
        LocalTensor<T> l1DyBuf[2] = {Std::get<DY_BUF_IDX>(l1BufPing), Std::get<DY_BUF_IDX>(l1BufPong)};

        ub2l1.Init(l1FmapBuf, l1DyBuf);
    }

    inline void __aicore__ End() { winoMmad_.End(); }

    __aicore__ inline void IterateK(const CoutCinRange& blockRange, SegmentTileKIterator<TilingT>& kIter,
                                    FwdTransformGM2L1Queue<T>& gm2l1, FwdTransformUB2L1Queue<T>& ub2l1,
                                    bool waitResidentTransform)
    {
        if (blockRange.NotEmpty()) {
            RunMmad<true>(blockRange, kIter, gm2l1, ub2l1, waitResidentTransform);
        } else {
            // 闲置核仅参与 Queue 信号同步，维持集群流水线运转，不进行实际 Compute
            RunMmad<false>(blockRange, kIter, gm2l1, ub2l1, waitResidentTransform);
        }
    }

    __aicore__ inline void Fixpipe2UB(InvTransformL0C2UBSyncQueue<TilingT>& syncQue, const CoutCinRange& localBlock,
                                      const LocalTensor<float>& invBuf)
    {
        winoMmad_.Fixpipe2UB(syncQue, localBlock.coutLength, localBlock.cinLength, invBuf);
    }

private:
    template <bool NotIdle>
    __aicore__ inline void RunMmad(const CoutCinRange& cRange, SegmentTileKIterator<TilingT>& iter,
                                   FwdTransformGM2L1Queue<T>& gm2l1, FwdTransformUB2L1Queue<T>& ub2l1,
                                   bool waitResidentTransform)
    {
        static constexpr uint16_t SingleShapeResidentC = BlockConfig::SingleShapeResidentC<TilingT>();
        uint32_t coutC1Length = 0;
        uint32_t cinC1Length = 0;
        uint32_t residentC1Idx = 0;
        uint32_t residentC1Length = 0;

        if constexpr (NotIdle) {
            coutC1Length = Ops::Base::CeilDiv(cRange.coutLength, C0<T>());
            cinC1Length = Ops::Base::CeilDiv(cRange.cinLength, C0<T>());

            residentC1Idx = cRange.GetIdx<ResidentTarget>() / C0<T>();
            residentC1Length = Ops::Base::CeilDiv(Std::min(cRange.GetLen<ResidentTarget>(), SingleShapeResidentC),
                                                  C0<T>());
        }

        if (likely(iter.More())) {
            HWBox tiles = iter.TileBox();

            // ================= 阶段 1: Prologue (预载入第一轮数据) =================
            MmadLoadResident<NotIdle>(tiles, gm2l1, iter.BatchIdx(), iter.TileKIdx(), residentC1Idx, residentC1Length,
                                      waitResidentTransform, loadPingPong_);

            iter.Next();
            bool firstK = true;
            // ================= 阶段 2: Steady State (计算当前轮 + 预载入下一轮) =================
            while (iter.More()) {
                // winograd每个点位需要执行16次独立的mad计算
                // 由于dav上cube的issue queue大小为16,算上wait flag
                // 如果一次最多塞入8条mad指令后就会阻塞,进而block住整个scalar
                // 即便按批一次处理4个点，那么加上一个wait flag,也最多处理12个点就block住
                // 让下一轮的mte2无法执行,导致整体串行化
                // 所以这里用预取下一轮的数据的方式来解决
                //
                //  首次Compute前直接下发PingPong两块L1的搬运指令:
                //   LoadL1 Ping
                //   LoadL1 Pong
                //
                //  然后在L1Ping上做计算,此时scalar单元会由于issue queue满而被阻塞
                //   Compute Ping (block scalar)
                //
                //  ComputePing的scalar执行完后在下发L1Ping的搬运指令,即便scalar被卡主也没关系,因为
                //  L1Ping搬入时为了下下轮计算,下一轮所需要的L1Pong已经被预载了
                //   LoadL1 Ping
                //
                //  下发L1Pong的计算指令,由于L1Pong的搬运指令已经提前下发,所以在vector正变换更得上的情况下L1Pong应该搬运的差不多了
                //  ComputePong可以立刻执行
                //   Compute Pong
                HWBox nextTiles = iter.TileBox();

                MmadLoadResident<NotIdle>(nextTiles, gm2l1, iter.BatchIdx(), iter.TileKIdx(), residentC1Idx,
                                          residentC1Length, waitResidentTransform, loadPingPong_);

                MmadCompute<NotIdle>(tiles, ub2l1, cRange.coutLength, coutC1Length, cRange.cinLength, cinC1Length,
                                     firstK, computePingPong_);

                firstK = false;
                tiles = nextTiles;

                iter.Next();
            }

            // ================= 阶段 3: Epilogue (计算最后一轮数据) =================
            MmadCompute<NotIdle>(tiles, ub2l1, cRange.coutLength, coutC1Length, cRange.cinLength, cinC1Length, firstK,
                                 computePingPong_);
        }
    }

    template <bool NotIdle>
    __aicore__ inline void MmadLoadResident(const HWBox& tiles, FwdTransformGM2L1Queue<T>& gm2l1, uint32_t batchIdx,
                                            uint32_t k1Idx, uint32_t c1Idx, uint32_t c1Length,
                                            bool waitResidentFinished, bool& l1PingPongFlag)
    {
        if (waitResidentFinished) {
            gm2l1.WaitData();
        }

        if constexpr (NotIdle) {
            NK1C1K0C0::CopyK0Params params;
            params.tiles = tiles.elements;
            params.batchIdx = batchIdx;
            params.k1Idx = k1Idx;
            params.c1Idx = c1Idx;
            params.c1Length = c1Length;

            winoMmad_.template LoadL1<ResidentTarget>(gm2l1.GetGlobalTensor(), gm2l1.GetGMShape(), params,
                                                      l1PingPongFlag);

            l1PingPongFlag = !l1PingPongFlag;
        }

        if (waitResidentFinished) {
            gm2l1.DeQue();
        }
    }

    template <bool NotIdle>
    __aicore__ inline void MmadCompute(const HWBox& tiles, FwdTransformUB2L1Queue<T>& ub2l1, uint32_t cout,
                                       uint32_t coutC1, uint32_t cin, uint32_t cinC1, bool firstK, bool& l1PingPongFlag)
    {
        // 阻塞等待 AIV 的 DY 生产信号
        ub2l1.WaitData();

        if constexpr (NotIdle) {
            winoMmad_.Compute(tiles, cout, coutC1, cin, cinC1, firstK, l1PingPongFlag);
            // pingpong和ub2l1更新同步
            l1PingPongFlag = !l1PingPongFlag;
        }

        ub2l1.DeQue();
    }

    WinoMMAD<T, TilingT>& winoMmad_;
    bool loadPingPong_ = false;
    bool computePingPong_ = false;
};
} // namespace WinoDetail

#endif // CONV_BP_WINO_DETAIL_H
