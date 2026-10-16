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
 * \file bn3_d_training_update_channel_first.h
 * \brief channel-first(NCHW / NCDHW)布局类:x 视图 [N][C][S],S = 非通道轴之积。
 */

#ifndef BN3_D_TRAINING_UPDATE_CHANNEL_FIRST_H
#define BN3_D_TRAINING_UPDATE_CHANNEL_FIRST_H

#include <cstdint>
#include "kernel_operator.h"
#include "bn3_d_training_update_common.h"
#include "bn3_d_training_update_stat.h"
#include "bn3_d_training_update_tiling_data.h"

namespace Bn3dTrainingUpdateOps {

template <typename T_X>
class Bn3dTrainingUpdateChannelFirst {
public:
    __aicore__ inline Bn3dTrainingUpdateChannelFirst() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sum, GM_ADDR squareSum, GM_ADDR scale, GM_ADDR offset, GM_ADDR mean,
                                GM_ADDR variance, GM_ADDR y, GM_ADDR meanOut, GM_ADDR varianceOut, GM_ADDR batchMean,
                                GM_ADDR batchVariance, const Bn3dTrainingUpdateChannelFirstTilingData* tilingData,
                                TPipe* pipe)
    {
        tilingData_ = tilingData;
        pipe_ = pipe;
        xGm_.SetGlobalBuffer((__gm__ T_X*)x);
        yGm_.SetGlobalBuffer((__gm__ T_X*)y);
        sumGm_.SetGlobalBuffer((__gm__ float*)sum, tilingData->c);
        squareSumGm_.SetGlobalBuffer((__gm__ float*)squareSum, tilingData->c);
        scaleGm_.SetGlobalBuffer((__gm__ float*)scale, tilingData->c);
        offsetGm_.SetGlobalBuffer((__gm__ float*)offset, tilingData->c);

        const int64_t vlFp32 = static_cast<int64_t>(VL_FP32);
        const int64_t tileElements = tilingData->channelWindow * tilingData->spatialTile;
        pipe_->InitBuffer(xQueue_, kXyTileBufferNum,
                          XyTileBufferBytes(tileElements, static_cast<int64_t>(sizeof(T_X)), vlFp32, UB_BLOCK_BYTES));
        pipe_->InitBuffer(yQueue_, kXyTileBufferNum,
                          XyTileBufferBytes(tileElements, static_cast<int64_t>(sizeof(T_X)), vlFp32, UB_BLOCK_BYTES));
        chunkSlotBytes_ = ChunkSlotBytes(vlFp32, UB_BLOCK_BYTES);
        pipe_->InitBuffer(foldInBuf_, kFoldInputSlots * chunkSlotBytes_);
        pipe_->InitBuffer(foldScratchBuf_, kFoldScratchSlots * chunkSlotBytes_);
        // 系数窗 coeffMult / coeffBm / coeffOffset[0..K)(三路,非折叠式):按 roundUp(K, VL) 定价。
        coeffTileBytes_ = CoeffTileBytes(CoeffTileCap(tilingData->channelWindow, vlFp32), UB_BLOCK_BYTES);
        pipe_->InitBuffer(coeffBuf_, kCoeffTileBufferNum * coeffTileBytes_);
        chunkSlotElems_ = chunkSlotBytes_ / static_cast<int64_t>(sizeof(float));
        coeffTileElems_ = coeffTileBytes_ / static_cast<int64_t>(sizeof(float));

        Bn3dTrainingUpdateStatDesc desc;
        desc.channel = tilingData->c;
        desc.statUnitCount = tilingData->statUnitCount;
        desc.statChunk = tilingData->statChunk;
        desc.baseStatPerCore = tilingData->baseStatPerCore;
        desc.extraStatCount = tilingData->extraStatCount;
        desc.numRec = tilingData->numRec;
        desc.besselScaler = tilingData->besselScaler;
        desc.factor = tilingData->factor;
        desc.oneMinusFactor = tilingData->oneMinusFactor;
        desc.epsilon = tilingData->epsilon;
        stat_.Init(sum, squareSum, scale, offset, mean, variance, meanOut, varianceOut, batchMean, batchVariance, desc,
                   pipe_);
    }

    __aicore__ inline void Process()
    {
        // 相 S 先跑:它是唯一读 mean/variance、唯一写 4 路 (C,) 输出的地方,且与 x 的切分解耦
        // (num == 0 时 y 无元素,但 4 路输出仍被完整写出)。
        stat_.Process();
        PhaseX();
    }

private:
    __aicore__ inline __ubuf__ float* ChunkSlot(int64_t slot)
    {
        return (__ubuf__ float*)foldInBuf_.Get<float>()[slot * chunkSlotElems_].GetPhyAddr();
    }

    __aicore__ inline __ubuf__ float* CoeffSlot(int64_t slot)
    {
        return (__ubuf__ float*)coeffBuf_.Get<float>()[slot * coeffTileElems_].GetPhyAddr();
    }

    __aicore__ inline void StageFoldInput(const GlobalTensor<float>& gm, int64_t slot, int64_t gmOffset, int64_t count)
    {
        LocalTensor<float> ub = foldInBuf_.Get<float>()[slot * chunkSlotElems_];
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(ub, gm[gmOffset], params, pad);
    }

    // 通道窗 [cWin·K, cWin·K + planes) 的系数:4 路输入一次 staging + VFFoldChunk(len = planes)
    // (K ≤ kParamChunkLen ⇒ 一次折叠处理完整个窗,无需分块循环),结果直写 coeffMult/coeffBm/coeffOffset[0..planes)。
    // 缓存键是 cWin(系数只依赖通道窗);窗口长度变了(尾窗较短)也一并刷新。
    __aicore__ inline void EnsureCoeff(int64_t cWin, int64_t planes, int64_t& cachedCWin, int64_t& cachedPlanes)
    {
        if (cWin == cachedCWin && planes == cachedPlanes) {
            return;
        }
        event_t waitV2Mte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(waitV2Mte2);
        WaitFlag<HardEvent::V_MTE2>(waitV2Mte2);
        const int64_t cBegin = cWin * tilingData_->channelWindow;
        StageFoldInput(sumGm_, kFoldInSum, cBegin, planes);
        StageFoldInput(squareSumGm_, kFoldInSquareSum, cBegin, planes);
        StageFoldInput(scaleGm_, kFoldInScale, cBegin, planes);
        StageFoldInput(offsetGm_, kFoldInOffset, cBegin, planes);
        event_t waitMte2V = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(waitMte2V);
        WaitFlag<HardEvent::MTE2_V>(waitMte2V);
        asc_vf_call<VFFoldChunk>(
            ChunkSlot(kFoldInSum), ChunkSlot(kFoldInSquareSum), ChunkSlot(kFoldInScale), ChunkSlot(kFoldInOffset),
            CoeffSlot(kCoeffMultSlot), CoeffSlot(kCoeffBmSlot), CoeffSlot(kCoeffOffsetSlot),
            (__ubuf__ float*)foldScratchBuf_.Get<float>()[kFoldSaveVar * chunkSlotElems_].GetPhyAddr(),
            tilingData_->numRec, tilingData_->epsilon, static_cast<uint32_t>(planes));
        cachedCWin = cWin;
        cachedPlanes = planes;
    }

    // K 个连续通道平面 × count 个空间元素:一次跨步多行搬入(与 CL 同一个 idiom;
    // count == S 时 srcStride == 0,退化为连续搬运,同一份代码、无对齐/路由分支)。
    __aicore__ inline void CopyInContiguous(int64_t gmOffset, int64_t planes, int64_t count)
    {
        LocalTensor<T_X> x = xQueue_.AllocTensor<T_X>();
        DataCopyExtParams params{static_cast<uint16_t>(planes), static_cast<uint32_t>(count * sizeof(T_X)),
                                 (tilingData_->spatial - count) * static_cast<int64_t>(sizeof(T_X)), 0, 0};
        DataCopyPadExtParams<T_X> pad{false, 0, 0, 0};
        DataCopyPad<T_X, PaddingMode::Compact>(x, xGm_[gmOffset], params, pad);
        xQueue_.EnQue(x);
    }

    __aicore__ inline void ComputeChannelFirst(int64_t planes, int64_t count)
    {
        LocalTensor<T_X> x = xQueue_.DeQue<T_X>();
        LocalTensor<T_X> y = yQueue_.AllocTensor<T_X>();
        asc_vf_call<VFChannelFirst<T_X>>((__ubuf__ T_X*)x.GetPhyAddr(), (__ubuf__ T_X*)y.GetPhyAddr(),
                                         CoeffSlot(kCoeffMultSlot), CoeffSlot(kCoeffBmSlot),
                                         CoeffSlot(kCoeffOffsetSlot), static_cast<uint32_t>(planes),
                                         static_cast<uint32_t>(count));
        yQueue_.EnQue(y);
        xQueue_.FreeTensor(x);
    }

    __aicore__ inline void CopyOutContiguous(int64_t gmOffset, int64_t planes, int64_t count)
    {
        LocalTensor<T_X> y = yQueue_.DeQue<T_X>();
        DataCopyExtParams params{static_cast<uint16_t>(planes), static_cast<uint32_t>(count * sizeof(T_X)), 0,
                                 (tilingData_->spatial - count) * static_cast<int64_t>(sizeof(T_X)), 0};
        DataCopyPad<T_X, PaddingMode::Compact>(yGm_[gmOffset], y, params);
        yQueue_.FreeTensor(y);
    }

    __aicore__ inline void PhaseX()
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t tileCount = tilingData_->baseTilesPerCore + (blockIdx < tilingData_->extraCoreCount ? 1 : 0);
        if (tileCount <= 0) {
            return; // num == 0(空 batch):无 X 单元,相 S 已把 4 路输出写完
        }
        int64_t tileId = blockIdx * tilingData_->baseTilesPerCore +
                         (blockIdx < tilingData_->extraCoreCount ? blockIdx : tilingData_->extraCoreCount);
        const int64_t cWindows = (tilingData_->c + tilingData_->channelWindow - 1) / tilingData_->channelWindow;
        int64_t cachedCWin = -1;
        int64_t cachedPlanes = -1;
        for (int64_t i = 0; i < tileCount; ++i, ++tileId) {
            // 单元 = (n, 通道窗, 空间块):空间块快变 ⇒ 同一通道窗的所有空间块共享系数窗。
            const int64_t sIdx = tileId % tilingData_->spatialTiles;
            const int64_t w = tileId / tilingData_->spatialTiles;
            const int64_t cWin = w % cWindows;
            const int64_t n = w / cWindows;
            const int64_t cBegin = cWin * tilingData_->channelWindow;
            const int64_t remainChannels = tilingData_->c - cBegin;
            const int64_t planes = remainChannels < tilingData_->channelWindow ? remainChannels :
                                                                                 tilingData_->channelWindow;
            const int64_t sBegin = sIdx * tilingData_->spatialTile;
            const int64_t remain = tilingData_->spatial - sBegin;
            const int64_t count = remain < tilingData_->spatialTile ? remain : tilingData_->spatialTile;
            const int64_t gmOffset = (n * tilingData_->c + cBegin) * tilingData_->spatial + sBegin;
            EnsureCoeff(cWin, planes, cachedCWin, cachedPlanes);
            CopyInContiguous(gmOffset, planes, count);
            ComputeChannelFirst(planes, count);
            CopyOutContiguous(gmOffset, planes, count);
        }
    }

private:
    const Bn3dTrainingUpdateChannelFirstTilingData* tilingData_ = nullptr;
    TPipe* pipe_ = nullptr;
    int64_t chunkSlotBytes_ = 0;
    int64_t chunkSlotElems_ = 0;
    int64_t coeffTileBytes_ = 0;
    int64_t coeffTileElems_ = 0;
    Bn3dTrainingUpdateStat stat_;
    TBuf<TPosition::VECCALC> foldInBuf_;
    TBuf<TPosition::VECCALC> foldScratchBuf_;
    TBuf<TPosition::VECCALC> coeffBuf_;
    TQue<QuePosition::VECIN, kXyTileBufferNum> xQueue_;
    TQue<QuePosition::VECOUT, kXyTileBufferNum> yQueue_;
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_X> yGm_;
    GlobalTensor<float> sumGm_;
    GlobalTensor<float> squareSumGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> offsetGm_;
};

} // namespace Bn3dTrainingUpdateOps

#endif // BN3_D_TRAINING_UPDATE_CHANNEL_FIRST_H
