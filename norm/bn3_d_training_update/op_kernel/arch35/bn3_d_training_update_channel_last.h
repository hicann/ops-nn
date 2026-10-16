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
 * \file bn3_d_training_update_channel_last.h
 * \brief channel-last(NHWC / NDHWC)布局类:x 视图 [num][C],num = N*S(拍平成 2D,与 rank 无关)。
 */

#ifndef BN3_D_TRAINING_UPDATE_CHANNEL_LAST_H
#define BN3_D_TRAINING_UPDATE_CHANNEL_LAST_H

#include <cstdint>
#include "kernel_operator.h"
#include "bn3_d_training_update_common.h"
#include "bn3_d_training_update_stat.h"
#include "bn3_d_training_update_tiling_data.h"

namespace Bn3dTrainingUpdateOps {

template <typename T_X>
class Bn3dTrainingUpdateChannelLast {
public:
    __aicore__ inline Bn3dTrainingUpdateChannelLast() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sum, GM_ADDR squareSum, GM_ADDR scale, GM_ADDR offset, GM_ADDR mean,
                                GM_ADDR variance, GM_ADDR y, GM_ADDR meanOut, GM_ADDR varianceOut, GM_ADDR batchMean,
                                GM_ADDR batchVariance, const Bn3dTrainingUpdateChannelLastTilingData* tilingData,
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
        const int64_t tileElements = tilingData->rowsPerTile * tilingData->channelBlock;
        pipe_->InitBuffer(xQueue_, kXyTileBufferNum,
                          XyTileBufferBytes(tileElements, static_cast<int64_t>(sizeof(T_X)), vlFp32, UB_BLOCK_BYTES));
        pipe_->InitBuffer(yQueue_, kXyTileBufferNum,
                          XyTileBufferBytes(tileElements, static_cast<int64_t>(sizeof(T_X)), vlFp32, UB_BLOCK_BYTES));
        chunkSlotBytes_ = ChunkSlotBytes(vlFp32, UB_BLOCK_BYTES);
        pipe_->InitBuffer(foldInBuf_, kFoldInputSlots * chunkSlotBytes_);
        pipe_->InitBuffer(foldScratchBuf_, kFoldScratchSlots * chunkSlotBytes_);
        coeffTileBytes_ = CoeffTileBytes(tilingData->coeffCap, UB_BLOCK_BYTES);
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

    // 折叠通道块 [cBegin, cBegin + cLen) 的系数:按 kParamChunk 分块调 VFFoldChunk,结果直写
    // coeffMult/coeffBm/coeffOffset 的 [0, cLen)(三路系数,非折叠式)。**不再做任何周期展开** ——
    // 热循环逐行处理,行内系数下标 与行内元素下标一一对应,直接用本体即可(见 VFChannelLast 的说明)。
    __aicore__ inline void EnsureCoeff(int64_t cBegin, int64_t cLen, int64_t& cachedCBegin, int64_t& cachedCLen)
    {
        if (cBegin == cachedCBegin && cLen == cachedCLen) {
            return;
        }
        __ubuf__ float* coeffMult = CoeffSlot(kCoeffMultSlot);
        __ubuf__ float* coeffBm = CoeffSlot(kCoeffBmSlot);
        __ubuf__ float* coeffOffset = CoeffSlot(kCoeffOffsetSlot);
        const int64_t chunk = kParamChunkLen(static_cast<int64_t>(VL_FP32));
        for (int64_t c0 = 0; c0 < cLen; c0 += chunk) {
            int64_t count = cLen - c0;
            if (count > chunk) {
                count = chunk;
            }
            event_t waitV2Mte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(waitV2Mte2);
            WaitFlag<HardEvent::V_MTE2>(waitV2Mte2);
            StageFoldInput(sumGm_, kFoldInSum, cBegin + c0, count);
            StageFoldInput(squareSumGm_, kFoldInSquareSum, cBegin + c0, count);
            StageFoldInput(scaleGm_, kFoldInScale, cBegin + c0, count);
            StageFoldInput(offsetGm_, kFoldInOffset, cBegin + c0, count);
            event_t waitMte2V = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(waitMte2V);
            WaitFlag<HardEvent::MTE2_V>(waitMte2V);
            asc_vf_call<VFFoldChunk>(
                ChunkSlot(kFoldInSum), ChunkSlot(kFoldInSquareSum), ChunkSlot(kFoldInScale), ChunkSlot(kFoldInOffset),
                coeffMult + c0, coeffBm + c0, coeffOffset + c0,
                (__ubuf__ float*)foldScratchBuf_.Get<float>()[kFoldSaveVar * chunkSlotElems_].GetPhyAddr(),
                tilingData_->numRec, tilingData_->epsilon, static_cast<uint32_t>(count));
        }
        cachedCBegin = cBegin;
        cachedCLen = cLen;
    }

    __aicore__ inline void CopyInChannelLast(int64_t gmOffset, int64_t rows, int64_t cLen)
    {
        LocalTensor<T_X> x = xQueue_.AllocTensor<T_X>();
        DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cLen * sizeof(T_X)),
                                 (tilingData_->c - cLen) * static_cast<int64_t>(sizeof(T_X)), 0, 0};
        DataCopyPadExtParams<T_X> pad{false, 0, 0, 0};
        DataCopyPad<T_X, PaddingMode::Compact>(x, xGm_[gmOffset], params, pad);
        xQueue_.EnQue(x);
    }

    __aicore__ inline void ComputeChannelLast(int64_t rows, int64_t cLen)
    {
        LocalTensor<T_X> x = xQueue_.DeQue<T_X>();
        LocalTensor<T_X> y = yQueue_.AllocTensor<T_X>();
        asc_vf_call<VFChannelLast<T_X>>((__ubuf__ T_X*)x.GetPhyAddr(), (__ubuf__ T_X*)y.GetPhyAddr(),
                                        CoeffSlot(kCoeffMultSlot), CoeffSlot(kCoeffBmSlot), CoeffSlot(kCoeffOffsetSlot),
                                        static_cast<uint32_t>(rows), static_cast<uint32_t>(cLen));
        yQueue_.EnQue(y);
        xQueue_.FreeTensor(x);
    }

    __aicore__ inline void CopyOutChannelLast(int64_t gmOffset, int64_t rows, int64_t cLen)
    {
        LocalTensor<T_X> y = yQueue_.DeQue<T_X>();
        DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(cLen * sizeof(T_X)), 0,
                                 (tilingData_->c - cLen) * static_cast<int64_t>(sizeof(T_X)), 0};
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
        int64_t cachedCBegin = -1;
        int64_t cachedCLen = -1;
        for (int64_t i = 0; i < tileCount; ++i, ++tileId) {
            const int64_t rowIdx = tileId % tilingData_->tilesPerBlock;
            const int64_t cBlock = tileId / tilingData_->tilesPerBlock;
            const int64_t rowStart = rowIdx * tilingData_->rowsPerTile;
            const int64_t remainRows = tilingData_->num - rowStart;
            const int64_t rows = remainRows < tilingData_->rowsPerTile ? remainRows : tilingData_->rowsPerTile;
            const int64_t cBegin = cBlock * tilingData_->channelBlock;
            const int64_t remainChannels = tilingData_->c - cBegin;
            const int64_t cLen = remainChannels < tilingData_->channelBlock ? remainChannels :
                                                                              tilingData_->channelBlock;
            const int64_t gmOffset = rowStart * tilingData_->c + cBegin;
            EnsureCoeff(cBegin, cLen, cachedCBegin, cachedCLen);
            CopyInChannelLast(gmOffset, rows, cLen);
            ComputeChannelLast(rows, cLen);
            CopyOutChannelLast(gmOffset, rows, cLen);
        }
    }

private:
    const Bn3dTrainingUpdateChannelLastTilingData* tilingData_ = nullptr;
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

#endif // BN3_D_TRAINING_UPDATE_CHANNEL_LAST_H
