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
 * \file in_training_update_grad_stream.h
 * \brief TilingKey 200000: R*C0 exceeds UB. Stream each D-slice in row chunks and keep the C0-wide
 *        accumulators resident (in the output UB tensors) across chunks.
 */
#ifndef IN_TRAINING_UPDATE_GRAD_STREAM_H_
#define IN_TRAINING_UPDATE_GRAD_STREAM_H_

#include "in_training_update_grad_common.h"

namespace InTrainingUpdateGrad {
using namespace AscendC;

template <typename T_DY>
class InTrainingUpdateGradStream {
public:
    __aicore__ inline InTrainingUpdateGradStream(const InTrainingUpdateGradStreamTilingData* tilingData)
    {
        numC1_ = tilingData->numC1;
        numD_ = tilingData->numD;
        numHW_ = tilingData->numHW;
        numC0_ = tilingData->numC0;
        groupNum_ = tilingData->groupNum;
        usedCoreNum_ = tilingData->usedCoreNum;
        perCoreGroups_ = tilingData->perCoreGroups;
        blockLenElem_ = tilingData->blockLenElem;
        streamTileRows_ = tilingData->streamTileRows;
        epsilon_ = tilingData->epsilon;
    }

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR variance, GM_ADDR mean, GM_ADDR resGamma,
                                GM_ADDR resBeta)
    {
        dyGm_.SetGlobalBuffer((__gm__ T_DY*)dy);
        xGm_.SetGlobalBuffer((__gm__ T_DY*)x);
        varianceGm_.SetGlobalBuffer((__gm__ float*)variance);
        meanGm_.SetGlobalBuffer((__gm__ float*)mean);
        resGammaGm_.SetGlobalBuffer((__gm__ float*)resGamma);
        resBetaGm_.SetGlobalBuffer((__gm__ float*)resBeta);

        uint32_t chunkBytes = streamTileRows_ * numC0_ * sizeof(T_DY) + VECTOR_REG_WIDTH;
        pipe_.InitBuffer(dyQueue_, BUFFER_NUM, chunkBytes);
        pipe_.InitBuffer(xQueue_, BUFFER_NUM, chunkBytes);
        pipe_.InitBuffer(varQueue_, 1, numC0_ * sizeof(float) + VECTOR_REG_WIDTH);
        pipe_.InitBuffer(meanQueue_, 1, numC0_ * sizeof(float) + VECTOR_REG_WIDTH);
        pipe_.InitBuffer(gammaOutQueue_, BUFFER_NUM, numC0_ * sizeof(float));
        pipe_.InitBuffer(betaOutQueue_, BUFFER_NUM, numC0_ * sizeof(float));
        pipe_.InitBuffer(rstdBuf_, numC0_ * sizeof(float));
        // Kahan compensation, resident across chunks (same lifetime as the C0 accumulators).
        pipe_.InitBuffer(cGammaBuf_, numC0_ * sizeof(float));
        pipe_.InitBuffer(cBetaBuf_, numC0_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= usedCoreNum_) {
            return;
        }
        uint32_t startGroup = blockIdx * perCoreGroups_;
        uint32_t endGroup = startGroup + perCoreGroups_;
        endGroup = endGroup > groupNum_ ? groupNum_ : endGroup;

        for (uint32_t g = startGroup; g < endGroup; g++) {
            ProcessGroup(g);
        }
    }

private:
    __aicore__ inline void ProcessGroup(uint32_t g)
    {
        uint32_t n = g / numC1_;
        uint32_t c1 = g % numC1_;
        uint64_t spatialBase = (static_cast<uint64_t>(n) * numD_ * numC1_ + c1) * blockLenElem_;
        uint64_t scalarOffset = static_cast<uint64_t>(g) * numC0_;

        // variance/mean and rstd (resident for the whole group)
        LocalTensor<float> varLocal = varQueue_.AllocTensor<float>();
        LocalTensor<float> meanLocal = meanQueue_.AllocTensor<float>();
        DataCopyExtParams c0Params;
        c0Params.blockCount = 1;
        c0Params.blockLen = numC0_ * sizeof(float);
        c0Params.srcStride = 0;
        c0Params.dstStride = 0;
        DataCopyPadExtParams<float> c0Pad;
        c0Pad.isPad = false;
        c0Pad.leftPadding = 0;
        c0Pad.rightPadding = 0;
        c0Pad.paddingValue = 0;
        DataCopyPad(varLocal, varianceGm_[scalarOffset], c0Params, c0Pad);
        DataCopyPad(meanLocal, meanGm_[scalarOffset], c0Params, c0Pad);
        varQueue_.EnQue(varLocal);
        meanQueue_.EnQue(meanLocal);
        varLocal = varQueue_.DeQue<float>();
        meanLocal = meanQueue_.DeQue<float>();

        LocalTensor<float> rstdLocal = rstdBuf_.Get<float>();
        __local_mem__ float* meanAddr = (__local_mem__ float*)meanLocal.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        ComputeRstd((__local_mem__ float*)varLocal.GetPhyAddr(), rstdAddr, numC0_, epsilon_);

        // C0 accumulators live in the output UB tensors; init to 0, accumulate across chunks.
        LocalTensor<float> gammaOut = gammaOutQueue_.AllocTensor<float>();
        LocalTensor<float> betaOut = betaOutQueue_.AllocTensor<float>();
        __local_mem__ float* gammaAddr = (__local_mem__ float*)gammaOut.GetPhyAddr();
        __local_mem__ float* betaAddr = (__local_mem__ float*)betaOut.GetPhyAddr();
        // Kahan compensation, resident across chunks (must persist like the accumulators, else a
        // huge-M stream reduction drops the compensation per chunk and regresses to naive error).
        __local_mem__ float* cGammaAddr = (__local_mem__ float*)cGammaBuf_.Get<float>().GetPhyAddr();
        __local_mem__ float* cBetaAddr = (__local_mem__ float*)cBetaBuf_.Get<float>().GetPhyAddr();
        ZeroC0(gammaAddr, numC0_);
        ZeroC0(betaAddr, numC0_);
        ZeroC0(cGammaAddr, numC0_);
        ZeroC0(cBetaAddr, numC0_);

        uint32_t loops = CeilDiv(numHW_, streamTileRows_);
        for (uint32_t d = 0; d < numD_; d++) {
            uint64_t dBase = spatialBase + static_cast<uint64_t>(d) * numC1_ * blockLenElem_;
            for (uint32_t c = 0; c < loops; c++) {
                uint32_t rows = (c == loops - 1) ? (numHW_ - c * streamTileRows_) : streamTileRows_;
                uint64_t chunkOffset = dBase + static_cast<uint64_t>(c) * streamTileRows_ * numC0_;
                AccumulateChunk(chunkOffset, rows, meanAddr, rstdAddr, gammaAddr, betaAddr, cGammaAddr, cBetaAddr);
            }
        }

        varQueue_.FreeTensor(varLocal);
        meanQueue_.FreeTensor(meanLocal);
        gammaOutQueue_.EnQue(gammaOut);
        betaOutQueue_.EnQue(betaOut);
        CopyOut(scalarOffset);
    }

    __aicore__ inline void AccumulateChunk(uint64_t chunkOffset, uint32_t rows, __local_mem__ float* meanAddr,
                                           __local_mem__ float* rstdAddr, __local_mem__ float* gammaAddr,
                                           __local_mem__ float* betaAddr, __local_mem__ float* cGammaAddr,
                                           __local_mem__ float* cBetaAddr)
    {
        LocalTensor<T_DY> dyLocal = dyQueue_.AllocTensor<T_DY>();
        LocalTensor<T_DY> xLocal = xQueue_.AllocTensor<T_DY>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = rows * numC0_ * sizeof(T_DY);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<T_DY> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
        DataCopyPad<T_DY, PaddingMode::Normal>(dyLocal, dyGm_[chunkOffset], copyParams, padParams);
        DataCopyPad<T_DY, PaddingMode::Normal>(xLocal, xGm_[chunkOffset], copyParams, padParams);
        dyQueue_.EnQue(dyLocal);
        xQueue_.EnQue(xLocal);
        dyLocal = dyQueue_.template DeQue<T_DY>();
        xLocal = xQueue_.template DeQue<T_DY>();

        AccumulateGroupC0<T_DY>((__local_mem__ T_DY*)dyLocal.GetPhyAddr(), (__local_mem__ T_DY*)xLocal.GetPhyAddr(),
                                meanAddr, rstdAddr, gammaAddr, betaAddr, cGammaAddr, cBetaAddr, rows, numC0_, false);

        dyQueue_.FreeTensor(dyLocal);
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t scalarOffset)
    {
        LocalTensor<float> gammaOut = gammaOutQueue_.template DeQue<float>();
        LocalTensor<float> betaOut = betaOutQueue_.template DeQue<float>();
        DataCopyExtParams c0Params;
        c0Params.blockCount = 1;
        c0Params.blockLen = numC0_ * sizeof(float);
        c0Params.srcStride = 0;
        c0Params.dstStride = 0;
        DataCopyPad(resGammaGm_[scalarOffset], gammaOut, c0Params);
        DataCopyPad(resBetaGm_[scalarOffset], betaOut, c0Params);
        gammaOutQueue_.FreeTensor(gammaOut);
        betaOutQueue_.FreeTensor(betaOut);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> dyQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue_;
    TQue<QuePosition::VECIN, 1> varQueue_;
    TQue<QuePosition::VECIN, 1> meanQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> gammaOutQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> betaOutQueue_;
    TBuf<TPosition::VECCALC> rstdBuf_;
    TBuf<TPosition::VECCALC> cGammaBuf_;
    TBuf<TPosition::VECCALC> cBetaBuf_;

    GlobalTensor<T_DY> dyGm_;
    GlobalTensor<T_DY> xGm_;
    GlobalTensor<float> varianceGm_;
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> resGammaGm_;
    GlobalTensor<float> resBetaGm_;

    uint32_t numC1_;
    uint32_t numD_;
    uint32_t numHW_;
    uint32_t numC0_;
    uint32_t groupNum_;
    uint32_t usedCoreNum_;
    uint32_t perCoreGroups_;
    uint32_t blockLenElem_;
    uint32_t streamTileRows_;
    float epsilon_;
};
} // namespace InTrainingUpdateGrad
#endif // IN_TRAINING_UPDATE_GRAD_STREAM_H_
