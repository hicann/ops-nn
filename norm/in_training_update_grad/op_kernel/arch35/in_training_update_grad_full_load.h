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
 * \file in_training_update_grad_full_load.h
 * \brief TilingKey 100000: one (n,c1) group's whole spatial block (R*C0) is loaded to UB at once.
 */
#ifndef IN_TRAINING_UPDATE_GRAD_FULL_LOAD_H_
#define IN_TRAINING_UPDATE_GRAD_FULL_LOAD_H_

#include "in_training_update_grad_common.h"

namespace InTrainingUpdateGrad {
using namespace AscendC;

template <typename T_DY>
class InTrainingUpdateGradFullLoad {
public:
    __aicore__ inline InTrainingUpdateGradFullLoad(const InTrainingUpdateGradFullLoadTilingData* tilingData)
    {
        numC1_ = tilingData->numC1;
        numD_ = tilingData->numD;
        numC0_ = tilingData->numC0;
        reduceR_ = tilingData->reduceR;
        groupNum_ = tilingData->groupNum;
        usedCoreNum_ = tilingData->usedCoreNum;
        perCoreGroups_ = tilingData->perCoreGroups;
        blockLenElem_ = tilingData->blockLenElem;
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

        // dy/x block for a group = reduceR * C0 elements, padded by a vector register to keep the
        // C0-wide register loads (which fetch a full 256B register) inside the buffer.
        uint32_t spatialBytes = reduceR_ * numC0_ * sizeof(T_DY) + VECTOR_REG_WIDTH;
        pipe_.InitBuffer(dyQueue_, BUFFER_NUM, spatialBytes);
        pipe_.InitBuffer(xQueue_, BUFFER_NUM, spatialBytes);
        pipe_.InitBuffer(varQueue_, 1, numC0_ * sizeof(float) + VECTOR_REG_WIDTH);
        pipe_.InitBuffer(meanQueue_, 1, numC0_ * sizeof(float) + VECTOR_REG_WIDTH);
        pipe_.InitBuffer(gammaOutQueue_, BUFFER_NUM, numC0_ * sizeof(float));
        pipe_.InitBuffer(betaOutQueue_, BUFFER_NUM, numC0_ * sizeof(float));
        pipe_.InitBuffer(rstdBuf_, numC0_ * sizeof(float));
        pipe_.InitBuffer(cGammaBuf_, numC0_ * sizeof(float)); // Kahan compensation scratch (single call)
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
            uint32_t n = g / numC1_;
            uint32_t c1 = g % numC1_;
            // dy/x base (elements) for (n, c1); D-slice d adds d * numC1 * blockLenElem.
            uint64_t spatialBase = (static_cast<uint64_t>(n) * numD_ * numC1_ + c1) * blockLenElem_;
            uint64_t scalarOffset = static_cast<uint64_t>(g) * numC0_;
            CopyIn(spatialBase, scalarOffset);
            Compute();
            CopyOut(scalarOffset);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t spatialBase, uint64_t scalarOffset)
    {
        LocalTensor<T_DY> dyLocal = dyQueue_.AllocTensor<T_DY>();
        LocalTensor<T_DY> xLocal = xQueue_.AllocTensor<T_DY>();
        // Gather the D slices of this (n,c1) into a contiguous (R,C0) UB block. Each slice is
        // blockLenElem = H*W*C0 contiguous; consecutive slices are numC1*blockLenElem apart, so the
        // DataCopyPad srcStride gap is (numC1-1)*blockLenElem. C0*sizeof(T) is a 32B multiple, so
        // blockLen is 32B aligned and rightPadding stays 0.
        DataCopyPadExtParams<T_DY> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
        DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(numD_);
        copyParams.blockLen = blockLenElem_ * sizeof(T_DY);
        copyParams.srcStride = static_cast<uint32_t>((numC1_ - 1)) * blockLenElem_ * sizeof(T_DY);
        copyParams.dstStride = 0;
        DataCopyPad<T_DY, PaddingMode::Normal>(dyLocal, dyGm_[spatialBase], copyParams, padParams);
        DataCopyPad<T_DY, PaddingMode::Normal>(xLocal, xGm_[spatialBase], copyParams, padParams);
        dyQueue_.EnQue(dyLocal);
        xQueue_.EnQue(xLocal);

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
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<T_DY> dyLocal = dyQueue_.template DeQue<T_DY>();
        LocalTensor<T_DY> xLocal = xQueue_.template DeQue<T_DY>();
        LocalTensor<float> varLocal = varQueue_.template DeQue<float>();
        LocalTensor<float> meanLocal = meanQueue_.template DeQue<float>();
        LocalTensor<float> gammaOut = gammaOutQueue_.AllocTensor<float>();
        LocalTensor<float> betaOut = betaOutQueue_.AllocTensor<float>();
        LocalTensor<float> rstdLocal = rstdBuf_.Get<float>();
        LocalTensor<float> cGammaLocal = cGammaBuf_.Get<float>();
        LocalTensor<float> cBetaLocal = cBetaBuf_.Get<float>();

        __local_mem__ float* varAddr = (__local_mem__ float*)varLocal.GetPhyAddr();
        __local_mem__ float* meanAddr = (__local_mem__ float*)meanLocal.GetPhyAddr();
        __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        ComputeRstd(varAddr, rstdAddr, numC0_, epsilon_);

        AccumulateGroupC0<T_DY>((__local_mem__ T_DY*)dyLocal.GetPhyAddr(), (__local_mem__ T_DY*)xLocal.GetPhyAddr(),
                                meanAddr, rstdAddr, (__local_mem__ float*)gammaOut.GetPhyAddr(),
                                (__local_mem__ float*)betaOut.GetPhyAddr(),
                                (__local_mem__ float*)cGammaLocal.GetPhyAddr(),
                                (__local_mem__ float*)cBetaLocal.GetPhyAddr(), reduceR_, numC0_, true);

        dyQueue_.FreeTensor(dyLocal);
        xQueue_.FreeTensor(xLocal);
        varQueue_.FreeTensor(varLocal);
        meanQueue_.FreeTensor(meanLocal);
        gammaOutQueue_.EnQue(gammaOut);
        betaOutQueue_.EnQue(betaOut);
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
    uint32_t numC0_;
    uint32_t reduceR_;
    uint32_t groupNum_;
    uint32_t usedCoreNum_;
    uint32_t perCoreGroups_;
    uint32_t blockLenElem_;
    float epsilon_;
};
} // namespace InTrainingUpdateGrad
#endif // IN_TRAINING_UPDATE_GRAD_FULL_LOAD_H_
