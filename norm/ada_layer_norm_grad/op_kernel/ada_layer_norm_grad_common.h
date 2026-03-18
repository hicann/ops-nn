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
 * \file ada_layer_norm_grad_common.h
 * \brief
 */

#ifndef ADA_LAYER_NORM_GRAD_COMMON_H
#define ADA_LAYER_NORM_GRAD_COMMON_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "ada_layer_norm_grad_determinstic_compute.h"
#include "ada_layer_norm_grad_common_constants.h"

namespace AdaLayerNormGrad {
using namespace AscendC;

template <typename T, typename U, bool isDeterministic>
class AdaLayerNormGradCommon {
public:
    __aicore__ inline AdaLayerNormGradCommon() {}

    __aicore__ inline void Init(
        GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR scale, GM_ADDR gamma, GM_ADDR beta,
        GM_ADDR pdX, GM_ADDR pdScale, GM_ADDR pdShift, GM_ADDR pdGamma, GM_ADDR pdBeta,
        GM_ADDR workspace, const AdaLayerNormGradTilingDataCommon* tilingData, TPipe& pipeIn);

    __aicore__ inline void Process(const AdaLayerNormGradTilingDataCommon* tilingData);

private:
    __aicore__ inline void CopyInBeta(const AdaLayerNormGradTilingDataCommon* tilingData);
    __aicore__ inline void CopyInGamma(const AdaLayerNormGradTilingDataCommon* tilingData);
    __aicore__ inline void CopyInPhase0(const AdaLayerNormGradTilingDataCommon* tilingData, const int64_t curRowsNum, const int64_t startRow);
    __aicore__ inline void ComputePhase0(const AdaLayerNormGradTilingDataCommon* tilingData);

    __aicore__ inline void CopyInPhase1(
        const int64_t outerIdx, const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData);
    __aicore__ inline void ComputePhase1(
        const int64_t outerIdx, const int64_t curRowsNum, const int64_t startRow,
        const AdaLayerNormGradTilingDataCommon* tilingData);

    __aicore__ inline void CopyInPhase2(
        const int64_t outerIdx, const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData);
    __aicore__ inline void ComputePhase2(
        const int64_t outerIdx, const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData);

    __aicore__ inline void CopyInPhase3(
        const int64_t outerIdx, const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData);
    __aicore__ inline void ComputePhase3(
        const int64_t outerIdx, const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData);

    __aicore__ inline void CopyInPhase4(
        const int64_t outerIdx, const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData);
    __aicore__ inline void ComputePhase4(
        const int64_t outerIdx, const int64_t curRowsNum, const int64_t startRow,
        const AdaLayerNormGradTilingDataCommon* tilingData);

    __aicore__ inline void CopyOutPhase0(
        const int64_t outerIdx, const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData);
    __aicore__ inline void CopyOutPhase1(
        const int64_t startRow, const AdaLayerNormGradTilingDataCommon* tilingData);
    __aicore__ inline void CopyOutPhase1Deterministic(const AdaLayerNormGradTilingDataCommon* tilingData);
    __aicore__ inline void CastWithCopyOutSWT(
        const int64_t outIdx, const int64_t rowOffset, const int64_t rowOfCp, const AdaLayerNormGradTilingDataCommon* tilingData);
    __aicore__ inline void CopyOutPhase2(const AdaLayerNormGradTilingDataCommon* tilingData);
    __aicore__ inline void CopyOutPhase2Deterministic(const AdaLayerNormGradTilingDataCommon* tilingData);

    __aicore__ inline void CastToFloat(
        const LocalTensor<float>& buffer, const LocalTensor<float>& tmpBuffer, const int64_t curRowsNum,
        const AdaLayerNormGradTilingDataCommon* tilingData, const int64_t bufferElemNums);

    __aicore__ inline void CastToB16(
        const LocalTensor<float>& buffer, const LocalTensor<float>& tmpBuffer, const int64_t curRowsNum,
        const AdaLayerNormGradTilingDataCommon* tilingData);

    template <typename dType>
    __aicore__ inline void BlockBroadcast(
        const LocalTensor<dType>& dst, const LocalTensor<dType>& src, const int64_t curRowsNum);

    __aicore__ inline void BinElemWithInlinedLastBrcFP32(
        const LocalTensor<float>& dst, const LocalTensor<float>& src0, const LocalTensor<float>& src1,
        const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData,
        void (*func)(
            const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t,
            const BinaryRepeatParams&));

    __aicore__ inline void BinElemWithInlinedNLastBrcFP32(
        const LocalTensor<float>& dst, const LocalTensor<float>& src0, const LocalTensor<float>& src1,
        const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData,
        void (*func)(
            const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t,
            const BinaryRepeatParams&));

    __aicore__ inline void NlastBatchReduceSum(
        const LocalTensor<float>& dst, const LocalTensor<float>& src, const int64_t curRowsNum,
        const int64_t startRow, const AdaLayerNormGradTilingDataCommon* tilingData);

    __aicore__ inline void NlastBatchMul(
        const LocalTensor<float>& dst, const LocalTensor<float>& src0, const LocalTensor<float>& src1,
        const int64_t curRowsNum, const int64_t startRow, const AdaLayerNormGradTilingDataCommon* tilingData);

    __aicore__ inline void NlastReduceSum(
        const LocalTensor<float>& dst, const LocalTensor<float>& src, const int64_t curRowsNum,
        const AdaLayerNormGradTilingDataCommon* tilingData);

    __aicore__ inline void LastReduceSum(
        const LocalTensor<float>& dst, const LocalTensor<float>& src, const LocalTensor<float>& tmp,
        const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData);

private:
    TPipe pipe;
    constexpr static uint16_t SYNC_AIV_ONLY_ALL = 14;

    TQue<QuePosition::VECIN, 1> queue0_;
    TQue<QuePosition::VECIN, 1> queue1_;
    TQue<QuePosition::VECIN, 1> queue2_;
    TQue<QuePosition::VECIN, 1> queue3_;
    TQue<QuePosition::VECIN, 1> queue5_;
    TQue<QuePosition::VECIN, 1> queue4_;
    TQue<QuePosition::VECOUT, 1> queue6_;
    TQue<QuePosition::VECOUT, 1> queue7_;
    TQue<QuePosition::VECOUT, 1> queue8_;
    TQue<QuePosition::VECOUT, 1> queue9_;
    TQue<QuePosition::VECOUT, 1> queue10_;
    TBuf<> tmpTensor0_;
    TBuf<> tmpTensor1_;

    LocalTensor<float> buffer0_;
    LocalTensor<float> buffer1_;
    LocalTensor<float> buffer2_;
    LocalTensor<float> buffer3_;
    LocalTensor<float> buffer5_;
    LocalTensor<float> buffer4_;
    LocalTensor<float> buffer6_;
    LocalTensor<float> buffer7_;
    LocalTensor<float> buffer8_;
    LocalTensor<float> buffer9_;
    LocalTensor<float> buffer10_;
    LocalTensor<float> buffer11_;
    LocalTensor<float> buffer12_;
    LocalTensor<float> buffer13_;

    GlobalTensor<T> dyInTensorGM_;
    GlobalTensor<T> xInTensorGM_;
    GlobalTensor<float> rstdInTensorGM_;
    GlobalTensor<float> meanInTensorGM_;
    GlobalTensor<U> gammaInTensorGM_;
    GlobalTensor<T> scaleInTensorGM_;
    GlobalTensor<U> betaInTensorGM_;

    GlobalTensor<T> pdXOutTensorGM_;
    GlobalTensor<T> pdScaleOutTensorGM;
    GlobalTensor<T> pdShiftOutTensorGM;
    GlobalTensor<U> pdGammaOutTensorGM;
    GlobalTensor<U> pdBetaOutTensorGM;
    GlobalTensor<float> dScaleWorkspaceGM;
    GlobalTensor<float> dShiftWorkspaceGM;
    GlobalTensor<float> workspaceGM;
    GlobalTensor<float> workspaceGMOri;

    float coff_;
    int64_t blockStartBatch_;
    int64_t ScaleReEleNum_;
    int64_t ScaleReRowNum_;
};

} // namespace AdaLayerNormGrad

#include "ada_layer_norm_grad_common_init.inl"
#include "ada_layer_norm_grad_common_phase.inl"
#include "ada_layer_norm_grad_common_output.inl"
#include "ada_layer_norm_grad_common_utils.inl"

#endif