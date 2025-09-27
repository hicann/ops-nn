/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_reduce_sum_mixcore.h
 * \brief
 */
#ifndef ASCENDC_QUANT_MATMUL_REDUCE_SUM_QUANT_MIXCORE_H
#define ASCENDC_QUANT_MATMUL_REDUCE_SUM_QUANT_MIXCORE_H

#include "quant_matmul_reduce_sum_utils.h"
#include "quant_matmul_reduce_sum.h"

namespace QUANT_MATMUL_REDUCE_SUM {
/*@brief store variables for core split configuration
 */
constexpr int32_t PIPELINE_NUM = 4;
constexpr uint32_t FP32_PER_REPEAT = 64;

/** @brief intenal computation class
 */
template <class mmType, bool sync = false>
class QuantMatmulReduceSumQuantMixCoreCompute : public QbmmRSCompute<mmType, sync>
{
public:
    using AT = typename mmType::AT::T;
    using BT = typename mmType::BT::T;
    using B = typename mmType::BT;
    using CT = typename mmType::CT::T;
    using BiasT = typename mmType::BiasT::T;
    constexpr static bool transposeX1 = mmType::AT::isTrans;
    constexpr static bool transposeX2 = mmType::BT::isTrans;

    /** @brief constructor */
    __aicore__ inline QuantMatmulReduceSumQuantMixCoreCompute(typename mmType::MT& mm_)
        : QbmmRSCompute<mmType, sync>(mm_)
    {}

    __aicore__ inline void Init(
        const QbmmRSComputeInitParams* __restrict initParams,
        const QuantMatmulReduceSumParams* __restrict qbmmReduceSumParams, const TCubeTiling* __restrict mmTilingData,
        TPipe* tPipe);

    __aicore__ inline void MMCompute(uint32_t batchIdx, MNConfig& mnConfig, uint32_t coreIdx);

    __aicore__ inline void VectorCompute(uint32_t batchIdx, MNConfig& mnConfig);

    __aicore__ inline void PostCompute();

private:
    __aicore__ inline void Dequant(uint32_t batchIdx, MNConfig& mnConfig);

    __aicore__ inline void SetPerTokenQuantStaticBuffer(
        const QuantMatmulReduceSumParams* __restrict qbmmReduceSumParams, const TCubeTiling* __restrict mmTilingData,
        GM_ADDR workspace);

    __aicore__ inline void DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN, uint64_t x2ScaleOffset);

    __aicore__ inline void DataCopyPerTokenScaleAndBrcb(
        MNConfig& mnConfig, uint32_t curBaseM, uint32_t alignBaseN, uint32_t offsetM);

    __aicore__ inline void SetPerTokenQuantRefreshedBuffer(const MNConfig mnConfig);

    __aicore__ inline void ActivationCompute(
        uint32_t computeSize, LocalTensor<float> preResUb, LocalTensor<uint8_t> actTmpLocal);

    __aicore__ inline void ComputeDequantAndActivate(
        MNConfig& mnConfig, uint32_t curVecBaseM, uint32_t alignBaseN, uint32_t curVecBaseN, uint32_t offsetM);

    __aicore__ inline void DataCopyOut(
        MNConfig& mnConfig, uint32_t curVecBaseM, uint32_t curVecBaseN, uint32_t alignBaseN, uint64_t outOffset);

    __aicore__ inline void VectorTilingCalc(
        MNConfig& mnConfig, uint32_t& curCubeSingleN, uint32_t& curCubeSingleM, uint32_t& vecBaseN, uint32_t& vecBaseM);

    GM_ADDR x1ScaleTensorPtr;
    GM_ADDR x2ScaleTensorPtr;
    GlobalTensor<float> x1ScaleGm;
    GlobalTensor<bfloat16_t> x2ScaleGm;
    GlobalTensor<CT> mmOutGm;
    // define the que
    TQue<QuePosition::VECIN, 1> vecInQueue;
    TQue<QuePosition::VECOUT, 1> vecOutQueue;
    TQue<QuePosition::VECIN, 1> x1ScaleInQueue;
    TQue<QuePosition::VECIN, 1> x2ScaleInQueue;
    TBuf<TPosition::VECCALC> tmpBuff;
    LocalTensor<CT> mmOutInUb;
    LocalTensor<float> x1ScaleInUb;
    LocalTensor<bfloat16_t> x2ScaleInUb;
    LocalTensor<float> dequantMiddleResult;
    LocalTensor<uint8_t> sharedTmpLocal;
    LocalTensor<float> mulsResultLocal;
    LocalTensor<float> x1ScaleBrcbLocal;
    LocalTensor<float> actResultLocal;
    bool sequentialWrite = true;
    uint32_t cubeNum; // Matmul completions on the kernel
    uint32_t nOffset; // antiquant n offset
};

template <typename mmType, bool sync>
__aicore__ inline void QuantMatmulReduceSumQuantMixCoreCompute<mmType, sync>::Init(
    const QbmmRSComputeInitParams* __restrict initParams,
    const QuantMatmulReduceSumParams* __restrict qbmmReduceSumParams, const TCubeTiling* __restrict mmTilingData,
    TPipe* tPipe)
{
    this->QbmmRSCompute<mmType, sync>::Init(initParams, qbmmReduceSumParams, mmTilingData, tPipe);
    x1ScaleTensorPtr = initParams->x1Scale;
    x2ScaleTensorPtr = initParams->x2Scale;
    cubeNum = 0;
    SetPerTokenQuantStaticBuffer(qbmmReduceSumParams, mmTilingData, initParams->workspace);
}

template <typename mmType, bool sync>
__aicore__ inline void QuantMatmulReduceSumQuantMixCoreCompute<mmType, sync>::PostCompute()
{
    if ASCEND_IS_AIC {
        for (int32_t idx = 0; idx < Min<int32_t>(cubeNum, PIPELINE_NUM); ++idx) {
            CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG);
        }
    }
}

template <typename mmType, bool sync>
__aicore__ inline void QuantMatmulReduceSumQuantMixCoreCompute<mmType, sync>::MMCompute(
    uint32_t batchIdx, MNConfig& mnConfig, uint32_t coreIdx)
{
    uint32_t tailN = mnConfig.nIdx * mnConfig.baseN;
    uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.baseN : mnConfig.n - tailN;
    uint32_t curSingleM =
        mnConfig.mIdx < mnConfig.blockDimM - 1 ? mnConfig.baseM : mnConfig.m - mnConfig.mIdx * mnConfig.baseM;
    uint64_t xOffset = mnConfig.mIdx * mnConfig.baseM * mnConfig.k;
    uint64_t outOffset = mnConfig.mIdx * mnConfig.baseM * mnConfig.n + tailN;
    // init global buffer
    this->x1Gm.SetGlobalBuffer((__gm__ AT*)this->x1TensorPtr + mnConfig.x1BaseOffset);
    GlobalTensor<BT> x2Gm = this->SetGlobalBufferW(batchIdx, tailN, mnConfig);
    if ASCEND_IS_AIC {
        this->mm.SetOrgShape(mnConfig.m, mnConfig.n, mnConfig.k);
        this->mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
        this->mm.SetTensorA(this->x1Gm[xOffset], transposeX1);
        this->mm.SetTensorB(x2Gm, transposeX2);
        while (this->mm.Iterate()) {
            if (sequentialWrite) {
                mnConfig.workSpaceOffset =
                    mnConfig.baseN * mnConfig.baseM * (coreIdx + (cubeNum % PIPELINE_NUM) * this->coreNum);
            } else {
                mnConfig.workSpaceOffset = outOffset + mnConfig.yBaseOffset;
            }
            if (this->cubeNum >= PIPELINE_NUM) {
                CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG);
            }
            this->mm.GetTensorC(mmOutGm[mnConfig.workSpaceOffset], 0, sequentialWrite);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);
            cubeNum++;
        }
    }
}

template <typename mmType, bool sync>
__aicore__ inline void QuantMatmulReduceSumQuantMixCoreCompute<mmType, sync>::VectorCompute(
    uint32_t batchIdx, MNConfig& mnConfig)
{
    nOffset = 0;
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN : mnConfig.n - tailN;
    uint32_t curSingleM =
        mnConfig.mIdx < mnConfig.blockDimM - 1 ? mnConfig.singleM : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    int nDim = Ceil(curSingleN, mnConfig.baseN);
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
    if ASCEND_IS_AIV {
        SetPerTokenQuantRefreshedBuffer(mnConfig);
        for (int i = 0; i < nDim; i++) {
            if (sequentialWrite) {
                mnConfig.workSpaceOffset = mnConfig.baseN * mnConfig.baseM *
                                           (GetBlockIdx() / GetTaskRation() + (cubeNum % PIPELINE_NUM) * this->coreNum);
            } else {
                mnConfig.workSpaceOffset = outOffset + mnConfig.yBaseOffset;
            }
            cubeNum++;
            CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
            Dequant(batchIdx, mnConfig);
            nOffset += mnConfig.baseN;
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG);
        }
    }
}

template <typename mmType, bool sync>
__aicore__ inline void QuantMatmulReduceSumQuantMixCoreCompute<mmType, sync>::ComputeDequantAndActivate(
    MNConfig& mnConfig, uint32_t curVecBaseM, uint32_t alignBaseN, uint32_t curVecBaseN, uint32_t offsetM)
{
    DataCopyPerTokenScaleAndBrcb(mnConfig, curVecBaseM, alignBaseN, offsetM);
    mmOutInUb = vecInQueue.DeQue<CT>();
    LocalTensor<DTYPE_Y> yLocalInUb = vecOutQueue.AllocTensor<DTYPE_Y>();

    AscendDequant(dequantMiddleResult, mmOutInUb, x2ScaleInUb, sharedTmpLocal, {curVecBaseM, alignBaseN, curVecBaseN});
    PipeBarrier<PIPE_V>();
    LocalTensor<float> preResUb = dequantMiddleResult;
    LocalTensor<float> yFP32LocalInUb = dequantMiddleResult;
    LocalTensor<uint8_t> actTmpLocal = sharedTmpLocal;
    // pertoken antiquant
    uint32_t tailNum = alignBaseN % FP32_PER_REPEAT;
    uint8_t repeatStride = alignBaseN * sizeof(float) / UB_BLOCK_UNIT_SIZE;
    uint64_t perchannelResOffset = 0;
    uint64_t alignedN = alignBaseN - tailNum;
    while (perchannelResOffset < alignedN) {
        Mul(mulsResultLocal[perchannelResOffset], dequantMiddleResult[perchannelResOffset], x1ScaleBrcbLocal,
            FP32_PER_REPEAT, curVecBaseM, {1, 1, 0, repeatStride, repeatStride, 1});
        perchannelResOffset += FP32_PER_REPEAT;
    }
    if (tailNum != 0) {
        Mul(mulsResultLocal[perchannelResOffset], dequantMiddleResult[perchannelResOffset], x1ScaleBrcbLocal, tailNum,
            curVecBaseM, {1, 1, 0, repeatStride, repeatStride, 1});
    }
    PipeBarrier<PIPE_V>();
    preResUb = mulsResultLocal;
    yFP32LocalInUb = mulsResultLocal;
    actTmpLocal = tmpBuff.GetWithOffset<uint8_t>(2 * this->ubCalSize * sizeof(float), 0);
    // get final output after Cast
    Cast(yLocalInUb, yFP32LocalInUb, RoundMode::CAST_RINT, curVecBaseM * alignBaseN);
    PipeBarrier<PIPE_V>();
    vecInQueue.FreeTensor(mmOutInUb);
    vecOutQueue.EnQue(yLocalInUb);
    return;
}

template <typename mmType, bool sync>
__aicore__ inline void QuantMatmulReduceSumQuantMixCoreCompute<mmType, sync>::VectorTilingCalc(
    MNConfig& mnConfig, uint32_t& curCubeSingleN, uint32_t& curCubeSingleM, uint32_t& vecBaseN, uint32_t& vecBaseM)
{
    curCubeSingleN =
        mnConfig.nIdx == mnConfig.blockDimN - 1 ? mnConfig.n - mnConfig.nIdx * mnConfig.baseN : mnConfig.baseN;
    curCubeSingleM =
        mnConfig.mIdx == mnConfig.blockDimM - 1 ? mnConfig.m - mnConfig.mIdx * mnConfig.baseM : mnConfig.baseM;
    vecBaseN = mnConfig.baseN;
    vecBaseM = this->ubCalSize / AlignUp(vecBaseN, static_cast<uint32_t>(UB_BLOCK_DOUBLE_UNIT_SIZE / sizeof(int32_t)));
    vecBaseM = Min(vecBaseM, curCubeSingleM);
}

template <typename mmType, bool sync>
__aicore__ inline void QuantMatmulReduceSumQuantMixCoreCompute<mmType, sync>::Dequant(
    uint32_t batchIdx, MNConfig& mnConfig)
{
    uint32_t curCubeSingleN;
    uint32_t curCubeSingleM;
    uint32_t vecBaseN;
    uint32_t vecBaseM;
    VectorTilingCalc(mnConfig, curCubeSingleN, curCubeSingleM, vecBaseN, vecBaseM);
    uint32_t curVecBaseN = vecBaseN;
    uint32_t curVecBaseM;
    uint32_t vecCount = 0;
    if (nOffset + vecBaseN >= curCubeSingleN) {
        curVecBaseN = curCubeSingleN - nOffset;
    }
    uint32_t rowLength = sequentialWrite ? curVecBaseN : mnConfig.n;
    uint32_t taskRation = GetTaskRation();
    for (uint32_t offsetN = nOffset; offsetN < curCubeSingleN; offsetN += vecBaseN) {
        uint32_t alignBaseN = AlignUp(curVecBaseN, static_cast<uint32_t>(UB_BLOCK_DOUBLE_UNIT_SIZE / sizeof(int32_t)));
        uint64_t x2ScaleOffset = mnConfig.nIdx * mnConfig.singleN + offsetN;
        DataCopyScale(curVecBaseN, alignBaseN, x2ScaleOffset);
        curVecBaseM = vecBaseM;
        for (uint32_t offsetM = 0; offsetM < curCubeSingleM; offsetM += vecBaseM) {
            vecCount++;
            if (vecCount % taskRation != this->subBlockIdx) {
                continue;
            }
            if (unlikely(offsetM + vecBaseM >= curCubeSingleM)) {
                curVecBaseM = curCubeSingleM - offsetM;
            }
            // use AscendDequant interface to do perchannel dequant
            uint64_t mmOutOffset = mnConfig.workSpaceOffset + offsetM * static_cast<uint64_t>(rowLength);
            LocalTensor<CT> mmOutLocal = vecInQueue.AllocTensor<CT>();
            DataCopyPad2D(mmOutLocal, mmOutGm[mmOutOffset], curVecBaseM, curVecBaseN, rowLength);
            vecInQueue.EnQue(mmOutLocal);
            ComputeDequantAndActivate(mnConfig, curVecBaseM, alignBaseN, curVecBaseN, offsetM);
            uint64_t outOffset =
                (mnConfig.mIdx * mnConfig.singleM + offsetM) * mnConfig.n + mnConfig.nIdx * mnConfig.singleN + offsetN;
            // use AtomicAdd
            AscendC::SetAtomicAdd<DTYPE_Y>();
            DataCopyOut(mnConfig, curVecBaseM, curVecBaseN, alignBaseN, outOffset);
            AscendC::SetAtomicNone();
        }
        x2ScaleInQueue.FreeTensor(x2ScaleInUb);
        // once a base block
        break;
    }
}

template <typename mmType, bool sync>
__aicore__ inline void QuantMatmulReduceSumQuantMixCoreCompute<mmType, sync>::DataCopyOut(
    MNConfig& mnConfig, uint32_t curVecBaseM, uint32_t curVecBaseN, uint32_t alignBaseN, uint64_t outOffset)
{
    // Copy the result of vector to yGm.
    LocalTensor<DTYPE_Y> yLocal = vecOutQueue.DeQue<DTYPE_Y>();
    DataCopyExtParams params;
    params.blockCount = curVecBaseM;
    params.blockLen = curVecBaseN * sizeof(DTYPE_Y);
    params.srcStride = static_cast<uint32_t>((alignBaseN - curVecBaseN) * sizeof(DTYPE_Y) / UB_BLOCK_UNIT_SIZE);
    params.dstStride = (mnConfig.n - curVecBaseN) * sizeof(DTYPE_Y);
    DataCopyPad(this->yGm[outOffset], yLocal, params);
    vecOutQueue.FreeTensor(yLocal);
}

template <typename mmType, bool sync>
__aicore__ inline void QuantMatmulReduceSumQuantMixCoreCompute<mmType, sync>::SetPerTokenQuantStaticBuffer(
    const QuantMatmulReduceSumParams* __restrict qbmmReduceSumParams, const TCubeTiling* __restrict mmTilingData,
    GM_ADDR workspace)
{
    // Initialize ub and gm memories that do not need to be reinitialized due to changes in batchidx.
    if ASCEND_IS_AIV {
        this->pipe->InitBuffer(x1ScaleInQueue, 2, mmTilingData->baseM * sizeof(float));      // 2: double buffer
        this->pipe->InitBuffer(x2ScaleInQueue, 2, mmTilingData->baseN * sizeof(bfloat16_t)); // 2: double buffer
        this->pipe->InitBuffer(vecInQueue, 2, this->ubCalSize * sizeof(CT));                 // 2: double buffer
        this->pipe->InitBuffer(vecOutQueue, 1, this->ubCalSize * sizeof(DTYPE_Y));
        this->pipe->InitBuffer(tmpBuff, qbmmReduceSumParams->ubRestBytes);
        dequantMiddleResult = tmpBuff.GetWithOffset<float>(this->ubCalSize, 0);
        uint32_t factor = 2; // 2: Indicates the first two blocks of ub are already occupied.
        uint32_t ubCalSizeFloat = this->ubCalSize * sizeof(float);
        uint32_t offset = factor * ubCalSizeFloat;
        sharedTmpLocal = tmpBuff.GetWithOffset<uint8_t>(2 * ubCalSizeFloat, offset);         // 2: temporary space
        mulsResultLocal = tmpBuff.GetWithOffset<float>(this->ubCalSize, 2 * ubCalSizeFloat); // 2: first two blocks
        x1ScaleBrcbLocal = tmpBuff.GetWithOffset<float>(this->ubCalSize, ubCalSizeFloat);
    }
    mmOutGm.SetGlobalBuffer((__gm__ int32_t*)workspace);
}

template <typename mmType, bool sync>
__aicore__ inline void QuantMatmulReduceSumQuantMixCoreCompute<mmType, sync>::SetPerTokenQuantRefreshedBuffer(
    const MNConfig mnConfig)
{
    // Initialize gm memories that need to be reinitialized due to changes in batchidx.
    // Currently, pertoken quant only supports single-tensor mode,
    // hence set according to x and weight single-tensor mode.
    // Add an if branch if multi-tensor mode for weght is required.
    x2ScaleGm.SetGlobalBuffer((__gm__ bfloat16_t*)this->x2ScaleTensorPtr);
    x1ScaleGm.SetGlobalBuffer((__gm__ float*)x1ScaleTensorPtr + mnConfig.mAxisBaseOffset);
    this->yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)this->yTensorPtr);
}

template <typename mmType, bool sync>
__aicore__ inline void QuantMatmulReduceSumQuantMixCoreCompute<mmType, sync>::DataCopyScale(
    uint32_t curBaseN, uint32_t alignBaseN, uint64_t x2ScaleOffset)
{
    // GM copy scale
    DataCopyPadExtParams<bfloat16_t> padParams;
    DataCopyExtParams scaleParams;
    scaleParams.blockLen = curBaseN * sizeof(bfloat16_t);
    scaleParams.blockCount = 1;
    scaleParams.srcStride = 0;
    scaleParams.dstStride = 0;
    LocalTensor<bfloat16_t> x2ScaleLocal = x2ScaleInQueue.AllocTensor<bfloat16_t>();
    DataCopyPad(x2ScaleLocal, x2ScaleGm[x2ScaleOffset], scaleParams, padParams);
    x2ScaleInQueue.EnQue(x2ScaleLocal);

    x2ScaleInUb = x2ScaleInQueue.DeQue<bfloat16_t>();
    x2ScaleInUb.SetSize(alignBaseN);
}

template <typename mmType, bool sync>
__aicore__ inline void QuantMatmulReduceSumQuantMixCoreCompute<mmType, sync>::DataCopyPerTokenScaleAndBrcb(
    MNConfig& mnConfig, uint32_t curBaseM, uint32_t alignBaseN, uint32_t offsetM)
{
    uint64_t perTokenScaleOffset = mnConfig.mIdx * mnConfig.baseM + offsetM;
    // GM copy per token scale
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams perTokenScaleParams;
    perTokenScaleParams.blockLen = curBaseM * sizeof(float);
    perTokenScaleParams.blockCount = 1;
    perTokenScaleParams.srcStride = 0;
    perTokenScaleParams.dstStride = 0;
    LocalTensor<float> perTokenScaleLocal = x1ScaleInQueue.AllocTensor<float>();
    DataCopyPad(perTokenScaleLocal, x1ScaleGm[perTokenScaleOffset], perTokenScaleParams, padParams);
    x1ScaleInQueue.EnQue(perTokenScaleLocal);

    x1ScaleInUb = x1ScaleInQueue.DeQue<float>();
    uint8_t repeatTimes = Ceil(curBaseM, 8); // curBaseM is 8 aligned;
    Brcb(x1ScaleBrcbLocal, x1ScaleInUb, repeatTimes, {1, 8});
    x1ScaleInQueue.FreeTensor(x1ScaleInUb);
}

} // namespace QUANT_MATMUL_REDUCE_SUM

#endif // ASCENDC_QUANT_MATMUL_REDUCE_SUM_QUANT_MIXCORE_H
