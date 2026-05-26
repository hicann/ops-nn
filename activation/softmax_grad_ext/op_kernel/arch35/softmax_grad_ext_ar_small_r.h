/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_grad_ext_ar_small_r.h
 * \brief
 */

#ifndef SOFTMAX_GRAD_EXT_AR_SMALL_R_H
#define SOFTMAX_GRAD_EXT_AR_SMALL_R_H

#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "softmax_grad_ext_base.h"

namespace SoftmaxGradExt {
using namespace AscendC;

template <typename T>
class SoftmaxGradExtARSmallR {
    static constexpr int32_t BUFFER_NUM_TWO = 2;
    static constexpr int32_t BUFFER_NUM_ONE = 1;
    static constexpr int32_t BUFFER_DEPTH = 1;
    static constexpr int64_t DATA_BLOCK_COUNT = 16;
    static constexpr int64_t DATA_BLOCK_COUNT_HALF = 8;
    static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint32_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();
    static constexpr float DUPLICATE_ZERO = 0.0;

public:
    __aicore__ inline SoftmaxGradExtARSmallR(){};
    __aicore__ inline void Init(
        GM_ADDR grad, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const SoftmaxGradExtARSmallRTilingData* tilingDataIn);
    __aicore__ inline void Process();
    __aicore__ inline SoftmaxGradExtARSmallR(TPipe* pipeIn)
    {
        pipe_ = pipeIn;
    }

private:
    __aicore__ inline void CopyInAndTransPose(int64_t xGmOffset, uint32_t curTileALen);
    __aicore__ inline void CalcReduceSum(
        const __ubuf__ T* gradLocalAddr, const __ubuf__ T* x1LocalAddr, __ubuf__ float* xSumLocalAddr,
        uint32_t curTileALen);
    __aicore__ inline void CalcOutput(
        const __ubuf__ T* gradLocalAddr, const __ubuf__ T* x1LocalAddr, const __ubuf__ T* x2LocalAddr,
        const __ubuf__ float* xSumLocalAddr, __ubuf__ T* yTempLocalAddr, uint32_t curTileALen);
    __aicore__ inline void CalcTranspose(LocalTensor<T> yTempLocal, uint32_t curTileALen);
    __aicore__ inline void CalcTransposeB16(LocalTensor<T> yTempLocal, uint32_t curTileALen);
    __aicore__ inline void CalcTransposeB32(LocalTensor<T> yTempLocal, uint32_t curTileALen);
    __aicore__ inline void CopyOutY(int64_t yGmOffset, uint32_t curTileALen);

private:
    const SoftmaxGradExtARSmallRTilingData* tilingData_;

    TPipe* pipe_;

    GlobalTensor<T> yGm_;
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;

    TQue<QuePosition::VECIN, BUFFER_NUM_TWO> gradQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM_TWO> x1Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM_TWO> x2Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM_ONE> x2ScalarQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_TWO> yQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM_TWO> xSumQueue_, yTempQueue_;

    static constexpr bool xIsFp32_ = IsSameType<T, float>::value;
    static constexpr int64_t rTileBase_ = IsSameType<T, float>::value ? DATA_BLOCK_COUNT_HALF : DATA_BLOCK_COUNT;
    int64_t aTileBase_ = DATA_BLOCK_COUNT;
    int64_t rLen_ = 0;
    int64_t rAligned_ = 0;
    int64_t ubFactor_ = 0;
    int64_t ubFactorTail_ = 0;

    int64_t curCoreANum_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t ubLoopNum_ = 0;
    int64_t xShapeLen_ = 0;
    int64_t blockIdx_ = 0;
    float x2ScalarValueB32 = 0;
    T x2ScalarValueB16 = 0;
};

template <typename T>
__aicore__ inline void SoftmaxGradExtARSmallR<T>::Init(
    GM_ADDR grad, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const SoftmaxGradExtARSmallRTilingData* tilingDataIn)
{
    tilingData_ = tilingDataIn;
    blockIdx_ = GetBlockIdx();
    usedCoreNum_ = tilingData_->usedCoreNums;
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    if (blockIdx_ == usedCoreNum_ - 1) {
        curCoreANum_ = tilingData_->aPerTailCore;
    } else {
        curCoreANum_ = tilingData_->aPerHeadCore;
    }

    ubFactor_ = tilingData_->ubFactor;
    ubLoopNum_ = ops::CeilDiv(curCoreANum_, ubFactor_);
    ubFactorTail_ = curCoreANum_ - (ubLoopNum_ - 1) * ubFactor_;
    rLen_ = tilingData_->r;
    rAligned_ = ops::CeilAlign(rLen_, rTileBase_);
    xShapeLen_ = ubFactor_ * rAligned_;

    gradGm_.SetGlobalBuffer((__gm__ T*)grad);
    x1Gm_.SetGlobalBuffer((__gm__ T*)x1);
    x2Gm_.SetGlobalBuffer((__gm__ T*)x2);
    yGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_->InitBuffer(x2ScalarQueue_, BUFFER_NUM_ONE, DATA_BLOCK_COUNT * sizeof(T));
    pipe_->InitBuffer(gradQueue_, BUFFER_NUM_TWO, xShapeLen_ * sizeof(T));
    pipe_->InitBuffer(x1Queue_, BUFFER_NUM_TWO, xShapeLen_ * sizeof(T));
    pipe_->InitBuffer(x2Queue_, BUFFER_NUM_TWO, xShapeLen_ * sizeof(T));
    pipe_->InitBuffer(xSumQueue_, BUFFER_NUM_TWO, ubFactor_ * sizeof(float));
    pipe_->InitBuffer(yTempQueue_, BUFFER_NUM_TWO, xShapeLen_ * sizeof(T));
    pipe_->InitBuffer(yQueue_, BUFFER_NUM_TWO, xShapeLen_ * sizeof(T));
}

template <typename T>
__aicore__ inline void SoftmaxGradExtARSmallR<T>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    if (tilingData_->x2IsScalar) {
        LocalTensor<T> x2Local_ = x2ScalarQueue_.AllocTensor<T>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T)), static_cast<uint32_t>(0),
            static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
        DataCopyPadExtParams<T> padParams{true, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0.0)};
        DataCopyPad<T>(x2Local_, x2Gm_[0], copyParams, padParams);
        event_t eventIdMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
        if constexpr (xIsFp32_) {
            x2ScalarValueB32 = x2Local_.GetValue(0);
        } else {
            x2ScalarValueB16 = x2Local_.GetValue(0);
        }
        x2ScalarQueue_.FreeTensor(x2Local_);
    }

    int64_t xOffsetBase = blockIdx_ * tilingData_->aPerHeadCore * rLen_;
    uint32_t curTileALen = (ubLoopNum_ == 1) ? ubFactorTail_ : ubFactor_;
    uint32_t nextTileALen = curTileALen;
    int32_t loopIdx = 0;
    CopyInAndTransPose(xOffsetBase, curTileALen);

    for (; loopIdx < ubLoopNum_ - 1; loopIdx++) {
        int64_t xOffset = xOffsetBase + loopIdx * ubFactor_ * rLen_;
        if (loopIdx + 1 == ubLoopNum_ - 1) {
            nextTileALen = ubFactorTail_;
        }

        LocalTensor<T> gradTensor = gradQueue_.DeQue<T>();
        LocalTensor<T> x1Tensor = x1Queue_.DeQue<T>();
        LocalTensor<T> x2Tensor = x2Queue_.DeQue<T>();
        LocalTensor<float> xSumTensor = xSumQueue_.AllocTensor<float>();
        LocalTensor<T> yTempTensor = yTempQueue_.AllocTensor<T>();
        __ubuf__ T* gradLocalAddr = (__ubuf__ T*)gradTensor.GetPhyAddr();
        __ubuf__ T* x1LocalAddr = (__ubuf__ T*)x1Tensor.GetPhyAddr();
        __ubuf__ T* x2LocalAddr = (__ubuf__ T*)x2Tensor.GetPhyAddr();
        __ubuf__ float* xSumLocalAddr = (__ubuf__ float*)xSumTensor.GetPhyAddr();
        __ubuf__ T* yTempLocalAddr = (__ubuf__ T*)yTempTensor.GetPhyAddr();

        Duplicate(xSumTensor, DUPLICATE_ZERO, curTileALen);
        CalcReduceSum(gradLocalAddr, x1LocalAddr, xSumLocalAddr, curTileALen);
        CopyInAndTransPose(xOffset + ubFactor_ * rLen_, nextTileALen);

        CalcOutput(gradLocalAddr, x1LocalAddr, x2LocalAddr, xSumLocalAddr, yTempLocalAddr, curTileALen);
        CalcTranspose(yTempTensor, curTileALen);
        gradQueue_.FreeTensor(gradTensor);
        x1Queue_.FreeTensor(x1Tensor);
        x2Queue_.FreeTensor(x2Tensor);
        xSumQueue_.FreeTensor(xSumTensor);
        yTempQueue_.FreeTensor(yTempTensor);
        CopyOutY(xOffset, curTileALen);
    }

    if (loopIdx == ubLoopNum_ - 1) {
        curTileALen = ubFactorTail_;
    }
    int64_t xOffset = xOffsetBase + loopIdx * ubFactor_ * rLen_;
    LocalTensor<T> gradTensor = gradQueue_.DeQue<T>();
    LocalTensor<T> x1Tensor = x1Queue_.DeQue<T>();
    LocalTensor<T> x2Tensor = x2Queue_.DeQue<T>();

    LocalTensor<float> xSumTensor = xSumQueue_.AllocTensor<float>();
    LocalTensor<T> yTempTensor = yTempQueue_.AllocTensor<T>();

    __ubuf__ T* gradLocalAddr = (__ubuf__ T*)gradTensor.GetPhyAddr();
    __ubuf__ T* x1LocalAddr = (__ubuf__ T*)x1Tensor.GetPhyAddr();
    __ubuf__ T* x2LocalAddr = (__ubuf__ T*)x2Tensor.GetPhyAddr();
    __ubuf__ float* xSumLocalAddr = (__ubuf__ float*)xSumTensor.GetPhyAddr();
    __ubuf__ T* yTempLocalAddr = (__ubuf__ T*)yTempTensor.GetPhyAddr();
    Duplicate(xSumTensor, DUPLICATE_ZERO, curTileALen);
    CalcReduceSum(gradLocalAddr, x1LocalAddr, xSumLocalAddr, curTileALen);
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    CalcOutput(gradLocalAddr, x1LocalAddr, x2LocalAddr, xSumLocalAddr, yTempLocalAddr, curTileALen);
    CalcTranspose(yTempTensor, curTileALen);
    gradQueue_.FreeTensor(gradTensor);
    x1Queue_.FreeTensor(x1Tensor);
    x2Queue_.FreeTensor(x2Tensor);
    xSumQueue_.FreeTensor(xSumTensor);
    yTempQueue_.FreeTensor(yTempTensor);
    CopyOutY(xOffset, curTileALen);
}

template <typename T>
__aicore__ inline void SoftmaxGradExtARSmallR<T>::CalcReduceSum(
    const __ubuf__ T* gradLocalAddr, const __ubuf__ T* x1LocalAddr, __ubuf__ float* xSumLocalAddr, uint32_t curTileALen)
{
    uint32_t aLen = ops::CeilAlign(curTileALen, static_cast<uint32_t>(aTileBase_));
    uint16_t rLen = static_cast<uint16_t>(rLen_);
    uint16_t loopA0Num = ops::CeilDiv(aLen, VL_FP32);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregX0;
        MicroAPI::RegTensor<float> vregX1;
        MicroAPI::RegTensor<float> vregSum;
        MicroAPI::RegTensor<T> vregX0B16;
        MicroAPI::RegTensor<T> vregX1B16;
        MicroAPI::MaskReg pregMask;
        for (uint16_t i = 0; i < rLen; i++) {
            uint32_t sreg = curTileALen;
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            for (uint16_t k = 0; k < loopA0Num; k++) {
                pregMask = MicroAPI::UpdateMask<float>(sreg);
                uint32_t xOffset = i * aLen + k * VL_FP32;
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                    vregSum, (__ubuf__ float*)xSumLocalAddr + k * VL_FP32);
                if constexpr (xIsFp32_) {
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                        vregX0, (__ubuf__ float*)gradLocalAddr + xOffset);
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                        vregX1, (__ubuf__ float*)x1LocalAddr + xOffset);
                } else { // fp16, bf16
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregX0B16, ((__ubuf__ T*)gradLocalAddr + xOffset));
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregX1B16, ((__ubuf__ T*)x1LocalAddr + xOffset));
                    MicroAPI::Cast<float, T, castTraitFp16ToFp32>(vregX0, vregX0B16, pregMask);
                    MicroAPI::Cast<float, T, castTraitFp16ToFp32>(vregX1, vregX1B16, pregMask);
                }
                MicroAPI::MulAddDst(vregSum, vregX0, vregX1, pregMask);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM>(
                    (__ubuf__ float*)xSumLocalAddr + k * VL_FP32, vregSum, pregMask);
            }
        }
    }
}

template <typename T>
__aicore__ inline void SoftmaxGradExtARSmallR<T>::CalcOutput(
    const __ubuf__ T* gradLocalAddr, const __ubuf__ T* x1LocalAddr, const __ubuf__ T* x2LocalAddr,
    const __ubuf__ float* xSumLocalAddr, __ubuf__ T* yTempLocalAddr, uint32_t curTileALen)
{
    uint32_t aLen = ops::CeilAlign(curTileALen, static_cast<uint32_t>(aTileBase_));
    uint16_t rLen = static_cast<uint16_t>(rLen_);
    uint16_t loopA0Num = static_cast<uint16_t>(ops::CeilDiv(aLen, VL_FP32));
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> vregX0;
        MicroAPI::RegTensor<float> vregX1;
        MicroAPI::RegTensor<float> vregX2;
        MicroAPI::RegTensor<float> vregSum;
        MicroAPI::RegTensor<T> vregX0B16;
        MicroAPI::RegTensor<T> vregX1B16;
        MicroAPI::RegTensor<T> vregX2B16;

        MicroAPI::RegTensor<float> vregSub;
        MicroAPI::RegTensor<float> vregMul0;
        MicroAPI::RegTensor<float> vregMul1;
        MicroAPI::RegTensor<float> vregResult;
        MicroAPI::RegTensor<T> vregResultB16;

        MicroAPI::MaskReg pregMask;
        for (uint16_t i = 0; i < rLen; i++) {
            uint32_t sreg = curTileALen;
            for (uint16_t k = 0; k < loopA0Num; k++) {
                pregMask = MicroAPI::UpdateMask<float>(sreg);
                uint32_t xOffset = i * aLen + k * VL_FP32;
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                    vregSum, (__ubuf__ float*)xSumLocalAddr + k * VL_FP32);
                if constexpr (xIsFp32_) {
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                        vregX0, (__ubuf__ float*)gradLocalAddr + xOffset);
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                        vregX1, (__ubuf__ float*)x1LocalAddr + xOffset);
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                        vregX2, (__ubuf__ float*)x2LocalAddr + xOffset);
                } else {
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregX0B16, ((__ubuf__ T*)gradLocalAddr + xOffset));
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregX1B16, ((__ubuf__ T*)x1LocalAddr + xOffset));
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregX2B16, ((__ubuf__ T*)x2LocalAddr + xOffset));
                    MicroAPI::Cast<float, T, castTraitFp16ToFp32>(vregX0, vregX0B16, pregMask);
                    MicroAPI::Cast<float, T, castTraitFp16ToFp32>(vregX1, vregX1B16, pregMask);
                    MicroAPI::Cast<float, T, castTraitFp16ToFp32>(vregX2, vregX2B16, pregMask);
                }
                MicroAPI::Mul(vregMul0, vregX1, vregX0, pregMask);
                MicroAPI::Mul(vregMul1, vregX1, vregSum, pregMask);
                MicroAPI::Sub(vregSub, vregMul0, vregMul1, pregMask);
                MicroAPI::Mul(vregResult, vregSub, vregX2, pregMask);
                if constexpr (xIsFp32_) {
                    MicroAPI::DataCopy((__ubuf__ float*)yTempLocalAddr + xOffset, vregResult, pregMask);
                } else { // fp16、bf16
                    MicroAPI::Cast<T, float, castTraitFp32ToFp16>(vregResultB16, vregResult, pregMask);
                    MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(
                        (__ubuf__ T*)yTempLocalAddr + xOffset, vregResultB16, pregMask);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void SoftmaxGradExtARSmallR<T>::CalcTranspose(LocalTensor<T> yTempLocal, uint32_t curTileALen)
{
    if constexpr (xIsFp32_) {
        CalcTransposeB32(yTempLocal, curTileALen);
    } else {
        CalcTransposeB16(yTempLocal, curTileALen);
    }
}

template <typename T>
__aicore__ inline void SoftmaxGradExtARSmallR<T>::CalcTransposeB32(LocalTensor<T> yTempLocal, uint32_t curTileALen)
{
    LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();
    uint32_t aLen = ops::CeilAlign(curTileALen, static_cast<uint32_t>(aTileBase_));
    int32_t rRepeartTimes = ops::CeilDiv(static_cast<int64_t>(rAligned_), rTileBase_);
    for (int32_t i = 0; i < rRepeartTimes; i++) {
        TransDataTo5HDParams params;
        LocalTensor<T> srcLocalList[DATA_BLOCK_COUNT];
        LocalTensor<T> dstLocalList[DATA_BLOCK_COUNT];

        uint32_t aRepeartTimes = ops::CeilDiv(aLen, uint32_t(DATA_BLOCK_COUNT));
        params.repeatTimes = aRepeartTimes;
        params.srcRepStride = aRepeartTimes == 1 ? 0 : CONST_TWO;
        params.dstRepStride = aRepeartTimes == 1 ? 0 : DATA_BLOCK_COUNT * rRepeartTimes;
        for (int32_t j = 0; j < DATA_BLOCK_COUNT_HALF; j++) {
            uint32_t offset = DATA_BLOCK_COUNT_HALF * aLen * i + aLen * j;
            srcLocalList[j] = yTempLocal[offset];
            srcLocalList[j + DATA_BLOCK_COUNT_HALF] = yTempLocal[offset + DATA_BLOCK_COUNT_HALF];
        }
        for (int32_t j = 0; j < DATA_BLOCK_COUNT_HALF; j++) {
            uint32_t offset = DATA_BLOCK_COUNT_HALF * i + DATA_BLOCK_COUNT_HALF * rRepeartTimes * j;
            dstLocalList[j * CONST_TWO] = yLocal[offset];
            dstLocalList[j * CONST_TWO + 1] =
                yLocal[offset + DATA_BLOCK_COUNT_HALF * DATA_BLOCK_COUNT_HALF * rRepeartTimes];
        }

        AscendC::TransDataTo5HD(dstLocalList, srcLocalList, params);
    }
    yQueue_.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void SoftmaxGradExtARSmallR<T>::CalcTransposeB16(LocalTensor<T> yTempLocal, uint32_t curTileALen)
{
    LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();
    uint32_t aLen = ops::CeilAlign(curTileALen, static_cast<uint32_t>(aTileBase_));
    int32_t rRepeartTimes = ops::CeilDiv(static_cast<int64_t>(rAligned_), rTileBase_);

    for (int32_t i = 0; i < rRepeartTimes; i++) {
        TransDataTo5HDParams params;
        LocalTensor<T> srcLocalList[DATA_BLOCK_COUNT];
        LocalTensor<T> dstLocalList[DATA_BLOCK_COUNT];

        uint32_t aRepeartTimes = ops::CeilDiv(aLen, static_cast<uint32_t>(rTileBase_));
        params.repeatTimes = aRepeartTimes;
        params.srcRepStride = aRepeartTimes == 1 ? 0 : 1;
        params.dstRepStride = aRepeartTimes == 1 ? 0 : (rTileBase_ * rRepeartTimes);

        for (int32_t j = 0; j < DATA_BLOCK_COUNT; j++) {
            uint32_t offset = rTileBase_ * aLen * i + aLen * j;
            srcLocalList[j] = yTempLocal[offset];
        }
        for (int32_t j = 0; j < DATA_BLOCK_COUNT; j++) {
            uint32_t offset = rTileBase_ * i + rAligned_ * j;
            dstLocalList[j] = yLocal[offset];
        }
        AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, params);
    }
    yQueue_.EnQue(yLocal);
}

template <typename T>
__aicore__ inline void SoftmaxGradExtARSmallR<T>::CopyInAndTransPose(int64_t xGmOffset, uint32_t curTileALen)
{
    static constexpr MultiCopyConfig config = {false};
    uint32_t aAligned = ops::CeilAlign(curTileALen, static_cast<uint32_t>(aTileBase_));
    MultiCopyLoopInfo<CONST_TWO> copyLoopInfo;
    copyLoopInfo.loopSrcStride[0] = 1;
    copyLoopInfo.loopSrcStride[1] = rLen_;
    copyLoopInfo.loopDstStride[0] = aAligned;
    copyLoopInfo.loopDstStride[1] = 1;
    copyLoopInfo.loopSize[0] = rLen_;
    copyLoopInfo.loopSize[1] = curTileALen;
    copyLoopInfo.loopRpSize[0] = aAligned - curTileALen;
    copyLoopInfo.loopRpSize[1] = 0;
    MultiCopyParams<T, CONST_TWO> params = {copyLoopInfo, 0};

    LocalTensor<T> gradLocal_ = gradQueue_.AllocTensor<T>();
    DataCopy<T, CONST_TWO, config>(gradLocal_, gradGm_[xGmOffset], params);
    gradQueue_.EnQue(gradLocal_);

    LocalTensor<T> x1Local_ = x1Queue_.AllocTensor<T>();
    DataCopy<T, CONST_TWO, config>(x1Local_, x1Gm_[xGmOffset], params);
    x1Queue_.EnQue(x1Local_);

    if (tilingData_->x2IsScalar == 1) {
        LocalTensor<T> x2Local_ = x2Queue_.AllocTensor<T>();
        if constexpr (xIsFp32_) {
            Duplicate<T>(x2Local_, x2ScalarValueB32, aAligned * rLen_);
        } else {
            Duplicate<T>(x2Local_, x2ScalarValueB16, aAligned * rLen_);
        }
        x2Queue_.EnQue(x2Local_);
    } else {
        LocalTensor<T> x2Local_ = x2Queue_.AllocTensor<T>();
        DataCopy<T, CONST_TWO, config>(x2Local_, x2Gm_[xGmOffset], params);
        x2Queue_.EnQue(x2Local_);
    }
}

template <typename T>
__aicore__ inline void SoftmaxGradExtARSmallR<T>::CopyOutY(int64_t yGmOffset, uint32_t curTileALen)
{
    LocalTensor<T> yLocal = yQueue_.DeQue<T>();
    DataCopyParams copyOutParams;
    copyOutParams.blockCount = curTileALen;
    copyOutParams.blockLen = rLen_ * sizeof(T);
    copyOutParams.dstStride = 0;
    copyOutParams.srcStride = 0;
    DataCopyPad(yGm_[yGmOffset], yLocal, copyOutParams);
    yQueue_.FreeTensor(yLocal);
}

} // namespace SoftmaxGradExt

#endif