/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file quant_max_per_tensor_regbase.h
 * @brief QuantMax Per-Tensor Kernel
 *
 */

#ifndef QUANT_MAX_PER_TENSOR_REGBASE_H
#define QUANT_MAX_PER_TENSOR_REGBASE_H
#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "quant_max_tiling_data.h"
#include "quant_max_common.h"

namespace QuantMax {
using namespace AscendC;

template <typename T, typename U, uint64_t RoundMode>
class QuantMaxPerTensorRegbase : public QuantMaxBase<T, U, RoundMode> {
public:
    __aicore__ inline QuantMaxPerTensorRegbase(const QuantMaxTilingData* tilingData) : tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR y, GM_ADDR amax, GM_ADDR workspace);
    __aicore__ inline void Process();
    __aicore__ inline void MergeAmax();

private:
    __aicore__ inline void CopyXAndCompute(int64_t dataCount, int64_t offset);
    __aicore__ inline void CopyInX(int64_t xLen, int64_t xInOffset);
    __aicore__ inline void Compute(int64_t dataCount);
    __aicore__ inline void CopyOutY(int64_t yLen, int64_t yOutOffset);

private:
    constexpr static int32_t bufferNum_ = 2;
    constexpr static int64_t WS_RESERVE = 1024 * 1024;
    constexpr static int32_t REDUCE_TMP_SIZE = 8 * 1024;

    TPipe pipe_;
    TQue<QuePosition::VECIN, bufferNum_> inQueueX_;
    TQue<QuePosition::VECOUT, bufferNum_> outQueueY_;
    TBuf<TPosition::VECCALC> tileMaxBuf_;
    TBuf<TPosition::VECCALC> amaxBuf_;
    TBuf<TPosition::VECCALC> reduceDstBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<U> yGm_;
    GlobalTensor<T> amaxGm_;
    GlobalTensor<float> gmWorkspaceFp32_;
    GlobalTensor<uint32_t> gmCounter_;

    const QuantMaxTilingData* tilingData_;
    int32_t blockIdx_ = 0;
    int64_t gmXOffset_ = 0;
    int64_t blockLen_ = 1;
    float scaleValue_ = 0.0f;
    uint16_t VL = 0;
};

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxPerTensorRegbase<T, U, RoundMode>::Init(
    GM_ADDR x, GM_ADDR scale, GM_ADDR y, GM_ADDR amax, GM_ADDR workspace)
{
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(y));
    amaxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(amax));

    gmWorkspaceFp32_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + WS_RESERVE), 64);

    scaleValue_ = (reinterpret_cast<__gm__ float*>(scale))[0];

    if (blockIdx_ == tilingData_->numCore - 1) {
        blockLen_ = tilingData_->blockTailFactor;
    } else {
        blockLen_ = tilingData_->blockFactor;
    }

    pipe_.InitBuffer(inQueueX_, bufferNum_, tilingData_->baseLen * sizeof(T));
    pipe_.InitBuffer(outQueueY_, bufferNum_, tilingData_->baseLen * sizeof(U));

    pipe_.InitBuffer(tileMaxBuf_, VECTOR_REG_WIDTH);
    LocalTensor<float> initLocal = tileMaxBuf_.Get<float>();
    Duplicate<float>(initLocal, -1.0f, VECTOR_REG_WIDTH / sizeof(float));
    pipe_.InitBuffer(amaxBuf_, 64 * sizeof(float));
    pipe_.InitBuffer(reduceDstBuf_, this->BLOCK_SIZE);
    VL = VECTOR_REG_WIDTH / sizeof(float);
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxPerTensorRegbase<T, U, RoundMode>::Process()
{
    if (blockIdx_ >= tilingData_->numCore) {
        return;
    }

    gmXOffset_ = blockIdx_ * tilingData_->blockFactor;

    int64_t lenLoopNum = blockLen_ / tilingData_->baseLen;
    int64_t lenLoopTail = blockLen_ % tilingData_->baseLen;

    for (int64_t i = 0; i < lenLoopNum; ++i) {
        CopyXAndCompute(tilingData_->baseLen, gmXOffset_ + i * tilingData_->baseLen);
    }
    if (lenLoopTail != 0) {
        CopyXAndCompute(lenLoopTail, gmXOffset_ + lenLoopNum * tilingData_->baseLen);
    }

    LocalTensor<float> tileMaxLocal = tileMaxBuf_.Get<float>();

    // V → MTE3 同步
    int32_t eventIDVToMTE3 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3);
    SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

    DataCopyExtParams amaxCopyParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(gmWorkspaceFp32_[blockIdx_], tileMaxLocal, amaxCopyParams);
    PipeBarrier<PIPE_ALL>();
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxPerTensorRegbase<T, U, RoundMode>::CopyXAndCompute(int64_t dataCount, int64_t offset)
{
    CopyInX(dataCount, offset);
    Compute(dataCount);
    CopyOutY(dataCount, offset);
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxPerTensorRegbase<T, U, RoundMode>::CopyInX(int64_t xLen, int64_t xInOffset)
{
    LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(xLen * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    DataCopyPad<T>(xLocal, xGm_[xInOffset], copyParams, padParams);
    inQueueX_.EnQue(xLocal);
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxPerTensorRegbase<T, U, RoundMode>::Compute(int64_t dataCount)
{
    LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
    LocalTensor<U> outLocal = outQueueY_.AllocTensor<U>();
    LocalTensor<float> tileMaxLocal = tileMaxBuf_.Get<float>();

    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ U* outLocalAddr = (__local_mem__ U*)outLocal.GetPhyAddr();
    __local_mem__ float* tileMaxLocalAddr = (__local_mem__ float*)tileMaxLocal.GetPhyAddr();

    uint16_t vfLoopNum = (dataCount + VL - 1) / VL;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> vregX;
        AscendC::Reg::RegTensor<float> vregFloatAbsT;
        AscendC::Reg::RegTensor<float> vregTileMaxFp32;
        AscendC::Reg::RegTensor<float> vregFloatX;
        AscendC::Reg::RegTensor<float> vregS;
        AscendC::Reg::RegTensor<float> vregFloatY;
        AscendC::Reg::RegTensor<U> vregY;

        AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float>();
        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vregTileMaxFp32, tileMaxLocalAddr);

        AscendC::Reg::Duplicate<float>(vregS, scaleValue_, mask);

        uint32_t count = dataCount;
        for (uint16_t i = 0; i < vfLoopNum; i++) {
            mask = AscendC::Reg::UpdateMask<float>(count);

            if constexpr (IsSameType<T, float>::value) {
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vregFloatX, xLocalAddr + i * VL);
            } else if constexpr (IsSameType<T, half>::value) {
                AscendC::Reg::DataCopy<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX, xLocalAddr + i * VL);
                AscendC::Reg::Cast<float, half, QuantMaxBase<T, U, RoundMode>::CAST_TRAIT_HALF_TO_FP32>(
                    vregFloatX, vregX, mask);
            } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                AscendC::Reg::DataCopy<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX, xLocalAddr + i * VL);
                AscendC::Reg::Cast<float, bfloat16_t, QuantMaxBase<T, U, RoundMode>::CAST_TRAIT_BF16_TO_FP32>(
                    vregFloatX, vregX, mask);
            }

            AscendC::Reg::Abs<float>(vregFloatAbsT, vregFloatX, mask);
            AscendC::Reg::Max<float>(vregTileMaxFp32, vregTileMaxFp32, vregFloatAbsT, mask);
            AscendC::Reg::Mul(vregFloatY, vregFloatX, vregS, mask);

            if constexpr (IsSameType<U, hifloat8_t>::value) {
                AscendC::Reg::Cast<U, float, QuantMaxBase<T, U, RoundMode>::CAST_TRAIT_FP32_TO_HIFP8>(
                    vregY, vregFloatY, mask);
                AscendC::Reg::DataCopy<U, AscendC::Reg::StoreDist::DIST_PACK4_B32>(outLocalAddr + i * VL, vregY, mask);
            } else if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
                AscendC::Reg::Cast<U, float, QuantMaxBase<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E5M2>(
                    vregY, vregFloatY, mask);
                AscendC::Reg::DataCopy<U, AscendC::Reg::StoreDist::DIST_PACK4_B32>(outLocalAddr + i * VL, vregY, mask);
            } else if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
                AscendC::Reg::Cast<U, float, QuantMaxBase<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E4M3>(
                    vregY, vregFloatY, mask);
                AscendC::Reg::DataCopy<U, AscendC::Reg::StoreDist::DIST_PACK4_B32>(outLocalAddr + i * VL, vregY, mask);
            }
        }

        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(vregTileMaxFp32, vregTileMaxFp32, mask);
        AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM>(tileMaxLocalAddr, vregTileMaxFp32, mask);
    }

    inQueueX_.FreeTensor(xLocal);
    outQueueY_.EnQue(outLocal);
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxPerTensorRegbase<T, U, RoundMode>::CopyOutY(int64_t yLen, int64_t yOutOffset)
{
    LocalTensor<U> outLocal = outQueueY_.DeQue<U>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(yLen * sizeof(U)), 0, 0, 0};
    DataCopyPad(yGm_[yOutOffset], outLocal, copyParams);
    outQueueY_.FreeTensor(outLocal);
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void QuantMaxPerTensorRegbase<T, U, RoundMode>::MergeAmax()
{
    SyncAll();
    if (blockIdx_ != 0)
        return;

    LocalTensor<float> amaxFp32Local = amaxBuf_.Get<float>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(tilingData_->numCore * sizeof(float)), 0, 0, 0};
    DataCopyPad(amaxFp32Local, gmWorkspaceFp32_[0], copyParams, {false, 0, 0, 0});

    int32_t eventIDMTE2ToV = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V);
    SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);

    LocalTensor<float> dstLocal = reduceDstBuf_.Get<float>();
    ReduceMax<float>(dstLocal, amaxFp32Local, amaxFp32Local, static_cast<int32_t>(tilingData_->numCore), false);

    if constexpr (IsSameType<T, float>::value) {
        int32_t eventIDVToMTE3 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3);
        SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

        DataCopyExtParams amaxCopyParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(amaxGm_, dstLocal, amaxCopyParams);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        LocalTensor<T> amaxBf16Local = amaxBuf_.Get<T>();
        Cast<T, float>(amaxBf16Local, dstLocal, RoundMode::CAST_RINT, 1);
        
        int32_t eventIDVToMTE3 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3);
        SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopyExtParams amaxCopyParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(amaxGm_, amaxBf16Local, amaxCopyParams);
    }
}

} // namespace QuantMax
#endif