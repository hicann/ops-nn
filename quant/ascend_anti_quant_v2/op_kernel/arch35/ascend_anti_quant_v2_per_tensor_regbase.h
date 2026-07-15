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
 * \file ascend_anti_quant_v2_per_tensor_regbase.h
 * \brief ascendantiquantv2 kernel
 */

#ifndef ASCEND_ANTI_QUANT_V2_PER_TENSOR_REGBASE_H_
#define ASCEND_ANTI_QUANT_V2_PER_TENSOR_REGBASE_H_

#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "ascend_anti_quant_v2_common.h"
#include "ascend_anti_quant_v2_struct.h"

namespace AscendAntiQuantV2 {
using namespace AscendC;
template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset = TPL_HAS_OFFSET>
class AscendAntiQuantV2PerTensorRegbase : public AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode> {
public:
    __aicore__ inline AscendAntiQuantV2PerTensorRegbase(const AscendAntiQuantV2TilingData* tilingData)
        : tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyXAndCompute(int64_t dataCount, int64_t offset, LocalTensor<T1>& sLocal,
                                           LocalTensor<T2>& oLocal);
    __aicore__ inline void CopyInScale();
    __aicore__ inline void CopyInOffset();
    __aicore__ inline void CopyInX(int64_t xLen, int64_t xInOffset);
    __aicore__ inline void CopyOutY(int64_t yLen, int64_t yOutOffset);
    __aicore__ inline void Compute(int64_t dataCount, LocalTensor<T1>& sLocal, LocalTensor<T2>& oLocal);

private:
    using xCopyDtype = std::conditional_t<IsSameType<T, int4b_t>::value, uint8_t, T>;
    constexpr static int32_t bufferNum_ = 2;
    TPipe pipe_;
    TQue<QuePosition::VECIN, bufferNum_> inQueueX_;
    TQue<QuePosition::VECIN, bufferNum_> inQueueScale_;
    TQue<QuePosition::VECIN, bufferNum_> inQueueOffset_;
    TQue<QuePosition::VECOUT, bufferNum_> outQueueY_;
    GlobalTensor<uint8_t> xGm_;
    GlobalTensor<T1> scaleGm_;
    GlobalTensor<T2> offsetGm_;
    GlobalTensor<U> yGm_;

    const AscendAntiQuantV2TilingData* tilingData_;
    int32_t blockIdx_ = 0;
    int64_t gmXOffset_ = 0;
    int64_t blockN_ = 1;
    int64_t blockLen_ = 1;
};

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset>
__aicore__ inline void AscendAntiQuantV2PerTensorRegbase<T, T1, T2, U, SqrtMode, HasOffset>::Init(GM_ADDR x,
                                                                                                  GM_ADDR scale,
                                                                                                  GM_ADDR offset,
                                                                                                  GM_ADDR y)
{
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T1*>(scale));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(y));

    blockN_ = tilingData_->dim0;
    if (blockIdx_ == tilingData_->numCore - 1) {
        blockLen_ = tilingData_->blockTailFactor;
    } else {
        blockLen_ = tilingData_->blockFactor;
    }

    // calc n size to alloc queue
    pipe_.InitBuffer(inQueueX_, bufferNum_, tilingData_->baseN * tilingData_->baseLen * sizeof(xCopyDtype));
    pipe_.InitBuffer(inQueueScale_, bufferNum_, this->BLOCK_SIZE);

    if constexpr (HasOffset) {
        offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T2*>(offset));
        pipe_.InitBuffer(inQueueOffset_, bufferNum_, this->BLOCK_SIZE);
    }

    pipe_.InitBuffer(outQueueY_, bufferNum_, tilingData_->baseN * tilingData_->baseLen * sizeof(U));
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset>
__aicore__ inline void AscendAntiQuantV2PerTensorRegbase<T, T1, T2, U, SqrtMode, HasOffset>::Process()
{
    if (blockIdx_ >= tilingData_->numCore) {
        return;
    }

    gmXOffset_ = blockIdx_ * tilingData_->blockFactor;

    // main loop with column, for scale and offset only need copy once
    int64_t lenLoopNum = blockLen_ / tilingData_->baseLen;
    int64_t lenLoopTail = blockLen_ % tilingData_->baseLen;
    CopyInScale();

    LocalTensor<T1> sLocal = inQueueScale_.DeQue<T1>();

    if constexpr (HasOffset) {
        CopyInOffset();
        LocalTensor<T2> oLocal = inQueueOffset_.DeQue<T2>();
        for (int64_t i = 0; i < lenLoopNum; ++i) {
            CopyXAndCompute(tilingData_->baseLen, gmXOffset_ + i * tilingData_->baseLen, sLocal, oLocal);
        }
        if (lenLoopTail != 0) {
            CopyXAndCompute(lenLoopTail, gmXOffset_ + lenLoopNum * tilingData_->baseLen, sLocal, oLocal);
        }
        inQueueOffset_.FreeTensor(oLocal);
    } else {
        LocalTensor<T2> oLocal;
        for (int64_t i = 0; i < lenLoopNum; ++i) {
            CopyXAndCompute(tilingData_->baseLen, gmXOffset_ + i * tilingData_->baseLen, sLocal, oLocal);
        }
        if (lenLoopTail != 0) {
            CopyXAndCompute(lenLoopTail, gmXOffset_ + lenLoopNum * tilingData_->baseLen, sLocal, oLocal);
        }
    }
    inQueueScale_.FreeTensor(sLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset>
__aicore__ inline void AscendAntiQuantV2PerTensorRegbase<T, T1, T2, U, SqrtMode, HasOffset>::CopyInScale()
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = sizeof(T1);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    copyParams.rsv = 0;
    LocalTensor<T1> sLocal = inQueueScale_.AllocTensor<T1>();
    DataCopyPad(sLocal, scaleGm_, copyParams, {false, 0, 0, 0});
    inQueueScale_.EnQue(sLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset>
__aicore__ inline void AscendAntiQuantV2PerTensorRegbase<T, T1, T2, U, SqrtMode, HasOffset>::CopyInOffset()
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = sizeof(T2);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    copyParams.rsv = 0;
    LocalTensor<T2> oLocal = inQueueOffset_.AllocTensor<T2>();
    DataCopyPad(oLocal, offsetGm_, copyParams, {false, 0, 0, 0});
    inQueueOffset_.EnQue(oLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset>
__aicore__ inline void AscendAntiQuantV2PerTensorRegbase<T, T1, T2, U, SqrtMode, HasOffset>::CopyXAndCompute(
    int64_t dataCount, int64_t offset, LocalTensor<T1>& sLocal, LocalTensor<T2>& oLocal)
{
    CopyInX(dataCount, offset);
    Compute(dataCount, sLocal, oLocal);
    CopyOutY(dataCount, offset);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset>
__aicore__ inline void AscendAntiQuantV2PerTensorRegbase<T, T1, T2, U, SqrtMode, HasOffset>::CopyInX(int64_t xLen,
                                                                                                     int64_t xInOffset)
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        xInOffset = xInOffset >> 1;
    }

    LocalTensor<uint8_t> xLocal = inQueueX_.AllocTensor<uint8_t>();
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<uint8_t> padParams = {false, 0, 0, 0};
    this->GetXInCopyParams(tilingData_->dim1, tilingData_->baseLen, tilingData_->baseN, xLen, copyParams);
    DataCopyPad<uint8_t>(xLocal, xGm_[xInOffset], copyParams, padParams);
    inQueueX_.EnQue(xLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset>
__aicore__ inline void AscendAntiQuantV2PerTensorRegbase<T, T1, T2, U, SqrtMode, HasOffset>::Compute(
    int64_t dataCount, LocalTensor<T1>& sLocal, LocalTensor<T2>& oLocal)
{
    LocalTensor<xCopyDtype> xLocal = inQueueX_.DeQue<xCopyDtype>();
    LocalTensor<U> outLocal = outQueueY_.AllocTensor<U>();

    __local_mem__ xCopyDtype* xLocalAddr = (__local_mem__ xCopyDtype*)xLocal.GetPhyAddr();
    __local_mem__ T1* scaleLocalAddr = (__local_mem__ T1*)sLocal.GetPhyAddr();
    __local_mem__ U* outLocalAddr = (__local_mem__ U*)outLocal.GetPhyAddr();

    uint16_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint16_t HalfVL = VL / 2;
    uint16_t vfLoopNum = (dataCount + VL - 1) / VL;

    __VEC_SCOPE__
    {
        // x: int4, int8, hifp8, fp8_e5m2, fp8_e4m3
        AscendC::Reg::RegTensor<xCopyDtype> vregX;
        AscendC::Reg::RegTensor<float> vregFloatX;
        // scales: fp32, bf16
        AscendC::Reg::RegTensor<T1> vregS;
        AscendC::Reg::RegTensor<float> vregFloatS;

        // offset: fp32, bf16 (only used with offset)
        AscendC::Reg::RegTensor<T2> vregO;
        AscendC::Reg::RegTensor<float> vregFloatO;

        // y: fp16, bf16
        AscendC::Reg::RegTensor<float> vregFloatY;
        AscendC::Reg::RegTensor<U> vregY;

        AscendC::Reg::RegTensor<float> vregTmp1;
        AscendC::Reg::MaskReg mask;

        mask = AscendC::Reg::CreateMask<float>();
        uint32_t count = dataCount;
        for (uint16_t i = 0; i < vfLoopNum; i++) {
            mask = AscendC::Reg::UpdateMask<float>(count);
            // ld and cast for x
            __local_mem__ xCopyDtype* xSrc = IsSameType<T, int4b_t>::value ? xLocalAddr + i * HalfVL :
                                                                             xLocalAddr + i * VL;
            this->template LoadCastXToFloat<T>(vregX, vregFloatX, xSrc, mask);

            // ld and cast for scale
            if constexpr (IsSameType<T1, float>::value) {
                // fp32
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vregFloatS, scaleLocalAddr);
            } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
                // bf16
                AscendC::Reg::DataCopy<T1, AscendC::Reg::LoadDist::DIST_BRC_B16>(vregS, scaleLocalAddr);
                AscendC::Reg::Cast<float, T1, AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CAST_TRAIT_BF16_TO_FP32>(
                    vregFloatS, vregS, mask);
            }

            // compute
            if constexpr (HasOffset) {
                // ld and cast for offset
                __local_mem__ T2* offsetLocalAddr = (__local_mem__ T2*)oLocal.GetPhyAddr();
                if constexpr (IsSameType<T2, float>::value) {
                    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vregFloatO, offsetLocalAddr);
                } else if constexpr (IsSameType<T2, bfloat16_t>::value) {
                    AscendC::Reg::DataCopy<T2, AscendC::Reg::LoadDist::DIST_BRC_B16>(vregO, offsetLocalAddr);
                    AscendC::Reg::Cast<float, T2,
                                       AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CAST_TRAIT_BF16_TO_FP32>(
                        vregFloatO, vregO, mask);
                }
                AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregTmp1, vregFloatX, vregFloatO, mask);
                if constexpr (SqrtMode == TPL_SQRT_MODE) {
                    AscendC::Reg::Mul(vregFloatS, vregFloatS, vregFloatS, mask);
                }
                AscendC::Reg::Mul(vregFloatY, vregTmp1, vregFloatS, mask);
            } else {
                if constexpr (SqrtMode == TPL_SQRT_MODE) {
                    AscendC::Reg::Mul(vregFloatS, vregFloatS, vregFloatS, mask);
                }
                AscendC::Reg::Mul(vregFloatY, vregFloatX, vregFloatS, mask);
            }

            // cast and sd for y
            if constexpr (IsSameType<U, half>::value) {
                AscendC::Reg::Cast<half, float, AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CAST_TRAIT_FP32_TO_HALF>(
                    vregY, vregFloatY, mask);
                AscendC::Reg::DataCopy<U, AscendC::Reg::StoreDist::DIST_PACK_B32>(outLocalAddr + i * VL, vregY, mask);
            } else if constexpr (IsSameType<U, bfloat16_t>::value) {
                AscendC::Reg::Cast<U, float, AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CAST_TRAIT_FP32_TO_BF16>(
                    vregY, vregFloatY, mask);
                AscendC::Reg::DataCopy<U, AscendC::Reg::StoreDist::DIST_PACK_B32>(outLocalAddr + i * VL, vregY, mask);
            }
        }
    }
    inQueueX_.FreeTensor(xLocal);
    outQueueY_.EnQue(outLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset>
__aicore__ inline void AscendAntiQuantV2PerTensorRegbase<T, T1, T2, U, SqrtMode, HasOffset>::CopyOutY(
    int64_t yLen, int64_t yOutOffset)
{
    LocalTensor<U> outLocal = outQueueY_.DeQue<U>();
    DataCopyExtParams copyParams;
    this->GetOutCopyParams(tilingData_->dim1, tilingData_->baseLen, tilingData_->baseN, yLen, copyParams);
    DataCopyPad<U>(yGm_[yOutOffset], outLocal, copyParams);
    outQueueY_.FreeTensor(outLocal);
}
} // namespace AscendAntiQuantV2
#endif
