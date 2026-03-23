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
 * \file ascend_anti_quant_v2_per_channel_nddma_regbase.h
 * \brief ascend_anti_quant_v2 kernel
 */

#ifndef ASCEND_ANTI_QUANT_V2_PER_CHANNEL_NDDMA_REGBASE_H_
#define ASCEND_ANTI_QUANT_V2_PER_CHANNEL_NDDMA_REGBASE_H_

#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "ascend_anti_quant_v2_common.h"

namespace AscendAntiQuantV2 {
using namespace AscendC;
template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
class AscendAntiQuantV2PerChannelNddmaRegbase : public AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode> {
public:
    __aicore__ inline AscendAntiQuantV2PerChannelNddmaRegbase(const AscendAntiQuantV2TilingData* tilingData)
        : tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyXAndCompute(
        int64_t dataCount, int64_t offset, LocalTensor<T1>& sLocal, LocalTensor<T2>& oLocal);
    __aicore__ inline void CopyInScale(int64_t sLen, int64_t sInOffset);
    __aicore__ inline void CopyInOffset(int64_t sLen, int64_t sInOffset);
    __aicore__ inline void CopyInX(int64_t xN, int64_t xLen, int64_t xInOffset);
    __aicore__ inline void CopyOutY(int64_t yN, int64_t yLen, int64_t yOutOffset);
    __aicore__ inline void Compute(int64_t nRow, int64_t dataCount, LocalTensor<T1>& sLocal, LocalTensor<T2>& oLocal);
    __aicore__ inline void ParseCoreBlocks(const AscendAntiQuantV2TilingData* tilingData, int32_t blockIdx, int64_t& blockN, int64_t& blockLen);

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
    int64_t gmSOffset_ = 0;
    int64_t blockN_ = 1;
    int64_t blockLen_ = 1;
};

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaRegbase<T, T1, T2, U, SqrtMode>::Init(
    GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y)
{
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T1*>(scale));
    offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T2*>(offset));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(y));

    ParseCoreBlocks(tilingData_, blockIdx_, blockN_, blockLen_);

    // calc n size to alloc queue
    pipe_.InitBuffer(
        inQueueX_, bufferNum_, this->CeilAlign(tilingData_->baseN * tilingData_->baseLen * sizeof(xCopyDtype), this->BLOCK_SIZE));
    pipe_.InitBuffer(
        inQueueScale_, bufferNum_,
        this->CeilAlign(tilingData_->baseN * tilingData_->baseLen * sizeof(T1), this->BLOCK_SIZE));
    pipe_.InitBuffer(
        inQueueOffset_, bufferNum_,
        this->CeilAlign(tilingData_->baseN * tilingData_->baseLen * sizeof(T2), this->BLOCK_SIZE));

    pipe_.InitBuffer(
        outQueueY_, bufferNum_, this->CeilAlign(tilingData_->baseN * tilingData_->baseLen * sizeof(U), this->BLOCK_SIZE));
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaRegbase<T, T1, T2, U, SqrtMode>::Process()
{
    if (blockIdx_ >= tilingData_->numCore) {
        return;
    }
    if (tilingData_->blockAxis == 0) {
        gmXOffset_ = blockIdx_ * tilingData_->blockFactor * tilingData_->dim1;
        gmSOffset_ = 0;
    } else {
        gmXOffset_ = blockIdx_ * tilingData_->blockFactor;
        gmSOffset_ = blockIdx_ * tilingData_->blockFactor;
    }

    // main loop with column, for scale and offset only need copy once
    int64_t lenLoopNum = blockLen_ / tilingData_->baseLen;
    int64_t lenLoopTail = blockLen_ % tilingData_->baseLen;
    for (int64_t i = 0; i < lenLoopNum; ++i) {
        CopyInScale(tilingData_->baseLen, gmSOffset_ + i * tilingData_->baseLen);
        CopyInOffset(tilingData_->baseLen, gmSOffset_ + i * tilingData_->baseLen);
        LocalTensor<T1> sLocal = inQueueScale_.DeQue<T1>();
        LocalTensor<T2> oLocal = inQueueOffset_.DeQue<T2>();
        CopyXAndCompute(tilingData_->baseLen, gmXOffset_ + i * tilingData_->baseLen, sLocal, oLocal);
        inQueueScale_.FreeTensor(sLocal);
        inQueueOffset_.FreeTensor(oLocal);
    }
    if (lenLoopTail != 0) {
        CopyInScale(lenLoopTail, gmSOffset_ + lenLoopNum * tilingData_->baseLen);
        CopyInOffset(lenLoopTail, gmSOffset_ + lenLoopNum * tilingData_->baseLen);
        LocalTensor<T1> sLocal = inQueueScale_.DeQue<T1>();
        LocalTensor<T2> oLocal = inQueueOffset_.DeQue<T2>();
        CopyXAndCompute(lenLoopTail, gmXOffset_ + lenLoopNum * tilingData_->baseLen, sLocal, oLocal);
        inQueueScale_.FreeTensor(sLocal);
        inQueueOffset_.FreeTensor(oLocal);
    }
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaRegbase<T, T1, T2, U, SqrtMode>::ParseCoreBlocks(
    const AscendAntiQuantV2TilingData* tilingData, int32_t blockIdx, int64_t& blockN, int64_t& blockLen)
{
    if (tilingData->blockAxis == 0) {
        if (blockIdx == tilingData->numCore - 1) {
            blockN = tilingData->blockTailFactor;
        } else {
            blockN = tilingData->blockFactor;
        }
        blockLen = tilingData->dim1;
    } else if (tilingData->blockAxis == 1) {
        blockN = tilingData->dim0;
        if (blockIdx == tilingData->numCore - 1) {
            blockLen = tilingData->blockTailFactor;
        } else {
            blockLen = tilingData->blockFactor;
        }
    }
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaRegbase<T, T1, T2, U, SqrtMode>::CopyInScale(
    int64_t sLen, int64_t sInOffset)
{
    LocalTensor<T1> sLocal = inQueueScale_.AllocTensor<T1>();
    static constexpr AscendC::MultiCopyConfig copyConfig = {false, 0, 0, false};
    MultiCopyLoopInfo<AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::MULTI_COPY_DIM> copyLoopInfo;
    copyLoopInfo.loopSrcStride[0] = 1;
    copyLoopInfo.loopSrcStride[1] = 0;
    copyLoopInfo.loopDstStride[0] = 1;
    copyLoopInfo.loopDstStride[1] = sLen;
    copyLoopInfo.loopSize[0] = sLen;
    copyLoopInfo.loopSize[1] = tilingData_->baseN;

    T1 constValue = 0;
    AscendC::MultiCopyParams<T1, AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::MULTI_COPY_DIM> copyParams =
        {copyLoopInfo, constValue};
    AscendC::DataCopy<T1, AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::MULTI_COPY_DIM, copyConfig>(
        sLocal, scaleGm_[sInOffset], copyParams);
    inQueueScale_.EnQue(sLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaRegbase<T, T1, T2, U, SqrtMode>::CopyInOffset(
    int64_t sLen, int64_t sInOffset)
{
    LocalTensor<T2> oLocal = inQueueOffset_.AllocTensor<T2>();
    static constexpr AscendC::MultiCopyConfig copyConfig = {false, 0, 0, false};
    MultiCopyLoopInfo<AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::MULTI_COPY_DIM> copyLoopInfo;
    copyLoopInfo.loopSrcStride[0] = 1;
    copyLoopInfo.loopSrcStride[1] = 0;
    copyLoopInfo.loopDstStride[0] = 1;
    copyLoopInfo.loopDstStride[1] = sLen;
    copyLoopInfo.loopSize[0] = sLen;
    copyLoopInfo.loopSize[1] = tilingData_->baseN;

    T2 constValue = 0;
    AscendC::MultiCopyParams<T2, AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::MULTI_COPY_DIM> copyParams =
        {copyLoopInfo, constValue};
    AscendC::DataCopy<T2, AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::MULTI_COPY_DIM, copyConfig>(
        oLocal, offsetGm_[sInOffset], copyParams);
    inQueueOffset_.EnQue(oLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaRegbase<T, T1, T2, U, SqrtMode>::CopyXAndCompute(
    int64_t dataCount, int64_t offset, LocalTensor<T1>& sLocal, LocalTensor<T2>& oLocal)
{
    int64_t nLoopNum = blockN_ / tilingData_->baseN;
    int64_t nLoopTail = blockN_ % tilingData_->baseN;
    int64_t xOffset = offset;
    for (int64_t nIdx = 0; nIdx < nLoopNum; ++nIdx) {
        xOffset = offset + nIdx * tilingData_->baseN * tilingData_->dim1;
        CopyInX(tilingData_->baseN, dataCount, xOffset);
        Compute(tilingData_->baseN, dataCount, sLocal, oLocal);
        CopyOutY(tilingData_->baseN, dataCount, xOffset);
    }
    if (nLoopTail != 0) {
        xOffset = offset + nLoopNum * tilingData_->baseN * tilingData_->dim1;
        CopyInX(nLoopTail, dataCount, xOffset);
        Compute(nLoopTail, dataCount, sLocal, oLocal);
        CopyOutY(nLoopTail, dataCount, xOffset);
    }
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaRegbase<T, T1, T2, U, SqrtMode>::CopyInX(
    int64_t xN, int64_t xLen, int64_t xInOffset)
{
    int64_t xLenReal = xLen;
    if constexpr (IsSameType<T, int4b_t>::value) {
        xInOffset = xInOffset / this->INT4_NUMS_IN_INT8_SPACE;
        xLenReal = xLenReal / this-> INT4_NUMS_IN_INT8_SPACE;
    }

    LocalTensor<uint8_t> xLocal = inQueueX_.AllocTensor<uint8_t>();
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<uint8_t> padParams = {false, 0, 0, 0};
    copyParams.blockCount = 1;
    copyParams.blockLen = xN * xLenReal * sizeof(xCopyDtype);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    DataCopyPad<uint8_t>(xLocal, xGm_[xInOffset], copyParams, padParams);
    inQueueX_.EnQue(xLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaRegbase<T, T1, T2, U, SqrtMode>::Compute(
    int64_t nRow, int64_t dataCount, LocalTensor<T1>& sLocal, LocalTensor<T2>& oLocal)
{
    LocalTensor<xCopyDtype> xLocal = inQueueX_.DeQue<xCopyDtype>();
    LocalTensor<U> outLocal = outQueueY_.AllocTensor<U>();

    __local_mem__ xCopyDtype* xLocalAddr = (__local_mem__ xCopyDtype*)xLocal.GetPhyAddr();
    __local_mem__ T1* scaleLocalAddr = (__local_mem__ T1*)sLocal.GetPhyAddr();
    __local_mem__ U* outLocalAddr = (__local_mem__ U*)outLocal.GetPhyAddr();

    uint16_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint16_t HalfVL = VL / 2;
    __local_mem__ T2* offsetLocalAddr = (__local_mem__ T2*)oLocal.GetPhyAddr();

    // has offset
    __VEC_SCOPE__
    {
        // x: int4, int8, hifp8, fp8_e5m2, fp8_e4m3
        AscendC::Reg::RegTensor<xCopyDtype> vregX;
        AscendC::Reg::RegTensor<half> vregHalfX;
        AscendC::Reg::RegTensor<float> vregFloatX;
        // scales: fp32, bf16
        AscendC::Reg::RegTensor<T1> vregS;
        AscendC::Reg::RegTensor<float> vregFloatS;
        // offset: fp32, bf16
        AscendC::Reg::RegTensor<T2> vregO;
        AscendC::Reg::RegTensor<float> vregFloatO;
        // y: fp16, bf16
        AscendC::Reg::RegTensor<float> vregFloatY;
        AscendC::Reg::RegTensor<U> vregY;

        AscendC::Reg::RegTensor<float> vregTmp1;
        AscendC::Reg::MaskReg mask;

        mask = AscendC::Reg::CreateMask<float>();
        uint32_t count = dataCount * nRow;
        uint16_t vfLoopNum = (count + VL - 1) / VL;
        for (uint16_t i = 0; i < vfLoopNum; i++) {
            mask = AscendC::Reg::UpdateMask<float>(count);
            // ld and cast for x
            if constexpr (IsSameType<T, hifloat8_t>::value) {
                // hifp8
                AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(
                    vregX, xLocalAddr + i * VL);
                AscendC::Reg::Cast<
                    float, T, AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CAST_TRAIT_HIFP8_TO_FP32>(
                    vregFloatX, vregX, mask);
            } else if constexpr (IsSameType<T, fp8_e5m2_t>::value) {
                // fp8_e5m2
                AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(
                    vregX, xLocalAddr + i * VL);
                AscendC::Reg::Cast<
                    float, T, AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CAST_TRAIT_FP8E5M2_TO_FP32>(
                    vregFloatX, vregX, mask);
            } else if constexpr (IsSameType<T, fp8_e4m3fn_t>::value) {
                // fp8_e4m3
                AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(
                    vregX, xLocalAddr + i * VL);
                AscendC::Reg::Cast<
                    float, T, AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CAST_TRAIT_FP8E4M3_TO_FP32>(
                    vregFloatX, vregX, mask);
            } else if constexpr (IsSameType<T, int8_t>::value) {
                // int8
                AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(
                    vregX, xLocalAddr + i * VL);
                AscendC::Reg::Cast<
                    half, T, AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CAST_TRAIT_INT8_TO_HALF>(
                    vregHalfX, vregX, mask);
                AscendC::Reg::Cast<
                    float, half, AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CAST_TRAIT_HALF_TO_FP32>(
                    vregFloatX, vregHalfX, mask);    
            } else if constexpr (IsSameType<T, int4b_t>::value) {
                // int4
                AscendC::Reg::RegTensor<half> vregTmpX;
                AscendC::Reg::RegTensor<uint16_t> vregTmpU16X;
                AscendC::Reg::DataCopy<xCopyDtype, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(
                    vregX, xLocalAddr + i * HalfVL);
                AscendC::Reg::Cast<
                    half, int4x2_t, AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CAST_TRAIT_INT8_TO_HALF>(
                    (AscendC::Reg::RegTensor<half>&)vregTmpU16X, 
                    (AscendC::Reg::RegTensor<int4x2_t>&)vregX, mask);
                AscendC::Reg::UnPack((AscendC::Reg::RegTensor<uint32_t>&)vregTmpX, vregTmpU16X);
                AscendC::Reg::Cast<
                    float, half, AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CAST_TRAIT_HALF_TO_FP32>(
                    vregFloatX, vregTmpX, mask);
            }

            // ld and cast for scale
            if constexpr (IsSameType<T1, float>::value) {
                // fp32
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(
                    vregFloatS, scaleLocalAddr + i * VL);
            } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
                // bf16
                AscendC::Reg::DataCopy<T1, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregS, scaleLocalAddr + i * VL);
                AscendC::Reg::Cast<
                    float, T1, AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CAST_TRAIT_BF16_TO_FP32>(
                    vregFloatS, vregS, mask);
            }

            // ld and cast for offset
            if constexpr (IsSameType<T2, float>::value) {
                // fp32
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(
                    vregFloatO, offsetLocalAddr + i * VL);
            } else if constexpr (IsSameType<T2, bfloat16_t>::value) {
                // bf16
                AscendC::Reg::DataCopy<T2, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregO, offsetLocalAddr + i * VL);
                AscendC::Reg::Cast<
                    float, T2, AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CAST_TRAIT_BF16_TO_FP32>(
                    vregFloatO, vregO, mask);
            }

            // compute
            AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregTmp1, vregFloatX, vregFloatO, mask);
            if constexpr (SqrtMode == TPL_SQRT_MODE) {
                AscendC::Reg::Mul(vregFloatS, vregFloatS, vregFloatS, mask);
            }
            AscendC::Reg::Mul(vregFloatY, vregTmp1, vregFloatS, mask);

            // cast and sd for y
            if constexpr (IsSameType<U, half>::value) {
                // fp16
                AscendC::Reg::Cast<
                    half, float, AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CAST_TRAIT_FP32_TO_HALF>(
                    vregY, vregFloatY, mask);
                AscendC::Reg::DataCopy<U, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                    outLocalAddr + i * VL, vregY, mask);
            } else if constexpr (IsSameType<U, bfloat16_t>::value) {
                // bf16
                AscendC::Reg::Cast<
                    U, float, AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CAST_TRAIT_FP32_TO_BF16>(
                    vregY, vregFloatY, mask);
                AscendC::Reg::DataCopy<U, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                    outLocalAddr + i * VL, vregY, mask);
            }
        }
    }
    inQueueX_.FreeTensor(xLocal);
    outQueueY_.EnQue(outLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNddmaRegbase<T, T1, T2, U, SqrtMode>::CopyOutY(
    int64_t yN, int64_t yLen, int64_t yOutOffset)
{
    LocalTensor<U> outLocal = outQueueY_.DeQue<U>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = yN * yLen * sizeof(U);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    DataCopyPad<U>(yGm_[yOutOffset], outLocal, copyParams);
    outQueueY_.FreeTensor(outLocal);
}
} // namespace AscendAntiQuantV2
#endif