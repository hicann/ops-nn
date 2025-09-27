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
 * \file quantize_per_channel_no_offset_regbase.h
 * \brief quantize kernel
 */

#ifndef QUANTIZE_PER_CHANNEL_NO_OFFSET_REGBASE_H_
#define QUANTIZE_PER_CHANNEL_NO_OFFSET_REGBASE_H_

#include "quantize.h"

namespace QuantizeOp {
using namespace AscendC;
template <typename T, typename T1, typename T2, typename U, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
class QuantizePerChannelNoOffsetRegbase : public QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode> {
public:
    __aicore__ inline QuantizePerChannelNoOffsetRegbase(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, const QuantizeTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyXAndCompute(int64_t dataCount, int64_t offset, LocalTensor<T1>& sLocal);
    __aicore__ inline void CopyInScale(int64_t sLen, int64_t sInOffset);
    __aicore__ inline void CopyInX(int64_t xN, int64_t xLen, int64_t xInOffset);
    __aicore__ inline void CopyOutY(int64_t yN, int64_t yLen, int64_t yOutOffset);
    __aicore__ inline void Compute(int64_t nRow, int64_t dataCount, LocalTensor<T1>& sLocal);

private:
    using yCopyDtype = std::conditional_t<IsSameType<U, int4b_t>::value, uint8_t, U>;
    constexpr static int32_t bufferNum_ = 2;
    TPipe pipe_;
    TQue<QuePosition::VECIN, bufferNum_> inQueueX_;
    TQue<QuePosition::VECIN, bufferNum_> inQueueScale_;
    TQue<QuePosition::VECIN, bufferNum_> inQueueOffset_;
    TQue<QuePosition::VECOUT, bufferNum_> outQueueY_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T1> scaleGm_;
    GlobalTensor<yCopyDtype> yGm_;

    QuantizeTilingData tilingData_;
    int32_t blockIdx_ = 0;
    int64_t gmXOffset_ = 0;
    int64_t gmSOffset_ = 0;
    int64_t blockN_ = 1;
    int64_t blockLen_ = 1;
};

template <typename T, typename T1, typename T2, typename U, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void QuantizePerChannelNoOffsetRegbase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::Init(
    GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, const QuantizeTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T1*>(scale));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ yCopyDtype*>(y));

    this->ParseTilingData(tilingData, tilingData_);
    this->ParseCoreBlocks(tilingData_, blockIdx_, blockN_, blockLen_);

    // calc n size to alloc queue
    pipe_.InitBuffer(inQueueX_, bufferNum_, tilingData_.baseN * tilingData_.baseLen * sizeof(T));
    pipe_.InitBuffer(inQueueScale_, bufferNum_, tilingData_.baseLen * sizeof(T1));

    pipe_.InitBuffer(outQueueY_, bufferNum_, tilingData_.baseN * tilingData_.baseLen * sizeof(U));
}

template <typename T, typename T1, typename T2, typename U, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void QuantizePerChannelNoOffsetRegbase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::Process()
{
    if (blockIdx_ >= tilingData_.numCore) {
        return;
    }
    if (tilingData_.blockAxis == 0) {
        gmXOffset_ = blockIdx_ * tilingData_.blockFactor * tilingData_.dim1;
        gmSOffset_ = 0;
    } else {
        gmXOffset_ = blockIdx_ * tilingData_.blockFactor;
        gmSOffset_ = blockIdx_ * tilingData_.blockFactor;
    }

    // main loop with column, for scale and offset only need copy once
    int64_t lenLoopNum = blockLen_ / tilingData_.baseLen;
    int64_t lenLoopTail = blockLen_ % tilingData_.baseLen;
    for (int64_t i = 0; i < lenLoopNum; ++i) {
        CopyInScale(tilingData_.baseLen, gmSOffset_ + i * tilingData_.baseLen);
        LocalTensor<T1> sLocal = inQueueScale_.DeQue<T1>();
        CopyXAndCompute(tilingData_.baseLen, gmXOffset_ + i * tilingData_.baseLen, sLocal);
        inQueueScale_.FreeTensor(sLocal);
    }
    if (lenLoopTail != 0) {
        CopyInScale(lenLoopTail, gmSOffset_ + lenLoopNum * tilingData_.baseLen);
        LocalTensor<T1> sLocal = inQueueScale_.DeQue<T1>();
        CopyXAndCompute(lenLoopTail, gmXOffset_ + lenLoopNum * tilingData_.baseLen, sLocal);
        inQueueScale_.FreeTensor(sLocal);
    }
}

template <typename T, typename T1, typename T2, typename U, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void QuantizePerChannelNoOffsetRegbase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CopyInScale(
    int64_t sLen, int64_t sInOffset)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = sLen * sizeof(T1);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    copyParams.rsv = 0;
    LocalTensor<T1> sLocal = inQueueScale_.AllocTensor<T1>();
    DataCopyPad(sLocal, scaleGm_[sInOffset], copyParams, {false, 0, 0, 0});
    inQueueScale_.EnQue(sLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void QuantizePerChannelNoOffsetRegbase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CopyXAndCompute(
    int64_t dataCount, int64_t offset, LocalTensor<T1>& sLocal)
{
    int64_t nLoopNum = blockN_ / tilingData_.baseN;
    int64_t nLoopTail = blockN_ % tilingData_.baseN;
    int64_t xOffset = offset;
    for (int64_t nIdx = 0; nIdx < nLoopNum; ++nIdx) {
        xOffset = offset + nIdx * tilingData_.baseN * tilingData_.dim1;
        CopyInX(tilingData_.baseN, dataCount, xOffset);
        Compute(tilingData_.baseN, dataCount, sLocal);
        CopyOutY(tilingData_.baseN, dataCount, xOffset);
    }
    if (nLoopTail != 0) {
        xOffset = offset + nLoopNum * tilingData_.baseN * tilingData_.dim1;
        CopyInX(nLoopTail, dataCount, xOffset);
        Compute(nLoopTail, dataCount, sLocal);
        CopyOutY(nLoopTail, dataCount, xOffset);
    }
}

template <typename T, typename T1, typename T2, typename U, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void QuantizePerChannelNoOffsetRegbase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CopyInX(
    int64_t xN, int64_t xLen, int64_t xInOffset)
{
    LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    this->GetXInCopyParams(tilingData_, xN, xLen, copyParams);
    DataCopyPad(xLocal, xGm_[xInOffset], copyParams, padParams);
    inQueueX_.EnQue(xLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void QuantizePerChannelNoOffsetRegbase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::Compute(
    int64_t nRow, int64_t dataCount, LocalTensor<T1>& sLocal)
{
    LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
    LocalTensor<yCopyDtype> outLocal = outQueueY_.AllocTensor<yCopyDtype>();

    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ T1* scaleLocalAddr = (__local_mem__ T1*)sLocal.GetPhyAddr();
    __local_mem__ yCopyDtype* outLocalAddr = (__local_mem__ yCopyDtype*)outLocal.GetPhyAddr();

    uint16_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint32_t xLocalOffset = static_cast<uint32_t>(tilingData_.baseLen);

    // no offset
    __VEC_SCOPE__
    {
        // x: fp32, fp16, bf16
        AscendC::MicroAPI::RegTensor<T> vregX;
        AscendC::MicroAPI::RegTensor<float> vregFloatX;
        // scales: fp32, bp16
        AscendC::MicroAPI::RegTensor<T1> vregS;
        AscendC::MicroAPI::RegTensor<float> vregFloatS;
        // y: int8, uint8, int32, hifp8
        AscendC::MicroAPI::RegTensor<float> vregFloatY;
        AscendC::MicroAPI::RegTensor<int16_t> vregInt16Y;
        AscendC::MicroAPI::RegTensor<half> vregHalfY;
        AscendC::MicroAPI::RegTensor<yCopyDtype> vregY;

        AscendC::MicroAPI::MaskReg mask;
        AscendC::MicroAPI::MaskReg mask4Int4 =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::H>();
        for (uint16_t j = 0; j < static_cast<uint16_t>(nRow); ++j) {
            uint32_t count = dataCount;
            uint16_t vfLoopNum = (dataCount + VL - 1) / VL;
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                mask = AscendC::MicroAPI::UpdateMask<float>(count);
                // ld for x
                if constexpr (IsSameType<T, float>::value) {
                    // fp32
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                        vregFloatX, xLocalAddr + i * VL + j * xLocalOffset);
                } else if constexpr (IsSameType<T, half>::value) {
                    // fp16
                    AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregX, xLocalAddr + i * VL + j * xLocalOffset);
                    AscendC::MicroAPI::Cast<
                        float, half, QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_HALF_TO_FP32>(
                        vregFloatX, vregX, mask);
                } else if constexpr (IsSameType<T, bfloat16_t>::value) {
                    // bf16
                    AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregX, xLocalAddr + i * VL + j * xLocalOffset);
                    AscendC::MicroAPI::Cast<
                        float, T, QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_BF16_TO_FP32>(
                        vregFloatX, vregX, mask);
                }

                // ld for scale
                if constexpr (IsSameType<T1, float>::value) {
                    // fp32
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                        vregFloatS, scaleLocalAddr + i * VL);
                } else if (IsSameType<T1, half>::value) {
                    // fp16
                    AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregS, scaleLocalAddr + i * VL);
                    AscendC::MicroAPI::Cast<
                        float, T1, QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_HALF_TO_FP32>(
                        vregFloatS, vregS, mask);
                } else if (IsSameType<T1, bfloat16_t>::value) {
                    // bf16
                    AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregS, scaleLocalAddr + i * VL);
                    AscendC::MicroAPI::Cast<
                        float, T1, QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_BF16_TO_FP32>(
                        vregFloatS, vregS, mask);
                }
                if constexpr (SqrtMode == TPL_SQRT_MODE) {
                    AscendC::MicroAPI::Mul(vregFloatS, vregFloatS, vregFloatS, mask);
                }
                if constexpr (DivMode == TPL_DIV_MODE_DIV) {
                    static constexpr AscendC::MicroAPI::DivSpecificMode divMode = {
                        AscendC::MicroAPI::MaskMergeMode::ZEROING, false};
                    AscendC::MicroAPI::Div<float, &divMode>(vregFloatY, vregFloatX, vregFloatS, mask);
                } else {
                    AscendC::MicroAPI::Mul(vregFloatY, vregFloatX, vregFloatS, mask);
                }

                // cast and sd for y
                if constexpr (IsSameType<U, hifloat8_t>::value) {
                    // hifp8
                    AscendC::MicroAPI::Cast<
                        U, float, QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_FP32_TO_HIFP8>(
                        vregY, vregFloatY, mask);
                    AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                        outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                } else if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
                    // fp8_e5m2
                    AscendC::MicroAPI::Cast<
                        U, float, QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_FP32_TO_FP8E5M2>(
                        vregY, vregFloatY, mask);
                    AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                        outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                } else if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
                    // fp8_e4m3
                    AscendC::MicroAPI::Cast<
                        U, float, QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_FP32_TO_FP8E4M3>(
                        vregY, vregFloatY, mask);
                    AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                        outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                } else if constexpr (IsSameType<U, int8_t>::value) {
                    // int8, float->int16->half->int8
                    AscendC::MicroAPI::Cast<
                        int16_t, float,
                        QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_FP32_TO_INT16>(
                        vregInt16Y, vregFloatY, mask);
                    AscendC::MicroAPI::Cast<
                        half, int16_t,
                        QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_INT16_TO_HALF>(
                        vregHalfY, vregInt16Y, mask);
                    AscendC::MicroAPI::Cast<
                        int8_t, half,
                        QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_HALF_TO_INT8>(
                        vregY, vregHalfY, mask);
                    AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                        outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                } else if constexpr (IsSameType<U, uint8_t>::value) {
                    // uint8, float->int16->half->int8
                    AscendC::MicroAPI::Cast<
                        int16_t, float,
                        QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_FP32_TO_INT16>(
                        vregInt16Y, vregFloatY, mask);
                    AscendC::MicroAPI::Cast<
                        half, int16_t,
                        QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_INT16_TO_HALF>(
                        vregHalfY, vregInt16Y, mask);
                    AscendC::MicroAPI::Cast<
                        uint8_t, half,
                        QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_HALF_TO_UINT8>(
                        vregY, vregHalfY, mask);
                    AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                        outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                } else if constexpr (IsSameType<U, int32_t>::value) {
                    // int32
                    AscendC::MicroAPI::Cast<
                        U, float, QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_FP32_TO_INT32>(
                        vregY, vregFloatY, mask);
                    AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::StoreDist::DIST_NORM>(
                        outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                } else if constexpr (IsSameType<U, int4b_t>::value) {
                    AscendC::MicroAPI::RegTensor<int16_t> vregInt16Y;
                    AscendC::MicroAPI::RegTensor<uint16_t> vregTmp1Y;
                    AscendC::MicroAPI::RegTensor<yCopyDtype> vregTmp2Y;
                    AscendC::MicroAPI::Cast<
                        int16_t, float,
                        QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_FP32_TO_INT16>(
                        vregInt16Y, vregFloatY, mask);
                    AscendC::MicroAPI::Cast<
                        half, int16_t,
                        QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_INT16_TO_HALF>(
                        vregHalfY, vregInt16Y, mask);

                    AscendC::MicroAPI::Pack(vregTmp1Y, (AscendC::MicroAPI::RegTensor<uint32_t>&)vregHalfY);
                    AscendC::MicroAPI::Cast<
                        int4x2_t, half, QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CAST_TRAIT_F16_TO_I8>(
                        (AscendC::MicroAPI::RegTensor<int4x2_t>&)vregTmp2Y,
                        (AscendC::MicroAPI::RegTensor<half>&)vregTmp1Y, mask);
                    AscendC::MicroAPI::DataCopy<yCopyDtype, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                        outLocalAddr + (i * VL / 2 + j * tilingData_.baseLen), vregTmp2Y, mask4Int4);
                }
            }
        }
    }
    inQueueX_.FreeTensor(xLocal);
    outQueueY_.EnQue(outLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void QuantizePerChannelNoOffsetRegbase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CopyOutY(
    int64_t yN, int64_t yLen, int64_t yOutOffset)
{
    if constexpr (IsSameType<U, int4b_t>::value) {
        yOutOffset = yOutOffset >> 1;
    }
    LocalTensor<yCopyDtype> outLocal = outQueueY_.DeQue<yCopyDtype>();
    DataCopyExtParams copyParams;
    this->GetOutCopyParams(tilingData_, yN, yLen, copyParams);
    DataCopyPad(yGm_[yOutOffset], outLocal, copyParams);
    outQueueY_.FreeTensor(outLocal);
}
} // namespace QuantizeOp
#endif