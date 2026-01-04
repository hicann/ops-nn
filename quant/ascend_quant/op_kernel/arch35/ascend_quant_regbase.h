/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ascend_quant_regbase.h
 * \brief ascend_quant kernel
 */

#ifndef ASCEND_QUANT_REGBASE_H_
#define ASCEND_QUANT_REGBASE_H_

#include "ascend_quant.h"

namespace AscendQuantOp {
using namespace AscendC;
template <typename T, typename U, uint64_t RoundMode>
class AscendQuantPerTensorRegbase : public AscendQuantBase<T, U, RoundMode> {
public:
    __aicore__ inline AscendQuantPerTensorRegbase(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const AscendQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyXAndCompute(int64_t dataCount, int64_t xOffset);
    __aicore__ inline void CopyInX(int64_t xLen, int64_t xInOffset);
    __aicore__ inline void CopyOutY(int64_t yLen, int64_t yOutOffset);
    __aicore__ inline void Compute(int64_t dataCount);

private:
    using yCopyDtype = std::conditional_t<IsSameType<U, int4b_t>::value, uint8_t, U>;
    constexpr static int32_t bufferNum_ = 2;
    TPipe pipe_;
    TQue<QuePosition::VECIN, bufferNum_> inQueueX_;
    TQue<QuePosition::VECOUT, bufferNum_> outQueueY_;
    GlobalTensor<T> xGm_;
    GlobalTensor<yCopyDtype> yGm_;

    AscendQuantTilingData tilingData_;
    int32_t blockIdx_ = 0;
    int64_t gmXOffset_ = 0;
    int64_t blockLen_ = 1;
};

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void AscendQuantPerTensorRegbase<T, U, RoundMode>::Init(
    GM_ADDR x, GM_ADDR y, const AscendQuantTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ yCopyDtype*>(y));

    this->ParseTilingData(tilingData, tilingData_);
    this->ParseCoreBlocks(tilingData_, blockIdx_, blockLen_);

    // calc n size to alloc queue
    pipe_.InitBuffer(inQueueX_, bufferNum_, tilingData_.baseLen * sizeof(T));
    pipe_.InitBuffer(outQueueY_, bufferNum_, tilingData_.baseLen * sizeof(U));
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void AscendQuantPerTensorRegbase<T, U, RoundMode>::Process()
{
    if (blockIdx_ >= tilingData_.numCore) {
        return;
    }
    gmXOffset_ = blockIdx_ * tilingData_.blockFactor;

    // main loop with column, for scale and offset only need copy once
    int64_t lenLoopNum = blockLen_ / tilingData_.baseLen;
    int64_t lenLoopTail = blockLen_ % tilingData_.baseLen;

    for (int64_t i = 0; i < lenLoopNum; ++i) {
        CopyXAndCompute(tilingData_.baseLen, gmXOffset_ + i * tilingData_.baseLen);
    }
    if (lenLoopTail != 0) {
        CopyXAndCompute(lenLoopTail, gmXOffset_ + lenLoopNum * tilingData_.baseLen);
    }
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void AscendQuantPerTensorRegbase<T, U, RoundMode>::CopyXAndCompute(int64_t dataCount, int64_t xOffset)
{
    CopyInX(dataCount, xOffset);
    Compute(dataCount);
    CopyOutY(dataCount, xOffset);
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void AscendQuantPerTensorRegbase<T, U, RoundMode>::CopyInX(int64_t xLen, int64_t xInOffset)
{
    LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    this->GetXInCopyParams(tilingData_, xLen, copyParams);
    DataCopyPad(xLocal, xGm_[xInOffset], copyParams, padParams);
    inQueueX_.EnQue(xLocal);
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void AscendQuantPerTensorRegbase<T, U, RoundMode>::Compute(int64_t dataCount)
{
    LocalTensor<T> xLocal = inQueueX_.DeQue<T>();
    LocalTensor<yCopyDtype> outLocal = outQueueY_.AllocTensor<yCopyDtype>();

    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ yCopyDtype* outLocalAddr = (__local_mem__ yCopyDtype*)outLocal.GetPhyAddr();

    uint16_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);

    float ATTR_SCALE = tilingData_.scale;
    float ATTR_OFFSET = tilingData_.offset;

    __VEC_SCOPE__
    {
        // x: fp32, fp16
        AscendC::MicroAPI::RegTensor<T> vregX;
        AscendC::MicroAPI::RegTensor<float> vregFloatX;
        // y: int8, int4
        AscendC::MicroAPI::RegTensor<float> vregFloatY;
        AscendC::MicroAPI::RegTensor<int16_t> vregInt16Y;
        AscendC::MicroAPI::RegTensor<half> vregHalfY;
        AscendC::MicroAPI::RegTensor<yCopyDtype> vregY;

        AscendC::MicroAPI::RegTensor<float> vregTmp1;
        AscendC::MicroAPI::MaskReg mask;
        AscendC::MicroAPI::MaskReg mask4Int4 =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::H>();

        mask = AscendC::MicroAPI::CreateMask<float>();
        uint32_t count = dataCount;
        uint16_t vfLoopNum = (dataCount + VL - 1) / VL;
        for (uint16_t i = 0; i < vfLoopNum; i++) {
            mask = AscendC::MicroAPI::UpdateMask<float>(count);
            // ld and cast for x
            if constexpr (IsSameType<T, float>::value) {
                // fp32
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                    vregFloatX, xLocalAddr + i * VL);
            } else if constexpr (IsSameType<T, half>::value) {
                // fp16
                AscendC::MicroAPI::DataCopy<half, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    vregX, xLocalAddr + i * VL);
                AscendC::MicroAPI::Cast<float, half, AscendQuantBase<T, U, RoundMode>::CAST_TRAIT_HALF_TO_FP32>(
                    vregFloatX, vregX, mask);
            }

            AscendC::MicroAPI::Muls(vregTmp1, vregFloatX, ATTR_SCALE, mask);
            AscendC::MicroAPI::Adds(vregFloatY, vregTmp1, ATTR_OFFSET, mask);

            // cast and sd for y
            if constexpr (IsSameType<U, hifloat8_t>::value) {
                // hifp8
                AscendC::MicroAPI::Cast<U, float, AscendQuantBase<T, U, RoundMode>::CAST_TRAIT_FP32_TO_HIFP8>(
                    vregY, vregFloatY, mask);
                AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    outLocalAddr + i * VL, vregY, mask);
            } else if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
                // fp8_e5m2
                AscendC::MicroAPI::Cast<U, float, AscendQuantBase<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E5M2>(
                    vregY, vregFloatY, mask);
                AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    outLocalAddr + i * VL, vregY, mask);
            } else if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
                // fp8_e4m3
                AscendC::MicroAPI::Cast<U, float, AscendQuantBase<T, U, RoundMode>::CAST_TRAIT_FP32_TO_FP8E4M3>(
                    vregY, vregFloatY, mask);
                AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    outLocalAddr + i * VL, vregY, mask);
            } else if constexpr (IsSameType<U, int8_t>::value) {
                // int8
                AscendC::MicroAPI::Cast<int16_t, float, AscendQuantBase<T, U, RoundMode>::CAST_TRAIT_FP32_TO_INT16>(
                    vregInt16Y, vregFloatY, mask);
                AscendC::MicroAPI::Cast<half, int16_t, AscendQuantBase<T, U, RoundMode>::CAST_TRAIT_INT16_TO_HALF>(
                    vregHalfY, vregInt16Y, mask);
                AscendC::MicroAPI::Cast<int8_t, half, AscendQuantBase<T, U, RoundMode>::CAST_TRAIT_HALF_TO_INT8>(
                    vregY, vregHalfY, mask);
                AscendC::MicroAPI::DataCopy<U, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    outLocalAddr + i * VL, vregY, mask);
            } else if constexpr (IsSameType<U, int4b_t>::value) {
                AscendC::MicroAPI::RegTensor<int16_t> vregInt16Y;
                AscendC::MicroAPI::RegTensor<uint16_t> vregTmp1Y;
                AscendC::MicroAPI::RegTensor<yCopyDtype> vregTmp2Y;
                AscendC::MicroAPI::Cast<int16_t, float, AscendQuantBase<T, U, RoundMode>::CAST_TRAIT_FP32_TO_INT16>(
                    vregInt16Y, vregFloatY, mask);
                AscendC::MicroAPI::Cast<half, int16_t, AscendQuantBase<T, U, RoundMode>::CAST_TRAIT_INT16_TO_HALF>(
                    vregHalfY, vregInt16Y, mask);

                AscendC::MicroAPI::Pack(vregTmp1Y, (AscendC::MicroAPI::RegTensor<uint32_t>&)vregHalfY);
                AscendC::MicroAPI::Cast<int4x2_t, half, AscendQuantBase<T, U, RoundMode>::CAST_TRAIT_F16_TO_I8>(
                    (AscendC::MicroAPI::RegTensor<int4x2_t>&)vregTmp2Y, (AscendC::MicroAPI::RegTensor<half>&)vregTmp1Y,
                    mask);
                AscendC::MicroAPI::DataCopy<yCopyDtype, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    outLocalAddr + (i * VL / 2), vregTmp2Y, mask4Int4);
            }
        }
    }
    inQueueX_.FreeTensor(xLocal);
    outQueueY_.EnQue(outLocal);
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void AscendQuantPerTensorRegbase<T, U, RoundMode>::CopyOutY(int64_t yLen, int64_t yOutOffset)
{
    if constexpr (IsSameType<U, int4b_t>::value) {
        yOutOffset = yOutOffset >> 1;
    }
    LocalTensor<yCopyDtype> outLocal = outQueueY_.DeQue<yCopyDtype>();
    DataCopyExtParams copyParams;
    this->GetOutCopyParams(tilingData_, yLen, copyParams);
    DataCopyPad(yGm_[yOutOffset], outLocal, copyParams);
    outQueueY_.FreeTensor(outLocal);
}
} // namespace AscendQuantOp
#endif