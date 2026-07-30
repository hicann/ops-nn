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
 * \file ascend_anti_quant_v2_per_channel_no_offset_base.h
 * \brief ascendantiquantv2 kernel
 */

#ifndef ASCEND_ANTI_QUANT_V2_PER_CHANNEL_NO_OFFSET_BASE_H_
#define ASCEND_ANTI_QUANT_V2_PER_CHANNEL_NO_OFFSET_BASE_H_

#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "ascend_anti_quant_v2_common.h"

namespace AscendAntiQuantV2 {
using namespace AscendC;
template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
class AscendAntiQuantV2PerChannelNoOffsetBase : public AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode> {
public:
    __aicore__ inline AscendAntiQuantV2PerChannelNoOffsetBase(const AscendAntiQuantV2TilingData* tilingData)
        : tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyXAndCompute(int64_t dataCount, int64_t offset, LocalTensor<float>& sFPLocal);
    __aicore__ inline void CopyInScale(int64_t sLen, int64_t sInOffset);
    __aicore__ inline void ParseCoreBlocks(const AscendAntiQuantV2TilingData* tilingData, int32_t blockIdx,
                                           int64_t& blockN, int64_t& blockLen);
    __aicore__ inline void CopyInX(int64_t xN, int64_t xLen, int64_t xInOffset);
    __aicore__ inline void CopyOutY(int64_t yN, int64_t yLen, int64_t yOutOffset);
    __aicore__ inline void Compute(int64_t nRow, int64_t dataCount, LocalTensor<float>& sFPLocal);

private:
    using xCopyDtype = std::conditional_t<IsSameType<T, int4b_t>::value, uint8_t, T>;
    constexpr static int32_t bufferNum_ = 2;

    TQue<QuePosition::VECIN, bufferNum_> inQueueX_;
    TQue<QuePosition::VECIN, bufferNum_> inQueueScale_;
    TQue<QuePosition::VECOUT, bufferNum_> outQueueY_;

    TBuf<TPosition::VECCALC> fp32Buf;
    TBuf<TPosition::VECCALC> fp16Buf;
    TBuf<TPosition::VECCALC> tmpBuf;

    GlobalTensor<uint8_t> xGm_;
    GlobalTensor<T1> scaleGm_;
    GlobalTensor<U> yGm_;

    LocalTensor<float> floats;

    const AscendAntiQuantV2TilingData* tilingData_;
    int32_t blockIdx_ = 0;
    int64_t gmXOffset_ = 0;
    int64_t gmSOffset_ = 0;
    int64_t blockN_ = 1;
    int64_t blockLen_ = 1;
};

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNoOffsetBase<T, T1, T2, U, SqrtMode>::Init(GM_ADDR x, GM_ADDR scale,
                                                                                             GM_ADDR offset, GM_ADDR y,
                                                                                             TPipe* pipeIn)
{
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T1*>(scale));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(y));

    ParseCoreBlocks(tilingData_, blockIdx_, blockN_, blockLen_);

    // calc n size to alloc queue
    pipeIn->InitBuffer(inQueueX_, bufferNum_, tilingData_->baseN * tilingData_->baseLen * sizeof(xCopyDtype));
    pipeIn->InitBuffer(inQueueScale_, 1, tilingData_->baseLen * sizeof(T1));

    pipeIn->InitBuffer(outQueueY_, bufferNum_, tilingData_->baseN * tilingData_->baseLen * sizeof(U));

    pipeIn->InitBuffer(fp16Buf, tilingData_->baseLen * sizeof(half));
    pipeIn->InitBuffer(fp32Buf, tilingData_->baseLen * sizeof(float));

    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        pipeIn->InitBuffer(tmpBuf, tilingData_->baseLen * sizeof(float));
    }
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNoOffsetBase<T, T1, T2, U, SqrtMode>::Process()
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

    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        this->floats = tmpBuf.Get<float>();
    }

    // main loop with column, for scale and offset only need copy once
    int64_t lenLoopNum = blockLen_ / tilingData_->baseLen;
    int64_t lenLoopTail = blockLen_ % tilingData_->baseLen;
    for (int64_t i = 0; i < lenLoopNum; ++i) {
        CopyInScale(tilingData_->baseLen, gmSOffset_ + i * tilingData_->baseLen);
        LocalTensor<T1> sLocal = inQueueScale_.DeQue<T1>();
        // ld and cast for scale
        if constexpr (IsSameType<T1, bfloat16_t>::value) {
            // bf16
            AscendC::Cast<float, bfloat16_t>(this->floats, sLocal, AscendC::RoundMode::CAST_NONE, tilingData_->baseLen);
        }

        if constexpr (SqrtMode == TPL_SQRT_MODE) {
            if constexpr (IsSameType<T1, float>::value) {
                // fp32
                AscendC::Mul(sLocal, sLocal, sLocal, tilingData_->baseLen);
            } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
                AscendC::Mul(this->floats, this->floats, this->floats, tilingData_->baseLen);
            }
        }

        if constexpr (IsSameType<T1, float>::value) {
            // fp32
            CopyXAndCompute(tilingData_->baseLen, gmXOffset_ + i * tilingData_->baseLen, sLocal);
        } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
            CopyXAndCompute(tilingData_->baseLen, gmXOffset_ + i * tilingData_->baseLen, this->floats);
        }

        inQueueScale_.FreeTensor(sLocal);
    }
    if (lenLoopTail != 0) {
        CopyInScale(lenLoopTail, gmSOffset_ + lenLoopNum * tilingData_->baseLen);
        LocalTensor<T1> sLocal = inQueueScale_.DeQue<T1>();
        // ld and cast for scale
        if constexpr (IsSameType<T1, bfloat16_t>::value) {
            // bf16
            AscendC::Cast<float, bfloat16_t>(this->floats, sLocal, AscendC::RoundMode::CAST_NONE, lenLoopTail);
        }

        if constexpr (SqrtMode == TPL_SQRT_MODE) {
            if constexpr (IsSameType<T1, float>::value) {
                // fp32
                AscendC::Mul(sLocal, sLocal, sLocal, lenLoopTail);
            } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
                AscendC::Mul(this->floats, this->floats, this->floats, lenLoopTail);
            }
        }

        if constexpr (IsSameType<T1, float>::value) {
            // fp32
            CopyXAndCompute(lenLoopTail, gmXOffset_ + lenLoopNum * tilingData_->baseLen, sLocal);
        } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
            CopyXAndCompute(lenLoopTail, gmXOffset_ + lenLoopNum * tilingData_->baseLen, this->floats);
        }

        inQueueScale_.FreeTensor(sLocal);
    }
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNoOffsetBase<T, T1, T2, U, SqrtMode>::ParseCoreBlocks(
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
__aicore__ inline void AscendAntiQuantV2PerChannelNoOffsetBase<T, T1, T2, U, SqrtMode>::CopyInScale(int64_t sLen,
                                                                                                    int64_t sInOffset)
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

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNoOffsetBase<T, T1, T2, U, SqrtMode>::CopyXAndCompute(
    int64_t dataCount, int64_t offset, LocalTensor<float>& sFPLocal)
{
    int64_t nLoopNum = blockN_ / tilingData_->baseN;
    int64_t nLoopTail = blockN_ % tilingData_->baseN;
    int64_t xOffset = offset;
    for (int64_t nIdx = 0; nIdx < nLoopNum; ++nIdx) {
        xOffset = offset + nIdx * tilingData_->baseN * tilingData_->dim1;
        CopyInX(tilingData_->baseN, dataCount, xOffset);
        Compute(tilingData_->baseN, dataCount, sFPLocal);
        CopyOutY(tilingData_->baseN, dataCount, xOffset);
    }
    if (nLoopTail != 0) {
        xOffset = offset + nLoopNum * tilingData_->baseN * tilingData_->dim1;
        CopyInX(nLoopTail, dataCount, xOffset);
        Compute(nLoopTail, dataCount, sFPLocal);
        CopyOutY(nLoopTail, dataCount, xOffset);
    }
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNoOffsetBase<T, T1, T2, U, SqrtMode>::CopyInX(int64_t xN,
                                                                                                int64_t xLen,
                                                                                                int64_t xInOffset)
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        xInOffset = xInOffset >> 1;
    }

    LocalTensor<uint8_t> xLocal = inQueueX_.AllocTensor<uint8_t>();
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<uint8_t> padParams = {false, 0, 0, 0};
    this->GetXInCopyParams(tilingData_->dim1, tilingData_->baseLen, xN, xLen, copyParams);
    DataCopyPad<uint8_t>(xLocal, xGm_[xInOffset], copyParams, padParams);
    inQueueX_.EnQue(xLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNoOffsetBase<T, T1, T2, U, SqrtMode>::Compute(
    int64_t nRow, int64_t dataCount, LocalTensor<float>& sFPLocal)
{
    LocalTensor<xCopyDtype> xLocal = inQueueX_.DeQue<xCopyDtype>();
    LocalTensor<U> outLocal = outQueueY_.AllocTensor<U>();

    LocalTensor<float> floatX = fp32Buf.Get<float>();

    LocalTensor<half> halfX = fp16Buf.Get<half>();

    for (uint16_t j = 0; j < static_cast<uint16_t>(nRow); ++j) {
        if constexpr (IsSameType<T, int8_t>::value) {
            // int8
            AscendC::Cast(halfX, xLocal[j * tilingData_->baseLen], AscendC::RoundMode::CAST_NONE, dataCount);
            AscendC::Cast(floatX, halfX, AscendC::RoundMode::CAST_NONE, dataCount);
        } else if constexpr (IsSameType<T, int4b_t>::value) {
            // int4
            LocalTensor<int4b_t> int4_xLocal = xLocal.template ReinterpretCast<int4b_t>();
            AscendC::Cast<half, AscendC::int4b_t>(halfX, int4_xLocal[2 * j * tilingData_->baseLen],
                                                  AscendC::RoundMode::CAST_NONE, dataCount);
            AscendC::Cast<float, half>(floatX, halfX, AscendC::RoundMode::CAST_NONE, dataCount);
        }
        AscendC::Mul(floatX, floatX, sFPLocal, dataCount);

        // cast and sd for y
        if constexpr (IsSameType<U, half>::value) {
            // fp16
            AscendC::Cast<half, float>(outLocal[j * tilingData_->baseLen], floatX, AscendC::RoundMode::CAST_RINT,
                                       dataCount);
        } else if constexpr (IsSameType<U, bfloat16_t>::value) {
            // bf16
            AscendC::Cast<bfloat16_t, float>(outLocal[j * tilingData_->baseLen], floatX, AscendC::RoundMode::CAST_RINT,
                                             dataCount);
        }
    }
    inQueueX_.FreeTensor(xLocal);
    outQueueY_.EnQue(outLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2PerChannelNoOffsetBase<T, T1, T2, U, SqrtMode>::CopyOutY(int64_t yN,
                                                                                                 int64_t yLen,
                                                                                                 int64_t yOutOffset)
{
    LocalTensor<U> outLocal = outQueueY_.DeQue<U>();
    DataCopyExtParams copyParams;
    this->GetOutCopyParams(tilingData_->dim1, tilingData_->baseLen, yN, yLen, copyParams);
    DataCopyPad<U>(yGm_[yOutOffset], outLocal, copyParams);
    outQueueY_.FreeTensor(outLocal);
}
} // namespace AscendAntiQuantV2
#endif
