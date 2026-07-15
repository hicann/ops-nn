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
 * \file ascend_anti_quant_v2_per_head_regbase.h
 * \brief ascendantiquantv2 kernel
 */

#ifndef ASCEND_ANTI_QUANT_V2_PER_HEAD_REGBASE_H
#define ASCEND_ANTI_QUANT_V2_PER_HEAD_REGBASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "ascend_anti_quant_v2_common.h"
#include "ascend_anti_quant_v2_struct.h"

namespace AscendAntiQuantV2 {
using namespace AscendC;
template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset = TPL_HAS_OFFSET>
class AscendAntiQuantV2PerHeadRegbase : public AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode> {
public:
    __aicore__ inline AscendAntiQuantV2PerHeadRegbase(const AscendAntiQuantV2TilingData* tilingData)
        : tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR y);
    __aicore__ inline void Process();

private:
    using xCopyDtype = std::conditional_t<IsSameType<T, int4b_t>::value, uint8_t, T>;
    TPipe pipe_;
    constexpr static int32_t bufferNum = 2;
    TQue<QuePosition::VECIN, bufferNum> inQueueX_;
    TQue<QuePosition::VECIN, bufferNum> inQueueScale_;
    TQue<QuePosition::VECIN, bufferNum> inQueueOffset_;
    TQue<QuePosition::VECOUT, bufferNum> outQueueY_;
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
    int64_t blockS_ = 1;
    int64_t nLoopLen = 0;
    int64_t nLoopNum = 0;
    int64_t nLoopTail = 0;
    int64_t loopLen = 0;
    int64_t lenLoopNum = 0;
    int64_t lenLoopTail = 0;

private:
    __aicore__ inline void ProcessInputLoop(int64_t nLoopLen, int64_t baseXOffset, LocalTensor<T1>& scaleLocal,
                                            LocalTensor<T2>& offsetLocal);
    template <typename dtypeCopyIn>
    __aicore__ inline void CopyInParam(TQue<QuePosition::VECIN, bufferNum>& inQueue, GlobalTensor<dtypeCopyIn>& inGm,
                                       int64_t paramLen, int64_t paramOffset);
    __aicore__ inline void ParseCoreBlocks(const AscendAntiQuantV2TilingData* tilingData, int32_t blockIdx,
                                           int64_t& blockN, int64_t& blockS, int64_t& blockLen);
    __aicore__ inline void ProcessParamOneLoop(int64_t nLoopLen, int64_t baseSOffset, int64_t baseXOffset);
    __aicore__ inline void GetXInCopyParams(const AscendAntiQuantV2TilingData* tilingData, int64_t xN, int64_t xLen,
                                            int64_t lastDimLen, DataCopyExtParams& copyParams);
    __aicore__ inline void GetOutCopyParams(const AscendAntiQuantV2TilingData* tilingData, int64_t yN, int64_t yLen,
                                            int64_t lastDimLen, DataCopyExtParams& copyParams);
    __aicore__ inline void CopyInX(int64_t xN, int64_t xLen, int64_t xInOffset);
    __aicore__ inline void CopyOutY(int64_t yN, int64_t yLen, int64_t yOutOffset);
    __aicore__ inline void Compute(int64_t nRow, int64_t dataCount, LocalTensor<T1>& sLocal, LocalTensor<T2>& oLocal);
};

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset>
__aicore__ inline void AscendAntiQuantV2PerHeadRegbase<T, T1, T2, U, SqrtMode, HasOffset>::Init(GM_ADDR x,
                                                                                                GM_ADDR scale,
                                                                                                GM_ADDR offset,
                                                                                                GM_ADDR y)
{
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T1*>(scale));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(y));

    ParseCoreBlocks(tilingData_, blockIdx_, blockN_, blockS_, blockLen_);

    // calc n size to alloc queue
    pipe_.InitBuffer(inQueueX_, bufferNum, tilingData_->baseN * tilingData_->baseLen * sizeof(xCopyDtype));
    pipe_.InitBuffer(inQueueScale_, bufferNum, tilingData_->baseN * sizeof(T1));
    if constexpr (HasOffset) {
        offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T2*>(offset));
        pipe_.InitBuffer(inQueueOffset_, bufferNum, tilingData_->baseN * sizeof(T2));
    }

    pipe_.InitBuffer(outQueueY_, bufferNum, tilingData_->baseN * tilingData_->baseLen * sizeof(U));
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset>
__aicore__ inline void AscendAntiQuantV2PerHeadRegbase<T, T1, T2, U, SqrtMode, HasOffset>::Process()
{
    if (blockIdx_ >= tilingData_->numCore || blockN_ == 0) {
        return;
    }
    if (tilingData_->blockAxis == 0) {
        // only split axis 0
        gmXOffset_ = blockIdx_ * tilingData_->blockFactor * tilingData_->dim1 * tilingData_->dim2;
        gmSOffset_ = 0;
    } else if (tilingData_->blockAxis == 1) {
        // only split axis 1, blockUnion means factor per block on split axis
        gmXOffset_ = blockIdx_ / tilingData_->blockUnion * tilingData_->dim1 * tilingData_->dim2 +
                     blockIdx_ % tilingData_->blockUnion * tilingData_->blockFactor * tilingData_->dim2;
        gmSOffset_ = blockIdx_ % tilingData_->blockUnion * tilingData_->blockFactor;
    } else {
        gmXOffset_ = (blockIdx_ / tilingData_->blockUnion * tilingData_->dim2 +
                      blockIdx_ % tilingData_->blockUnion * tilingData_->blockFactor);
        gmSOffset_ = blockIdx_ / tilingData_->blockUnion;
    }

    nLoopLen = tilingData_->baseN;
    nLoopNum = blockN_ / nLoopLen;
    nLoopTail = blockN_ % nLoopLen;

    loopLen = tilingData_->baseLen;
    lenLoopNum = blockLen_ / loopLen;
    lenLoopTail = blockLen_ % loopLen;

    for (int64_t i = 0; i < blockS_; ++i) {
        // scale allows start from begin on each core
        int64_t baseSOffset = gmSOffset_;
        int64_t baseXOffset = gmXOffset_;

        for (int64_t i = 0; i < nLoopNum; ++i) {
            ProcessParamOneLoop(nLoopLen, baseSOffset, baseXOffset);
            baseXOffset += nLoopLen * tilingData_->dim2;
            baseSOffset += tilingData_->baseN;
        }
        if (nLoopTail != 0) {
            ProcessParamOneLoop(nLoopTail, baseSOffset, baseXOffset);
        }
        gmXOffset_ += tilingData_->dim1 * tilingData_->dim2;
    }
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset>
__aicore__ inline void AscendAntiQuantV2PerHeadRegbase<T, T1, T2, U, SqrtMode, HasOffset>::ProcessParamOneLoop(
    int64_t nLoopLen, int64_t baseSOffset, int64_t baseXOffset)
{
    // copy in scale
    CopyInParam<T1>(inQueueScale_, scaleGm_, nLoopLen, baseSOffset % tilingData_->dim1);
    auto scaleLocal = inQueueScale_.DeQue<T1>();

    if constexpr (HasOffset) {
        CopyInParam<T2>(inQueueOffset_, offsetGm_, nLoopLen, baseSOffset % tilingData_->dim1);
        auto offsetLocal = inQueueOffset_.DeQue<T2>();
        ProcessInputLoop(nLoopLen, baseXOffset, scaleLocal, offsetLocal);
        inQueueOffset_.FreeTensor(offsetLocal);
    } else {
        LocalTensor<T2> offsetLocal;
        ProcessInputLoop(nLoopLen, baseXOffset, scaleLocal, offsetLocal);
    }

    inQueueScale_.FreeTensor(scaleLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset>
template <typename dtypeCopyIn>
__aicore__ inline void AscendAntiQuantV2PerHeadRegbase<T, T1, T2, U, SqrtMode, HasOffset>::CopyInParam(
    TQue<QuePosition::VECIN, bufferNum>& inQueue, GlobalTensor<dtypeCopyIn>& inGm, int64_t paramLen,
    int64_t paramOffset)
{
    auto paramLocal = inQueue.AllocTensor<dtypeCopyIn>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = paramLen * sizeof(dtypeCopyIn);
    copyParams.dstStride = 0;
    copyParams.srcStride = 0;
    copyParams.rsv = 0;
    DataCopyPad(paramLocal, inGm[paramOffset], copyParams, {false, 0, 0, 0});
    inQueue.EnQue(paramLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset>
__aicore__ inline void AscendAntiQuantV2PerHeadRegbase<T, T1, T2, U, SqrtMode, HasOffset>::ParseCoreBlocks(
    const AscendAntiQuantV2TilingData* tilingData, int32_t blockIdx, int64_t& blockN, int64_t& blockS,
    int64_t& blockLen)
{
    if (tilingData->blockAxis == 0) {
        // blockFactor is in [1, S]
        if (blockIdx == tilingData->numCore - 1) {
            blockS = tilingData->blockTailFactor;
        } else {
            blockS = tilingData->blockFactor;
        }
        blockN = tilingData->dim1;
        blockLen = tilingData->dim2;
    } else if (tilingData->blockAxis == 1) {
        // blockFactor is in [1, N], blockUnion is in [1, N/blockFactor]
        if (blockIdx % tilingData->blockUnion == tilingData->blockUnion - 1) {
            blockN = tilingData->blockTailFactor;
        } else {
            blockN = tilingData->blockFactor;
        }
        blockLen = tilingData->dim2;
    } else {
        // blockFactor is in [1, D]
        blockN = 1;
        if (blockIdx % tilingData->blockUnion == tilingData->blockUnion - 1) {
            blockLen = tilingData->blockTailFactor;
        } else {
            blockLen = tilingData->blockFactor;
        }
    }
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset>
__aicore__ inline void AscendAntiQuantV2PerHeadRegbase<T, T1, T2, U, SqrtMode, HasOffset>::ProcessInputLoop(
    int64_t nLoopLen, int64_t baseXOffset, LocalTensor<T1>& scaleLocal, LocalTensor<T2>& offsetLocal)
{
    for (auto i = 0; i < lenLoopNum; ++i) {
        CopyInX(nLoopLen, loopLen, baseXOffset);
        Compute(nLoopLen, loopLen, scaleLocal, offsetLocal);
        CopyOutY(nLoopLen, loopLen, baseXOffset);
        baseXOffset += tilingData_->baseLen;
    }
    if (lenLoopTail != 0) {
        CopyInX(nLoopLen, lenLoopTail, baseXOffset);
        Compute(nLoopLen, lenLoopTail, scaleLocal, offsetLocal);
        CopyOutY(nLoopLen, lenLoopTail, baseXOffset);
    }
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset>
__aicore__ inline void AscendAntiQuantV2PerHeadRegbase<T, T1, T2, U, SqrtMode, HasOffset>::GetXInCopyParams(
    const AscendAntiQuantV2TilingData* tilingData, int64_t xN, int64_t xLen, int64_t lastDimLen,
    DataCopyExtParams& copyParams)
{
    int64_t xLenReal = xLen;
    if constexpr (IsSameType<T, int4b_t>::value) {
        xLenReal = xLenReal / this->INT4_NUMS_IN_INT8_SPACE;
    }

    copyParams.blockCount = xN;
    copyParams.blockLen = xLenReal * sizeof(xCopyDtype);
    if (lastDimLen > xLen) {
        if constexpr (IsSameType<T, int4b_t>::value) {
            copyParams.srcStride = (lastDimLen - xLen) * sizeof(xCopyDtype) / this->INT4_NUMS_IN_INT8_SPACE;
        } else {
            copyParams.srcStride = (lastDimLen - xLen) * sizeof(xCopyDtype);
        }
    } else {
        copyParams.srcStride = 0;
    }
    if (tilingData->baseLen > xLenReal) {
        copyParams.dstStride = (tilingData->baseLen - xLenReal) * sizeof(xCopyDtype) / this->BLOCK_SIZE;
    } else {
        copyParams.dstStride = 0;
    }
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset>
__aicore__ inline void AscendAntiQuantV2PerHeadRegbase<T, T1, T2, U, SqrtMode, HasOffset>::CopyInX(int64_t xN,
                                                                                                   int64_t xLen,
                                                                                                   int64_t xInOffset)
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        xInOffset = xInOffset >> 1;
    }

    LocalTensor<uint8_t> xLocal = inQueueX_.AllocTensor<uint8_t>();
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<uint8_t> padParams = {false, 0, 0, 0};
    GetXInCopyParams(tilingData_, xN, xLen, tilingData_->dim2, copyParams);
    DataCopyPad<uint8_t>(xLocal, xGm_[xInOffset], copyParams, padParams);
    inQueueX_.EnQue(xLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset>
__aicore__ inline void AscendAntiQuantV2PerHeadRegbase<T, T1, T2, U, SqrtMode, HasOffset>::Compute(
    int64_t nRow, int64_t dataCount, LocalTensor<T1>& sLocal, LocalTensor<T2>& oLocal)
{
    LocalTensor<xCopyDtype> xLocal = inQueueX_.DeQue<xCopyDtype>();
    LocalTensor<U> outLocal = outQueueY_.AllocTensor<U>();

    __local_mem__ xCopyDtype* xLocalAddr = (__local_mem__ xCopyDtype*)xLocal.GetPhyAddr();
    __local_mem__ T1* scaleLocalAddr = (__local_mem__ T1*)sLocal.GetPhyAddr();
    __local_mem__ U* outLocalAddr = (__local_mem__ U*)outLocal.GetPhyAddr();

    uint16_t VL = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    uint16_t HalfVL = VL / 2;
    uint16_t vfLoopNum = (dataCount + VL - 1) / VL;

    uint32_t xLocalOffset = static_cast<uint32_t>(tilingData_->baseLen);

    __VEC_SCOPE__
    {
        // x: int4, int8, hifp8, fp8_e5m2, fp8_e4m3
        AscendC::Reg::RegTensor<xCopyDtype> vregX;
        AscendC::Reg::RegTensor<float> vregFloatX;
        // scales: fp32, bf16
        AscendC::Reg::RegTensor<T1> vregS;
        AscendC::Reg::RegTensor<float> vregFloatS;
        // y: fp16, bf16
        AscendC::Reg::RegTensor<float> vregFloatY;
        AscendC::Reg::RegTensor<U> vregY;
        // offset: fp32, bf16
        AscendC::Reg::RegTensor<T2> vregO;
        AscendC::Reg::RegTensor<float> vregFloatO;

        AscendC::Reg::RegTensor<float> vregTmp1;
        AscendC::Reg::MaskReg mask;

        mask = AscendC::Reg::CreateMask<float>();
        for (uint16_t j = 0; j < static_cast<uint16_t>(nRow); ++j) {
            uint32_t count = dataCount;
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                mask = AscendC::Reg::UpdateMask<float>(count);
                __local_mem__ xCopyDtype* xSrc = IsSameType<T, int4b_t>::value ?
                                                     xLocalAddr + i * HalfVL + j * xLocalOffset :
                                                     xLocalAddr + i * VL + j * xLocalOffset;
                this->template LoadCastXToFloat<T>(vregX, vregFloatX, xSrc, mask);

                // ld and cast for scale
                if constexpr (IsSameType<T1, float>::value) {
                    // fp32
                    AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vregFloatS, scaleLocalAddr + j);
                } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
                    // bf16
                    AscendC::Reg::DataCopy<T1, AscendC::Reg::LoadDist::DIST_BRC_B16>(vregS, scaleLocalAddr + j);
                    AscendC::Reg::Cast<float, T1,
                                       AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CAST_TRAIT_BF16_TO_FP32>(
                        vregFloatS, vregS, mask);
                }

                if constexpr (HasOffset) {
                    // ld and cast for offset
                    __local_mem__ T2* offsetLocalAddr = (__local_mem__ T2*)oLocal.GetPhyAddr();
                    if constexpr (IsSameType<T2, float>::value) {
                        // fp32
                        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vregFloatO,
                                                                                            offsetLocalAddr + j);
                    } else if constexpr (IsSameType<T2, bfloat16_t>::value) {
                        // bf16
                        AscendC::Reg::DataCopy<T2, AscendC::Reg::LoadDist::DIST_BRC_B16>(vregO, offsetLocalAddr + j);
                        AscendC::Reg::Cast<float, T2,
                                           AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CAST_TRAIT_BF16_TO_FP32>(
                            vregFloatO, vregO, mask);
                    }

                    // compute
                    AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(vregTmp1, vregFloatX, vregFloatO,
                                                                                   mask);
                    if constexpr (SqrtMode == TPL_SQRT_MODE) {
                        AscendC::Reg::Mul(vregFloatS, vregFloatS, vregFloatS, mask);
                    }
                    AscendC::Reg::Mul(vregFloatY, vregTmp1, vregFloatS, mask);
                } else {
                    // compute
                    if constexpr (SqrtMode == TPL_SQRT_MODE) {
                        AscendC::Reg::Mul(vregFloatS, vregFloatS, vregFloatS, mask);
                    }
                    AscendC::Reg::Mul(vregFloatY, vregFloatX, vregFloatS, mask);
                }

                if constexpr (IsSameType<U, half>::value) {
                    AscendC::Reg::Cast<half, float,
                                       AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CAST_TRAIT_FP32_TO_HALF>(
                        vregY, vregFloatY, mask);
                    AscendC::Reg::DataCopy<U, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                        outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                } else if constexpr (IsSameType<U, bfloat16_t>::value) {
                    AscendC::Reg::Cast<U, float,
                                       AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CAST_TRAIT_FP32_TO_BF16>(
                        vregY, vregFloatY, mask);
                    AscendC::Reg::DataCopy<U, AscendC::Reg::StoreDist::DIST_PACK_B32>(
                        outLocalAddr + i * VL + j * xLocalOffset, vregY, mask);
                }
            }
        }
    }
    inQueueX_.FreeTensor(xLocal);
    outQueueY_.EnQue(outLocal);
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset>
__aicore__ inline void AscendAntiQuantV2PerHeadRegbase<T, T1, T2, U, SqrtMode, HasOffset>::GetOutCopyParams(
    const AscendAntiQuantV2TilingData* tilingData, int64_t yN, int64_t yLen, int64_t lastDimLen,
    DataCopyExtParams& copyParams)
{
    copyParams.blockCount = yN;
    copyParams.blockLen = yLen * sizeof(U);
    if (tilingData->baseLen > yLen) {
        copyParams.srcStride = (tilingData->baseLen - yLen) * sizeof(U) / this->BLOCK_SIZE;
    } else {
        copyParams.srcStride = 0;
    }
    if (lastDimLen > yLen) {
        copyParams.dstStride = (lastDimLen - yLen) * sizeof(U);
    } else {
        copyParams.dstStride = 0;
    }
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode, uint64_t HasOffset>
__aicore__ inline void AscendAntiQuantV2PerHeadRegbase<T, T1, T2, U, SqrtMode, HasOffset>::CopyOutY(int64_t yN,
                                                                                                    int64_t yLen,
                                                                                                    int64_t yOutOffset)
{
    LocalTensor<U> outLocal = outQueueY_.DeQue<U>();
    DataCopyExtParams copyParams;
    GetOutCopyParams(tilingData_, yN, yLen, tilingData_->dim2, copyParams);
    DataCopyPad<U>(yGm_[yOutOffset], outLocal, copyParams);
    outQueueY_.FreeTensor(outLocal);
}
} // namespace AscendAntiQuantV2
#endif // ASCEND_ANTI_QUANT_V2_PER_HEAD_REGBASE_H
