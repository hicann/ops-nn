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
 * \file dynamic_mx_quant_not_tail_axis_base.h
 * \brief
 */

#ifndef DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_BASE_H
#define DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_BASE_H
#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "dynamic_mx_quant_common.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"

namespace DynamicMxQuant {
using namespace AscendC;
template <typename T, typename U, const bool ISTAIL>
class DynamicMxQuantBase {
public:
    __aicore__ inline DynamicMxQuantBase(){};

protected:
    __aicore__ inline void BaseInit(
        GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workspace, const DynamicMxQuantTilingData* tilingData);
    __aicore__ inline void ParseTilingData(const DynamicMxQuantTilingData* tilingData);
    __aicore__ inline void InitCopyParams(int64_t blockCount, int64_t dataLen);
    __aicore__ inline void CopyIn(int64_t offset);
    __aicore__ inline void CopyOut(int64_t xOffset, int64_t scaleOffset, int64_t dataLen);
    __aicore__ inline int64_t CalcDataLen(
        int64_t curLoopBlockSizeNum, int64_t blockSizeNumHandled, int64_t scaleDataLen);
    __aicore__ inline int64_t BlockCountInCurCompute(int64_t blockSizeIdx);
    template <typename ComType>
    __aicore__ inline void LoadData(
        __ubuf__ T* xAddr, uint64_t offset, AscendC::MicroAPI::RegTensor<ComType>& xRegTensor,
        AscendC::MicroAPI::MaskReg& mask);

protected:
    TPipe pipe_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> mxScaleQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint8_t> mxScaleGm_;
    GlobalTensor<uint8_t> workspaceGm_;
    DataCopyExtParams inCopyParams_;
    DataCopyExtParams outCopyParams_;
    DataCopyPadExtParams<T> padParams_;
    int64_t blockIdx_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t uo_ = 0;
    int64_t ubDim_ = 0;
    int64_t ubFactor_ = 0;
    int64_t tailUbFactor_ = 0;
    int64_t blockFactor_ = 0;
    int64_t tailBlockFactor_ = 0;
    int64_t roundMode_ = 0;
    int64_t dstType_ = 0;
    int64_t blockSize_ = 0;
    int64_t blockSizeNumInAxis_ = 0;
    int64_t fullBlockSizeNumInAxis_ = 0;
    int64_t tailBlockSize_ = 0;
    int64_t postAxisSize_ = 0;
    int64_t mxScaleSize_ = 0;
    bool isPad_ = false;
    bool isTailBlock_ = false;
    using intCalcType = typename std::conditional<IsSame<T, half>::value, uint32_t, uint16_t>::type;
    constexpr static int16_t shrNum_ = GetShrNum<T>();
    constexpr static intCalcType maxExp_ = GetMaxExp<intCalcType>();
    constexpr static intCalcType f4Emax_ = GetFp4MaxExp<intCalcType>();
    constexpr static intCalcType f8Emax_ = GetFp8MaxExp<intCalcType>();
    constexpr static intCalcType maxBias_ = GetMaxBias<intCalcType>();
    constexpr static intCalcType nanValue_ = GetNanValue<intCalcType>();
    constexpr static intCalcType specialExp_ = GetSpecialExp<intCalcType>();
};

template <typename T, typename U, const bool ISTAIL>
__aicore__ inline void DynamicMxQuantBase<T, U, ISTAIL>::BaseInit(
    GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workspace, const DynamicMxQuantTilingData* tilingData)
{
#if (__NPU_ARCH__ == 3101)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    blockIdx_ = GetBlockIdx();
    ParseTilingData(tilingData);
    isTailBlock_ = blockIdx_ == (usedCoreNum_ - 1);
    fullBlockSizeNumInAxis_ = blockSizeNumInAxis_ - 1;
}

template <typename T, typename U, const bool ISTAIL>
__aicore__ inline void DynamicMxQuantBase<T, U, ISTAIL>::ParseTilingData(const DynamicMxQuantTilingData* tilingData)
{
    usedCoreNum_ = tilingData->usedCoreNum;
    totalCoreNum_ = tilingData->totalCoreNum;
    uo_ = tilingData->uo;
    ubDim_ = tilingData->ubDim;
    ubFactor_ = tilingData->ubFactor;
    tailUbFactor_ = tilingData->tailUbFactor;
    blockFactor_ = tilingData->blockFactor;
    tailBlockFactor_ = tilingData->tailBlockFactor;
    roundMode_ = tilingData->roundMode;
    dstType_ = tilingData->dstType;
    blockSize_ = tilingData->blockSize;
    tailBlockSize_ = tilingData->tailBlockSize;
    blockSizeNumInAxis_ = tilingData->blockSizeNumInAxis;
    postAxisSize_ = tilingData->postAxisSize;
    isPad_ = tilingData->isPad == 1;
    mxScaleSize_ = tilingData->mxScaleSize;
}

template <typename T, typename U, const bool ISTAIL>
__aicore__ inline void DynamicMxQuantBase<T, U, ISTAIL>::CopyIn(int64_t offset)
{
    LocalTensor<T> x = inQueue_.AllocTensor<T>();

    auto dst = (__ubuf__ T*)x.GetPhyAddr();
    auto src = (__gm__ T*)xGm_[offset].GetPhyAddr();
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;
    DataCopyPad<T, PaddingMode::Compact>(x, xGm_[offset], inCopyParams_, padParams);

    inQueue_.EnQue(x);
}

template <typename T, typename U, const bool ISTAIL>
__aicore__ inline void DynamicMxQuantBase<T, U, ISTAIL>::CopyOut(int64_t xOffset, int64_t scaleOffset, int64_t dataLen)
{
    DataCopyExtParams scaleCopyOutParams = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(uint8_t)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    LocalTensor<uint8_t> mxScale = mxScaleQueue_.DeQue<uint8_t>();
    DataCopyPad(workspaceGm_[scaleOffset], mxScale, scaleCopyOutParams);
    mxScaleQueue_.FreeTensor(mxScale);

    LocalTensor<uint8_t> y = outQueue_.DeQue<uint8_t>();
    DataCopyPad(yGm_[xOffset / DIGIT_TWO], y, outCopyParams_);
    outQueue_.FreeTensor(y);
}

template <typename T, typename U, const bool ISTAIL>
__aicore__ inline void DynamicMxQuantBase<T, U, ISTAIL>::InitCopyParams(int64_t blockCount, int64_t dataLen)
{
    if (ubDim_ == DIM2) {
        inCopyParams_ = {
            static_cast<uint16_t>(blockCount), static_cast<uint32_t>(dataLen * sizeof(T)),
            static_cast<uint32_t>((postAxisSize_) * sizeof(T)), static_cast<uint32_t>(dataLen * sizeof(T)),
            static_cast<uint32_t>(0)};
        outCopyParams_ = {
            static_cast<uint16_t>(blockCount), static_cast<uint32_t>(dataLen / DIGIT_TWO * sizeof(uint8_t)),
            static_cast<uint32_t>(0), static_cast<uint32_t>((postAxisSize_ - dataLen) / DIGIT_TWO * sizeof(uint8_t)),
            static_cast<uint32_t>(0)};
    } else {
        inCopyParams_ = {
            static_cast<uint16_t>(blockCount), static_cast<uint32_t>(dataLen * sizeof(T)), static_cast<uint32_t>(0),
            static_cast<uint32_t>(dataLen * sizeof(T)), static_cast<uint32_t>(0)};
        outCopyParams_ = {
            static_cast<uint16_t>(blockCount), static_cast<uint32_t>(dataLen / DIGIT_TWO * sizeof(uint8_t)),
            static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    }
    inCopyParams_.srcStride -= inCopyParams_.blockLen;
    padParams_ = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
}

template <typename T, typename U, const bool ISTAIL>
__aicore__ inline int64_t DynamicMxQuantBase<T, U, ISTAIL>::CalcDataLen(
    int64_t curLoopBlockSizeNum, int64_t blockSizeNumHandled, int64_t scaleDataLen)
{
    if (!isPad_) {
        return scaleDataLen * blockSize_;
    }
    int64_t dataLen = curLoopBlockSizeNum / blockSizeNumInAxis_ * postAxisSize_ *
                      (fullBlockSizeNumInAxis_ * blockSize_ + tailBlockSize_);
    int64_t blockSizeMod = curLoopBlockSizeNum % blockSizeNumInAxis_;
    if (blockSizeNumHandled % blockSizeNumInAxis_ + blockSizeMod < blockSizeNumInAxis_) {
        dataLen += blockSizeMod * postAxisSize_ * blockSize_;
    } else {
        dataLen += ((blockSizeMod - 1) * blockSize_ + tailBlockSize_) * postAxisSize_;
    }
    return dataLen;
}

template <typename T, typename U, const bool ISTAIL>
__aicore__ inline int64_t DynamicMxQuantBase<T, U, ISTAIL>::BlockCountInCurCompute(int64_t blockSizeIdx)
{
    if (!isPad_) {
        return blockSize_;
    }
    return (blockSizeIdx >= blockSizeNumInAxis_ && blockSizeIdx % blockSizeNumInAxis_ == 0) ? tailBlockSize_ :
                                                                                              blockSize_;
}

template <typename T, typename U, const bool ISTAIL>
template <typename ComType>
__aicore__ inline void DynamicMxQuantBase<T, U, ISTAIL>::LoadData(
    __ubuf__ T* xAddr, uint64_t offset, AscendC::MicroAPI::RegTensor<ComType>& xRegTensor,
    AscendC::MicroAPI::MaskReg& mask)
{
    AscendC::MicroAPI::UnalignReg uReg;
    if constexpr (IsSame<T, half>::value) {
        AscendC::MicroAPI::RegTensor<T> fp16Tensor;
        AscendC::MicroAPI::DataCopyUnAlignPre(uReg, xAddr + offset);
        AscendC::MicroAPI::DataCopyUnAlign(fp16Tensor, uReg, xAddr + offset);
        static constexpr AscendC::MicroAPI::CastTrait castTraitHalf2Fp32 = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        AscendC::MicroAPI::UnPack(
            (AscendC::MicroAPI::RegTensor<uint32_t>&)fp16Tensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)fp16Tensor);
        AscendC::MicroAPI::Cast<ComType, T, castTraitHalf2Fp32>(xRegTensor, fp16Tensor, mask);
    } else {
        AscendC::MicroAPI::DataCopyUnAlignPre(uReg, xAddr + offset);
        AscendC::MicroAPI::DataCopyUnAlign(xRegTensor, uReg, xAddr + offset);
    }
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_NOT_TAIL_AXIS_BASE_H