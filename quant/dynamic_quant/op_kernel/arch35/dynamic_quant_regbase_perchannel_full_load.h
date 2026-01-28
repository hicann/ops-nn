/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the 'License').
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_quant_regbase_full_load.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_REGBASE_PERCHANNEL_FULL_LOAD_H
#define DYNAMIC_QUANT_REGBASE_PERCHANNEL_FULL_LOAD_H

#include "dynamic_quant_regbase_base.h"
#include "dynamic_quant_regbase_perchannel_base.h"
#include "../inc/kernel_utils.h"

namespace DynamicQuantPerChannel {
using namespace AscendC;
constexpr uint32_t ALIGN_NUMBER_FP32 = 8;
constexpr uint32_t ALIGN_NUMBER_FP16 = 16;
constexpr uint32_t ALIGN_NUMBER_FP8 = 32;
constexpr uint32_t ALIGN_NUMBER_IN4 = 64;

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
class DynamicQuantRegbasePerChannnelFullLoad {
public:
    __aicore__ inline DynamicQuantRegbasePerChannnelFullLoad(TPipe* pipe) {
        pPipe = pipe;
    }
    // 没有group_index输入
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset,
                                GM_ADDR workSpace, const DynamicQuantTilingData* __restrict tilingData);
    __aicore__ inline void Process();

private:
    using yCopyDtype = std::conditional_t<IsSameType<yDtype, int4b_t>::value, uint8_t, yDtype>;
    __aicore__ inline void ParseTilingData(const DynamicQuantTilingData* tilingData);
    __aicore__ inline void ProcessSingleBlock(uint32_t batchBlockIdx, uint32_t nBlockIdx, uint32_t bBlockSize, uint32_t nBlockSize);
    __aicore__ inline void CopyIn(uint32_t bBlockSize, uint32_t nBlockSize, uint64_t  xOffset, uint64_t smoothOffset);
    __aicore__ inline void Compute(uint32_t bBlockSize, uint32_t nBlockSize);
    __aicore__ inline void ComputeVFforSymmetric(
        __local_mem__ xDtype* inAddr, __local_mem__ xDtype* smoothAddr, __local_mem__ yCopyDtype* yAddr,
        __local_mem__ float* scaleAddr, uint32_t bBlockSize, uint32_t nBlockSize);
    __aicore__ inline void ComputeVFforNoSymmetric(
        __local_mem__ xDtype* inAddr, __local_mem__ xDtype* smoothAddr, __local_mem__ yCopyDtype* yAddr,
        __local_mem__ float* scaleAddr, __local_mem__ float* offsetAddr, uint32_t bBlockSize, uint32_t nBlockSize);
    __aicore__ inline void CopyOut(uint32_t bBlockSize, uint32_t nBlockSize, uint64_t xOffset, uint64_t scaleOffset);

    /* ascendc variable */
    TQue<QuePosition::VECIN, USE_BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECIN, USE_BUFFER_NUM> smoothQueue_;
    TQue<QuePosition::VECOUT, USE_BUFFER_NUM> offsetQueue_;
    TQue<QuePosition::VECOUT, USE_BUFFER_NUM> outQueue_;
    TQue<QuePosition::VECOUT, USE_BUFFER_NUM> scaleQueue_;
    TPipe* pPipe = nullptr;
    /* global memory address */
    GlobalTensor<xDtype> inGm, smoothGm;
    
    GlobalTensor<float> offsetGm, scaleGm;
    GlobalTensor<yCopyDtype> outGm;

    uint32_t coreNum_ = 0;
    uint32_t headCoreNum_ = 0;
    uint32_t totalBatchLen_ = 0;
    uint32_t mLen_ = 0;
    uint32_t nLen_ = 0;
    uint32_t nBlockSize_ = 0;
    uint32_t nTailBlockSize_ = 0;
    uint32_t nBlockNum_ = 0;
    uint32_t nBaseLoopNum_ = 0;
    uint32_t blockPerHead_ = 0;
    uint32_t blockPerTail_ = 0;
    uint32_t totalBlockNum_ = 0;
    uint32_t nBaseSize_ = 0;
    uint32_t blockIdx =0;
    uint32_t curCoreProcessNum_ = 0;
    uint32_t blockStart_ = 0;
    uint32_t blockEnd_ = 0;
    uint32_t outBufferSize_ = 0;
    float maxValue = 0.0;
    float maxValueDiv = 0.0;
    float offsetDivValue = 0.0;
    uint32_t batchBlockSize_ = 1;
    uint32_t batchTailBlockSize_ = 1;
    uint32_t batchBlockNum_ = 1;
};

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelFullLoad<xDtype, yDtype, hasSmooth, isSymmetrical>::Init(
    GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset,
    GM_ADDR workSpace, const DynamicQuantTilingData* __restrict tilingData)
{
    DynamicQuantNDOpt::SetFloatOverflowModeForRegbase<yDtype>();
    ParseTilingData(tilingData);
    blockIdx = GetBlockIdx();
    if (blockIdx >= coreNum_) {
        return;
    }
    SetMaxValue<yDtype>(maxValueDiv, maxValue, offsetDivValue);

    // calc params
    curCoreProcessNum_ = blockIdx < headCoreNum_ ? blockPerHead_ : blockPerTail_;
    blockStart_ = blockIdx < headCoreNum_ ? blockIdx * blockPerHead_ :
                    headCoreNum_ * blockPerHead_ + (blockIdx - headCoreNum_) * blockPerTail_;
    blockEnd_ = blockStart_ + curCoreProcessNum_; 

    // init buffers
    inGm.SetGlobalBuffer((__gm__ xDtype*)x);
    outGm.SetGlobalBuffer((__gm__ yCopyDtype*)y);
    scaleGm.SetGlobalBuffer((__gm__ float*)scale);

    if constexpr (hasSmooth) {
        smoothGm.SetGlobalBuffer((__gm__ xDtype*)smooth_scales);
        // smooth_scale is col-wise
        pPipe->InitBuffer(smoothQueue_, USE_BUFFER_NUM, mLen_ * sizeof(xDtype));
    }

    pPipe->InitBuffer(inQueue_, USE_BUFFER_NUM, batchBlockSize_ * mLen_ * nBlockSize_ * sizeof(xDtype));
    pPipe->InitBuffer(outQueue_, USE_BUFFER_NUM, outBufferSize_ * sizeof(yCopyDtype));
    pPipe->InitBuffer(scaleQueue_, USE_BUFFER_NUM, batchBlockSize_ * nBlockSize_ * sizeof(float));
    if constexpr (isSymmetrical == false) {
        offsetGm.SetGlobalBuffer((__gm__ float*)offset);
        pPipe->InitBuffer(offsetQueue_, USE_BUFFER_NUM, batchBlockSize_ * nBlockSize_ * sizeof(float));
    }
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelFullLoad<xDtype, yDtype, hasSmooth, isSymmetrical>::ParseTilingData(const DynamicQuantTilingData* tilingData)
{
    coreNum_ = tilingData->coreNum;
    headCoreNum_ = tilingData->headCoreNum;
    totalBatchLen_ = tilingData->totalBatchLen;
    mLen_ = tilingData->mLen;
    nLen_ = tilingData->nLen;
    nBlockSize_ = tilingData->nBlockSize;
    nTailBlockSize_ = tilingData->nTailBlockSize;
    nBlockNum_ = tilingData->nBlockNum;
    nBaseSize_ = tilingData->nBaseSize;
    nBaseLoopNum_ = tilingData->nBaseLoopNum;
    blockPerHead_ = tilingData->blockPerHead;
    blockPerTail_ = tilingData->blockPerTail;
    totalBlockNum_ = tilingData->totalBlockNum;
    batchBlockSize_ = tilingData->batchBlockSize;
    batchTailBlockSize_ = tilingData->batchTailBlockSize;
    batchBlockNum_ = tilingData->batchBlockNum;
    if constexpr(IsSameType<yDtype, int4b_t>::value) {
        outBufferSize_ = batchBlockSize_ * mLen_* nBlockSize_ / 2;
    } else {
        outBufferSize_ = batchBlockSize_ * mLen_ * nBlockSize_;
    }
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelFullLoad<xDtype, yDtype, hasSmooth, isSymmetrical>::Process()
{
    if (blockIdx >= coreNum_) {
        return;
    }
    for (uint32_t i = blockStart_; i < blockEnd_; i++) {
        uint32_t nBlockIdx = i / batchBlockNum_;
        uint32_t batchBlockIdx = i - nBlockIdx * batchBlockNum_;
        uint32_t nBlockSize =  (nBlockIdx + 1) == nBlockNum_ ? nTailBlockSize_ : nBlockSize_;
        uint32_t bBlockSize =  (batchBlockIdx + 1) == batchBlockNum_ ? batchTailBlockSize_ : batchBlockSize_;
        ProcessSingleBlock(batchBlockIdx, nBlockIdx, bBlockSize, nBlockSize);
    }
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelFullLoad<xDtype, yDtype, hasSmooth, isSymmetrical>::ProcessSingleBlock(uint32_t batchBlockIdx, uint32_t nBlockIdx, uint32_t bBlockSize, uint32_t nBlockSize)
{   
    uint64_t xOffset = batchBlockIdx * batchBlockSize_ * nLen_ * mLen_ + nBlockIdx * nBlockSize_;
    uint64_t smoothOffset = 0;
    uint64_t scaleOffset = batchBlockIdx * batchBlockSize_ * nLen_  + nBlockIdx * nBlockSize_;
    CopyIn(bBlockSize, nBlockSize, xOffset, smoothOffset);
    Compute(bBlockSize, nBlockSize);
    CopyOut(bBlockSize, nBlockSize, xOffset, scaleOffset);
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelFullLoad<xDtype, yDtype, hasSmooth, isSymmetrical>::CopyIn(uint32_t bBlockSize, uint32_t nBlockSize,  uint64_t xOffset, uint64_t smoothOffset)
{
    uint8_t copyPadNumPerRow = ops::CeilAlign(nBlockSize, ALIGN_NUMBER_FP16) - nBlockSize;
    LocalTensor<xDtype> inLocal = inQueue_.AllocTensor<xDtype>();
    DataCopyExtParams copyParams = {
        static_cast<uint16_t>(mLen_ * bBlockSize),
        static_cast<uint32_t>(nBlockSize * sizeof(xDtype)), static_cast<uint32_t>((nLen_ - nBlockSize)* sizeof(xDtype)), 0, 0};
    DataCopyPadExtParams<xDtype> padParams{true, 0, copyPadNumPerRow, 0};
    DataCopyPad(inLocal, inGm[xOffset], copyParams, padParams);
    inQueue_.EnQue(inLocal);
    if constexpr (hasSmooth) {
        LocalTensor<xDtype> smoothLocal = smoothQueue_.AllocTensor<xDtype>();
        DataCopyExtParams smoothCopyParams = {1, static_cast<uint32_t>(mLen_ * sizeof(xDtype)), 0, 0, 0};
        DataCopyPadExtParams<xDtype> smoothPadParams{true, 0, 0, 0};
        DataCopyPad(smoothLocal, smoothGm[smoothOffset], smoothCopyParams, smoothPadParams);
        smoothQueue_.EnQue(smoothLocal);
    }
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelFullLoad<xDtype, yDtype, hasSmooth, isSymmetrical>::Compute(uint32_t bBlockSize, uint32_t nBlockSize)
{
    LocalTensor<xDtype> inLocal = inQueue_.DeQue<xDtype>();
    __ubuf__ xDtype* inAddr = (__ubuf__ xDtype*)inLocal.GetPhyAddr();
    __ubuf__ xDtype* smoothAddr{nullptr};
    LocalTensor<xDtype> smoothLocal;
    LocalTensor<yCopyDtype> outLocal = outQueue_.AllocTensor<yCopyDtype>();
    LocalTensor<float> scaleLocal = scaleQueue_.AllocTensor<float>();

    if constexpr (hasSmooth) {
        smoothLocal = smoothQueue_.DeQue<xDtype>();
        smoothAddr = (__ubuf__ xDtype*)smoothLocal.GetPhyAddr();
    }
    __ubuf__ yCopyDtype* outAddr = (__ubuf__ yCopyDtype*)outLocal.GetPhyAddr();
    __ubuf__ float* scaleAddr = (__ubuf__ float*)scaleLocal.GetPhyAddr();
    __ubuf__ float* offsetAddr;
     
    LocalTensor<float> offsetLocal;
    if constexpr (isSymmetrical == false) {
        offsetLocal = offsetQueue_.AllocTensor<float>();
        offsetAddr = (__ubuf__ float*)offsetLocal.GetPhyAddr();
    }
     if constexpr (isSymmetrical )
        ComputeVFforSymmetric(inAddr, smoothAddr, outAddr, scaleAddr, bBlockSize, nBlockSize);
    else
        ComputeVFforNoSymmetric(inAddr, smoothAddr, outAddr, scaleAddr, offsetAddr, bBlockSize, nBlockSize);

    outQueue_.EnQue(outLocal);
    scaleQueue_.EnQue(scaleLocal);
    if constexpr (hasSmooth) {
        smoothQueue_.FreeTensor(smoothLocal);
    }
    if constexpr (isSymmetrical == false) {
        offsetQueue_.EnQue(offsetLocal);
    }
    inQueue_.FreeTensor(inLocal);
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelFullLoad<xDtype, yDtype, hasSmooth, isSymmetrical>::ComputeVFforSymmetric(
    __local_mem__ xDtype* inAddr, __local_mem__ xDtype* smoothAddr, __local_mem__ yCopyDtype* yAddr,
    __local_mem__ float* scaleAddr, uint32_t bBlockSize, uint32_t nBlockSize)
{    
    uint32_t nSizeScale = ops::CeilAlign(nBlockSize, ALIGN_NUMBER_FP32);
    uint32_t nSize = ops::CeilAlign(nBlockSize, ALIGN_NUMBER_FP16);
    uint32_t nSizeOut = ops::CeilAlign(nBlockSize, ALIGN_NUMBER_FP8);
    if constexpr (IsSameType<yDtype, int4b_t>::value) {
        nSizeOut = ops::CeilAlign(nBlockSize, ALIGN_NUMBER_IN4);
    }
    uint16_t mLoopNum = static_cast<uint16_t>(mLen_);
    uint16_t nLoopNum = static_cast<uint16_t>(ops::CeilDiv(nSize, REG_LEN));
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> vregIn;
        MicroAPI::RegTensor<float> vregInFp32;
        MicroAPI::RegTensor<xDtype> vregSmooth;
        MicroAPI::RegTensor<float> vregSmoothFp32;
        MicroAPI::RegTensor<float> vregAbs;
        MicroAPI::RegTensor<float> vregOutScale;
        MicroAPI::RegTensor<float> vregColMax;
        MicroAPI::RegTensor<int16_t> vregCastI16;
        MicroAPI::RegTensor<half> vregCastF16;
        MicroAPI::RegTensor<yCopyDtype> vregOut;
        MicroAPI::RegTensor<float> vregOutFp32;
        MicroAPI::RegTensor<float> vregMaxFactor;
        MicroAPI::MaskReg preg0;
        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregH = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::H>();
        MicroAPI::Duplicate<float>(vregMaxFactor, maxValueDiv, pregAll);
        for (uint16_t bIdx = 0; bIdx < bBlockSize; bIdx++){
            uint32_t sregN = nSize;
            for (uint16_t i = 0; i < nLoopNum; i++) {
                preg0 = MicroAPI::UpdateMask<float>(sregN);
                MicroAPI::Duplicate<float>(vregColMax, NEG_INFINITY, preg0);
                for (uint16_t j = 0; j < mLoopNum; j++) {
                    MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregIn, (__ubuf__ xDtype*)(inAddr + i * REG_LEN + j * nSize + bIdx * mLen_ * nSize));
                    MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregInFp32, vregIn, preg0);
                    if constexpr (hasSmooth) {
                       MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_BRC_B16>(vregSmooth, (__ubuf__ xDtype*)(smoothAddr + j));
                       MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregSmoothFp32, vregSmooth, preg0);
                       MicroAPI::Mul<float>(vregInFp32, vregInFp32, vregSmoothFp32, preg0);
                    }
                    MicroAPI::Abs<float>(vregAbs, vregInFp32, preg0);
                    MicroAPI::Max<float>(vregColMax, vregAbs, vregColMax, preg0);
                }
                MicroAPI::Mul(vregOutScale, vregColMax, vregMaxFactor, preg0);
                for (uint16_t k = 0; k < mLoopNum; k++) {
                    auto addr = yAddr + i * REG_LEN + (bIdx * mLoopNum + k ) * nSizeOut;
                    MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregIn, (__ubuf__ xDtype*)(inAddr + i * REG_LEN + k *  nSize + bIdx * mLen_ * nSize));
                    MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregInFp32, vregIn, preg0);
                    if constexpr (hasSmooth) {
                       MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_BRC_B16>(vregSmooth, (__ubuf__ xDtype*)(smoothAddr + k));
                       MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregSmoothFp32, vregSmooth, preg0);
                       MicroAPI::Mul<float>(vregInFp32, vregInFp32, vregSmoothFp32, preg0);
                    }
                    MicroAPI::Div<float>(vregOutFp32, vregInFp32, vregOutScale, preg0);
                    CastToDstType<yDtype, yCopyDtype>(vregOutFp32, vregOut, preg0);
                    if constexpr (IsSameType<yDtype, int4b_t>::value) {
                       addr = yAddr + (i * REG_LEN + (bIdx * mLen_ + k) * nSizeOut) / 2;
                       MicroAPI::DataCopy<yCopyDtype, MicroAPI::StoreDist::DIST_PACK4_B32>(addr, vregOut, pregH);
                    } else {
                       MicroAPI::DataCopy<yCopyDtype, MicroAPI::StoreDist::DIST_PACK4_B32>(addr, vregOut, preg0);
                    }
                }
                   MicroAPI::DataCopy<float>((__ubuf__ float*)(scaleAddr + i * REG_LEN + bIdx * nSizeScale), vregOutScale, preg0);
            }
       }
}
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelFullLoad<xDtype, yDtype, hasSmooth, isSymmetrical>::ComputeVFforNoSymmetric(
    __local_mem__ xDtype* inAddr, __local_mem__ xDtype* smoothAddr, __local_mem__ yCopyDtype* yAddr,
    __local_mem__ float* scaleAddr, __local_mem__ float* offsetAddr, uint32_t bBlockSize, uint32_t nBlockSize)
{       
        uint32_t nSizeScale = ops::CeilAlign(nBlockSize, ALIGN_NUMBER_FP32);
        uint32_t nSize = ops::CeilAlign(nBlockSize, ALIGN_NUMBER_FP16);
        uint32_t nSizeOut = ops::CeilAlign(nBlockSize, ALIGN_NUMBER_FP8);
        if constexpr (IsSameType<yDtype, int4b_t>::value){
             nSizeOut = ops::CeilAlign(nBlockSize, ALIGN_NUMBER_IN4);
       }
        uint16_t mLoopNum = static_cast<uint16_t>(mLen_);
        uint16_t nLoopNum = static_cast<uint16_t>(ops::CeilDiv(nSize, REG_LEN));  
        
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> vregIn;
        MicroAPI::RegTensor<float> vregInFp32;
        MicroAPI::RegTensor<xDtype> vregSmooth;
        MicroAPI::RegTensor<float> vregSmoothFp32;
        MicroAPI::RegTensor<float> vregResult;
        MicroAPI::RegTensor<float> vregOutScale;
        MicroAPI::RegTensor<float> vregColMax;
        MicroAPI::RegTensor<float> vregColMin;
        MicroAPI::RegTensor<float> vregDivScale;
        MicroAPI::RegTensor<float> vregOffset;
        MicroAPI::RegTensor<float> vregMaxFactor;
        MicroAPI::RegTensor<int16_t> vregCastI16;
        MicroAPI::RegTensor<half> vregCastF16;
        MicroAPI::RegTensor<yCopyDtype> vregOut;
        MicroAPI::RegTensor<float> vregOutFp32;
        MicroAPI::RegTensor<float> vregDiv;
        MicroAPI::RegTensor<float> vregOffsetDivVal;
        MicroAPI::MaskReg preg0;
        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregH = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::H>();
        MicroAPI::Duplicate<float>(vregMaxFactor, maxValue, pregAll);
      
        MicroAPI::Duplicate<float>(vregOffsetDivVal, offsetDivValue, pregAll);
       
        for(uint16_t bIdx =0; bIdx < bBlockSize; bIdx++){
            uint32_t sregN =nSize;
            for (uint16_t i = 0; i < nLoopNum; i++) {
                preg0 = MicroAPI::UpdateMask<float>(sregN);
                MicroAPI::Duplicate<float>(vregColMax, NEG_INFINITY, preg0);
                MicroAPI::Duplicate<float>(vregColMin, POS_INFINITY, preg0);
                for (uint16_t j = 0; j < mLoopNum; j++) {
                    MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregIn, (__ubuf__ xDtype*)(inAddr + i * REG_LEN + (j + bIdx * mLen_) * nSize));
                    MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregInFp32, vregIn, preg0);
                    if constexpr (hasSmooth) {
                        MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_BRC_B16>(vregSmooth, (__ubuf__ xDtype*)(smoothAddr + j));
                        MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregSmoothFp32, vregSmooth, preg0);
                        MicroAPI::Mul<float>(vregInFp32, vregInFp32, vregSmoothFp32, preg0);
                    }
                    MicroAPI::Max<float>(vregColMax, vregInFp32, vregColMax, preg0);
                    MicroAPI::Min<float>(vregColMin, vregInFp32, vregColMin, preg0);
                }
                MicroAPI::Sub(vregResult, vregColMax, vregColMin, preg0);
                MicroAPI::Mul(vregOutScale, vregResult, vregOffsetDivVal, preg0);
                MicroAPI::Div<float,  &divHighPrecisionMode>(vregDivScale, vregColMax, vregOutScale, preg0);
                MicroAPI::Sub<float>(vregOffset, vregMaxFactor, vregDivScale, preg0);

                for (uint16_t k = 0; k < mLoopNum; k++) {
                    auto addr = yAddr + i * REG_LEN + (k + bIdx * mLen_ ) * nSizeOut;
                    MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(vregIn, (__ubuf__ xDtype*)(inAddr + i * REG_LEN + (k + bIdx * mLen_) * nSize));
                    MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregInFp32, vregIn, preg0);
                    if constexpr (hasSmooth) {
                        MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_BRC_B16>(vregSmooth, (__ubuf__ xDtype*)(smoothAddr + k));
                        MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregSmoothFp32, vregSmooth, preg0);
                        MicroAPI::Mul<float>(vregInFp32, vregInFp32, vregSmoothFp32, preg0);
                    }
                    MicroAPI::Div<float>(vregDiv, vregInFp32, vregOutScale,preg0);
                    MicroAPI::Add<float>(vregOutFp32, vregDiv, vregOffset, preg0);

                    CastToDstType<yDtype, yCopyDtype>(vregOutFp32, vregOut, preg0);
                    if constexpr (IsSameType<yDtype, int4b_t>::value) {
                        addr = yAddr + (i * REG_LEN + (k + bIdx * mLen_) * nSizeOut) / 2;
                        MicroAPI::DataCopy<yCopyDtype, MicroAPI::StoreDist::DIST_PACK4_B32>(addr, vregOut, pregH);
                    } else {
                        MicroAPI::DataCopy<yCopyDtype, MicroAPI::StoreDist::DIST_PACK4_B32>(addr, vregOut, preg0);
                    }
                }
                MicroAPI::DataCopy<float>((__ubuf__ float*)scaleAddr + i * REG_LEN + bIdx * nSizeScale, vregOutScale, preg0);
                MicroAPI::DataCopy<float>((__ubuf__ float*)offsetAddr + i * REG_LEN + bIdx * nSizeScale, vregOffset, preg0);
            }
    }
}
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void
DynamicQuantRegbasePerChannnelFullLoad<xDtype, yDtype, hasSmooth, isSymmetrical>::CopyOut(
     uint32_t bBlockSize, uint32_t nBlockSize, uint64_t xOffset, uint64_t scaleOffset)
{
    LocalTensor<yCopyDtype> yLocal = outQueue_.DeQue<yCopyDtype>();
    LocalTensor<float> scaleLocal = scaleQueue_.DeQue<float>();
    DataCopyExtParams scaleCopyParams{
        static_cast<uint16_t>(bBlockSize), static_cast<uint32_t>(nBlockSize* sizeof(float)), 0, static_cast<uint32_t>((nLen_ - nBlockSize)* sizeof(float)), 0};
    DataCopyPad(scaleGm[scaleOffset], scaleLocal, scaleCopyParams);

    LocalTensor<float> offsetLocal;
    if constexpr (isSymmetrical == false) {
        offsetLocal = offsetQueue_.DeQue<float>();
        DataCopyPad(offsetGm[scaleOffset], offsetLocal, scaleCopyParams);
        offsetQueue_.FreeTensor(offsetLocal);
    }
    DataCopyExtParams copyParams{
         static_cast<uint16_t>(mLen_ * bBlockSize), static_cast<uint32_t>(nBlockSize* sizeof(yCopyDtype)), 0, static_cast<uint32_t>((nLen_ - nBlockSize)* sizeof(yCopyDtype)), 0};
    if constexpr (IsSameType<yDtype, int4b_t>::value) {
        copyParams.blockLen = copyParams.blockLen >> 1;
        copyParams.dstStride = copyParams.dstStride >> 1;
        DataCopyPad(outGm[xOffset / 2], yLocal, copyParams);
    } else {
        DataCopyPad(outGm[xOffset], yLocal, copyParams);
    }
    outQueue_.FreeTensor(yLocal);
    scaleQueue_.FreeTensor(scaleLocal);
}

}  // namespace DynamicQuantPerChannelFullLoad
#endif  // DYNAMIC_QUANT_REGBASE_PERCHANNEL_FULL_LOAD_H
