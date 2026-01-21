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
 * \file pool_3d_ksize_one_kernel.h
 * \brief
 */
#ifndef POOL_3D_KSIZE_ONE_KERNEL_H_
#define POOL_3D_KSIZE_ONE_KERNEL_H_

#include "pool_3d_common.h"

#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"

namespace Pool3D
{
using namespace AscendC;

template <typename T, int32_t OP_TYPE>
class Pool3DKsizeOneKernel
{
public:
    __aicore__ inline Pool3DKsizeOneKernel(TPipe* pipe, const Pool3DOneKsizeTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInMultiRows(int64_t offset, const ShapeInfo &inputInfo);
    __aicore__ inline void CopyMaxOut(int64_t offset, uint32_t blockLen);
    __aicore__ inline void ComputeAvg(uint32_t datalen);
    __aicore__ inline void Compute(uint32_t datalen);
    __aicore__ inline int64_t min(int64_t a, int64_t b)
    {
        return (a > b) ? b : a;
    }

    TPipe* pipe_;
    // 输入队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    // 输出ub
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> maxGm_;
    const Pool3DOneKsizeTilingData* tilingData_;
};

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DKsizeOneKernel<T, OP_TYPE>::Init(GM_ADDR x, GM_ADDR y)
{
    // GM
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    maxGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->inUbSize * sizeof(T));
    pipe_->InitBuffer(outputQue_, BUFFER_NUM, tilingData_->inUbSize * sizeof(T));
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DKsizeOneKernel<T, OP_TYPE>::Process()
{
    int64_t startIdx = 0;
    int64_t endIdx = 0;
    if (GetBlockIdx() < tilingData_->blockTail) {
        startIdx = GetBlockIdx() * (tilingData_->blockFactor + 1);
        endIdx = startIdx + tilingData_->blockFactor + 1;
    } else {
        startIdx = GetBlockIdx() * tilingData_->blockFactor + tilingData_->blockTail;
        endIdx = startIdx + tilingData_->blockFactor;
    }
    for (int64_t idx = startIdx; idx < endIdx; idx++) {
        int64_t nIdx = idx / (tilingData_->dLoop * tilingData_->hLoop * tilingData_->wLoop);
        int64_t tmpIdx = idx - nIdx * tilingData_->dLoop * tilingData_->hLoop * tilingData_->wLoop;
        int64_t dIdx = tmpIdx / (tilingData_->hLoop * tilingData_->wLoop);
        int64_t hIdx = (tmpIdx - dIdx * tilingData_->hLoop * tilingData_->wLoop) / tilingData_->wLoop;
        int64_t wIdx = tmpIdx % tilingData_->wLoop;
        uint32_t n = nIdx == tilingData_->nLoop - 1 ? tilingData_->nOutDim - nIdx * tilingData_->ubFactorN
                                                   : tilingData_->ubFactorN;
        int64_t depthStart = dIdx * tilingData_->sD * tilingData_->outUbFactorD; 
        int64_t rowStart = hIdx * tilingData_->sH * tilingData_->outUbFactorH;
        int64_t colStart = wIdx * tilingData_->sW * tilingData_->outUbFactorW;

        int64_t srcOffset = nIdx * tilingData_->ubFactorN * tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim +
                            depthStart * tilingData_->hInDim * tilingData_->wInDim + rowStart * tilingData_->wInDim + colStart;
        int64_t dstOffset = nIdx * tilingData_->ubFactorN * tilingData_->dOutDim * tilingData_->hOutDim * tilingData_->wOutDim +
                            + dIdx * tilingData_->outUbFactorD  * tilingData_->hOutDim * tilingData_->wOutDim 
                            + hIdx * tilingData_->outUbFactorH * tilingData_->wOutDim + wIdx * tilingData_->outUbFactorW;
        srcOffset *= tilingData_->channel;
        dstOffset *= tilingData_->channel;
 
        uint32_t depths = dIdx == tilingData_->dLoop - 1 ? tilingData_->dOutDim - dIdx * tilingData_->outUbFactorD : tilingData_->outUbFactorD;                                           
        uint32_t rows = hIdx == tilingData_->hLoop - 1 ? tilingData_->hOutDim - hIdx * tilingData_->outUbFactorH : tilingData_->outUbFactorH;
        uint32_t cols = wIdx == tilingData_->wLoop - 1 ? tilingData_->wOutDim - wIdx * tilingData_->outUbFactorW : tilingData_->outUbFactorW;

        ShapeInfo tensorInfo = {n, depths, rows, cols, static_cast<uint32_t>(tilingData_->channel)};
        uint32_t totalNum = n*depths*rows*cols*tilingData_->channel;
        CopyInMultiRows(srcOffset, tensorInfo);
        Compute(totalNum);
        CopyMaxOut(dstOffset, totalNum);
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DKsizeOneKernel<T, OP_TYPE>::CopyInMultiRows(int64_t offset, const ShapeInfo &inputInfo)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
    if (tilingData_->dataFormat == 0) {    
        MultiCopyLoopInfo<Pool3D::FOUR> loopInfo;
        loopInfo.loopSize[ZERO] = inputInfo.width;
        loopInfo.loopSize[ONE] = inputInfo.height;
        loopInfo.loopSize[TWO] = inputInfo.depth;
        loopInfo.loopSize[THREE] = inputInfo.n;
        loopInfo.loopSrcStride[ZERO] = tilingData_->sW;
        loopInfo.loopSrcStride[ONE] = tilingData_->sH * tilingData_->wInDim;
        loopInfo.loopSrcStride[TWO] = tilingData_->sD * tilingData_->hInDim * tilingData_->wInDim;
        loopInfo.loopSrcStride[THREE] = tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim;
        loopInfo.loopDstStride[ZERO] = 1;
        loopInfo.loopDstStride[ONE] = inputInfo.width;
        loopInfo.loopDstStride[TWO] = inputInfo.width * inputInfo.height;
        loopInfo.loopDstStride[THREE] = inputInfo.width * inputInfo.height * inputInfo.depth; 
        static constexpr MultiCopyConfig config = {false};
        MultiCopyParams<T, Pool3D::FOUR> paramsMain = {loopInfo};
        DataCopy<T, Pool3D::FOUR, config>(xLocal, xGm_[offset], paramsMain);
    } else {
        MultiCopyLoopInfo<Pool3D::FIVE> loopInfo;
        loopInfo.loopSize[ZERO] = inputInfo.channel;
        loopInfo.loopSize[ONE] = inputInfo.width;
        loopInfo.loopSize[TWO] = inputInfo.height;
        loopInfo.loopSize[THREE] = inputInfo.depth;
        loopInfo.loopSize[FOUR] = inputInfo.n;
        loopInfo.loopSrcStride[ZERO] = 1;
        loopInfo.loopSrcStride[ONE] = tilingData_->sW * tilingData_->channel;
        loopInfo.loopSrcStride[TWO] = tilingData_->sH * tilingData_->wInDim * tilingData_->channel;
        loopInfo.loopSrcStride[THREE] = tilingData_->sD * tilingData_->hInDim * tilingData_->wInDim * tilingData_->channel;
        loopInfo.loopSrcStride[FOUR] = tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim * tilingData_->channel;
        loopInfo.loopDstStride[ZERO] = 1;
        loopInfo.loopDstStride[ONE] = inputInfo.channel;
        loopInfo.loopDstStride[TWO] = inputInfo.width * inputInfo.channel;
        loopInfo.loopDstStride[THREE] = inputInfo.width * inputInfo.height * inputInfo.channel; 
        loopInfo.loopDstStride[FOUR] = inputInfo.depth * inputInfo.height * inputInfo.width * inputInfo.channel; 
        static constexpr MultiCopyConfig config = {false};
        MultiCopyParams<T, Pool3D::FIVE> paramsMain = {loopInfo};
        DataCopy<T, Pool3D::FIVE, config>(xLocal, xGm_[offset], paramsMain);
    }
    inputQue_.EnQue(xLocal);
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DKsizeOneKernel<T, OP_TYPE>::CopyMaxOut(int64_t offset, uint32_t blockLen)
{
    LocalTensor<T> maxOutLocal = outputQue_.DeQue<T>();
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = blockLen * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    DataCopyPad<T>(maxGm_[offset], maxOutLocal, extParams);
    outputQue_.FreeTensor<T>(maxOutLocal);
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DKsizeOneKernel<T, OP_TYPE>::Compute(uint32_t datalen)
{
    if constexpr (OP_TYPE == OP_TYPE_AVG_POOL_3D) {
        ComputeAvg(datalen);
    } else {
        LocalTensor<T> maxOutLocal = outputQue_.AllocTensor<T>();
        LocalTensor<T> xLocal = inputQue_.DeQue<T>();   
        AscendC::Copy(maxOutLocal, xLocal, datalen);
        inputQue_.FreeTensor<T>(xLocal);
        outputQue_.EnQue<T>(maxOutLocal); 
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3DKsizeOneKernel<T, OP_TYPE>::ComputeAvg(uint32_t datalen)
{
    LocalTensor<T> maxOutLocal = outputQue_.AllocTensor<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();

    __local_mem__ T* srcAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ T* dstAddr = (__local_mem__ T*)maxOutLocal.GetPhyAddr();

    uint32_t repeatElm = platform::GetVRegSize() / sizeof(float32_t);
    uint16_t loopNum = static_cast<uint16_t>((datalen + repeatElm - 1) / repeatElm);
    float32_t divisor = static_cast<float32_t>(tilingData_->divisor);
    __VEC_SCOPE__
    {   
        MicroAPI::RegTensor<T> src;
        MicroAPI::RegTensor<float32_t> div;
        MicroAPI::RegTensor<float32_t> tmp;
        MicroAPI::RegTensor<T> res;
        MicroAPI::Duplicate(div, divisor);
        uint32_t sreg = datalen;
        for (uint16_t i = 0; i < loopNum; i++) {
            MicroAPI::AddrReg srcOffset = MicroAPI::CreateAddrReg<T>(i, repeatElm);
            MicroAPI::AddrReg dstOffset = MicroAPI::CreateAddrReg<T>(i, repeatElm);
            MicroAPI::MaskReg pMask = MicroAPI::UpdateMask<float32_t>(sreg);

            if constexpr(std::is_same<T, float32_t>::value) {
                MicroAPI::DataCopy(src, srcAddr, srcOffset);  
                MicroAPI::Div(res, src, div, pMask); 
                MicroAPI::DataCopy(dstAddr, res, dstOffset, pMask);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(src, srcAddr, srcOffset);  
                MicroAPI::Cast<float32_t, T, Pool3D::castTraitB16ToB32>(tmp, src, pMask);
                MicroAPI::Div(tmp, tmp, div, pMask); 
                MicroAPI::Cast<T, float32_t, Pool3D::castTraitB32ToB16>(res, tmp, pMask);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstAddr, res, dstOffset, pMask);
            }
        }
    } 
    inputQue_.FreeTensor<T>(xLocal);
    outputQue_.EnQue<T>(maxOutLocal);
}

}  // namespace Pool3D
#endif  // POOL_3D_KSIZE_ONE_KERNEL_H_
