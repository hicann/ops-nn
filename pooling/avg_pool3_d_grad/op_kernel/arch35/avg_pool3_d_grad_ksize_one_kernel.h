/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool3_d_grad_ksize_one_kernel.h
 * \brief KsizeOne kernel for 3D average pooling backward (arch35).
 *        When kD=kH=kW=1 and sD=sH=sW=1: grads and output have identical shape.
 *        Backward is simply: read grad → Div by divisor → write to output.
 *        Flat 1D tiling, no SyncAll, no zero-fill, no stride scatter.
 */

#ifndef AVG_POOL3_D_GRAD_KSIZE_ONE_KERNEL_H_
#define AVG_POOL3_D_GRAD_KSIZE_ONE_KERNEL_H_

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "avg_pool3_d_grad_base.h"
#include "avg_pool3_d_grad_tiling_data.h"

namespace AvgPool3DGradKsizeOneNameSpace {
using namespace AscendC;
using namespace AvgPool3DGrad;

template <typename T, bool HAS_DIVISOR>
class AvgPool3DGradKsizeOne {
public:
    __aicore__ inline AvgPool3DGradKsizeOne(TPipe* pipe, const AvgPool3DGradKsizeOneTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling)
    {}
    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR output);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t gmOffset, uint32_t elementCount);
    __aicore__ inline void ComputeDiv(uint32_t elementCount);
    __aicore__ inline void CopyOut(int64_t gmOffset, uint32_t elementCount);

    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue_;
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> outputGm_;
    const AvgPool3DGradKsizeOneTilingData* tilingData_;
};

template <typename T, bool HAS_DIVISOR>
__aicore__ inline void AvgPool3DGradKsizeOne<T, HAS_DIVISOR>::Init(GM_ADDR grads, GM_ADDR output)
{
    gradGm_.SetGlobalBuffer((__gm__ T*)grads);
    outputGm_.SetGlobalBuffer((__gm__ T*)output);
    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->ubBufferSize * sizeof(T));
    pipe_->InitBuffer(outputQue_, BUFFER_NUM, tilingData_->ubBufferSize * sizeof(T));
}

template <typename T, bool HAS_DIVISOR>
__aicore__ inline void AvgPool3DGradKsizeOne<T, HAS_DIVISOR>::Process()
{
    int64_t coreId = static_cast<int64_t>(GetBlockIdx());
    if (coreId >= tilingData_->usedCoreNum) {
        return;
    }

    int64_t startElem = 0;
    int64_t endElem = 0;
    if (coreId < tilingData_->tailCoreElements) {
        startElem = coreId * (tilingData_->elementsPerCore + 1);
        endElem = startElem + tilingData_->elementsPerCore + 1;
    } else {
        startElem = coreId * tilingData_->elementsPerCore + tilingData_->tailCoreElements;
        endElem = startElem + tilingData_->elementsPerCore;
    }

    int64_t ubCapacity = tilingData_->ubBufferSize;
    for (int64_t offset = startElem; offset < endElem; offset += ubCapacity) {
        uint32_t elementCount = static_cast<uint32_t>((offset + ubCapacity > endElem) ? (endElem - offset) :
                                                                                        ubCapacity);
        CopyIn(offset, elementCount);
        ComputeDiv(elementCount);
        CopyOut(offset, elementCount);
    }
}

template <typename T, bool HAS_DIVISOR>
__aicore__ inline void AvgPool3DGradKsizeOne<T, HAS_DIVISOR>::CopyIn(int64_t gmOffset, uint32_t elementCount)
{
    LocalTensor<T> gradLocal = inputQue_.AllocTensor<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = elementCount * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    DataCopyPadExtParams<T> padParams;
    DataCopyPad<T>(gradLocal, gradGm_[gmOffset], copyParams, padParams);
    inputQue_.EnQue(gradLocal);
}

template <typename T, bool HAS_DIVISOR>
__aicore__ inline void AvgPool3DGradKsizeOne<T, HAS_DIVISOR>::ComputeDiv(uint32_t elementCount)
{
    LocalTensor<T> outputLocal = outputQue_.AllocTensor<T>();
    LocalTensor<T> gradLocal = inputQue_.DeQue<T>();

    if constexpr (!HAS_DIVISOR) {
        AscendC::Copy(outputLocal, gradLocal, elementCount);
        inputQue_.FreeTensor(gradLocal);
        outputQue_.EnQue(outputLocal);
        return;
    }

    __local_mem__ T* gradAddr = (__local_mem__ T*)gradLocal.GetPhyAddr();
    __local_mem__ T* outputAddr = (__local_mem__ T*)outputLocal.GetPhyAddr();

    uint32_t vRegSize = platform::GetVRegSize() / sizeof(float32_t);
    uint16_t repeatCount = static_cast<uint16_t>(ops::CeilDiv(elementCount, vRegSize));
    float32_t divisorVal = static_cast<float32_t>(tilingData_->divisor);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> gradReg;
        MicroAPI::RegTensor<float32_t> divisorReg;
        MicroAPI::RegTensor<float32_t> tmpReg;
        MicroAPI::RegTensor<T> resultReg;
        MicroAPI::Duplicate(divisorReg, divisorVal);
        for (uint16_t repeatIdx = 0; repeatIdx < repeatCount; repeatIdx++) {
            MicroAPI::AddrReg srcAddrReg = MicroAPI::CreateAddrReg<T>(repeatIdx, vRegSize);
            MicroAPI::AddrReg dstAddrReg = MicroAPI::CreateAddrReg<T>(repeatIdx, vRegSize);
            uint32_t validCount = (repeatIdx == repeatCount - 1) ? (elementCount - repeatIdx * vRegSize) : vRegSize;
            MicroAPI::MaskReg maskReg = MicroAPI::UpdateMask<float32_t>(validCount);

            if constexpr (std::is_same<T, float32_t>::value) {
                MicroAPI::DataCopy(gradReg, gradAddr, srcAddrReg);
                MicroAPI::Div(resultReg, gradReg, divisorReg, maskReg);
                MicroAPI::DataCopy(outputAddr, resultReg, dstAddrReg, maskReg);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(gradReg, gradAddr, srcAddrReg);
                MicroAPI::Cast<float32_t, T, castTraitT1ComputeType>(tmpReg, gradReg, maskReg);
                MicroAPI::Div(tmpReg, tmpReg, divisorReg, maskReg);
                MicroAPI::Cast<T, float32_t, castTraitU32U16>(resultReg, tmpReg, maskReg);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(outputAddr, resultReg, dstAddrReg, maskReg);
            }
        }
    }
    inputQue_.FreeTensor(gradLocal);
    outputQue_.EnQue(outputLocal);
}

template <typename T, bool HAS_DIVISOR>
__aicore__ inline void AvgPool3DGradKsizeOne<T, HAS_DIVISOR>::CopyOut(int64_t gmOffset, uint32_t elementCount)
{
    LocalTensor<T> outputLocal = outputQue_.DeQue<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = elementCount * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    DataCopyPad<T>(outputGm_[gmOffset], outputLocal, copyParams);
    outputQue_.FreeTensor(outputLocal);
}

} // namespace AvgPool3DGradKsizeOneNameSpace
#endif // AVG_POOL3_D_GRAD_KSIZE_ONE_KERNEL_H_
