/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file non_zero_small_mask_trans_1.h
 * \brief
 */

#ifndef CANN_NON_ZERO_SMALL_MASK_ONE_DIM_H
#define CANN_NON_ZERO_SMALL_MASK_ONE_DIM_H

#include "non_zero_base.h"

namespace NonZero {
using namespace AscendC;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

template <typename T1, typename T2, int TILING_KEY>
class NonZeroSmallMask1 : public NonZeroBase<T1, T2, TILING_KEY> {
public:
    __aicore__ inline NonZeroSmallMask1(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR outShape, GM_ADDR workspace, const NonZeroTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPreCore(int32_t loopNum, int32_t tailNum, int32_t loopNumOut, int32_t loopTail);
    __aicore__ inline void ComputeOut(uint64_t arNum, uint64_t loopGm, LocalTensor<T2>& dstInt32);
    __aicore__ inline void ComputeIds(
        uint64_t& loopGm, int32_t loopCore, int32_t beforeNumOut, LocalTensor<uint32_t>& maskUbSize);
};

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroSmallMask1<T1, T2, TILING_KEY>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR outShape, GM_ADDR workspace, const NonZeroTilingData* tilingData)
{
    this->InitBase(x, y, outShape, workspace, tilingData);
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroSmallMask1<T1, T2, TILING_KEY>::ComputeOut(
    uint64_t arNum, uint64_t loopGm, LocalTensor<T2>& dstInt32)
{
    this->inQueX_.template EnQue<QuePosition::VECCALC, QuePosition::VECOUT>(dstInt32);
    LocalTensor<T2> outUbSize = this->inQueX_.template DeQue<QuePosition::VECCALC, QuePosition::VECOUT, T2>();

    this->copyParams.blockCount = 1;
    this->copyParams.blockLen = arNum * sizeof(T2);
    this->copyParams.srcStride = 0;
    this->copyParams.dstStride = 0;
    DataCopyPad(this->yGm_[this->gmOffset_ + loopGm - arNum], outUbSize, this->copyParams);
    this->inQueX_.template FreeTensor(outUbSize);
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroSmallMask1<T1, T2, TILING_KEY>::ComputeIds(
    uint64_t& loopGm, int32_t loopCore, int32_t beforeNumOut, LocalTensor<uint32_t>& maskUbSize)
{
    LocalTensor<T2> dstUbInt64 = this->inQueX_.template AllocTensor<T2>();
    this->inQueX_.template EnQue<QuePosition::VECIN, QuePosition::VECCALC>(dstUbInt64);
    LocalTensor<T2> xUbSize = this->inQueX_.template DeQue<QuePosition::VECIN, QuePosition::VECCALC, T2>();
    LocalTensor<int32_t> dstUbInt32 = xUbSize.template ReinterpretCast<int32_t>();
    uint64_t arNum = 0;

    this->VfComputeIds(loopCore, beforeNumOut, dstUbInt32, maskUbSize, arNum);
    loopGm = loopGm + arNum;
    if (arNum > 0) {
        if constexpr (IsSameType<T2, int32_t>::value) {
            ComputeOut(arNum, loopGm, xUbSize);
        } else if constexpr (IsSameType<T2, int64_t>::value) {
            uint32_t alignInt32 = ((arNum + DIM8 - 1) / DIM8) * DIM8;
            uint16_t repeatTimes = CeilDivision(arNum, VF_LEN_INT64);
            auto dstInt64Ptr = (__ubuf__ int32_t*)dstUbInt32.GetPhyAddr();
            auto dstInt32Ptr = (__ubuf__ int32_t*)dstUbInt32[this->tilingData_->offsetInt32Trans].GetPhyAddr();
            uint32_t sregInt32 = alignInt32 * DIM2;
            __VEC_SCOPE__
            {
                MaskReg preg;
                RegTensor<int32_t> int32Reg;
                CastInt32(repeatTimes, preg, sregInt32, int32Reg, dstInt32Ptr, dstInt64Ptr);
            }
            ComputeOut(arNum, loopGm, xUbSize);
        }
    } else {
        this->inQueX_.template FreeTensor(xUbSize);
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroSmallMask1<T1, T2, TILING_KEY>::ProcessPreCore(
    int32_t loopNum, int32_t tailNum, int32_t loopNumOut, int32_t loopTail)
{
    LocalTensor<uint32_t> addUbSize = this->addUb.template Get<uint32_t>();
    LocalTensor<uint32_t> maskUbSize = this->maskUb.template Get<uint32_t>();
    this->VfCleanUb(addUbSize);
    this->CopyInBefore(loopNum, this->tilingData_->ubFactorNum, tailNum, addUbSize, maskUbSize);

    uint64_t loopGm = 0;
    for (int32_t loopCore = 0; loopCore < loopNumOut; loopCore++) {
        ComputeIds(loopGm, loopCore, this->tilingData_->beforeNumO, maskUbSize);
    }
    if (loopTail != 0) {
        ComputeIds(loopGm, loopNumOut, loopTail, maskUbSize);
    }
}

template <typename T1, typename T2, int TILING_KEY>
__aicore__ inline void NonZeroSmallMask1<T1, T2, TILING_KEY>::Process()
{
    if (this->blockIdx_ == this->tilingData_->realCoreNum - 1) {
        ProcessPreCore(
            this->tilingData_->loopNumTailCore, this->tilingData_->loopTailTailCore, this->tilingData_->loopNumTo,
            this->tilingData_->loopTailTo);
    } else {
        ProcessPreCore(
            this->tilingData_->loopNumPerCore, this->tilingData_->loopTailPerCore, this->tilingData_->loopNumO,
            this->tilingData_->loopTailO);
    }
}
} // namespace NonZero

#endif // CANN_RESIZE_NEAREST_NEIGHBOR_V2_DATA_COPY_ND_SMALL_C_H
