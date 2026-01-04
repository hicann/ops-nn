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
 * \file non_zero_big_mask_dim4.h
 * \brief
 */

#ifndef CANN_NON_ZERO_BIG_MASK_DIM4_H
#define CANN_NON_ZERO_BIG_MASK_DIM4_H

#include "non_zero_big_mask.h"

namespace NonZero {
using namespace AscendC;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

template <typename T1, typename T2>
class NonZeroBigMaskDim4 : public NonZeroBigMask<T1, T2> {
public:
    __aicore__ inline NonZeroBigMaskDim4(){};

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR outShape, GM_ADDR workspace, const NonZeroTilingData* tilingData);
    template <typename CLS_NAME, void (CLS_NAME::*ComputeOutputPtr)(LocalTensor<int32_t>& yUb)>
    __aicore__ inline void Process(CLS_NAME* objPtr);
    __aicore__ inline void ComputeOutput(LocalTensor<int32_t>& yUb);
};

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMaskDim4<T1, T2>::ComputeOutput(LocalTensor<int32_t>& yUb)
{
    // Scalar
    LocalTensor<uint32_t> multipDim = this->multipDimUb.template Get<uint32_t>();
    uint32_t dim0SValue = multipDim.GetValue(IDX_NUM_0);
    uint32_t dim0MValue = multipDim.GetValue(IDX_NUM_16);
    int16_t dim0KValue = (int16_t)multipDim.GetValue(IDX_NUM_8);

    uint32_t dim1SValue = multipDim.GetValue(IDX_NUM_1);
    uint32_t dim1MValue = multipDim.GetValue(IDX_NUM_17);
    int16_t dim1KValue = (int16_t)multipDim.GetValue(IDX_NUM_9);

    uint32_t dim2SValue = multipDim.GetValue(IDX_NUM_2);
    uint32_t dim2MValue = multipDim.GetValue(IDX_NUM_18);
    int16_t dim2KValue = (int16_t)multipDim.GetValue(IDX_NUM_10);

    // address
    __ubuf__ uint32_t* dstPtr0 = nullptr;
    __ubuf__ uint32_t* dstPtr1 = nullptr;
    __ubuf__ uint32_t* dstPtr2 = nullptr;
    __ubuf__ uint32_t* dstLastPtr = nullptr;

    if constexpr (IsSameType<T2, int64_t>::value) {
        dstPtr0 = (__ubuf__ uint32_t*)yUb[this->processSize_ * (this->shapeDim_ + 1)].GetPhyAddr();
    } else {
        dstPtr0 = (__ubuf__ uint32_t*)yUb[this->processSize_].GetPhyAddr();
    }

    dstPtr1 = dstPtr0 + this->processSize_;
    dstPtr2 = dstPtr1 + this->processSize_;
    dstLastPtr = dstPtr0 + this->processSize_ * (this->shapeDim_ - 1);
    auto srcPtr = (__ubuf__ uint32_t*)yUb.GetPhyAddr();

    // params
    uint16_t repeatTimes = CeilDivision(this->arNum_, this->repeatElmB32_);
    uint16_t num = (uint32_t)(this->arNum_);
    uint16_t offsetI = this->repeatElmB32_;
    int32_t vfLenInt32 = VF_LEN_INT32;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg preg;
        AscendC::MicroAPI::RegTensor<uint32_t> srcReg, subReg, shapeReg;
        AscendC::MicroAPI::RegTensor<uint32_t> mulReg, mReg;
        AscendC::MicroAPI::RegTensor<uint32_t> divReg0, divReg1;

        // 处理2维
        uint32_t sreg = num;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            preg = AscendC::MicroAPI::UpdateMask<int32_t>(sreg);
            AscendC::MicroAPI::AddrReg vagReg = AscendC::MicroAPI::CreateAddrReg<uint32_t>(1, offsetI);
            this->ComputeOutputBaseFunc(
                srcPtr, dstPtr0, dstLastPtr, preg, vagReg, srcReg, subReg, shapeReg, mulReg, mReg, divReg0, divReg1,
                dim0SValue, dim0MValue, dim0KValue);
        }
        mem_bar(VST_VLD);
        // 处理3维
        sreg = num;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            preg = AscendC::MicroAPI::UpdateMask<int32_t>(sreg);
            AscendC::MicroAPI::AddrReg vagReg = AscendC::MicroAPI::CreateAddrReg<uint32_t>(1, offsetI);
            this->ComputeOutputBaseFunc(
                dstLastPtr, dstPtr1, dstLastPtr, preg, vagReg, srcReg, subReg, shapeReg, mulReg, mReg, divReg0, divReg1,
                dim1SValue, dim1MValue, dim1KValue);
        }
        mem_bar(VST_VLD);
        // 处理4维
        sreg = num;
        for (uint16_t i = 0; i < repeatTimes; i++) {
            preg = AscendC::MicroAPI::UpdateMask<int32_t>(sreg);
            AscendC::MicroAPI::AddrReg vagReg = AscendC::MicroAPI::CreateAddrReg<uint32_t>(1, offsetI);
            this->ComputeOutputBaseFunc(
                dstLastPtr, dstPtr2, dstLastPtr, preg, vagReg, srcReg, subReg, shapeReg, mulReg, mReg, divReg0, divReg1,
                dim2SValue, dim2MValue, dim2KValue);
        }
        mem_bar(VST_VLD);
    }
    this->CastOutput(yUb, this->processSize_ * (this->shapeDim_ + 1), this->arNum_, this->shapeDim_);
    this->outQueY_.EnQue(yUb);
}

template <typename T1, typename T2>
__aicore__ inline void NonZeroBigMaskDim4<T1, T2>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR outShape, GM_ADDR workspace, const NonZeroTilingData* tilingData)
{
    this->InitBase(x, y, outShape, workspace, tilingData);
}

template <typename T1, typename T2>
template <typename CLS_NAME, void (CLS_NAME::*ComputeOutputPtr)(LocalTensor<int32_t>& yUb)>
__aicore__ inline void NonZeroBigMaskDim4<T1, T2>::Process(CLS_NAME* objPtr)
{
    this->template ProcessBase<CLS_NAME, ComputeOutputPtr>(objPtr);
}
} // namespace NonZero

#endif // CANN_NON_ZERO_BIG_MASK_DIM4_H