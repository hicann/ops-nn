/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REPEAT_INTERLEAVE_GATHER_H
#define REPEAT_INTERLEAVE_GATHER_H

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"

namespace RepeatInterleave {
using namespace AscendC;

template <typename IdxT, typename GatherIdxT>
__simd_callee__ inline void CalcGatherSrcIdx(AscendC::Reg::RegTensor<GatherIdxT>& srcIdxReg,
                                             AscendC::Reg::RegTensor<IdxT>& idxReg, AscendC::Reg::MaskReg& maskIdx,
                                             GatherIdxT repeatsMulCpSize, GatherIdxT cpSize)
{
    AscendC::Reg::RegTensor<GatherIdxT> groupIdxReg;
    AscendC::Reg::RegTensor<GatherIdxT> cpSizeReg;
    AscendC::Reg::RegTensor<GatherIdxT> qReg;
    AscendC::Reg::RegTensor<GatherIdxT> qCpReg;
    AscendC::Reg::RegTensor<GatherIdxT> cpIdxCpReg;
    AscendC::Reg::RegTensor<GatherIdxT> offsetReg;
    AscendC::Reg::RegTensor<GatherIdxT> divReg;
    AscendC::Reg::Duplicate(divReg, repeatsMulCpSize, maskIdx);
    AscendC::Reg::Div(groupIdxReg, (AscendC::Reg::RegTensor<GatherIdxT>&)idxReg, divReg, maskIdx);
    AscendC::Reg::Duplicate(cpSizeReg, cpSize, maskIdx);
    AscendC::Reg::Div(qReg, (AscendC::Reg::RegTensor<GatherIdxT>&)idxReg, cpSizeReg, maskIdx);
    AscendC::Reg::Mul(qCpReg, qReg, cpSizeReg, maskIdx);
    AscendC::Reg::Sub(offsetReg, (AscendC::Reg::RegTensor<GatherIdxT>&)idxReg, qCpReg, maskIdx);
    AscendC::Reg::Mul(cpIdxCpReg, groupIdxReg, cpSizeReg, maskIdx);
    AscendC::Reg::Add(srcIdxReg, cpIdxCpReg, offsetReg, maskIdx);
}

template <typename T, typename IdxT, typename GatherIdxT>
__simd_callee__ inline void DataCopyGatherCpBatchUnalign(__ubuf__ T* xInLocalPtr, __ubuf__ T* xOutLocalPtr,
                                                         uint16_t repeatsScalarValue, uint32_t cpSize,
                                                         uint16_t fullBatches, uint32_t fullOutElems,
                                                         uint32_t tailOutElems, int32_t offsetStep)
{
    uint32_t sreg = tailOutElems;
    uint32_t sregFull = fullOutElems;
    AscendC::Reg::RegTensor<IdxT> idxReg;
    AscendC::Reg::RegTensor<GatherIdxT> srcIdxReg;
    AscendC::Reg::RegTensor<T> vDstReg;
    AscendC::Reg::UnalignRegForStore uOut;
    AscendC::Reg::MaskReg maskIdx = AscendC::Reg::CreateMask<GatherIdxT, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskFull = AscendC::Reg::UpdateMask<T>(sregFull);

    AscendC::Reg::Arange(idxReg, (IdxT)0);
    CalcGatherSrcIdx<IdxT, GatherIdxT>(srcIdxReg, idxReg, maskIdx, (GatherIdxT)(repeatsScalarValue * cpSize),
                                       (GatherIdxT)cpSize);

    for (uint16_t batch = 0; batch < fullBatches; batch++) {
        AscendC::Reg::Gather(vDstReg, (__ubuf__ T*)xInLocalPtr, srcIdxReg, maskFull);
        AscendC::Reg::StoreUnAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(xOutLocalPtr, vDstReg, uOut,
                                                                                   fullOutElems);
        AscendC::Reg::Adds(srcIdxReg, srcIdxReg, (GatherIdxT)offsetStep, maskIdx);
    }
    AscendC::Reg::MaskReg maskTailDyn = AscendC::Reg::UpdateMask<T>(sreg);
    AscendC::Reg::Gather(vDstReg, (__ubuf__ T*)xInLocalPtr, srcIdxReg, maskTailDyn);
    AscendC::Reg::StoreUnAlign<T, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(xOutLocalPtr, vDstReg, uOut,
                                                                               tailOutElems);
    AscendC::Reg::StoreUnAlignPost(xOutLocalPtr, uOut, 0);
}

template <typename T, typename WideT>
__simd_callee__ inline void DataCopyGatherCpBatchB8(__ubuf__ T* xInLocalPtr, __ubuf__ T* xOutLocalPtr,
                                                    uint16_t repeatsScalarValue, uint32_t cpSize, uint16_t fullBatches,
                                                    uint32_t fullOutElems, uint32_t tailOutElems, int32_t offsetStep)
{
    uint32_t sreg = tailOutElems;
    uint32_t sregFull = fullOutElems;
    AscendC::Reg::RegTensor<int16_t> idxReg;
    AscendC::Reg::RegTensor<uint16_t> srcIdxReg;
    AscendC::Reg::RegTensor<WideT> vWideReg;
    AscendC::Reg::RegTensor<uint8_t> vEvenReg;
    AscendC::Reg::RegTensor<uint8_t> vOddReg;
    AscendC::Reg::UnalignRegForStore uOut;
    AscendC::Reg::MaskReg maskIdx = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg maskFullU16 = AscendC::Reg::UpdateMask<uint16_t>(sregFull);

    AscendC::Reg::Arange(idxReg, (int16_t)0);
    CalcGatherSrcIdx<int16_t, uint16_t>(srcIdxReg, idxReg, maskIdx, (uint16_t)(repeatsScalarValue * cpSize),
                                        (uint16_t)cpSize);

    __ubuf__ uint8_t* outPtr = (__ubuf__ uint8_t*)xOutLocalPtr;
    for (uint16_t batch = 0; batch < fullBatches; batch++) {
        AscendC::Reg::Gather(vWideReg, (__ubuf__ T*)xInLocalPtr, srcIdxReg, maskFullU16);
        AscendC::Reg::DeInterleave<uint8_t>(vEvenReg, vOddReg, (AscendC::Reg::RegTensor<uint8_t>&)vWideReg,
                                            (AscendC::Reg::RegTensor<uint8_t>&)vWideReg);
        AscendC::Reg::StoreUnAlign<uint8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(outPtr, vEvenReg, uOut,
                                                                                         fullOutElems);
        AscendC::Reg::Adds(srcIdxReg, srcIdxReg, (uint16_t)offsetStep, maskIdx);
    }
    AscendC::Reg::MaskReg maskTailU16Dyn = AscendC::Reg::UpdateMask<uint16_t>(sreg);
    AscendC::Reg::Gather(vWideReg, (__ubuf__ T*)xInLocalPtr, srcIdxReg, maskTailU16Dyn);
    AscendC::Reg::DeInterleave<uint8_t>(vEvenReg, vOddReg, (AscendC::Reg::RegTensor<uint8_t>&)vWideReg,
                                        (AscendC::Reg::RegTensor<uint8_t>&)vWideReg);
    AscendC::Reg::StoreUnAlign<uint8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(outPtr, vEvenReg, uOut,
                                                                                     tailOutElems);
    AscendC::Reg::StoreUnAlignPost(outPtr, uOut, 0);
}

template <typename T>
__simd_vf__ inline void CopyCpToRepeatOutGatherVf(__ubuf__ T* xInLocalPtr, __ubuf__ T* xOutLocalPtr, uint32_t cpSize,
                                                  uint16_t repeatsScalarValue, uint16_t fullBatches, int32_t offsetStep,
                                                  uint32_t fullOutElems, uint32_t tailOutElems)
{
    if constexpr (sizeof(T) == 1) {
        using WideT = typename std::conditional<std::is_signed<T>::value, int16_t, uint16_t>::type;
        DataCopyGatherCpBatchB8<T, WideT>(xInLocalPtr, xOutLocalPtr, repeatsScalarValue, cpSize, fullBatches,
                                          fullOutElems, tailOutElems, offsetStep);
    } else {
        using IdxT = typename std::conditional<sizeof(T) == 2, int16_t, int32_t>::type;
        using GatherIdxT = typename std::conditional<sizeof(T) == 2, uint16_t, uint32_t>::type;
        DataCopyGatherCpBatchUnalign<T, IdxT, GatherIdxT>(xInLocalPtr, xOutLocalPtr, repeatsScalarValue, cpSize,
                                                          fullBatches, fullOutElems, tailOutElems, offsetStep);
    }
}

template <typename T>
__aicore__ inline void CopyCpToRepeatOutGatherAicore(__ubuf__ T* xInLocalPtr, __ubuf__ T* xOutLocalPtr, uint32_t cpSize,
                                                     uint16_t cpNum, uint16_t repeatsScalarValue)
{
    uint32_t dtypeSize = sizeof(T);
    uint32_t vRegBytes = static_cast<uint32_t>(Ops::Base::GetVRegSize());
    uint32_t elementsPerReg = (sizeof(T) == 1) ? (vRegBytes / 2) : (vRegBytes / dtypeSize);
    uint32_t cpPerBatch = elementsPerReg / (repeatsScalarValue * cpSize);
    if (cpPerBatch == 0) {
        cpPerBatch = 1;
    }
    uint16_t fullBatches = cpNum / cpPerBatch;
    int32_t offsetStep = static_cast<int32_t>(cpPerBatch * cpSize);
    uint32_t fullOutElems = cpPerBatch * repeatsScalarValue * cpSize;
    uint32_t totalOutElems = cpNum * repeatsScalarValue * cpSize;
    uint32_t tailOutElems = totalOutElems - fullOutElems * fullBatches;

    CopyCpToRepeatOutGatherVf<T>(xInLocalPtr, xOutLocalPtr, cpSize, repeatsScalarValue, fullBatches, offsetStep,
                                 fullOutElems, tailOutElems);
}
} // namespace RepeatInterleave
#endif // REPEAT_INTERLEAVE_GATHER_H
