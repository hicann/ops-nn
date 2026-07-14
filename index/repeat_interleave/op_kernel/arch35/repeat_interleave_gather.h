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
__simd_callee__ inline void CalcGatherSrcIdx(AscendC::MicroAPI::RegTensor<GatherIdxT>& srcIdxReg,
                                             AscendC::MicroAPI::RegTensor<IdxT>& idxReg,
                                             AscendC::MicroAPI::MaskReg& maskIdx, GatherIdxT repeatsMulCpSize,
                                             GatherIdxT cpSize)
{
    AscendC::MicroAPI::RegTensor<GatherIdxT> groupIdxReg;
    AscendC::MicroAPI::RegTensor<GatherIdxT> cpSizeReg;
    AscendC::MicroAPI::RegTensor<GatherIdxT> qReg;
    AscendC::MicroAPI::RegTensor<GatherIdxT> qCpReg;
    AscendC::MicroAPI::RegTensor<GatherIdxT> cpIdxCpReg;
    AscendC::MicroAPI::RegTensor<GatherIdxT> offsetReg;
    AscendC::MicroAPI::RegTensor<GatherIdxT> divReg;
    AscendC::MicroAPI::Duplicate(divReg, repeatsMulCpSize, maskIdx);
    AscendC::MicroAPI::Div(groupIdxReg, (AscendC::MicroAPI::RegTensor<GatherIdxT>&)idxReg, divReg, maskIdx);
    AscendC::MicroAPI::Duplicate(cpSizeReg, cpSize, maskIdx);
    AscendC::MicroAPI::Div(qReg, (AscendC::MicroAPI::RegTensor<GatherIdxT>&)idxReg, cpSizeReg, maskIdx);
    AscendC::MicroAPI::Mul(qCpReg, qReg, cpSizeReg, maskIdx);
    AscendC::MicroAPI::Sub(offsetReg, (AscendC::MicroAPI::RegTensor<GatherIdxT>&)idxReg, qCpReg, maskIdx);
    AscendC::MicroAPI::Mul(cpIdxCpReg, groupIdxReg, cpSizeReg, maskIdx);
    AscendC::MicroAPI::Add(srcIdxReg, cpIdxCpReg, offsetReg, maskIdx);
}

template <typename T, typename IdxT, typename GatherIdxT>
__simd_callee__ inline void DataCopyGatherCpBatchUnalign(__ubuf__ T* xInLocalPtr, __ubuf__ T* xOutLocalPtr,
                                                         uint16_t repeatsScalarValue, uint32_t cpSize,
                                                         uint16_t fullBatches, uint32_t fullOutElems,
                                                         uint32_t tailOutElems, int32_t offsetStep)
{
    uint32_t sreg = tailOutElems;
    uint32_t sregFull = fullOutElems;
    AscendC::MicroAPI::RegTensor<IdxT> idxReg;
    AscendC::MicroAPI::RegTensor<GatherIdxT> srcIdxReg;
    AscendC::MicroAPI::RegTensor<T> vDstReg;
    AscendC::MicroAPI::UnalignReg uOut;
    AscendC::MicroAPI::MaskReg
        maskIdx = AscendC::MicroAPI::CreateMask<GatherIdxT, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg maskFull = AscendC::MicroAPI::UpdateMask<T>(sregFull);

    AscendC::MicroAPI::Arange(idxReg, (IdxT)0);
    CalcGatherSrcIdx<IdxT, GatherIdxT>(srcIdxReg, idxReg, maskIdx, (GatherIdxT)(repeatsScalarValue * cpSize),
                                       (GatherIdxT)cpSize);

    for (uint16_t batch = 0; batch < fullBatches; batch++) {
        AscendC::MicroAPI::DataCopyGather(vDstReg, (__local_mem__ T*)xInLocalPtr, srcIdxReg, maskFull);
        AscendC::MicroAPI::DataCopyUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(xOutLocalPtr, vDstReg,
                                                                                                uOut, fullOutElems);
        AscendC::MicroAPI::Adds(srcIdxReg, srcIdxReg, (GatherIdxT)offsetStep, maskIdx);
    }
    AscendC::MicroAPI::MaskReg maskTailDyn = AscendC::MicroAPI::UpdateMask<T>(sreg);
    AscendC::MicroAPI::DataCopyGather(vDstReg, (__local_mem__ T*)xInLocalPtr, srcIdxReg, maskTailDyn);
    AscendC::MicroAPI::DataCopyUnAlign<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(xOutLocalPtr, vDstReg, uOut,
                                                                                            tailOutElems);
    AscendC::MicroAPI::DataCopyUnAlignPost(xOutLocalPtr, uOut, 0);
}

template <typename T, typename WideT>
__simd_callee__ inline void DataCopyGatherCpBatchB8(__ubuf__ T* xInLocalPtr, __ubuf__ T* xOutLocalPtr,
                                                    uint16_t repeatsScalarValue, uint32_t cpSize, uint16_t fullBatches,
                                                    uint32_t fullOutElems, uint32_t tailOutElems, int32_t offsetStep)
{
    uint32_t sreg = tailOutElems;
    uint32_t sregFull = fullOutElems;
    AscendC::MicroAPI::RegTensor<int16_t> idxReg;
    AscendC::MicroAPI::RegTensor<uint16_t> srcIdxReg;
    AscendC::MicroAPI::RegTensor<WideT> vWideReg;
    AscendC::MicroAPI::RegTensor<uint8_t> vEvenReg;
    AscendC::MicroAPI::RegTensor<uint8_t> vOddReg;
    AscendC::MicroAPI::UnalignReg uOut;
    AscendC::MicroAPI::MaskReg maskIdx = AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg maskFullU16 = AscendC::MicroAPI::UpdateMask<uint16_t>(sregFull);

    AscendC::MicroAPI::Arange(idxReg, (int16_t)0);
    CalcGatherSrcIdx<int16_t, uint16_t>(srcIdxReg, idxReg, maskIdx, (uint16_t)(repeatsScalarValue * cpSize),
                                        (uint16_t)cpSize);

    __ubuf__ uint8_t* outPtr = (__ubuf__ uint8_t*)xOutLocalPtr;
    for (uint16_t batch = 0; batch < fullBatches; batch++) {
        AscendC::MicroAPI::DataCopyGather(vWideReg, (__local_mem__ T*)xInLocalPtr, srcIdxReg, maskFullU16);
        AscendC::MicroAPI::DeInterleave<uint8_t>(vEvenReg, vOddReg, (AscendC::MicroAPI::RegTensor<uint8_t>&)vWideReg,
                                                 (AscendC::MicroAPI::RegTensor<uint8_t>&)vWideReg);
        AscendC::MicroAPI::DataCopyUnAlign<uint8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            outPtr, vEvenReg, uOut, fullOutElems);
        AscendC::MicroAPI::Adds(srcIdxReg, srcIdxReg, (uint16_t)offsetStep, maskIdx);
    }
    AscendC::MicroAPI::MaskReg maskTailU16Dyn = AscendC::MicroAPI::UpdateMask<uint16_t>(sreg);
    AscendC::MicroAPI::DataCopyGather(vWideReg, (__local_mem__ T*)xInLocalPtr, srcIdxReg, maskTailU16Dyn);
    AscendC::MicroAPI::DeInterleave<uint8_t>(vEvenReg, vOddReg, (AscendC::MicroAPI::RegTensor<uint8_t>&)vWideReg,
                                             (AscendC::MicroAPI::RegTensor<uint8_t>&)vWideReg);
    AscendC::MicroAPI::DataCopyUnAlign<uint8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(outPtr, vEvenReg,
                                                                                                  uOut, tailOutElems);
    AscendC::MicroAPI::DataCopyUnAlignPost(outPtr, uOut, 0);
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
