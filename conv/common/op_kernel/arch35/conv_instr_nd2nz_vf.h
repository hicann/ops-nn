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
 * \file conv_instr_nd2nz_vf.h
 * \brief ND2NZ vector function helpers shared by conv weight trans kernels.
 */

#ifndef CONV_INSTR_ND2NZ_VF_H
#define CONV_INSTR_ND2NZ_VF_H

#include "conv_util.h"

namespace conv {
using namespace AscendC;

template <typename SrcT, typename DstT, typename IndexT>
struct TransND2NZVfParams {
    uint16_t ciLoopTimes;
    uint16_t khkwLoopTimes;
    uint16_t coLoopTimes;
    uint32_t srcCiStride;
    uint32_t srcKhKwStride;
    uint32_t srcCoStride;
    uint32_t dstCiStride;
    uint32_t dstKhKwStride;
    uint32_t dstCoStride;
    __ubuf__ SrcT* srcAddr;
    __ubuf__ DstT* dstAddr;
    __ubuf__ IndexT* indexAddr;
};

struct TransND2NZKdVfParams {
    uint16_t kdLoopTimes;
    uint32_t srcKdStride;
    uint32_t dstKdStride;
};

template <typename T, typename IndexT>
struct TransFractalZVfParams {
    uint16_t kLoopTimes;
    uint16_t nLoopTimes;
    uint32_t srcKStride;
    uint32_t srcNStride;
    uint32_t dstKStride;
    uint32_t dstNStride;
    __ubuf__ T* srcAddr;
    __ubuf__ T* dstAddr;
    __ubuf__ IndexT* indexAddr;
};

template <typename IndexT>
__simd_vf__ inline void SetIndexVf(__ubuf__ IndexT* indexAddr, uint16_t repeatTimes, IndexT nStride, uint8_t k0)
{
    Reg::RegTensor<IndexT> indexReg;
    Reg::LoadAlign<IndexT>(indexReg, indexAddr);
    uint32_t maskL = k0;
    Reg::MaskReg maskReg = Reg::UpdateMask<IndexT>(maskL);

    uint8_t dstOffset = k0;
    uint8_t elesPerRepeat = k0;
    for (uint16_t repeat = 0; repeat < repeatTimes; ++repeat) {
        Reg::Adds<IndexT, IndexT>(indexReg, indexReg, nStride, maskReg);
        Reg::StoreAlign<IndexT>(indexAddr + dstOffset, indexReg, maskReg);
        dstOffset += elesPerRepeat;
    }
}

template <typename SrcT, typename DstT, typename RegT, typename IndexT, bool isQuantScene>
__simd_callee__ inline void GatherStoreNz(__ubuf__ SrcT* srcPtr, __ubuf__ DstT* dstPtr,
                                          Reg::RegTensor<IndexT>& indexReg, Reg::MaskReg& gatherMaskReg,
                                          Reg::MaskReg& vstsMaskReg)
{
    Reg::RegTensor<RegT> gatherReg;
    Reg::Gather<RegT, SrcT, IndexT>(gatherReg, srcPtr, indexReg, gatherMaskReg);
    if constexpr (isQuantScene) {
        // Remove the higher zeros of the int16_t data gathered by the Micro Gather instr
        Reg::Pack<uint8_t, RegT, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint8_t>&)gatherReg, gatherReg);
    }
    Reg::StoreAlign<DstT>(dstPtr, (Reg::RegTensor<DstT>&)gatherReg, vstsMaskReg);
}

template <typename SrcT, typename DstT, typename RegT, typename IndexT, bool isQuantScene>
__simd_vf__ inline void TransND2NZVf(const TransND2NZVfParams<SrcT, DstT, IndexT> params)
{
    Reg::RegTensor<IndexT> indexReg;
    Reg::MaskReg gatherMaskReg = Reg::CreateMask<RegT, Reg::MaskPattern::ALL>();
    Reg::MaskReg vstsMaskReg;
    if constexpr (isQuantScene) {
        vstsMaskReg = Reg::CreateMask<DstT, Reg::MaskPattern::H>();
    } else {
        vstsMaskReg = Reg::CreateMask<DstT, Reg::MaskPattern::ALL>();
    }
    Reg::LoadAlign<IndexT>(indexReg, params.indexAddr);

    for (uint16_t ci1OptIndex = 0; ci1OptIndex < params.ciLoopTimes; ++ci1OptIndex) {
        for (uint16_t khkwIndex = 0; khkwIndex < params.khkwLoopTimes; ++khkwIndex) {
            for (uint16_t coOptIndex = 0; coOptIndex < params.coLoopTimes; ++coOptIndex) {
                uint32_t srcOffset = ci1OptIndex * params.srcCiStride + khkwIndex * params.srcKhKwStride +
                                     coOptIndex * params.srcCoStride;
                uint32_t dstOffset = ci1OptIndex * params.dstCiStride + khkwIndex * params.dstKhKwStride +
                                     coOptIndex * params.dstCoStride;
                GatherStoreNz<SrcT, DstT, RegT, IndexT, isQuantScene>(
                    params.srcAddr + srcOffset, params.dstAddr + dstOffset, indexReg, gatherMaskReg, vstsMaskReg);
            }
        }
    }
}

template <typename SrcT, typename DstT, typename RegT, typename IndexT, bool isQuantScene>
__simd_vf__ inline void TransND2NZKdVf(const TransND2NZVfParams<SrcT, DstT, IndexT> params,
                                       const TransND2NZKdVfParams kdParams)
{
    Reg::RegTensor<IndexT> indexReg;
    Reg::MaskReg gatherMaskReg = Reg::CreateMask<RegT, Reg::MaskPattern::ALL>();
    Reg::MaskReg vstsMaskReg;
    if constexpr (isQuantScene) {
        vstsMaskReg = Reg::CreateMask<DstT, Reg::MaskPattern::H>();
    } else {
        vstsMaskReg = Reg::CreateMask<DstT, Reg::MaskPattern::ALL>();
    }
    Reg::LoadAlign<IndexT>(indexReg, params.indexAddr);

    for (uint16_t kdIndex = 0; kdIndex < kdParams.kdLoopTimes; ++kdIndex) {
        for (uint16_t ci1OptIndex = 0; ci1OptIndex < params.ciLoopTimes; ++ci1OptIndex) {
            for (uint16_t khkwIndex = 0; khkwIndex < params.khkwLoopTimes; ++khkwIndex) {
                for (uint16_t coOptIndex = 0; coOptIndex < params.coLoopTimes; ++coOptIndex) {
                    uint32_t srcOffset = kdIndex * kdParams.srcKdStride + ci1OptIndex * params.srcCiStride +
                                         khkwIndex * params.srcKhKwStride + coOptIndex * params.srcCoStride;
                    uint32_t dstOffset = kdIndex * kdParams.dstKdStride + ci1OptIndex * params.dstCiStride +
                                         khkwIndex * params.dstKhKwStride + coOptIndex * params.dstCoStride;
                    GatherStoreNz<SrcT, DstT, RegT, IndexT, isQuantScene>(
                        params.srcAddr + srcOffset, params.dstAddr + dstOffset, indexReg, gatherMaskReg, vstsMaskReg);
                }
            }
        }
    }
}

template <typename T, typename IndexT>
__simd_vf__ inline void TransFractalZVf(const TransFractalZVfParams<T, IndexT> params)
{
    Reg::RegTensor<IndexT> indexReg;
    Reg::MaskReg maskReg = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::LoadAlign<IndexT>(indexReg, params.indexAddr);

    for (uint16_t kIndex = 0; kIndex < params.kLoopTimes; ++kIndex) {
        for (uint16_t nIndex = 0; nIndex < params.nLoopTimes; ++nIndex) {
            uint32_t srcOffset = kIndex * params.srcKStride + nIndex * params.srcNStride;
            uint32_t dstOffset = kIndex * params.dstKStride + nIndex * params.dstNStride;
            GatherStoreNz<T, T, T, IndexT, false>(params.srcAddr + srcOffset, params.dstAddr + dstOffset, indexReg,
                                                  maskReg, maskReg);
        }
    }
}

} // namespace conv

#endif // CONV_INSTR_ND2NZ_VF_H
