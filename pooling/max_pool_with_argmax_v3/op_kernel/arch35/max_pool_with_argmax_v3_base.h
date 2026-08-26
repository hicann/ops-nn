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
 * \file max_pool_with_argmax_v3_base.h
 * \brief
 */

#ifndef MAX_POOL_WITH_ARGMAX_V3_BASE_H_
#define MAX_POOL_WITH_ARGMAX_V3_BASE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../pool_3d_common/arch35/pool_3d_common.h"

using namespace AscendC;
using Pool3D::FastDivImpl;

// 默认 rate1D = 1 生成 0 1 2 3 ...         rate1D = 0 生成  0 0 0 0 ...
template <typename T>
__aicore__ inline void GenGatterIndex2D(Reg::RegTensor<T>& indexReg, T rate2D, T num1D, T rate1D = 1)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(constReg, T(num1D));
    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segmentScalarReg, T(num1D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(indexReg, indexReg, T(rate1D), preg);
    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(rate2D), preg);

    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T>
__simd_callee__ inline void GenGatterIndex2DVF(Reg::RegTensor<T>& indexReg, T rate2D, T num1D, T rate1D = 1)
{
    Reg::Arange(indexReg, 0);
    Reg::RegTensor<T> segmentScalarReg;
    Reg::RegTensor<T> tmpReg;
    Reg::RegTensor<T> constReg;
    Reg::MaskReg preg = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::Duplicate(constReg, T(num1D));
    Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    Reg::Muls(tmpReg, segmentScalarReg, T(num1D), preg);
    Reg::Sub(indexReg, indexReg, tmpReg, preg);
    Reg::Muls(indexReg, indexReg, T(rate1D), preg);
    Reg::Muls(segmentScalarReg, segmentScalarReg, T(rate2D), preg);

    Reg::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void GenGatterIndex3D(Reg::RegTensor<T>& indexReg, T rate3D, T num2D, T rate2D, T num1D, T rate1D = 1)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentScalarReg2;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(constReg, T(num2D));
    AscendC::Reg::Div(segmentScalarReg2, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segmentScalarReg2, T(num2D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(segmentScalarReg2, segmentScalarReg2, T(rate3D), preg);

    AscendC::Reg::Duplicate(constReg, T(num1D));
    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segmentScalarReg, T(num1D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(indexReg, indexReg, T(rate1D), preg);
    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(rate2D), preg);

    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg, preg);
    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg2, preg);
}

template <typename T>
__simd_callee__ inline void GenGatterIndex3DVF(Reg::RegTensor<T>& indexReg, T rate3D, T num2D, T rate2D, T num1D,
                                               T rate1D = 1)
{
    Reg::Arange(indexReg, 0);
    Reg::RegTensor<T> segmentScalarReg;
    Reg::RegTensor<T> segmentScalarReg2;
    Reg::RegTensor<T> tmpReg;
    Reg::RegTensor<T> constReg;
    Reg::MaskReg preg = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    Reg::Duplicate(constReg, T(num2D));
    Reg::Div(segmentScalarReg2, indexReg, constReg, preg);
    Reg::Muls(tmpReg, segmentScalarReg2, T(num2D), preg);
    Reg::Sub(indexReg, indexReg, tmpReg, preg);
    Reg::Muls(segmentScalarReg2, segmentScalarReg2, T(rate3D), preg);

    Reg::Duplicate(constReg, T(num1D));
    Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    Reg::Muls(tmpReg, segmentScalarReg, T(num1D), preg);
    Reg::Sub(indexReg, indexReg, tmpReg, preg);
    Reg::Muls(indexReg, indexReg, T(rate1D), preg);
    Reg::Muls(segmentScalarReg, segmentScalarReg, T(rate2D), preg);

    Reg::Add(indexReg, indexReg, segmentScalarReg, preg);
    Reg::Add(indexReg, indexReg, segmentScalarReg2, preg);
}

template <typename T>
__aicore__ inline void GenGatterIndex4D(Reg::RegTensor<T>& indexReg, T rate4D, T num3D, T rate3D, T num2D, T rate2D,
                                        T num1D, T rate1D = 1)
{
    AscendC::Reg::Arange(indexReg, 0);
    AscendC::Reg::RegTensor<T> segmentScalarReg;
    AscendC::Reg::RegTensor<T> segmentScalarReg2;
    AscendC::Reg::RegTensor<T> segmentScalarReg3;
    AscendC::Reg::RegTensor<T> tmpReg;
    AscendC::Reg::RegTensor<T> constReg;
    AscendC::Reg::MaskReg preg = AscendC::Reg::CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(constReg, T(num3D));
    AscendC::Reg::Div(segmentScalarReg3, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segmentScalarReg3, T(num3D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(segmentScalarReg3, segmentScalarReg3, T(rate4D), preg);

    AscendC::Reg::Duplicate(constReg, T(num2D));
    AscendC::Reg::Div(segmentScalarReg2, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segmentScalarReg2, T(num2D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(segmentScalarReg2, segmentScalarReg2, T(rate3D), preg);

    AscendC::Reg::Duplicate(constReg, T(num1D));
    AscendC::Reg::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::Reg::Muls(tmpReg, segmentScalarReg, T(num1D), preg);
    AscendC::Reg::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::Reg::Muls(indexReg, indexReg, T(rate1D), preg);
    AscendC::Reg::Muls(segmentScalarReg, segmentScalarReg, T(rate2D), preg);

    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg, preg);
    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg2, preg);
    AscendC::Reg::Add(indexReg, indexReg, segmentScalarReg3, preg);
}

template <typename T>
__aicore__ inline void DuplicateNegInfReg(Reg::RegTensor<T>& negInfReg)
{
    // -inf
    constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000;
    constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;
    constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;
    using computeType = std::conditional_t<std::is_same<T, float>::value, uint32_t, uint16_t>;

    if constexpr (std::is_same<T, float>::value) {
        AscendC::Reg::Duplicate((AscendC::Reg::RegTensor<computeType>&)negInfReg, (FLOAT32_NEG_INF));
    } else if constexpr (std::is_same<T, half>::value) {
        AscendC::Reg::Duplicate((AscendC::Reg::RegTensor<computeType>&)negInfReg, (FLOAT16_NEG_INF));
    } else {
        AscendC::Reg::Duplicate((AscendC::Reg::RegTensor<computeType>&)negInfReg, (BFLOAT16_NEG_INF));
    }
}

template <typename T>
__simd_callee__ inline void DuplicateNegInfRegVF(Reg::RegTensor<T>& negInfReg)
{
    constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000;
    constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;
    constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;
    using computeType = std::conditional_t<std::is_same<T, float>::value, uint32_t, uint16_t>;

    if constexpr (std::is_same<T, float>::value) {
        Reg::Duplicate((Reg::RegTensor<computeType>&)negInfReg, (FLOAT32_NEG_INF));
    } else if constexpr (std::is_same<T, half>::value) {
        Reg::Duplicate((Reg::RegTensor<computeType>&)negInfReg, (FLOAT16_NEG_INF));
    } else {
        Reg::Duplicate((Reg::RegTensor<computeType>&)negInfReg, (BFLOAT16_NEG_INF));
    }
}

/**
 * \brief Fill buffer with negative infinity values (vectorized)
 * \param dstAddr Destination buffer address
 * \param repeatElm Number of elements per repeat
 * \param loop Number of full loops
 * \param tail Number of elements in tail
 */
template <typename T>
__aicore__ inline void DupBufferNegInfCommon(__ubuf__ T* dstAddr, uint32_t repeatElm, uint16_t loop, uint32_t tail)
{
    Reg::RegTensor<T> v0;
    DuplicateNegInfReg<T>(v0);
    Reg::MaskReg preg = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    for (uint16_t i = 0; i < loop; i++) {
        Reg::StoreAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(dstAddr, v0, repeatElm, preg);
    }
    preg = Reg::UpdateMask<T>(tail);
    Reg::StoreAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(dstAddr, v0, repeatElm, preg);
}

/**
 * \brief Copy data to calculation buffer with padding support (2D)
 */
template <typename T>
__aicore__ inline void CopyToCalcBuffer2DCommon(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint16_t batch, uint16_t rows,
                                                uint16_t loopCols, uint16_t tailCols, uint32_t repeatElm,
                                                uint32_t srcBatchStride, uint32_t srcRowStride, uint32_t dstBatchStride,
                                                uint32_t dstRowStride, uint32_t dstRowOffset, uint32_t dstColOffset)
{
    Reg::RegTensor<T> v0;
    Reg::UnalignRegForStore u0;
    for (uint16_t i = 0; i < batch; i++) {
        for (uint16_t j = 0; j < rows; j++) {
            __ubuf__ T* curSrcAddr = srcAddr + i * srcBatchStride + j * srcRowStride;
            __ubuf__ T* curDstAddr = dstAddr + i * dstBatchStride + (j + dstRowOffset) * dstRowStride + dstColOffset;
            for (uint16_t k = 0; k < loopCols; k++) {
                Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                Reg::StoreUnAlign(curDstAddr, v0, u0, repeatElm);
            }
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
            Reg::StoreUnAlign(curDstAddr, v0, u0, tailCols);
            Reg::StoreUnAlignPost(curDstAddr, u0, 0);
        }
    }
}

/**
 * \brief Copy data to calculation buffer with depth dimension support (3D)
 */
template <typename T>
__aicore__ inline void CopyToCalcBuffer3DCommon(__ubuf__ T* dstAddr, __ubuf__ T* srcAddr, uint16_t batch, uint16_t deps,
                                                uint16_t rows, uint16_t loopCols, uint16_t tailCols, uint32_t repeatElm,
                                                uint32_t srcBatchStride, uint32_t srcDepStride, uint32_t srcRowStride,
                                                uint32_t dstBatchStride, uint32_t dstDepStride, uint32_t dstRowStride,
                                                uint32_t dstDepOffset, uint32_t dstRowOffset, uint32_t dstColOffset)
{
    Reg::RegTensor<T> v0;
    Reg::UnalignRegForStore u0;
    for (uint16_t i = 0; i < batch; i++) {
        for (uint16_t t = 0; t < deps; t++) {
            for (uint16_t j = 0; j < rows; j++) {
                __ubuf__ T* curSrcAddr = srcAddr + i * srcBatchStride + t * srcDepStride + j * srcRowStride;
                __ubuf__ T* curDstAddr = dstAddr + i * dstBatchStride + (t + dstDepOffset) * dstDepStride +
                                         (j + dstRowOffset) * dstRowStride + dstColOffset;
                for (uint16_t k = 0; k < loopCols; k++) {
                    Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                    Reg::StoreUnAlign(curDstAddr, v0, u0, repeatElm);
                }
                Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                Reg::StoreUnAlign(curDstAddr, v0, u0, tailCols);
                Reg::StoreUnAlignPost(curDstAddr, u0, 0);
            }
        }
    }
}

/**
 * \brief Convert linear index to 2D (hIndex, wIndex) without padding alignment
 * \tparam T2 Index type (int32_t or int64_t)
 * \tparam IS_PAD Whether padding is enabled
 * \param srcReg Input linear index register
 * \param wStrideOffset Width stride offset
 * \param left Left padding offset
 * \param wInputActualNoPad Width input without padding
 * \param hIndexBase Height index base offset
 * \param dstReg Output converted index register
 * \param ncInputOffset NC batch input offset
 */
template <typename T2, const uint32_t IS_PAD>
__aicore__ inline void ConvertIndexWithoutPadAlignCommon(Reg::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset,
                                                         T2 left, T2 wInputActualNoPad, T2 hIndexBase,
                                                         Reg::RegTensor<T2>& dstReg, int32_t ncInputOffset)
{
    Reg::RegTensor<T2> hIndexReg;
    Reg::RegTensor<int32_t> constReg;
    Reg::RegTensor<int32_t> divResultReg;
    Reg::RegTensor<T2> divResultRegUnpack;
    Reg::RegTensor<T2> wIndexReg;
    Reg::RegTensor<int32_t> wIndexRegUnpack;
    Reg::RegTensor<T2> zeroReg;
    Reg::MaskReg negInfMask;
    Reg::MaskReg allMaskB32 = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg allMaskT2 = Reg::CreateMask<T2, Reg::MaskPattern::ALL>();
    Reg::Duplicate(constReg, static_cast<int32_t>(wStrideOffset));
    Reg::Duplicate(zeroReg, static_cast<T2>(0));
    Reg::Adds(srcReg, srcReg, -ncInputOffset, allMaskB32);
    Reg::Div(divResultReg, srcReg, constReg, allMaskB32);
    if constexpr (std::is_same<T2, int64_t>::value) {
        Reg::UnPack(divResultRegUnpack, divResultReg);
        Reg::Adds(hIndexReg, divResultRegUnpack, hIndexBase, allMaskT2);
    } else {
        Reg::Adds(hIndexReg, divResultReg, hIndexBase, allMaskB32);
    }
    if constexpr (IS_PAD) {
        Reg::Compare<T2, CMPMODE::LT>(negInfMask, hIndexReg, zeroReg, allMaskT2);
        Reg::Select(hIndexReg, zeroReg, hIndexReg, negInfMask);
    }
    Reg::Muls(hIndexReg, hIndexReg, wInputActualNoPad, allMaskT2);
    Reg::Mul(divResultReg, divResultReg, constReg, allMaskB32);
    Reg::Sub(wIndexRegUnpack, srcReg, divResultReg, allMaskB32);
    if constexpr (std::is_same<T2, int64_t>::value) {
        Reg::UnPack(wIndexReg, wIndexRegUnpack);
        Reg::Adds(wIndexReg, wIndexReg, left, allMaskT2);
    } else {
        Reg::Adds(wIndexReg, wIndexRegUnpack, left, allMaskB32);
    }
    if constexpr (IS_PAD) {
        Reg::Compare<T2, CMPMODE::LT>(negInfMask, wIndexReg, zeroReg, allMaskT2);
        Reg::Select(wIndexReg, zeroReg, wIndexReg, negInfMask);
    }
    Reg::Add(dstReg, hIndexReg, wIndexReg, allMaskT2);
    return;
}

/**
 * \brief Convert linear index to 2D (hIndex, wIndex) with NC batch support
 * \tparam T2 Index type (int32_t or int64_t)
 * \tparam IS_PAD Whether padding is enabled
 * \param srcReg Input linear index register
 * \param wStrideOffset Width stride offset
 * \param left Left padding offset
 * \param wInputActualNoPad Width input without padding
 * \param hIndexBase Height index base offset
 * \param dstReg Output converted index register
 * \param ncInputOffset NC batch input offset
 * \param ncOutputCount NC output count
 * \param inputNcSize Input NC size
 */
template <typename T2, const uint32_t IS_PAD>
__aicore__ inline void ConvertIndexWithoutPadAlignNcCommon(Reg::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset,
                                                           T2 left, T2 wInputActualNoPad, T2 hIndexBase,
                                                           Reg::RegTensor<T2>& dstReg, int32_t ncInputOffset,
                                                           int32_t ncOutputCount, int32_t inputNcSize)
{
    Reg::RegTensor<int32_t> ncIndexReg;
    Reg::RegTensor<int32_t> divResultReg;
    Reg::RegTensor<int32_t> constReg;
    Reg::MaskReg allMaskB32 = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
    Reg::Arange(ncIndexReg, static_cast<int32_t>(0));
    Reg::Duplicate(constReg, static_cast<int32_t>(ncOutputCount));
    Reg::Div(divResultReg, ncIndexReg, constReg, allMaskB32);
    Reg::Muls(divResultReg, divResultReg, inputNcSize, allMaskB32);
    Reg::Sub(srcReg, srcReg, divResultReg, allMaskB32);

    ConvertIndexWithoutPadAlignCommon<T2, IS_PAD>(srcReg, wStrideOffset, left, wInputActualNoPad, hIndexBase, dstReg,
                                                  ncInputOffset);
}

template <const uint32_t IS_PAD>
__aicore__ inline void ConvertIndexWithoutPadAlignCommonFastDiv(Reg::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset,
                                                                int32_t left, int32_t wInputActualNoPad,
                                                                int32_t hIndexBase, Reg::RegTensor<int32_t>& dstReg,
                                                                int32_t ncInputOffset, uint32_t magic, uint32_t shift)
{
    Reg::RegTensor<int32_t> hIndexReg;
    Reg::RegTensor<int32_t> wIndexReg;
    Reg::RegTensor<int32_t> zeroReg;
    Reg::RegTensor<uint32_t> divResultU32;
    Reg::RegTensor<uint32_t> magicReg;
    Reg::MaskReg negInfMask;
    Reg::MaskReg allMaskB32 = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();

    Reg::Duplicate(zeroReg, static_cast<int32_t>(0));
    Reg::Duplicate(magicReg, magic);
    Reg::Adds(srcReg, srcReg, -ncInputOffset, allMaskB32);

    FastDivImpl(divResultU32, (Reg::RegTensor<uint32_t>&)srcReg, magicReg, static_cast<int16_t>(shift), allMaskB32);

    Reg::Adds(hIndexReg, (Reg::RegTensor<int32_t>&)divResultU32, hIndexBase, allMaskB32);

    if constexpr (IS_PAD) {
        Reg::Compare<int32_t, CMPMODE::LT>(negInfMask, hIndexReg, zeroReg, allMaskB32);
        Reg::Select(hIndexReg, zeroReg, hIndexReg, negInfMask);
    }

    Reg::Muls(hIndexReg, hIndexReg, wInputActualNoPad, allMaskB32);

    Reg::Muls(divResultU32, divResultU32, wStrideOffset, allMaskB32);
    Reg::Sub((Reg::RegTensor<uint32_t>&)srcReg, (Reg::RegTensor<uint32_t>&)srcReg, divResultU32, allMaskB32);
    Reg::Adds(wIndexReg, srcReg, left, allMaskB32);

    if constexpr (IS_PAD) {
        Reg::Compare<int32_t, CMPMODE::LT>(negInfMask, wIndexReg, zeroReg, allMaskB32);
        Reg::Select(wIndexReg, zeroReg, wIndexReg, negInfMask);
    }

    Reg::Add(dstReg, hIndexReg, wIndexReg, allMaskB32);
}

template <const uint32_t IS_PAD>
__aicore__ inline void ConvertIndexWithoutPadAlignNcCommonFastDiv(
    Reg::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, int32_t left, int32_t wInputActualNoPad,
    int32_t hIndexBase, Reg::RegTensor<int32_t>& dstReg, int32_t ncInputOffset, int32_t ncOutputCount,
    int32_t inputNcSize, uint32_t magicNc, uint32_t shiftNc, uint32_t magicWStride, uint32_t shiftWStride)
{
    Reg::RegTensor<int32_t> ncIndexReg;
    Reg::RegTensor<uint32_t> divResultU32;
    Reg::RegTensor<uint32_t> magicReg;
    Reg::MaskReg allMaskB32 = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();

    Reg::Duplicate(magicReg, magicNc);
    Reg::Arange(ncIndexReg, static_cast<int32_t>(0));
    FastDivImpl(divResultU32, (Reg::RegTensor<uint32_t>&)ncIndexReg, magicReg, static_cast<int16_t>(shiftNc),
                allMaskB32);
    Reg::Muls(ncIndexReg, (Reg::RegTensor<int32_t>&)divResultU32, inputNcSize, allMaskB32);
    Reg::Sub(srcReg, srcReg, ncIndexReg, allMaskB32);

    ConvertIndexWithoutPadAlignCommonFastDiv<IS_PAD>(srcReg, wStrideOffset, left, wInputActualNoPad, hIndexBase, dstReg,
                                                     ncInputOffset, magicWStride, shiftWStride);
}

#endif // MAX_POOL_WITH_ARGMAX_V3_BASE_H_
