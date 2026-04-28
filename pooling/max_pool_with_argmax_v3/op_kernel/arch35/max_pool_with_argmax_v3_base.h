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

using namespace AscendC;

// 默认 rate1D = 1 生成 0 1 2 3 ...         rate1D = 0 生成  0 0 0 0 ...
template <typename T>
__aicore__ inline void GenGatterIndex2D(MicroAPI::RegTensor<T>& indexReg, T rate2D, T num1D, T rate1D = 1)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> tmpReg;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Duplicate(constReg, T(num1D));
    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segmentScalarReg, T(num1D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(indexReg, indexReg, T(rate1D), preg);
    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(rate2D), preg);

    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg, preg);
}

template <typename T>
__aicore__ inline void GenGatterIndex3D(
    MicroAPI::RegTensor<T>& indexReg, T rate3D, T num2D, T rate2D, T num1D, T rate1D = 1)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg2;
    AscendC::MicroAPI::RegTensor<T> tmpReg;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Duplicate(constReg, T(num2D));
    AscendC::MicroAPI::Div(segmentScalarReg2, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segmentScalarReg2, T(num2D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(segmentScalarReg2, segmentScalarReg2, T(rate3D), preg);

    AscendC::MicroAPI::Duplicate(constReg, T(num1D));
    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segmentScalarReg, T(num1D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(indexReg, indexReg, T(rate1D), preg);
    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(rate2D), preg);

    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg2, preg);
}

template <typename T>
__aicore__ inline void GenGatterIndex4D(
    MicroAPI::RegTensor<T>& indexReg, T rate4D, T num3D, T rate3D, T num2D, T rate2D, T num1D, T rate1D = 1)
{
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg;
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg2;
    AscendC::MicroAPI::RegTensor<T> segmentScalarReg3;
    AscendC::MicroAPI::RegTensor<T> tmpReg;
    AscendC::MicroAPI::RegTensor<T> constReg;
    AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::Duplicate(constReg, T(num3D));
    AscendC::MicroAPI::Div(segmentScalarReg3, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segmentScalarReg3, T(num3D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(segmentScalarReg3, segmentScalarReg3, T(rate4D), preg);

    AscendC::MicroAPI::Duplicate(constReg, T(num2D));
    AscendC::MicroAPI::Div(segmentScalarReg2, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segmentScalarReg2, T(num2D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(segmentScalarReg2, segmentScalarReg2, T(rate3D), preg);

    AscendC::MicroAPI::Duplicate(constReg, T(num1D));
    AscendC::MicroAPI::Div(segmentScalarReg, indexReg, constReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, segmentScalarReg, T(num1D), preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(indexReg, indexReg, T(rate1D), preg);
    AscendC::MicroAPI::Muls(segmentScalarReg, segmentScalarReg, T(rate2D), preg);

    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg2, preg);
    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg3, preg);
}

template <typename T>
__aicore__ inline void DuplicateNegInfReg(MicroAPI::RegTensor<T>& negInfReg)
{
    // -inf
    constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000;
    constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;
    constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;
    using computeType = std::conditional_t<std::is_same<T, float>::value, uint32_t, uint16_t>;

    if constexpr (std::is_same<T, float>::value) {
        AscendC::MicroAPI::Duplicate((AscendC::MicroAPI::RegTensor<computeType>&)negInfReg, (FLOAT32_NEG_INF));
    } else if constexpr (std::is_same<T, half>::value) {
        AscendC::MicroAPI::Duplicate((AscendC::MicroAPI::RegTensor<computeType>&)negInfReg, (FLOAT16_NEG_INF));
    } else {
        AscendC::MicroAPI::Duplicate((AscendC::MicroAPI::RegTensor<computeType>&)negInfReg, (BFLOAT16_NEG_INF));
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
__aicore__ inline void DupBufferNegInfCommon(__local_mem__ T* dstAddr, uint32_t repeatElm, uint16_t loop, uint32_t tail)
{
    MicroAPI::RegTensor<T> v0;
    DuplicateNegInfReg<T>(v0);
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    for (uint16_t i = 0; i < loop; i++) {
        MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, v0, repeatElm, preg);
    }
    preg = MicroAPI::UpdateMask<T>(tail);
    MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(dstAddr, v0, repeatElm, preg);
}

/**
 * \brief Copy data to calculation buffer with padding support (2D)
 */
template <typename T>
__aicore__ inline void CopyToCalcBuffer2DCommon(
    __local_mem__ T* dstAddr, __local_mem__ T* srcAddr, uint16_t batch, uint16_t rows, uint16_t loopCols,
    uint16_t tailCols, uint32_t repeatElm, uint32_t srcBatchStride, uint32_t srcRowStride, uint32_t dstBatchStride,
    uint32_t dstRowStride, uint32_t dstRowOffset, uint32_t dstColOffset)
{
    MicroAPI::RegTensor<T> v0;
    MicroAPI::UnalignReg u0;
    for (uint16_t i = 0; i < batch; i++) {
        for (uint16_t j = 0; j < rows; j++) {
            __local_mem__ T* curSrcAddr = srcAddr + i * srcBatchStride + j * srcRowStride;
            __local_mem__ T* curDstAddr =
                dstAddr + i * dstBatchStride + (j + dstRowOffset) * dstRowStride + dstColOffset;
            for (uint16_t k = 0; k < loopCols; k++) {
                MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                MicroAPI::DataCopyUnAlign(curDstAddr, v0, u0, repeatElm);
            }
            MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
            MicroAPI::DataCopyUnAlign(curDstAddr, v0, u0, tailCols);
            MicroAPI::DataCopyUnAlignPost(curDstAddr, u0, 0);
        }
    }
}

/**
 * \brief Copy data to calculation buffer with depth dimension support (3D)
 */
template <typename T>
__aicore__ inline void CopyToCalcBuffer3DCommon(
    __local_mem__ T* dstAddr, __local_mem__ T* srcAddr, uint16_t batch, uint16_t deps, uint16_t rows, uint16_t loopCols,
    uint16_t tailCols, uint32_t repeatElm, uint32_t srcBatchStride, uint32_t srcDepStride, uint32_t srcRowStride,
    uint32_t dstBatchStride, uint32_t dstDepStride, uint32_t dstRowStride, uint32_t dstDepOffset, uint32_t dstRowOffset,
    uint32_t dstColOffset)
{
    MicroAPI::RegTensor<T> v0;
    MicroAPI::UnalignReg u0;
    for (uint16_t i = 0; i < batch; i++) {
        for (uint16_t t = 0; t < deps; t++) {
            for (uint16_t j = 0; j < rows; j++) {
                __local_mem__ T* curSrcAddr = srcAddr + i * srcBatchStride + t * srcDepStride + j * srcRowStride;
                __local_mem__ T* curDstAddr = dstAddr + i * dstBatchStride + (t + dstDepOffset) * dstDepStride +
                                              (j + dstRowOffset) * dstRowStride + dstColOffset;
                for (uint16_t k = 0; k < loopCols; k++) {
                    MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                    MicroAPI::DataCopyUnAlign(curDstAddr, v0, u0, repeatElm);
                }
                MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v0, curSrcAddr, repeatElm);
                MicroAPI::DataCopyUnAlign(curDstAddr, v0, u0, tailCols);
                MicroAPI::DataCopyUnAlignPost(curDstAddr, u0, 0);
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
__aicore__ inline void ConvertIndexWithoutPadAlignCommon(
    MicroAPI::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, T2 left, T2 wInputActualNoPad, T2 hIndexBase,
    MicroAPI::RegTensor<T2>& dstReg, int32_t ncInputOffset)
{
    MicroAPI::RegTensor<T2> hIndexReg;
    MicroAPI::RegTensor<int32_t> constReg;
    MicroAPI::RegTensor<int32_t> divResultReg;
    MicroAPI::RegTensor<T2> divResultRegUnpack;
    MicroAPI::RegTensor<T2> wIndexReg;
    MicroAPI::RegTensor<int32_t> wIndexRegUnpack;
    MicroAPI::RegTensor<T2> zeroReg;
    MicroAPI::MaskReg negInfMask;
    MicroAPI::MaskReg allMaskB32 = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg allMaskT2 = MicroAPI::CreateMask<T2, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(constReg, static_cast<int32_t>(wStrideOffset));
    MicroAPI::Duplicate(zeroReg, static_cast<T2>(0));
    MicroAPI::Adds(srcReg, srcReg, -ncInputOffset, allMaskB32);
    MicroAPI::Div(divResultReg, srcReg, constReg, allMaskB32);
    if constexpr (std::is_same<T2, int64_t>::value) {
        MicroAPI::UnPack(divResultRegUnpack, divResultReg);
        MicroAPI::Adds(hIndexReg, divResultRegUnpack, hIndexBase, allMaskT2);
    } else {
        MicroAPI::Adds(hIndexReg, divResultReg, hIndexBase, allMaskB32);
    }
    if constexpr (IS_PAD) {
        MicroAPI::Compare<T2, CMPMODE::LT>(negInfMask, hIndexReg, zeroReg, allMaskT2);
        MicroAPI::Select(hIndexReg, zeroReg, hIndexReg, negInfMask);
    }
    MicroAPI::Muls(hIndexReg, hIndexReg, wInputActualNoPad, allMaskT2);
    MicroAPI::Mul(divResultReg, divResultReg, constReg, allMaskB32);
    MicroAPI::Sub(wIndexRegUnpack, srcReg, divResultReg, allMaskB32);
    if constexpr (std::is_same<T2, int64_t>::value) {
        MicroAPI::UnPack(wIndexReg, wIndexRegUnpack);
        MicroAPI::Adds(wIndexReg, wIndexReg, left, allMaskT2);
    } else {
        MicroAPI::Adds(wIndexReg, wIndexRegUnpack, left, allMaskB32);
    }
    if constexpr (IS_PAD) {
        MicroAPI::Compare<T2, CMPMODE::LT>(negInfMask, wIndexReg, zeroReg, allMaskT2);
        MicroAPI::Select(wIndexReg, zeroReg, wIndexReg, negInfMask);
    }
    MicroAPI::Add(dstReg, hIndexReg, wIndexReg, allMaskT2);
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
__aicore__ inline void ConvertIndexWithoutPadAlignNcCommon(
    MicroAPI::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, T2 left, T2 wInputActualNoPad, T2 hIndexBase,
    MicroAPI::RegTensor<T2>& dstReg, int32_t ncInputOffset, int32_t ncOutputCount, int32_t inputNcSize)
{
    MicroAPI::RegTensor<int32_t> ncIndexReg;
    MicroAPI::RegTensor<int32_t> divResultReg;
    MicroAPI::RegTensor<int32_t> constReg;
    MicroAPI::MaskReg allMaskB32 = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Arange(ncIndexReg, static_cast<int32_t>(0));
    MicroAPI::Duplicate(constReg, static_cast<int32_t>(ncOutputCount));
    MicroAPI::Div(divResultReg, ncIndexReg, constReg, allMaskB32);
    MicroAPI::Muls(divResultReg, divResultReg, inputNcSize, allMaskB32);
    MicroAPI::Sub(srcReg, srcReg, divResultReg, allMaskB32);

    ConvertIndexWithoutPadAlignCommon<T2, IS_PAD>(
        srcReg, wStrideOffset, left, wInputActualNoPad, hIndexBase, dstReg, ncInputOffset);
}

#endif // MAX_POOL_WITH_ARGMAX_V3_BASE_H_
