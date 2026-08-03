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
 * \file adaptive_avg_pool2d_split_h.h
 * \brief SplitH template: covers both W upsampling (wOut>wIn) and W downsampling
 *        (wOut<wIn), gated on H downsampling (hOut<hIn). Ho-outer loop: each Ho
 *        window's covered input rows are reduced in-register and written back to
 *        outBuf only once (or once per straddled hiBatch), cutting outBuf
 *        read-modify-write from O(hIn) down to O(hIn/hiFactor). Reuses
 *        CopyIn/Transpose/W-kernel machinery; only the accumulation loop structure
 *        differs. The W↓ path specializes rowCount==2/3 with kW==2/3 unrolled; the
 *        W↑ non-KW1 path uses a two-phase tempSumBuf_ scheme.
 *
 *        Shadows base class TransOut/CopyOut/CalAvgOneHo with compact wOut stride
 *        (no wOutAlign_ padding), enabling single-block CopyOut per nc.
 *
 * \arch Ascend950 / A5 / DAV_3510 only, RegBase (MicroAPI) main path, VL = 256 Byte.
 *       [RegBase-native] Host-side gate: AdaptivePool2dBaseTiling::GetShapeAttrsInfo
 *       rejects GetCurNpuArch() != NpuArch::DAV_3510.
 */

#ifndef ADAPTIVE_AVG_POOL2D_SPLIT_H_H_
#define ADAPTIVE_AVG_POOL2D_SPLIT_H_H_

#include "adaptive_avg_pool2d_pooling_base.h"

namespace AdaptivePool2dSplitHNamespace {
using namespace AscendC;
using namespace ops;
using namespace AdaptiveAvgPool2dOp;
using namespace AdaptiveAvgPool2dPoolingBaseNs;

template <typename ID_T>
__simd_vf__ inline void SplitHCalWiToWoInfoVf(__ubuf__ int32_t* wiWoStartAddr, __ubuf__ int32_t* wiWoCountAddr,
                                              uint16_t loopSize, uint32_t dataLen, uint16_t vfLen, int64_t wInDim,
                                              int64_t wOutDim)
{
    AapIndexRegType<ID_T> startIdxReg;
    AapIndexRegType<ID_T> endIdxReg;
    AapIndexRegType<ID_T> countReg;
    AapIndexRegType<ID_T> dupReg;
    // See CalWKernelInfo: Pack narrows b64->b32 and requires an unsigned destination.
    MicroAPI::RegTensor<uint32_t> startDstReg;
    MicroAPI::RegTensor<uint32_t> countDstReg;
    MicroAPI::MaskReg calMask;

    MicroAPI::Duplicate(dupReg, static_cast<ID_T>(wInDim));
    for (uint16_t i = 0; i < loopSize; i++) {
        if constexpr (IsSameType<ID_T, int64_t>::value) {
            calMask = MicroAPI::UpdateMask<ID_T, MicroAPI::RegTraitNumTwo>(dataLen);
        } else {
            calMask = MicroAPI::UpdateMask<ID_T>(dataLen);
        }
        ID_T startIdx = i * vfLen;
        MicroAPI::Arange(startIdxReg, startIdx);
        MicroAPI::Adds(endIdxReg, startIdxReg, static_cast<ID_T>(1), calMask);
        MicroAPI::Muls(startIdxReg, startIdxReg, static_cast<ID_T>(wOutDim), calMask);
        MicroAPI::Muls(endIdxReg, endIdxReg, static_cast<ID_T>(wOutDim), calMask);
        MicroAPI::Adds(endIdxReg, endIdxReg, static_cast<ID_T>(wInDim - 1), calMask);
        MicroAPI::Div(startIdxReg, startIdxReg, dupReg, calMask);
        MicroAPI::Div(endIdxReg, endIdxReg, dupReg, calMask);
        MicroAPI::Sub(countReg, endIdxReg, startIdxReg, calMask);

        if constexpr (IsSameType<ID_T, int64_t>::value) {
            // Narrow b64 indices to b32; see CalWKernelInfo. Under RegTraitNumTwo
            // reg[0] already holds the low words of all 64 elements.
            MicroAPI::Pack<uint32_t, ID_T, MicroAPI::HighLowPart::LOWEST>(startDstReg, startIdxReg);
            MicroAPI::Pack<uint32_t, ID_T, MicroAPI::HighLowPart::LOWEST>(countDstReg, countReg);
            MicroAPI::DataCopy((__ubuf__ uint32_t*)wiWoStartAddr + i * vfLen, startDstReg, calMask);
            MicroAPI::DataCopy((__ubuf__ uint32_t*)wiWoCountAddr + i * vfLen, countDstReg, calMask);
        } else {
            MicroAPI::DataCopy(wiWoStartAddr + i * vfLen, startIdxReg, calMask);
            MicroAPI::DataCopy(wiWoCountAddr + i * vfLen, countReg, calMask);
        }
    }
}

template <typename T, const uint32_t NC_FACTOR>
__simd_vf__ inline void SplitHAccumulateRowsToHoWUpsampleVf(__ubuf__ T* inputAddr, __ubuf__ float* outAddr,
                                                            __ubuf__ int32_t* wiWoStartAddr,
                                                            __ubuf__ int32_t* wiWoCountAddr, uint32_t outBase,
                                                            uint16_t wiNum, uint16_t rStart, uint16_t rEnd,
                                                            uint32_t vlNum, uint32_t wInAlign, uint32_t vfLenFp32)
{
    MicroAPI::RegTensor<float> inputReg;
    MicroAPI::RegTensor<float> sumReg;
    MicroAPI::RegTensor<float> outReg;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    for (uint16_t wi = 0; wi < wiNum; wi++) {
        MicroAPI::Duplicate(sumReg, 0.0f);
        for (uint16_t r = rStart; r < rEnd; r++) {
            uint32_t inOff = (static_cast<uint32_t>(r) * wInAlign + static_cast<uint32_t>(wi)) * vlNum;
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inOff);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
        }

        uint32_t woStart = static_cast<uint32_t>(wiWoStartAddr[wi]);
        uint16_t woCount = static_cast<uint16_t>(wiWoCountAddr[wi]);
        for (uint16_t j = 0; j < woCount; j++) {
            uint32_t wo = woStart + static_cast<uint32_t>(j);
            uint32_t sumOffset = outBase + wo * vlNum;
            MicroAPI::DataCopy(outReg, outAddr + sumOffset);
            MicroAPI::Add(outReg, outReg, sumReg, preg);
            MicroAPI::DataCopy(outAddr + sumOffset, outReg, preg);
        }

        if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
            MicroAPI::Duplicate(sumReg, 0.0f);
            for (uint16_t r = rStart; r < rEnd; r++) {
                uint32_t inOff = (static_cast<uint32_t>(r) * wInAlign + static_cast<uint32_t>(wi)) * vlNum + vfLenFp32;
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inOff);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            }
            for (uint16_t j = 0; j < woCount; j++) {
                uint32_t wo = woStart + static_cast<uint32_t>(j);
                uint32_t sumOffset = outBase + wo * vlNum + vfLenFp32;
                MicroAPI::DataCopy(outReg, outAddr + sumOffset);
                MicroAPI::Add(outReg, outReg, sumReg, preg);
                MicroAPI::DataCopy(outAddr + sumOffset, outReg, preg);
            }
        }
    }
}

__simd_vf__ inline void SplitHClearTempBufVf(__ubuf__ float* tempAddr, uint16_t loopSize, uint32_t remaining,
                                             uint32_t vfLenFp32)
{
    MicroAPI::RegTensor<float> zeroReg;
    MicroAPI::Duplicate(zeroReg, 0.0f);
    for (uint16_t i = 0; i < loopSize; i++) {
        MicroAPI::MaskReg mask = MicroAPI::UpdateMask<float>(remaining);
        MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<float>(i, vfLenFp32);
        MicroAPI::DataCopy(tempAddr, zeroReg, offset, mask);
    }
}

template <typename T, const uint32_t NC_FACTOR>
__simd_vf__ inline void SplitHAccumulateRowsToTempBufVf(__ubuf__ T* inputAddr, __ubuf__ float* tempAddr, uint16_t wiNum,
                                                        uint16_t rStart, uint16_t rEnd, uint32_t vlNum,
                                                        uint32_t wInAlign, uint32_t vfLenFp32)
{
    MicroAPI::RegTensor<float> inputReg;
    MicroAPI::RegTensor<float> sumReg;
    MicroAPI::RegTensor<float> tempReg;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    for (uint16_t wi = 0; wi < wiNum; wi++) {
        MicroAPI::Duplicate(sumReg, 0.0f);
        for (uint16_t r = rStart; r < rEnd; r++) {
            uint32_t inOff = (static_cast<uint32_t>(r) * wInAlign + static_cast<uint32_t>(wi)) * vlNum;
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inOff);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
        }

        uint32_t tempOff = static_cast<uint32_t>(wi) * vlNum;
        MicroAPI::DataCopy(tempReg, tempAddr + tempOff);
        MicroAPI::Add(tempReg, tempReg, sumReg, preg);
        MicroAPI::DataCopy(tempAddr + tempOff, tempReg, preg);

        if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
            MicroAPI::Duplicate(sumReg, 0.0f);
            for (uint16_t r = rStart; r < rEnd; r++) {
                uint32_t inOff = (static_cast<uint32_t>(r) * wInAlign + static_cast<uint32_t>(wi)) * vlNum + vfLenFp32;
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inOff);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            }
            uint32_t tempOff2 = tempOff + vfLenFp32;
            MicroAPI::DataCopy(tempReg, tempAddr + tempOff2);
            MicroAPI::Add(tempReg, tempReg, sumReg, preg);
            MicroAPI::DataCopy(tempAddr + tempOff2, tempReg, preg);
        }
    }
}

template <const uint32_t NC_FACTOR>
__simd_vf__ inline void SplitHScatterTempToOutBufVf(__ubuf__ float* tempAddr, __ubuf__ float* outAddr,
                                                    __ubuf__ int32_t* wiWoStartAddr, __ubuf__ int32_t* wiWoCountAddr,
                                                    uint32_t outBase, uint16_t wiNum, uint32_t vlNum,
                                                    uint32_t vfLenFp32)
{
    MicroAPI::RegTensor<float> tempReg;
    MicroAPI::RegTensor<float> outReg;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    for (uint16_t wi = 0; wi < wiNum; wi++) {
        uint32_t tempOff = static_cast<uint32_t>(wi) * vlNum;
        MicroAPI::DataCopy(tempReg, tempAddr + tempOff);

        uint32_t woStart = static_cast<uint32_t>(wiWoStartAddr[wi]);
        uint16_t woCount = static_cast<uint16_t>(wiWoCountAddr[wi]);
        for (uint16_t j = 0; j < woCount; j++) {
            uint32_t wo = woStart + static_cast<uint32_t>(j);
            uint32_t outOffset = outBase + wo * vlNum;
            MicroAPI::DataCopy(outReg, outAddr + outOffset);
            MicroAPI::Add(outReg, outReg, tempReg, preg);
            MicroAPI::DataCopy(outAddr + outOffset, outReg, preg);
        }

        if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
            uint32_t tempOff2 = tempOff + vfLenFp32;
            MicroAPI::DataCopy(tempReg, tempAddr + tempOff2);
            for (uint16_t j = 0; j < woCount; j++) {
                uint32_t wo = woStart + static_cast<uint32_t>(j);
                uint32_t outOffset = outBase + wo * vlNum + vfLenFp32;
                MicroAPI::DataCopy(outReg, outAddr + outOffset);
                MicroAPI::Add(outReg, outReg, tempReg, preg);
                MicroAPI::DataCopy(outAddr + outOffset, outReg, preg);
            }
        }
    }
}

template <typename T, const uint32_t NC_FACTOR, bool IS_FIRST>
__simd_vf__ inline void SplitHAccumulateRowsToHoRows2Kw2Vf(__ubuf__ T* inputAddr, __ubuf__ float* outAddr,
                                                           __ubuf__ int32_t* wStartAddr, uint32_t outBase,
                                                           uint16_t woStart, uint16_t woEnd, uint16_t r0, uint16_t r1,
                                                           uint32_t vlNum, uint32_t wInAlign, uint32_t vfLenFp32)
{
    if constexpr (IS_FIRST) {
        MicroAPI::RegTensor<float> inputReg;
        MicroAPI::RegTensor<float> sumReg;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        for (uint16_t wo = woStart; wo < woEnd; wo++) {
            uint32_t wStart = static_cast<uint32_t>(wStartAddr[wo]);
            uint32_t sumOffset = outBase + static_cast<uint32_t>(wo) * vlNum;
            uint32_t rowBase0 = (static_cast<uint32_t>(r0) * wInAlign + wStart) * vlNum;
            uint32_t rowBase1 = (static_cast<uint32_t>(r1) * wInAlign + wStart) * vlNum;
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, sumReg, preg, rowBase0);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase0 + vlNum);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase1);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase1 + vlNum);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            MicroAPI::DataCopy(outAddr + sumOffset, sumReg, preg);

            if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
                rowBase0 += vfLenFp32;
                rowBase1 += vfLenFp32;
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, sumReg, preg, rowBase0);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase0 + vlNum);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase1);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase1 + vlNum);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                MicroAPI::DataCopy(outAddr + sumOffset + vfLenFp32, sumReg, preg);
            }
        }
    } else {
        MicroAPI::RegTensor<float> inputReg;
        MicroAPI::RegTensor<float> sumReg;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        for (uint16_t wo = woStart; wo < woEnd; wo++) {
            uint32_t wStart = static_cast<uint32_t>(wStartAddr[wo]);
            uint32_t sumOffset = outBase + static_cast<uint32_t>(wo) * vlNum;
            uint32_t rowBase0 = (static_cast<uint32_t>(r0) * wInAlign + wStart) * vlNum;
            uint32_t rowBase1 = (static_cast<uint32_t>(r1) * wInAlign + wStart) * vlNum;
            MicroAPI::DataCopy(sumReg, outAddr + sumOffset);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase0);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase0 + vlNum);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase1);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase1 + vlNum);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            MicroAPI::DataCopy(outAddr + sumOffset, sumReg, preg);

            if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
                rowBase0 += vfLenFp32;
                rowBase1 += vfLenFp32;
                MicroAPI::DataCopy(sumReg, outAddr + sumOffset + vfLenFp32);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase0);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase0 + vlNum);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase1);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase1 + vlNum);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                MicroAPI::DataCopy(outAddr + sumOffset + vfLenFp32, sumReg, preg);
            }
        }
    }
}

template <typename T, const uint32_t NC_FACTOR, bool IS_FIRST>
__simd_vf__ inline void SplitHAccumulateRowsToHoRows3Kw2Vf(__ubuf__ T* inputAddr, __ubuf__ float* outAddr,
                                                           __ubuf__ int32_t* wStartAddr, uint32_t outBase,
                                                           uint16_t woStart, uint16_t woEnd, uint16_t r0, uint16_t r1,
                                                           uint16_t r2, uint32_t vlNum, uint32_t wInAlign,
                                                           uint32_t vfLenFp32)
{
    if constexpr (IS_FIRST) {
        MicroAPI::RegTensor<float> inputReg;
        MicroAPI::RegTensor<float> sumReg;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        for (uint16_t wo = woStart; wo < woEnd; wo++) {
            uint32_t wStart = static_cast<uint32_t>(wStartAddr[wo]);
            uint32_t sumOffset = outBase + static_cast<uint32_t>(wo) * vlNum;
            uint32_t rowBase0 = (static_cast<uint32_t>(r0) * wInAlign + wStart) * vlNum;
            uint32_t rowBase1 = (static_cast<uint32_t>(r1) * wInAlign + wStart) * vlNum;
            uint32_t rowBase2 = (static_cast<uint32_t>(r2) * wInAlign + wStart) * vlNum;
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, sumReg, preg, rowBase0);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase0 + vlNum);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase1);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase1 + vlNum);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase2);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase2 + vlNum);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            MicroAPI::DataCopy(outAddr + sumOffset, sumReg, preg);

            if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
                rowBase0 += vfLenFp32;
                rowBase1 += vfLenFp32;
                rowBase2 += vfLenFp32;
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, sumReg, preg, rowBase0);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase0 + vlNum);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase1);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase1 + vlNum);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase2);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase2 + vlNum);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                MicroAPI::DataCopy(outAddr + sumOffset + vfLenFp32, sumReg, preg);
            }
        }
    } else {
        MicroAPI::RegTensor<float> inputReg;
        MicroAPI::RegTensor<float> sumReg;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        for (uint16_t wo = woStart; wo < woEnd; wo++) {
            uint32_t wStart = static_cast<uint32_t>(wStartAddr[wo]);
            uint32_t sumOffset = outBase + static_cast<uint32_t>(wo) * vlNum;
            uint32_t rowBase0 = (static_cast<uint32_t>(r0) * wInAlign + wStart) * vlNum;
            uint32_t rowBase1 = (static_cast<uint32_t>(r1) * wInAlign + wStart) * vlNum;
            uint32_t rowBase2 = (static_cast<uint32_t>(r2) * wInAlign + wStart) * vlNum;
            MicroAPI::DataCopy(sumReg, outAddr + sumOffset);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase0);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase0 + vlNum);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase1);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase1 + vlNum);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase2);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase2 + vlNum);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            MicroAPI::DataCopy(outAddr + sumOffset, sumReg, preg);

            if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
                rowBase0 += vfLenFp32;
                rowBase1 += vfLenFp32;
                rowBase2 += vfLenFp32;
                MicroAPI::DataCopy(sumReg, outAddr + sumOffset + vfLenFp32);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase0);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase0 + vlNum);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase1);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase1 + vlNum);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase2);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, rowBase2 + vlNum);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                MicroAPI::DataCopy(outAddr + sumOffset + vfLenFp32, sumReg, preg);
            }
        }
    }
}

template <typename T, const uint32_t NC_FACTOR>
__simd_vf__ inline void SplitHAccumulateExtraColRows2Vf(__ubuf__ T* inputAddr, __ubuf__ float* outAddr,
                                                        __ubuf__ int32_t* wStartAddr, __ubuf__ int32_t* wKerSizeAddr,
                                                        __ubuf__ int32_t* extraWoIdxAddr, uint32_t outBase,
                                                        uint16_t extraCount, uint16_t r0, uint16_t r1, uint32_t vlNum,
                                                        uint32_t wInAlign, uint32_t vfLenFp32)
{
    MicroAPI::RegTensor<float> inputReg;
    MicroAPI::RegTensor<float> sumReg;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    for (uint16_t i = 0; i < extraCount; i++) {
        uint16_t wo = static_cast<uint16_t>(extraWoIdxAddr[i]);
        uint32_t wStart = static_cast<uint32_t>(wStartAddr[wo]);
        uint16_t kernelW = static_cast<uint16_t>(wKerSizeAddr[wo]);
        uint32_t sumOffset = outBase + static_cast<uint32_t>(wo) * vlNum;

        MicroAPI::DataCopy(sumReg, outAddr + sumOffset);
        for (uint16_t k = 2; k < kernelW; k++) {
            uint32_t colOff0 = (static_cast<uint32_t>(r0) * wInAlign + wStart + static_cast<uint32_t>(k)) * vlNum;
            uint32_t colOff1 = (static_cast<uint32_t>(r1) * wInAlign + wStart + static_cast<uint32_t>(k)) * vlNum;
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, colOff0);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, colOff1);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
        }
        MicroAPI::DataCopy(outAddr + sumOffset, sumReg, preg);

        if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
            MicroAPI::DataCopy(sumReg, outAddr + sumOffset + vfLenFp32);
            for (uint16_t k = 2; k < kernelW; k++) {
                uint32_t colOff0 = (static_cast<uint32_t>(r0) * wInAlign + wStart + static_cast<uint32_t>(k)) * vlNum +
                                   vfLenFp32;
                uint32_t colOff1 = (static_cast<uint32_t>(r1) * wInAlign + wStart + static_cast<uint32_t>(k)) * vlNum +
                                   vfLenFp32;
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, colOff0);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, colOff1);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            }
            MicroAPI::DataCopy(outAddr + sumOffset + vfLenFp32, sumReg, preg);
        }
    }
}

template <typename T, const uint32_t NC_FACTOR>
__simd_vf__ inline void SplitHAccumulateExtraColRows3Vf(__ubuf__ T* inputAddr, __ubuf__ float* outAddr,
                                                        __ubuf__ int32_t* wStartAddr, __ubuf__ int32_t* wKerSizeAddr,
                                                        __ubuf__ int32_t* extraWoIdxAddr, uint32_t outBase,
                                                        uint16_t extraCount, uint16_t r0, uint16_t r1, uint16_t r2,
                                                        uint32_t vlNum, uint32_t wInAlign, uint32_t vfLenFp32)
{
    MicroAPI::RegTensor<float> inputReg;
    MicroAPI::RegTensor<float> sumReg;
    MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    for (uint16_t i = 0; i < extraCount; i++) {
        uint16_t wo = static_cast<uint16_t>(extraWoIdxAddr[i]);
        uint32_t wStart = static_cast<uint32_t>(wStartAddr[wo]);
        uint16_t kernelW = static_cast<uint16_t>(wKerSizeAddr[wo]);
        uint32_t sumOffset = outBase + static_cast<uint32_t>(wo) * vlNum;

        MicroAPI::DataCopy(sumReg, outAddr + sumOffset);
        for (uint16_t k = 2; k < kernelW; k++) {
            uint32_t colOff0 = (static_cast<uint32_t>(r0) * wInAlign + wStart + static_cast<uint32_t>(k)) * vlNum;
            uint32_t colOff1 = (static_cast<uint32_t>(r1) * wInAlign + wStart + static_cast<uint32_t>(k)) * vlNum;
            uint32_t colOff2 = (static_cast<uint32_t>(r2) * wInAlign + wStart + static_cast<uint32_t>(k)) * vlNum;
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, colOff0);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, colOff1);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, colOff2);
            MicroAPI::Add(sumReg, sumReg, inputReg, preg);
        }
        MicroAPI::DataCopy(outAddr + sumOffset, sumReg, preg);

        if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
            MicroAPI::DataCopy(sumReg, outAddr + sumOffset + vfLenFp32);
            for (uint16_t k = 2; k < kernelW; k++) {
                uint32_t colOff0 = (static_cast<uint32_t>(r0) * wInAlign + wStart + static_cast<uint32_t>(k)) * vlNum +
                                   vfLenFp32;
                uint32_t colOff1 = (static_cast<uint32_t>(r1) * wInAlign + wStart + static_cast<uint32_t>(k)) * vlNum +
                                   vfLenFp32;
                uint32_t colOff2 = (static_cast<uint32_t>(r2) * wInAlign + wStart + static_cast<uint32_t>(k)) * vlNum +
                                   vfLenFp32;
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, colOff0);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, colOff1);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, colOff2);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            }
            MicroAPI::DataCopy(outAddr + sumOffset + vfLenFp32, sumReg, preg);
        }
    }
}

template <typename T, const uint32_t NC_FACTOR, bool IS_FIRST>
__simd_vf__ inline void SplitHAccumulateRowsToHoVf(__ubuf__ T* inputAddr, __ubuf__ float* outAddr,
                                                   __ubuf__ int32_t* wStartAddr, __ubuf__ int32_t* wKerSizeAddr,
                                                   uint32_t outBase, uint16_t woNum, uint16_t rStart, uint16_t rEnd,
                                                   uint32_t vlNum, uint32_t wInAlign, uint32_t vfLenFp32)
{
    if constexpr (IS_FIRST) {
        MicroAPI::RegTensor<float> inputReg;
        MicroAPI::RegTensor<float> sumReg;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        for (uint16_t wo = 0; wo < woNum; wo++) {
            uint32_t wStart = static_cast<uint32_t>(wStartAddr[wo]);
            uint16_t kernelW = static_cast<uint16_t>(wKerSizeAddr[wo]);
            uint32_t sumOffset = outBase + static_cast<uint32_t>(wo) * vlNum;

            uint32_t firstRowBase = (static_cast<uint32_t>(rStart) * wInAlign + wStart) * vlNum;
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, sumReg, preg, firstRowBase);
            for (uint16_t k = 1; k < kernelW; k++) {
                uint32_t inOff = firstRowBase + static_cast<uint32_t>(k) * vlNum;
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inOff);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            }
            for (uint16_t r = rStart + 1; r < rEnd; r++) {
                uint32_t rowBase = (static_cast<uint32_t>(r) * wInAlign + wStart) * vlNum;
                for (uint16_t k = 0; k < kernelW; k++) {
                    uint32_t inOff = rowBase + static_cast<uint32_t>(k) * vlNum;
                    ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inOff);
                    MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                }
            }
            MicroAPI::DataCopy(outAddr + sumOffset, sumReg, preg);

            if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
                firstRowBase = (static_cast<uint32_t>(rStart) * wInAlign + wStart) * vlNum + vfLenFp32;
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, sumReg, preg, firstRowBase);
                for (uint16_t k = 1; k < kernelW; k++) {
                    uint32_t inOff = firstRowBase + static_cast<uint32_t>(k) * vlNum;
                    ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inOff);
                    MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                }
                for (uint16_t r = rStart + 1; r < rEnd; r++) {
                    uint32_t rowBase = (static_cast<uint32_t>(r) * wInAlign + wStart) * vlNum + vfLenFp32;
                    for (uint16_t k = 0; k < kernelW; k++) {
                        uint32_t inOff = rowBase + static_cast<uint32_t>(k) * vlNum;
                        ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inOff);
                        MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                    }
                }
                MicroAPI::DataCopy(outAddr + sumOffset + vfLenFp32, sumReg, preg);
            }
        }
    } else {
        MicroAPI::RegTensor<float> inputReg;
        MicroAPI::RegTensor<float> sumReg;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        for (uint16_t wo = 0; wo < woNum; wo++) {
            uint32_t wStart = static_cast<uint32_t>(wStartAddr[wo]);
            uint16_t kernelW = static_cast<uint16_t>(wKerSizeAddr[wo]);
            uint32_t sumOffset = outBase + static_cast<uint32_t>(wo) * vlNum;

            MicroAPI::DataCopy(sumReg, outAddr + sumOffset);
            for (uint16_t r = rStart; r < rEnd; r++) {
                uint32_t rowBase = (static_cast<uint32_t>(r) * wInAlign + wStart) * vlNum;
                for (uint16_t k = 0; k < kernelW; k++) {
                    uint32_t inOff = rowBase + static_cast<uint32_t>(k) * vlNum;
                    ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inOff);
                    MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                }
            }
            MicroAPI::DataCopy(outAddr + sumOffset, sumReg, preg);

            if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
                MicroAPI::DataCopy(sumReg, outAddr + sumOffset + vfLenFp32);
                for (uint16_t r = rStart; r < rEnd; r++) {
                    uint32_t rowBase = (static_cast<uint32_t>(r) * wInAlign + wStart) * vlNum + vfLenFp32;
                    for (uint16_t k = 0; k < kernelW; k++) {
                        uint32_t inOff = rowBase + static_cast<uint32_t>(k) * vlNum;
                        ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inOff);
                        MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                    }
                }
                MicroAPI::DataCopy(outAddr + sumOffset + vfLenFp32, sumReg, preg);
            }
        }
    }
}

template <typename T, const uint32_t NC_FACTOR, bool IS_FIRST>
__simd_vf__ inline void SplitHAccumulateRowsToHoWUpsampleKW1Vf(__ubuf__ T* inputAddr, __ubuf__ float* outAddr,
                                                               __ubuf__ int32_t* wiWoStartAddr,
                                                               __ubuf__ int32_t* wiWoCountAddr, uint32_t outBase,
                                                               uint16_t wiNum, uint16_t rStart, uint16_t rEnd,
                                                               uint32_t vlNum, uint32_t wInAlign, uint32_t vfLenFp32)
{
    if constexpr (IS_FIRST) {
        MicroAPI::RegTensor<float> inputReg;
        MicroAPI::RegTensor<float> sumReg;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        for (uint16_t wi = 0; wi < wiNum; wi++) {
            uint32_t inOff0 = (static_cast<uint32_t>(rStart) * wInAlign + static_cast<uint32_t>(wi)) * vlNum;
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, sumReg, preg, inOff0);
            for (uint16_t r = rStart + 1; r < rEnd; r++) {
                uint32_t inOff = (static_cast<uint32_t>(r) * wInAlign + static_cast<uint32_t>(wi)) * vlNum;
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inOff);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            }

            uint32_t woStart = static_cast<uint32_t>(wiWoStartAddr[wi]);
            uint16_t woCount = static_cast<uint16_t>(wiWoCountAddr[wi]);
            for (uint16_t j = 0; j < woCount; j++) {
                uint32_t wo = woStart + static_cast<uint32_t>(j);
                uint32_t sumOffset = outBase + wo * vlNum;
                MicroAPI::DataCopy(outAddr + sumOffset, sumReg, preg);
            }

            if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
                inOff0 = (static_cast<uint32_t>(rStart) * wInAlign + static_cast<uint32_t>(wi)) * vlNum + vfLenFp32;
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, sumReg, preg, inOff0);
                for (uint16_t r = rStart + 1; r < rEnd; r++) {
                    uint32_t inOff = (static_cast<uint32_t>(r) * wInAlign + static_cast<uint32_t>(wi)) * vlNum +
                                     vfLenFp32;
                    ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inOff);
                    MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                }
                for (uint16_t j = 0; j < woCount; j++) {
                    uint32_t wo = woStart + static_cast<uint32_t>(j);
                    uint32_t sumOffset = outBase + wo * vlNum + vfLenFp32;
                    MicroAPI::DataCopy(outAddr + sumOffset, sumReg, preg);
                }
            }
        }
    } else {
        MicroAPI::RegTensor<float> inputReg;
        MicroAPI::RegTensor<float> sumReg;
        MicroAPI::RegTensor<float> outReg;
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        for (uint16_t wi = 0; wi < wiNum; wi++) {
            uint32_t inOff0 = (static_cast<uint32_t>(rStart) * wInAlign + static_cast<uint32_t>(wi)) * vlNum;
            ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, sumReg, preg, inOff0);
            for (uint16_t r = rStart + 1; r < rEnd; r++) {
                uint32_t inOff = (static_cast<uint32_t>(r) * wInAlign + static_cast<uint32_t>(wi)) * vlNum;
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inOff);
                MicroAPI::Add(sumReg, sumReg, inputReg, preg);
            }

            uint32_t woStart = static_cast<uint32_t>(wiWoStartAddr[wi]);
            uint16_t woCount = static_cast<uint16_t>(wiWoCountAddr[wi]);
            for (uint16_t j = 0; j < woCount; j++) {
                uint32_t wo = woStart + static_cast<uint32_t>(j);
                uint32_t sumOffset = outBase + wo * vlNum;
                MicroAPI::DataCopy(outReg, outAddr + sumOffset);
                MicroAPI::Add(outReg, outReg, sumReg, preg);
                MicroAPI::DataCopy(outAddr + sumOffset, outReg, preg);
            }

            if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
                inOff0 = (static_cast<uint32_t>(rStart) * wInAlign + static_cast<uint32_t>(wi)) * vlNum + vfLenFp32;
                ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, sumReg, preg, inOff0);
                for (uint16_t r = rStart + 1; r < rEnd; r++) {
                    uint32_t inOff = (static_cast<uint32_t>(r) * wInAlign + static_cast<uint32_t>(wi)) * vlNum +
                                     vfLenFp32;
                    ops_vf::LoadOneTensorForDtypeT<T>(inputAddr, inputReg, preg, inOff);
                    MicroAPI::Add(sumReg, sumReg, inputReg, preg);
                }
                for (uint16_t j = 0; j < woCount; j++) {
                    uint32_t wo = woStart + static_cast<uint32_t>(j);
                    uint32_t sumOffset = outBase + wo * vlNum + vfLenFp32;
                    MicroAPI::DataCopy(outReg, outAddr + sumOffset);
                    MicroAPI::Add(outReg, outReg, sumReg, preg);
                    MicroAPI::DataCopy(outAddr + sumOffset, outReg, preg);
                }
            }
        }
    }
}

template <const uint32_t NC_FACTOR>
__simd_vf__ inline void SplitHCalAvgOneHoKW1Vf(__ubuf__ float* outAddr, uint32_t outBaseOffset, uint16_t woNum,
                                               int32_t kh, uint32_t vlNum, uint32_t vfLenFp32)
{
    MicroAPI::RegTensor<float> sumReg;
    MicroAPI::RegTensor<float> avgReg;
    MicroAPI::RegTensor<int32_t> divisorReg;
    MicroAPI::RegTensor<float> divisorCastReg;
    MicroAPI::MaskReg calMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    MicroAPI::Duplicate(divisorReg, kh);
    MicroAPI::Cast<float, int32_t, aapCastTraitI32F32>(divisorCastReg, divisorReg, calMask);

    for (uint16_t wo = 0; wo < woNum; wo++) {
        uint32_t offset = outBaseOffset + static_cast<uint32_t>(wo) * vlNum;
        MicroAPI::DataCopy(sumReg, outAddr + offset);
        MicroAPI::Div(avgReg, sumReg, divisorCastReg, calMask);
        MicroAPI::DataCopy(outAddr + offset, avgReg, calMask);

        if constexpr (NC_FACTOR == TPL_NC_FACTOR_128) {
            MicroAPI::DataCopy(sumReg, outAddr + offset + vfLenFp32);
            MicroAPI::Div(avgReg, sumReg, divisorCastReg, calMask);
            MicroAPI::DataCopy(outAddr + offset + vfLenFp32, avgReg, calMask);
        }
    }
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
class AdaptiveAvgPool2dSplitH
    : protected AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, AdaptivePool2dSplitHTilingData> {
    using Base = AdaptiveAvgPool2dPoolingBase<T, ID_T, NC_FACTOR, AdaptivePool2dSplitHTilingData>;

public:
    __aicore__ inline AdaptiveAvgPool2dSplitH(const AdaptivePool2dSplitHTilingData* tilingData, TPipe* pipe)
        : Base(tilingData, pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    // IS_FIRST selects "store" vs "load+add+store" on the outBuf scatter. It is loop-invariant
    // across the whole VF, so it is a template parameter rather than a runtime flag: the branch
    // is resolved at compile time and each instantiation stays a single straight-line vector loop.
    template <bool IS_FIRST>
    __aicore__ inline void AccumulateRowsToHo(int64_t hoLocal, int64_t rowLoStart, int64_t rowLoEnd);
    template <bool IS_FIRST>
    __aicore__ inline void AccumulateRowsToHoRows2(int64_t hoLocal, int64_t rowLoStart);
    template <bool IS_FIRST>
    __aicore__ inline void AccumulateRowsToHoRows3(int64_t hoLocal, int64_t rowLoStart);
    template <bool IS_FIRST>
    __aicore__ inline void AccumulateRowsToHoRows2Kw2(int64_t hoLocal, uint16_t woStart, uint16_t woEnd,
                                                      int64_t rowLoStart);
    template <bool IS_FIRST>
    __aicore__ inline void AccumulateRowsToHoRows3Kw2(int64_t hoLocal, uint16_t woStart, uint16_t woEnd,
                                                      int64_t rowLoStart);
    __aicore__ inline void AccumulateExtraColRows2(int64_t hoLocal, uint16_t woStart, uint16_t woEnd,
                                                   int64_t rowLoStart);
    __aicore__ inline void AccumulateExtraColRows3(int64_t hoLocal, uint16_t woStart, uint16_t woEnd,
                                                   int64_t rowLoStart);
    __aicore__ inline void AccumulateRowsToHoWUpsample(int64_t hoLocal, int64_t rowLoStart, int64_t rowLoEnd);
    __aicore__ inline void ClearTempBuf();
    __aicore__ inline void AccumulateRowsToTempBuf(int64_t rowLoStart, int64_t rowLoEnd);
    __aicore__ inline void ScatterTempToOutBuf(int64_t hoLocal);
    template <bool IS_FIRST>
    __aicore__ inline void AccumulateRowsToHoWUpsampleKW1(int64_t hoLocal, int64_t rowLoStart, int64_t rowLoEnd);
    __aicore__ inline void CalAvgOneHoKW1(int64_t kernelH, int64_t hoLocal);
    __aicore__ inline void CalWiToWoInfo();
    __aicore__ inline void TransOut(int64_t hoNum);
    __aicore__ inline void CopyOut(int64_t ncIdx, int64_t ncNum, int64_t hoGlobal, int64_t hoNum);
    __aicore__ inline void ProcessOneBlock(const BlockParam& blockPara);
    __aicore__ inline void ProcessBlocksHIn1KW1();
    __aicore__ inline void ScatterTransposeSlim(int64_t hoNum);

    // wi→wo reverse mapping buffers for W-upsampling
    TBuf<QuePosition::VECCALC> wiWoStartBuf_;
    TBuf<QuePosition::VECCALC> wiWoCountBuf_;
    TBuf<QuePosition::VECCALC> extraWoIdxBuf_;
    TBuf<QuePosition::VECCALC> tempSumBuf_;
};

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::Init(GM_ADDR x, GM_ADDR y)
{
    if (!this->InitCommon(x, y)) {
        return;
    }
    uint64_t dataBlock = AAP_UB_BLOCK_SIZE;
    int64_t wIn = this->tilingData_->wIn;
    int64_t wOut = this->tilingData_->wOut;
    int64_t hOut = this->tilingData_->hOut;
    bool isSlim = (this->tilingData_->hIn == 1) && (wOut > wIn) && (wOut % wIn == 0) &&
                  (this->tilingData_->hoFactor >= hOut);

    if (isSlim) {
        this->pipe_->InitBuffer(this->inputQue_, 1, this->tilingData_->inputQueSize);
        this->pipe_->InitBuffer(this->resQue1_, 1, this->tilingData_->resQue1Size);
        this->pipe_->InitBuffer(this->resQue2_, 1, this->tilingData_->resQue2Size);
        uint64_t transRowAlign = ops::CeilAlign(static_cast<uint64_t>(this->tilingData_->hiFactor) * this->wInAlign_,
                                                static_cast<uint64_t>(AAP_TRANS_ADDR_LEN));
        uint64_t transBufSize = transRowAlign * this->vlNum_ * sizeof(T);
        this->pipe_->InitBuffer(this->transBuf_, transBufSize);
    } else if (wOut <= wIn) {
        // W↓: compact wOut layout + resQue sized for T (early Cast in TransOut). No sumBuf.
        uint64_t wBufSize = ops::CeilAlign(static_cast<uint64_t>(wOut) * sizeof(int32_t), dataBlock);
        uint64_t transRowAlign = ops::CeilAlign(static_cast<uint64_t>(this->tilingData_->hiFactor) * this->wInAlign_,
                                                static_cast<uint64_t>(AAP_TRANS_ADDR_LEN));
        uint64_t transBufSize = transRowAlign * this->vlNum_ * sizeof(T);
        uint64_t outRowAlign = ops::CeilAlign(
            static_cast<uint64_t>(this->tilingData_->hoFactor) * static_cast<uint64_t>(wOut),
            static_cast<uint64_t>(AAP_TRANS_ADDR_LEN));
        uint64_t outBufSize = outRowAlign * this->vlNum_ * sizeof(float);
        this->pipe_->InitBuffer(this->inputQue_, 1, this->tilingData_->inputQueSize);
        this->pipe_->InitBuffer(this->resQue1_, 1, this->tilingData_->resQue1Size);
        if (this->tilingData_->resQue2Size > 0) {
            this->pipe_->InitBuffer(this->resQue2_, 1, this->tilingData_->resQue2Size);
        }
        this->pipe_->InitBuffer(this->transBuf_, transBufSize);
        this->pipe_->InitBuffer(this->outBuf_, outBufSize);
        this->pipe_->InitBuffer(this->wStartBuf_, wBufSize);
        this->pipe_->InitBuffer(this->wKerSizeBuf_, wBufSize);
    } else {
        uint64_t wBufSize = ops::CeilAlign(static_cast<uint64_t>(wOut) * sizeof(int32_t), dataBlock);
        bool isKW1 = (wOut % wIn == 0);
        if (isKW1) {
            this->InitBuffers(wBufSize, this->wInAlign_);
        } else {
            uint64_t transRowAlign = ops::CeilAlign(
                static_cast<uint64_t>(this->tilingData_->hiFactor) * this->wInAlign_,
                static_cast<uint64_t>(AAP_TRANS_ADDR_LEN));
            uint64_t transBufSize = transRowAlign * this->vlNum_ * sizeof(T);
            uint64_t outRowAlign = ops::CeilAlign(static_cast<uint64_t>(this->tilingData_->hoFactor) * this->wOutAlign_,
                                                  static_cast<uint64_t>(AAP_TRANS_ADDR_LEN));
            uint64_t outBufSize = outRowAlign * this->vlNum_ * sizeof(float);
            this->pipe_->InitBuffer(this->inputQue_, 1, this->tilingData_->inputQueSize);
            this->pipe_->InitBuffer(this->resQue1_, 1, this->tilingData_->resQue1Size);
            if (this->tilingData_->resQue2Size > 0) {
                this->pipe_->InitBuffer(this->resQue2_, 1, this->tilingData_->resQue2Size);
            }
            this->pipe_->InitBuffer(this->transBuf_, transBufSize);
            this->pipe_->InitBuffer(this->outBuf_, outBufSize);
            this->pipe_->InitBuffer(this->wStartBuf_, wBufSize);
            this->pipe_->InitBuffer(this->wKerSizeBuf_, wBufSize);
            uint64_t tempBufSize = static_cast<uint64_t>(this->wInAlign_) * this->vlNum_ * sizeof(float);
            this->pipe_->InitBuffer(tempSumBuf_, tempBufSize);
        }
    }

    uint64_t wiBufSize = ops::CeilAlign(static_cast<uint64_t>(wIn) * sizeof(int32_t), dataBlock);
    uint64_t extraWoIdxBufSize = ops::CeilAlign(static_cast<uint64_t>(wOut) * sizeof(int32_t), dataBlock);
    if (wOut > wIn) {
        this->pipe_->InitBuffer(wiWoStartBuf_, wiBufSize);
        this->pipe_->InitBuffer(wiWoCountBuf_, wiBufSize);
    }
    if (wOut <= wIn) {
        this->pipe_->InitBuffer(extraWoIdxBuf_, extraWoIdxBufSize);
    }
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::CalWiToWoInfo()
{
    LocalTensor<int32_t> wiWoStartLocal = wiWoStartBuf_.template Get<int32_t>();
    LocalTensor<int32_t> wiWoCountLocal = wiWoCountBuf_.template Get<int32_t>();
    __ubuf__ int32_t* wiWoStartAddr = (__ubuf__ int32_t*)wiWoStartLocal.GetPhyAddr();
    __ubuf__ int32_t* wiWoCountAddr = (__ubuf__ int32_t*)wiWoCountLocal.GetPhyAddr();

    int32_t wIn = this->tilingData_->wIn;
    int64_t wOutDim = this->tilingData_->wOut;
    int64_t wInDim = this->tilingData_->wIn;
    uint16_t vfLen = AAP_V_REG_SIZE / sizeof(int32_t);
    uint16_t loopSize = ops::CeilDiv(static_cast<uint16_t>(wIn), vfLen);
    uint32_t dataLen = wIn;

    SplitHCalWiToWoInfoVf<ID_T>(wiWoStartAddr, wiWoCountAddr, loopSize, dataLen, vfLen, wInDim, wOutDim);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::AccumulateRowsToHoWUpsample(int64_t hoLocal,
                                                                                                int64_t rowLoStart,
                                                                                                int64_t rowLoEnd)
{
    LocalTensor<T> transLocal = this->transBuf_.template Get<T>();
    LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
    LocalTensor<int32_t> wiWoStartLocal = wiWoStartBuf_.template Get<int32_t>();
    LocalTensor<int32_t> wiWoCountLocal = wiWoCountBuf_.template Get<int32_t>();

    __ubuf__ T* inputAddr = (__ubuf__ T*)transLocal.GetPhyAddr();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();
    __ubuf__ int32_t* wiWoStartAddr = (__ubuf__ int32_t*)wiWoStartLocal.GetPhyAddr();
    __ubuf__ int32_t* wiWoCountAddr = (__ubuf__ int32_t*)wiWoCountLocal.GetPhyAddr();

    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint32_t outBase = static_cast<uint32_t>(hoLocal) * static_cast<uint32_t>(this->tilingData_->wOut) * this->vlNum_;
    uint16_t wiNum = static_cast<uint16_t>(this->tilingData_->wIn);
    uint16_t rStart = static_cast<uint16_t>(rowLoStart);
    uint16_t rEnd = static_cast<uint16_t>(rowLoEnd);

    SplitHAccumulateRowsToHoWUpsampleVf<T, NC_FACTOR>(inputAddr, outAddr, wiWoStartAddr, wiWoCountAddr, outBase, wiNum,
                                                      rStart, rEnd, this->vlNum_, this->wInAlign_, vfLenFp32);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::ClearTempBuf()
{
    LocalTensor<float> tempLocal = tempSumBuf_.template Get<float>();
    __ubuf__ float* tempAddr = (__ubuf__ float*)tempLocal.GetPhyAddr();

    uint32_t totalCount = static_cast<uint32_t>(this->wInAlign_) * this->vlNum_;
    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint16_t loopSize = ops::CeilDiv(totalCount, vfLenFp32);
    uint32_t remaining = totalCount;

    SplitHClearTempBufVf(tempAddr, loopSize, remaining, vfLenFp32);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::AccumulateRowsToTempBuf(int64_t rowLoStart,
                                                                                            int64_t rowLoEnd)
{
    LocalTensor<T> transLocal = this->transBuf_.template Get<T>();
    LocalTensor<float> tempLocal = tempSumBuf_.template Get<float>();

    __ubuf__ T* inputAddr = (__ubuf__ T*)transLocal.GetPhyAddr();
    __ubuf__ float* tempAddr = (__ubuf__ float*)tempLocal.GetPhyAddr();

    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint16_t wiNum = static_cast<uint16_t>(this->tilingData_->wIn);
    uint16_t rStart = static_cast<uint16_t>(rowLoStart);
    uint16_t rEnd = static_cast<uint16_t>(rowLoEnd);

    SplitHAccumulateRowsToTempBufVf<T, NC_FACTOR>(inputAddr, tempAddr, wiNum, rStart, rEnd, this->vlNum_,
                                                  this->wInAlign_, vfLenFp32);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::ScatterTempToOutBuf(int64_t hoLocal)
{
    LocalTensor<float> tempLocal = tempSumBuf_.template Get<float>();
    LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
    LocalTensor<int32_t> wiWoStartLocal = wiWoStartBuf_.template Get<int32_t>();
    LocalTensor<int32_t> wiWoCountLocal = wiWoCountBuf_.template Get<int32_t>();

    __ubuf__ float* tempAddr = (__ubuf__ float*)tempLocal.GetPhyAddr();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();
    __ubuf__ int32_t* wiWoStartAddr = (__ubuf__ int32_t*)wiWoStartLocal.GetPhyAddr();
    __ubuf__ int32_t* wiWoCountAddr = (__ubuf__ int32_t*)wiWoCountLocal.GetPhyAddr();

    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint32_t outBase = static_cast<uint32_t>(hoLocal) * static_cast<uint32_t>(this->tilingData_->wOut) * this->vlNum_;
    uint16_t wiNum = static_cast<uint16_t>(this->tilingData_->wIn);

    SplitHScatterTempToOutBufVf<NC_FACTOR>(tempAddr, outAddr, wiWoStartAddr, wiWoCountAddr, outBase, wiNum,
                                           this->vlNum_, vfLenFp32);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
template <bool IS_FIRST>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::AccumulateRowsToHoRows2(int64_t hoLocal,
                                                                                            int64_t rowLoStart)
{
    uint16_t woNum = static_cast<uint16_t>(this->tilingData_->wOut);

    // Pass 1: all wo treated as kW=2 (one VF entry)
    AccumulateRowsToHoRows2Kw2<IS_FIRST>(hoLocal, 0, woNum, rowLoStart);

    // Pass 2: all kW>2 positions in one single VF entry
    AccumulateExtraColRows2(hoLocal, 0, woNum, rowLoStart);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
template <bool IS_FIRST>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::AccumulateRowsToHoRows3(int64_t hoLocal,
                                                                                            int64_t rowLoStart)
{
    uint16_t woNum = static_cast<uint16_t>(this->tilingData_->wOut);

    // Pass 1: all wo treated as kW=2 (one VF entry)
    AccumulateRowsToHoRows3Kw2<IS_FIRST>(hoLocal, 0, woNum, rowLoStart);

    // Pass 2: all kW>2 positions in one single VF entry
    AccumulateExtraColRows3(hoLocal, 0, woNum, rowLoStart);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
template <bool IS_FIRST>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::AccumulateRowsToHoRows2Kw2(int64_t hoLocal,
                                                                                               uint16_t woStart,
                                                                                               uint16_t woEnd,
                                                                                               int64_t rowLoStart)
{
    LocalTensor<T> transLocal = this->transBuf_.template Get<T>();
    LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
    LocalTensor<int32_t> wStartLocal = this->wStartBuf_.template Get<int32_t>();
    __ubuf__ T* inputAddr = (__ubuf__ T*)transLocal.GetPhyAddr();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();
    __ubuf__ int32_t* wStartAddr = (__ubuf__ int32_t*)wStartLocal.GetPhyAddr();

    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint32_t outBase = static_cast<uint32_t>(hoLocal) * static_cast<uint32_t>(this->tilingData_->wOut) * this->vlNum_;
    uint16_t r0 = static_cast<uint16_t>(rowLoStart);
    uint16_t r1 = static_cast<uint16_t>(rowLoStart + 1);

    SplitHAccumulateRowsToHoRows2Kw2Vf<T, NC_FACTOR, IS_FIRST>(inputAddr, outAddr, wStartAddr, outBase, woStart, woEnd,
                                                               r0, r1, this->vlNum_, this->wInAlign_, vfLenFp32);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
template <bool IS_FIRST>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::AccumulateRowsToHoRows3Kw2(int64_t hoLocal,
                                                                                               uint16_t woStart,
                                                                                               uint16_t woEnd,
                                                                                               int64_t rowLoStart)
{
    LocalTensor<T> transLocal = this->transBuf_.template Get<T>();
    LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
    LocalTensor<int32_t> wStartLocal = this->wStartBuf_.template Get<int32_t>();
    __ubuf__ T* inputAddr = (__ubuf__ T*)transLocal.GetPhyAddr();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();
    __ubuf__ int32_t* wStartAddr = (__ubuf__ int32_t*)wStartLocal.GetPhyAddr();

    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint32_t outBase = static_cast<uint32_t>(hoLocal) * static_cast<uint32_t>(this->tilingData_->wOut) * this->vlNum_;
    uint16_t r0 = static_cast<uint16_t>(rowLoStart);
    uint16_t r1 = static_cast<uint16_t>(rowLoStart + 1);
    uint16_t r2 = static_cast<uint16_t>(rowLoStart + 2);

    SplitHAccumulateRowsToHoRows3Kw2Vf<T, NC_FACTOR, IS_FIRST>(inputAddr, outAddr, wStartAddr, outBase, woStart, woEnd,
                                                               r0, r1, r2, this->vlNum_, this->wInAlign_, vfLenFp32);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::AccumulateExtraColRows2(int64_t hoLocal,
                                                                                            uint16_t woStart,
                                                                                            uint16_t woEnd,
                                                                                            int64_t rowLoStart)
{
    LocalTensor<T> transLocal = this->transBuf_.template Get<T>();
    LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
    LocalTensor<int32_t> wStartLocal = this->wStartBuf_.template Get<int32_t>();
    LocalTensor<int32_t> wKerSizeLocal = this->wKerSizeBuf_.template Get<int32_t>();
    __ubuf__ T* inputAddr = (__ubuf__ T*)transLocal.GetPhyAddr();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();
    __ubuf__ int32_t* wStartAddr = (__ubuf__ int32_t*)wStartLocal.GetPhyAddr();
    __ubuf__ int32_t* wKerSizeAddr = (__ubuf__ int32_t*)wKerSizeLocal.GetPhyAddr();

    LocalTensor<int32_t> extraWoIdxLocal = extraWoIdxBuf_.template Get<int32_t>();
    __ubuf__ int32_t* extraWoIdxAddr = (__ubuf__ int32_t*)extraWoIdxLocal.GetPhyAddr();

    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint32_t outBase = static_cast<uint32_t>(hoLocal) * static_cast<uint32_t>(this->tilingData_->wOut) * this->vlNum_;
    uint16_t r0 = static_cast<uint16_t>(rowLoStart);
    uint16_t r1 = static_cast<uint16_t>(rowLoStart + 1);

    // Pre-build list of kW>2 indices (scalar, outside VF)
    uint16_t extraCount = 0;
    for (uint16_t wo = woStart; wo < woEnd; wo++) {
        if (static_cast<uint16_t>(wKerSizeAddr[wo]) > 2) {
            extraWoIdxAddr[extraCount++] = static_cast<int32_t>(wo);
        }
    }
    if (extraCount == 0) {
        return;
    }

    SplitHAccumulateExtraColRows2Vf<T, NC_FACTOR>(inputAddr, outAddr, wStartAddr, wKerSizeAddr, extraWoIdxAddr, outBase,
                                                  extraCount, r0, r1, this->vlNum_, this->wInAlign_, vfLenFp32);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::AccumulateExtraColRows3(int64_t hoLocal,
                                                                                            uint16_t woStart,
                                                                                            uint16_t woEnd,
                                                                                            int64_t rowLoStart)
{
    LocalTensor<T> transLocal = this->transBuf_.template Get<T>();
    LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
    LocalTensor<int32_t> wStartLocal = this->wStartBuf_.template Get<int32_t>();
    LocalTensor<int32_t> wKerSizeLocal = this->wKerSizeBuf_.template Get<int32_t>();
    __ubuf__ T* inputAddr = (__ubuf__ T*)transLocal.GetPhyAddr();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();
    __ubuf__ int32_t* wStartAddr = (__ubuf__ int32_t*)wStartLocal.GetPhyAddr();
    __ubuf__ int32_t* wKerSizeAddr = (__ubuf__ int32_t*)wKerSizeLocal.GetPhyAddr();

    LocalTensor<int32_t> extraWoIdxLocal = extraWoIdxBuf_.template Get<int32_t>();
    __ubuf__ int32_t* extraWoIdxAddr = (__ubuf__ int32_t*)extraWoIdxLocal.GetPhyAddr();

    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint32_t outBase = static_cast<uint32_t>(hoLocal) * static_cast<uint32_t>(this->tilingData_->wOut) * this->vlNum_;
    uint16_t r0 = static_cast<uint16_t>(rowLoStart);
    uint16_t r1 = static_cast<uint16_t>(rowLoStart + 1);
    uint16_t r2 = static_cast<uint16_t>(rowLoStart + 2);

    // Pre-build list of kW>2 indices (scalar, outside VF)
    uint16_t extraCount = 0;
    for (uint16_t wo = woStart; wo < woEnd; wo++) {
        if (static_cast<uint16_t>(wKerSizeAddr[wo]) > 2) {
            extraWoIdxAddr[extraCount++] = static_cast<int32_t>(wo);
        }
    }
    if (extraCount == 0) {
        return;
    }

    SplitHAccumulateExtraColRows3Vf<T, NC_FACTOR>(inputAddr, outAddr, wStartAddr, wKerSizeAddr, extraWoIdxAddr, outBase,
                                                  extraCount, r0, r1, r2, this->vlNum_, this->wInAlign_, vfLenFp32);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
template <bool IS_FIRST>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::AccumulateRowsToHo(int64_t hoLocal,
                                                                                       int64_t rowLoStart,
                                                                                       int64_t rowLoEnd)
{
    LocalTensor<T> transLocal = this->transBuf_.template Get<T>();
    LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
    LocalTensor<int32_t> wStartLocal = this->wStartBuf_.template Get<int32_t>();
    __ubuf__ T* inputAddr = (__ubuf__ T*)transLocal.GetPhyAddr();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();
    __ubuf__ int32_t* wStartAddr = (__ubuf__ int32_t*)wStartLocal.GetPhyAddr();
    LocalTensor<int32_t> wKerSizeLocal = this->wKerSizeBuf_.template Get<int32_t>();
    __ubuf__ int32_t* wKerSizeAddr = (__ubuf__ int32_t*)wKerSizeLocal.GetPhyAddr();

    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint32_t outBase = static_cast<uint32_t>(hoLocal) * static_cast<uint32_t>(this->tilingData_->wOut) * this->vlNum_;
    uint16_t woNum = static_cast<uint16_t>(this->tilingData_->wOut);
    uint16_t rStart = static_cast<uint16_t>(rowLoStart);
    uint16_t rEnd = static_cast<uint16_t>(rowLoEnd);

    // IS_FIRST is a template parameter, so this resolves at compile time into two
    // branch-free straight-line paths: keeping the if inside __VEC_SCOPE__
    // breaks the software-pipeline scheduling of the vector loop.
    // [RegBase-native] Observed with the CANN 9.0.0 (V100R001C10SPC001B250) compiler;
    // re-measure before assuming it still holds on a newer toolchain.
    SplitHAccumulateRowsToHoVf<T, NC_FACTOR, IS_FIRST>(inputAddr, outAddr, wStartAddr, wKerSizeAddr, outBase, woNum,
                                                       rStart, rEnd, this->vlNum_, this->wInAlign_, vfLenFp32);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::TransOut(int64_t hoNum)
{
    int64_t rowNum = hoNum * static_cast<int64_t>(this->tilingData_->wOut);
    uint64_t rowNumAlign = ops::CeilAlign(static_cast<uint64_t>(rowNum), AAP_TRANS_ADDR_LEN);

    LocalTensor<float> srcLocal = this->outBuf_.template Get<float>();
    if constexpr (IsSameType<T, float>::value) {
        LocalTensor<float> dstLocal = this->resQue1_.template AllocTensor<float>();
        this->template TransposeB32<float>(dstLocal, srcLocal, rowNumAlign, this->vlNum_);
        this->resQue1_.EnQue(dstLocal);
    } else {
        // bf16/fp16: cast fp32->T before transpose, then use B16 transpose so the
        // low-bitwidth data is moved. Numerically bit-equivalent to casting after
        // TransposeB32 (same per-element rounding, relocated ahead of transpose).
        // W↓ aliases the cast buffer onto inputQue (resQue2Size==0 flags the alias): by
        // now the whole hiBatch loop has finished and inputQue is freed, so it can be
        // borrowed, saving UB. Other paths (W↑) keep their own resQue2.
        bool reuseInputQue = (this->tilingData_->resQue2Size == 0);
        LocalTensor<T> castLocal = reuseInputQue ? this->inputQue_.template AllocTensor<T>() :
                                                   this->resQue2_.template AllocTensor<T>();
        if constexpr (IsSameType<T, half>::value) {
            Cast(castLocal, srcLocal, RoundMode::CAST_NONE, rowNumAlign * this->vlNum_);
        } else {
            Cast(castLocal, srcLocal, RoundMode::CAST_RINT, rowNumAlign * this->vlNum_);
        }
        LocalTensor<T> resLocal = this->resQue1_.template AllocTensor<T>();
        this->TransposeB16(resLocal, castLocal, rowNumAlign, this->vlNum_);
        this->resQue1_.EnQue(resLocal);
        if (reuseInputQue) {
            this->inputQue_.FreeTensor(castLocal);
        } else {
            this->resQue2_.FreeTensor(castLocal);
        }
    }
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::CopyOut(int64_t ncIdx, int64_t ncNum,
                                                                            int64_t hoGlobal, int64_t hoNum)
{
    uint64_t rowNumAlign = ops::CeilAlign(static_cast<uint64_t>(hoNum) * static_cast<uint64_t>(this->tilingData_->wOut),
                                          static_cast<uint64_t>(AAP_TRANS_ADDR_LEN));
    int64_t ncBase = ncIdx * this->tilingData_->ncFactor;

    LocalTensor<T> resOutLocal = this->resQue1_.template DeQue<T>();

    DataCopyExtParams valueParams;
    valueParams.blockCount = 1;
    valueParams.blockLen = static_cast<uint32_t>(hoNum * this->tilingData_->wOut * sizeof(T));
    valueParams.srcStride = 0;
    valueParams.dstStride = 0;

    LoopModeParams loopModeParams;
    loopModeParams.loop1Size = ncNum;
    loopModeParams.loop2Size = 1;
    loopModeParams.loop1SrcStride = rowNumAlign * sizeof(T);
    loopModeParams.loop2SrcStride = 0;
    loopModeParams.loop1DstStride = this->outHW_ * sizeof(T);
    loopModeParams.loop2DstStride = 0;

    int64_t yOff = ncBase * this->outHW_ + hoGlobal * this->tilingData_->wOut;
    SetLoopModePara(loopModeParams, DataCopyMVType::UB_TO_OUT);
    DataCopyPad(this->yGm_[yOff], resOutLocal, valueParams);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);

    this->resQue1_.FreeTensor(resOutLocal);
}

// kW=1 fast path: when wOut%wIn==0, each wo receives exactly one wi's contribution.
// Skips ClearOutBuf; IS_FIRST=true stores directly, IS_FIRST=false does load+add+store.
template <typename T, typename ID_T, const uint32_t NC_FACTOR>
template <bool IS_FIRST>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::AccumulateRowsToHoWUpsampleKW1(int64_t hoLocal,
                                                                                                   int64_t rowLoStart,
                                                                                                   int64_t rowLoEnd)
{
    LocalTensor<T> transLocal = this->transBuf_.template Get<T>();
    LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
    LocalTensor<int32_t> wiWoStartLocal = wiWoStartBuf_.template Get<int32_t>();
    LocalTensor<int32_t> wiWoCountLocal = wiWoCountBuf_.template Get<int32_t>();

    __ubuf__ T* inputAddr = (__ubuf__ T*)transLocal.GetPhyAddr();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();
    __ubuf__ int32_t* wiWoStartAddr = (__ubuf__ int32_t*)wiWoStartLocal.GetPhyAddr();
    __ubuf__ int32_t* wiWoCountAddr = (__ubuf__ int32_t*)wiWoCountLocal.GetPhyAddr();

    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint32_t outBase = static_cast<uint32_t>(hoLocal) * static_cast<uint32_t>(this->tilingData_->wOut) * this->vlNum_;
    uint16_t wiNum = static_cast<uint16_t>(this->tilingData_->wIn);
    uint16_t rStart = static_cast<uint16_t>(rowLoStart);
    uint16_t rEnd = static_cast<uint16_t>(rowLoEnd);

    // IS_FIRST is a compile-time parameter, giving two branch-free straight-line
    // paths so the vector loop's software pipeline is
    // not broken by an in-scope branch. The W-reduce part is identical in both
    // paths; only the scatter (store vs load+add+store) differs.
    SplitHAccumulateRowsToHoWUpsampleKW1Vf<T, NC_FACTOR, IS_FIRST>(inputAddr, outAddr, wiWoStartAddr, wiWoCountAddr,
                                                                   outBase, wiNum, rStart, rEnd, this->vlNum_,
                                                                   this->wInAlign_, vfLenFp32);
}

// kW=1: all wo have same divisor (kH only), hoist Duplicate+Cast out of wo loop.
template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::CalAvgOneHoKW1(int64_t kernelH, int64_t hoLocal)
{
    LocalTensor<float> outLocal = this->outBuf_.template Get<float>();
    __ubuf__ float* outAddr = (__ubuf__ float*)outLocal.GetPhyAddr();

    uint32_t vfLenFp32 = AAP_V_REG_SIZE / sizeof(float);
    uint32_t outBaseOffset = static_cast<uint32_t>(hoLocal) * static_cast<uint32_t>(this->tilingData_->wOut) *
                             this->vlNum_;
    uint16_t woNum = static_cast<uint16_t>(this->tilingData_->wOut);
    int32_t kh = static_cast<int32_t>(kernelH);

    SplitHCalAvgOneHoKW1Vf<NC_FACTOR>(outAddr, outBaseOffset, woNum, kh, this->vlNum_, vfLenFp32);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::ProcessOneBlock(const BlockParam& blockPara)
{
    int64_t ncIdx = blockPara.ncIdx;
    int64_t ncNum = blockPara.ncNum;
    int64_t hoGlobalStart = blockPara.hoIdx * this->tilingData_->hoFactor;
    int64_t hoNum = blockPara.hoNum;
    int64_t hIn = this->tilingData_->hIn;
    int64_t hOut = this->tilingData_->hOut;
    int64_t wIn = this->tilingData_->wIn;
    int64_t wOut = this->tilingData_->wOut;

    bool isKW1 = (wOut > wIn) && (wOut % wIn == 0);
    bool isWDown = (wOut <= wIn);
    // The rowCount==2/3 fast paths unroll exactly two W columns per wo and ignore
    // wKerSizeAddr[], so they are only valid when every window is at least 2 wide.
    // kW = ceil((wo+1)*wIn/wOut) - floor(wo*wIn/wOut) >= wIn/wOut, so wOut<wIn gives
    // kW>=2 for all wo, while wOut==wIn gives kW==1 everywhere and must take the
    // general path (which bounds its column loops by kernelW).
    bool isKwGe2 = (wOut < wIn);

    if (!isKW1 && !isWDown) {
        this->ClearOutBuf();
    }

    int64_t hiMin = 0;
    int64_t hiMax = 0;
    this->CalHiRange(hoGlobalStart, hoNum, hiMin, hiMax);
    bool useTempBuf = !isKW1 && !isWDown && (hoNum == 1) && (hiMax - hiMin > this->tilingData_->hiFactor);

    for (int64_t hiBase = hiMin; hiBase < hiMax; hiBase += this->tilingData_->hiFactor) {
        int64_t hiBatch = this->tilingData_->hiFactor;
        if (hiBase + hiBatch > hiMax) {
            hiBatch = hiMax - hiBase;
        }
        this->CopyInputBatch(ncIdx, ncNum, hiBase, hiBatch);
        this->TransInputBatch(hiBatch);

        int64_t batchLo = hiBase;
        int64_t batchHi = hiBase + hiBatch;

        int64_t hoScan = (batchLo * hOut) / hIn;
        if (hoScan < hoGlobalStart) {
            hoScan = hoGlobalStart;
        }
        for (int64_t hoGlobal = hoScan; hoGlobal < hoGlobalStart + hoNum; hoGlobal++) {
            int64_t hStart = (hoGlobal * hIn) / hOut;
            int64_t hEnd = ((hoGlobal + 1) * hIn + hOut - 1) / hOut;
            if (hEnd <= batchLo) {
                continue;
            }
            if (hStart >= batchHi) {
                break;
            }
            int64_t ovLo = hStart > batchLo ? hStart : batchLo;
            int64_t ovHi = hEnd < batchHi ? hEnd : batchHi;
            int64_t rowLoStart = ovLo - batchLo;
            int64_t rowLoEnd = ovHi - batchLo;
            int64_t rowCount = rowLoEnd - rowLoStart;
            bool isFirst = (hStart >= batchLo);
            if (isKW1) {
                if (isFirst) {
                    AccumulateRowsToHoWUpsampleKW1<true>(hoGlobal - hoGlobalStart, rowLoStart, rowLoEnd);
                } else {
                    AccumulateRowsToHoWUpsampleKW1<false>(hoGlobal - hoGlobalStart, rowLoStart, rowLoEnd);
                }
            } else if (wOut > wIn) {
                if (useTempBuf) {
                    if (isFirst) {
                        ClearTempBuf();
                    }
                    AccumulateRowsToTempBuf(rowLoStart, rowLoEnd);
                    if (hEnd <= batchHi) {
                        ScatterTempToOutBuf(hoGlobal - hoGlobalStart);
                    }
                } else {
                    AccumulateRowsToHoWUpsample(hoGlobal - hoGlobalStart, rowLoStart, rowLoEnd);
                }
            } else if (rowCount == 2 && isKwGe2) {
                if (isFirst) {
                    AccumulateRowsToHoRows2<true>(hoGlobal - hoGlobalStart, rowLoStart);
                } else {
                    AccumulateRowsToHoRows2<false>(hoGlobal - hoGlobalStart, rowLoStart);
                }
            } else if (rowCount == 3 && isKwGe2) {
                if (isFirst) {
                    AccumulateRowsToHoRows3<true>(hoGlobal - hoGlobalStart, rowLoStart);
                } else {
                    AccumulateRowsToHoRows3<false>(hoGlobal - hoGlobalStart, rowLoStart);
                }
            } else {
                if (isFirst) {
                    AccumulateRowsToHo<true>(hoGlobal - hoGlobalStart, rowLoStart, rowLoEnd);
                } else {
                    AccumulateRowsToHo<false>(hoGlobal - hoGlobalStart, rowLoStart, rowLoEnd);
                }
            }
        }
    }

    for (int64_t hoLocal = 0; hoLocal < hoNum; hoLocal++) {
        int64_t hoGlobal = hoGlobalStart + hoLocal;
        int64_t hStart = (hoGlobal * hIn) / hOut;
        int64_t hEnd = ((hoGlobal + 1) * hIn + hOut - 1) / hOut;
        int64_t kernelH = hEnd - hStart;
        if (isKW1 && kernelH == 1) {
            continue;
        }
        if (isKW1) {
            CalAvgOneHoKW1(kernelH, hoLocal);
        } else {
            this->CalAvgOneHo(kernelH, hoLocal, static_cast<uint32_t>(wOut));
        }
    }

    TransOut(hoNum);
    CopyOut(ncIdx, ncNum, hoGlobalStart, hoNum);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::ScatterTransposeSlim(int64_t hoNum)
{
    LocalTensor<T> transLocal = this->transBuf_.template Get<T>();
    LocalTensor<int32_t> wiWoStartLocal = wiWoStartBuf_.template Get<int32_t>();
    LocalTensor<int32_t> wiWoCountLocal = wiWoCountBuf_.template Get<int32_t>();
    __ubuf__ int32_t* wiWoStartAddr = (__ubuf__ int32_t*)wiWoStartLocal.GetPhyAddr();
    __ubuf__ int32_t* wiWoCountAddr = (__ubuf__ int32_t*)wiWoCountLocal.GetPhyAddr();

    int64_t wOut = this->tilingData_->wOut;
    uint64_t rowNumAlign = ops::CeilAlign(static_cast<uint64_t>(hoNum) * static_cast<uint64_t>(wOut),
                                          static_cast<uint64_t>(AAP_TRANS_ADDR_LEN));
    uint16_t wiNum = static_cast<uint16_t>(this->tilingData_->wIn);

    LocalTensor<T> scatterLocal = this->resQue2_.template AllocTensor<T>();

    for (uint16_t wi = 0; wi < wiNum; wi++) {
        uint32_t srcOff = static_cast<uint32_t>(wi) * this->vlNum_;
        uint32_t woStart = static_cast<uint32_t>(wiWoStartAddr[wi]);
        uint16_t woCount = static_cast<uint16_t>(wiWoCountAddr[wi]);
        for (uint16_t j = 0; j < woCount; j++) {
            uint32_t wo = woStart + static_cast<uint32_t>(j);
            uint32_t dstOff = wo * this->vlNum_;
            Adds(scatterLocal[dstOff], transLocal[srcOff], static_cast<T>(0), this->vlNum_);
        }
    }

    uint32_t rowElems = static_cast<uint32_t>(wOut) * this->vlNum_;
    for (int64_t ho = 1; ho < hoNum; ho++) {
        uint32_t dstOff = static_cast<uint32_t>(ho) * rowElems;
        Adds(scatterLocal[dstOff], scatterLocal[0], static_cast<T>(0), rowElems);
    }

    LocalTensor<T> resLocal = this->resQue1_.template AllocTensor<T>();
    if constexpr (IsSameType<T, float>::value) {
        this->template TransposeB32<float>(resLocal.template ReinterpretCast<float>(),
                                           scatterLocal.template ReinterpretCast<float>(),
                                           static_cast<uint32_t>(rowNumAlign), this->vlNum_);
    } else {
        this->TransposeB16(resLocal, scatterLocal, static_cast<uint32_t>(rowNumAlign), this->vlNum_);
    }
    this->resQue1_.EnQue(resLocal);
    this->resQue2_.FreeTensor(scatterLocal);
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::ProcessBlocksHIn1KW1()
{
    int64_t lastNcIdx = -1;

    BlockParam blockPara;
    for (int64_t curIdx = this->startBlockIdx_; curIdx < this->endBlockIdx_; curIdx++) {
        this->CalBlockPara(curIdx, blockPara);

        if (blockPara.ncIdx != lastNcIdx) {
            this->CopyInputBatch(blockPara.ncIdx, blockPara.ncNum, 0, 1);
            this->TransInputBatch(1);
            lastNcIdx = blockPara.ncIdx;
        }

        int64_t hoGlobalStart = blockPara.hoIdx * this->tilingData_->hoFactor;
        ScatterTransposeSlim(blockPara.hoNum);
        CopyOut(blockPara.ncIdx, blockPara.ncNum, hoGlobalStart, blockPara.hoNum);
    }
}

template <typename T, typename ID_T, const uint32_t NC_FACTOR>
__aicore__ inline void AdaptiveAvgPool2dSplitH<T, ID_T, NC_FACTOR>::Process()
{
    if (GetBlockIdx() >= this->tilingData_->useCoreNum) {
        return;
    }

    int64_t wIn = this->tilingData_->wIn;
    int64_t wOut = this->tilingData_->wOut;
    int64_t hOut = this->tilingData_->hOut;
    bool isSlim = (this->tilingData_->hIn == 1) && (wOut > wIn) && (wOut % wIn == 0) &&
                  (this->tilingData_->hoFactor >= hOut);

    if (!isSlim) {
        this->CalWKernelInfo();
    }
    if (wOut > wIn) {
        CalWiToWoInfo();
    }
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    if (isSlim) {
        ProcessBlocksHIn1KW1();
        return;
    }

    BlockParam blockPara;
    for (int64_t curIdx = this->startBlockIdx_; curIdx < this->endBlockIdx_; curIdx++) {
        this->CalBlockPara(curIdx, blockPara);
        ProcessOneBlock(blockPara);
    }
}

} // namespace AdaptivePool2dSplitHNamespace
#endif // ADAPTIVE_AVG_POOL2D_SPLIT_H_H_
