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
 * \file cla_gate_quant_vf.h
 * \brief Register-level vector functions for ClaGateQuant on Ascend 950.
 *
 * These __simd_vf__ functions operate on UB addresses and scalar arguments.
 * GM transfers, synchronization, and dispatch are handled by the caller.
 */
#ifndef OPS_NN_CLA_GATE_QUANT_VF_H
#define OPS_NN_CLA_GATE_QUANT_VF_H

#include "kernel_operator.h"
#include "../cla_gate_quant_common.h"

namespace ClaGateQuant {
using namespace AscendC;

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__simd_vf__ inline void ComputeClaGateFullTileVF(__ubuf__ xDtype* globalAddr, __ubuf__ xDtype* localAddr,
                                                 __ubuf__ xDtype* globalGateAddr, __ubuf__ xDtype* localGateAddr,
                                                 __ubuf__ xDtype* outputAddr, uint16_t calcRows, int64_t headSize,
                                                 __ubuf__ uint8_t* sigmoidBroadcastAddr, int64_t gateSideOffset)
{
    uint16_t vfsPerHead = static_cast<uint16_t>(headSize / VF_LEN_FP32);

    Reg::RegTensor<xDtype> globalReg, localReg, globalGateReg, localGateReg, outputReg;
    Reg::RegTensor<float> globalGateF, localGateF, globalTmpF, localTmpF, oneF;
    Reg::RegTensor<float> globalSigmoidF, localSigmoidF;
    Reg::RegTensor<float> globalRegF, localRegF;
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    // Sigmoid scratch stores FP32 values. Keep global/local regions separated.
    __ubuf__ uint8_t* sigmoidBytes = (__ubuf__ uint8_t*)sigmoidBroadcastAddr;
    __ubuf__ float* globalSigmoidUb = (__ubuf__ float*)sigmoidBytes;
    __ubuf__ float* localSigmoidUb = (__ubuf__ float*)(sigmoidBytes + gateSideOffset * sizeof(float));

    // This fallback handles one head only; full D=128 two-head tiles are
    // dispatched to ComputeClaGateD128TwoHeads by the dual-axis caller.
    Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(globalGateReg, globalGateAddr);
    Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(localGateReg, localGateAddr);
    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(globalGateF, globalGateReg, allMask);
    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(localGateF, localGateReg, allMask);
    Reg::Muls(globalTmpF, globalGateF, -1.0f, allMask);
    Reg::Muls(localTmpF, localGateF, -1.0f, allMask);
    Reg::Exp(globalTmpF, globalTmpF, allMask);
    Reg::Exp(localTmpF, localTmpF, allMask);
    Reg::Adds(globalTmpF, globalTmpF, 1.0f, allMask);
    Reg::Adds(localTmpF, localTmpF, 1.0f, allMask);
    Reg::Duplicate(oneF, 1.0f, allMask);
    Reg::Div(globalSigmoidF, oneF, globalTmpF, allMask);
    Reg::Div(localSigmoidF, oneF, localTmpF, allMask);
    Reg::StoreAlign<float>(globalSigmoidUb, globalSigmoidF, allMask);
    Reg::StoreAlign<float>(localSigmoidUb, localSigmoidF, allMask);

    // Phase 2: broadcast each per-row/per-head FP32 sigmoid over its D-sized span.
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    Reg::AddrReg offset;
    for (uint16_t row = 0; row < calcRows; ++row) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(globalSigmoidF, globalSigmoidUb + row);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(localSigmoidF, localSigmoidUb + row);

        for (uint16_t vf = 0; vf < vfsPerHead; ++vf) {
            offset = Reg::CreateAddrReg<xDtype>(row, ONCE_ROW_LEN, vf, VF_LEN_FP32);
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(globalReg, globalAddr, offset);
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(localReg, localAddr, offset);

            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(globalRegF, globalReg, allMask);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(localRegF, localReg, allMask);
            Reg::Mul(globalRegF, globalRegF, globalSigmoidF, allMask);
            Reg::MulAddDst(globalRegF, localRegF, localSigmoidF, allMask);
            Reg::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(outputReg, globalRegF, allMask);
            Reg::StoreAlign<xDtype, Reg::StoreDist::DIST_PACK_B32>(outputAddr, outputReg, offset, allMask);
        }
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__simd_vf__ inline void ComputeClaGateD128TwoHeadsVF(__ubuf__ xDtype* globalAddr, __ubuf__ xDtype* localAddr,
                                                     __ubuf__ xDtype* globalGateAddr, __ubuf__ xDtype* localGateAddr,
                                                     __ubuf__ xDtype* outputAddr, uint16_t calcRows,
                                                     __ubuf__ uint8_t* sigmoidBroadcastAddr, int64_t gateSideOffset)
{
    Reg::RegTensor<xDtype> globalReg0, localReg0, globalReg1, localReg1;
    Reg::RegTensor<xDtype> globalGateReg, localGateReg, outputReg0, outputReg1;
    Reg::RegTensor<float> globalGateF, localGateF, globalTmpF, localTmpF, oneF;
    Reg::RegTensor<float> globalSigmoidF0, localSigmoidF0;
    Reg::RegTensor<float> globalSigmoidF1, localSigmoidF1;
    Reg::RegTensor<float> globalRegF0, localRegF0;
    Reg::RegTensor<float> globalRegF1, localRegF1;
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    __ubuf__ uint8_t* sigmoidBytes = (__ubuf__ uint8_t*)sigmoidBroadcastAddr;
    __ubuf__ float* globalSigmoidUb = (__ubuf__ float*)sigmoidBytes;
    __ubuf__ float* localSigmoidUb = (__ubuf__ float*)(sigmoidBytes + gateSideOffset * sizeof(float));

    // Gate input is interleaved by row.  Two 64-value VFs cover the 128 gate
    // scalars of a full 64-row tile; keeping the sigmoid output interleaved lets
    // the merge address both heads directly without an extra gather/transpose.
    Reg::Duplicate(oneF, 1.0f, allMask);
    for (uint16_t gateVf = 0; gateVf < static_cast<uint16_t>(DIGIT_TWO); ++gateVf) {
        uint32_t gateOffset = static_cast<uint32_t>(gateVf) * VF_LEN_FP32;
        Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(globalGateReg, globalGateAddr + gateOffset);
        Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(localGateReg, localGateAddr + gateOffset);
        Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(globalGateF, globalGateReg, allMask);
        Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(localGateF, localGateReg, allMask);
        Reg::Muls(globalTmpF, globalGateF, -1.0f, allMask);
        Reg::Muls(localTmpF, localGateF, -1.0f, allMask);
        Reg::Exp(globalTmpF, globalTmpF, allMask);
        Reg::Exp(localTmpF, localTmpF, allMask);
        Reg::Adds(globalTmpF, globalTmpF, 1.0f, allMask);
        Reg::Adds(localTmpF, localTmpF, 1.0f, allMask);
        Reg::Div(globalSigmoidF0, oneF, globalTmpF, allMask);
        Reg::Div(localSigmoidF0, oneF, localTmpF, allMask);
        Reg::StoreAlign<float>(globalSigmoidUb + gateOffset, globalSigmoidF0, allMask);
        Reg::StoreAlign<float>(localSigmoidUb + gateOffset, localSigmoidF0, allMask);
    }

    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    for (uint16_t row = 0; row < calcRows; ++row) {
        uint32_t gateOffset = static_cast<uint32_t>(row) * DIGIT_TWO;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(globalSigmoidF0, globalSigmoidUb + gateOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(localSigmoidF0, localSigmoidUb + gateOffset);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(globalSigmoidF1, globalSigmoidUb + gateOffset + 1);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(localSigmoidF1, localSigmoidUb + gateOffset + 1);

        constexpr uint16_t d128VfsPerHead = static_cast<uint16_t>(D128_HEAD_ELEMS / VF_LEN_FP32);
        uint32_t rowOffset = static_cast<uint32_t>(row) * ONCE_ROW_LEN;
        __ubuf__ xDtype* globalHead0 = globalAddr + rowOffset;
        __ubuf__ xDtype* globalHead1 = globalHead0 + D128_HEAD_ELEMS;
        __ubuf__ xDtype* localHead0 = localAddr + rowOffset;
        __ubuf__ xDtype* localHead1 = localHead0 + D128_HEAD_ELEMS;
        __ubuf__ xDtype* outputHead0 = outputAddr + rowOffset;
        __ubuf__ xDtype* outputHead1 = outputHead0 + D128_HEAD_ELEMS;
        for (uint16_t vf = 0; vf < d128VfsPerHead; ++vf) {
            uint32_t vfOffset = static_cast<uint32_t>(vf) * VF_LEN_FP32;
            __ubuf__ xDtype* globalVf0 = globalHead0 + vfOffset;
            __ubuf__ xDtype* globalVf1 = globalHead1 + vfOffset;
            __ubuf__ xDtype* localVf0 = localHead0 + vfOffset;
            __ubuf__ xDtype* localVf1 = localHead1 + vfOffset;
            __ubuf__ xDtype* outputVf0 = outputHead0 + vfOffset;
            __ubuf__ xDtype* outputVf1 = outputHead1 + vfOffset;

            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(globalReg0, globalVf0);
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(globalReg1, globalVf1);
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(localReg0, localVf0);
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(localReg1, localVf1);

            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(globalRegF0, globalReg0, allMask);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(globalRegF1, globalReg1, allMask);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(localRegF0, localReg0, allMask);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(localRegF1, localReg1, allMask);

            Reg::Mul(globalRegF0, globalRegF0, globalSigmoidF0, allMask);
            Reg::Mul(globalRegF1, globalRegF1, globalSigmoidF1, allMask);
            Reg::MulAddDst(globalRegF0, localRegF0, localSigmoidF0, allMask);
            Reg::MulAddDst(globalRegF1, localRegF1, localSigmoidF1, allMask);
            Reg::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(outputReg0, globalRegF0, allMask);
            Reg::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(outputReg1, globalRegF1, allMask);
            Reg::StoreAlign<xDtype, Reg::StoreDist::DIST_PACK_B32>(outputVf0, outputReg0, allMask);
            Reg::StoreAlign<xDtype, Reg::StoreDist::DIST_PACK_B32>(outputVf1, outputReg1, allMask);
        }
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__simd_vf__ inline void FillZeroVF(__ubuf__ xDtype* outputAddr, uint32_t elementCount)
{
    uint16_t iterations = static_cast<uint16_t>((elementCount + VF_LEN_B16 - 1) / VF_LEN_B16);
    uint32_t remainingElements = elementCount;

    Reg::RegTensor<xDtype> zeroReg;
    AscendC::Reg::MaskReg mask;
    Reg::Duplicate(zeroReg, 0);
    for (uint16_t i = 0; i < iterations; i++) {
        mask = AscendC::Reg::UpdateMask<xDtype>(remainingElements);
        Reg::AddrReg offset = Reg::CreateAddrReg<xDtype>(i, VF_LEN_B16);
        AscendC::Reg::StoreAlign(outputAddr, zeroReg, offset, mask);
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__simd_vf__ inline void ComputeRowDataToFp8VF(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                              __ubuf__ uint16_t* rowScaleReciprocalAddr, __ubuf__ uint8_t* rowDataAddr)
{
    Reg::MaskReg maskAll = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskAllB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint16_t> scaleForMulFP16;
    Reg::RegTensor<float> scaleForMulFP32;
    Reg::RegTensor<xDtype> evenData;
    Reg::RegTensor<xDtype> oddData;
    Reg::RegTensor<bfloat16_t> evenBf16;
    Reg::RegTensor<bfloat16_t> oddBf16;
    Reg::RegTensor<float> evenZeroFp32;
    Reg::RegTensor<float> evenOneFp32;
    Reg::RegTensor<float> oddZeroFp32;
    Reg::RegTensor<float> oddOneFp32;
    Reg::RegTensor<rowDataDtype> evenZeroFp8;
    Reg::RegTensor<rowDataDtype> evenOneFp8;
    Reg::RegTensor<rowDataDtype> oddZeroFp8;
    Reg::RegTensor<rowDataDtype> oddOneFp8;

    for (uint16_t i = 0; i < blockCount; i++) {
        Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(evenData, oddData,
                                                                                                   xAddr, ONCE_ROW_LEN);
        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
            scaleForMulFP16, rowScaleReciprocalAddr, ROW_SCALE_RECIPROCAL_ROW_ELEMS);

        if constexpr (IsSameType<xDtype, half>::value) {
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(evenZeroFp32, evenData, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(evenOneFp32, evenData, maskAll);
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(scaleForMulFP32,
                                                              (Reg::RegTensor<bfloat16_t>&)scaleForMulFP16, maskAll);

            Reg::Mul(evenZeroFp32, evenZeroFp32, scaleForMulFP32, maskAll);
            Reg::Mul(evenOneFp32, evenOneFp32, scaleForMulFP32, maskAll);

            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(oddZeroFp32, oddData, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(oddOneFp32, oddData, maskAll);
            Reg::Mul(oddZeroFp32, oddZeroFp32, scaleForMulFP32, maskAll);
            Reg::Mul(oddOneFp32, oddOneFp32, scaleForMulFP32, maskAll);
        } else {
            Reg::Mul(evenData, evenData, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
            Reg::Mul(oddData, oddData, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);

            AscendC::Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(evenZeroFp32, evenData, maskAll);
            AscendC::Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(evenOneFp32, evenData, maskAll);
            AscendC::Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(oddZeroFp32, oddData, maskAll);
            AscendC::Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(oddOneFp32, oddData, maskAll);
        }
        AscendC::Reg::Cast<rowDataDtype, float, CAST_32_TO_80>(evenZeroFp8, evenZeroFp32, maskAll);
        AscendC::Reg::Cast<rowDataDtype, float, CAST_32_TO_82>(evenOneFp8, evenOneFp32, maskAll);
        AscendC::Reg::Cast<rowDataDtype, float, CAST_32_TO_81>(oddZeroFp8, oddZeroFp32, maskAll);
        AscendC::Reg::Cast<rowDataDtype, float, CAST_32_TO_83>(oddOneFp8, oddOneFp32, maskAll);

        AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t>&)evenZeroFp8,
                          (AscendC::Reg::RegTensor<uint8_t>&)evenZeroFp8, (AscendC::Reg::RegTensor<uint8_t>&)evenOneFp8,
                          maskAllB8);
        AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t>&)evenZeroFp8,
                          (AscendC::Reg::RegTensor<uint8_t>&)evenZeroFp8, (AscendC::Reg::RegTensor<uint8_t>&)oddZeroFp8,
                          maskAllB8);
        AscendC::Reg::Add((AscendC::Reg::RegTensor<uint8_t>&)evenZeroFp8,
                          (AscendC::Reg::RegTensor<uint8_t>&)evenZeroFp8, (AscendC::Reg::RegTensor<uint8_t>&)oddOneFp8,
                          maskAllB8);
        AscendC::Reg::StoreAlign<uint8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                 AscendC::Reg::StoreDist::DIST_NORM_B8>(
            rowDataAddr, (AscendC::Reg::RegTensor<uint8_t>&)evenZeroFp8, ONCE_ROW_LEN, maskAllB8);
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__simd_vf__ inline void ComputeRowDataToFp4VF(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                              __ubuf__ uint16_t* rowScaleReciprocalAddr, __ubuf__ uint8_t* rowDataAddr,
                                              uint16_t reciprocalStride)
{
    static constexpr Reg::CastTrait castTraitBF16toFp4 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                          Reg::MaskMergeMode::ZEROING, roundMode};
    static constexpr Reg::CastTrait castTraitFp32toBF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                           Reg::MaskMergeMode::ZEROING, roundMode};

    // Long-lived masks: create once, never inside the row loop.
    Reg::MaskReg dataMaskB8 = Reg::CreateMask<uint8_t>();
    Reg::MaskReg dataMaskB16 = Reg::CreateMask<half>();
    Reg::MaskReg dataMaskB32 = Reg::CreateMask<float>();

    Reg::RegTensor<uint16_t> scaleForMulFP16;
    Reg::RegTensor<xDtype> evenData;
    Reg::RegTensor<xDtype> oddData;

    Reg::RegTensor<float> evenZeroFp32;
    Reg::RegTensor<float> evenOneFp32;
    Reg::RegTensor<float> oddZeroFp32;
    Reg::RegTensor<float> oddOneFp32;
    Reg::RegTensor<float> scaleForMulZeroFP32;

    Reg::RegTensor<bfloat16_t> evenZeroBf16;
    Reg::RegTensor<bfloat16_t> evenOneBf16;
    Reg::RegTensor<bfloat16_t> oddZeroBf16;
    Reg::RegTensor<bfloat16_t> oddOneBf16;

    Reg::RegTensor<rowDataDtype> evenFp4;
    Reg::RegTensor<rowDataDtype> oddFp4;
    Reg::RegTensor<int32_t> negZero;

    Reg::Duplicate(negZero, NEG_ZERO);

    // E2M1 grid alignment masks the raw exponent field in place, so these three
    // constants are loop invariant: build them once per call, not once per row.
    Reg::RegTensor<int32_t> expMaskFP32;
    Reg::RegTensor<int32_t> expUnitFP32;
    Reg::RegTensor<int32_t> expZeroFP32;
    if constexpr (IsSameType<xDtype, half>::value) {
        Reg::Duplicate(expMaskFP32, MAX_EXP_FOR_FP32);
        Reg::Duplicate(expUnitFP32, FP32_EXP_UNIT_BITS);
        Reg::Duplicate(expZeroFP32, FP32_EXP_ZERO_BITS);
    }

    for (uint16_t i = 0; i < blockCount; i++) {
        Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(evenData, oddData,
                                                                                                   xAddr, ONCE_ROW_LEN);
        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
            scaleForMulFP16, rowScaleReciprocalAddr, reciprocalStride);

        if constexpr (IsSameType<xDtype, half>::value) {
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(
                scaleForMulZeroFP32, (Reg::RegTensor<bfloat16_t>&)scaleForMulFP16, dataMaskB16);

            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(evenZeroFp32, evenData, dataMaskB16);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(evenOneFp32, evenData, dataMaskB16);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(oddZeroFp32, oddData, dataMaskB16);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(oddOneFp32, oddData, dataMaskB16);

            Reg::Mul(evenZeroFp32, scaleForMulZeroFP32, evenZeroFp32, dataMaskB32);
            Reg::Mul(evenOneFp32, scaleForMulZeroFP32, evenOneFp32, dataMaskB32);
            Reg::Mul(oddZeroFp32, scaleForMulZeroFP32, oddZeroFp32, dataMaskB32);
            Reg::Mul(oddOneFp32, scaleForMulZeroFP32, oddOneFp32, dataMaskB32);

            // Pair 0: evenZeroFp32 + evenOneFp32, stage-major.
            {
                Reg::MaskReg negInfMaskA;
                Reg::MaskReg specialMaskA;
                Reg::MaskReg negInfMaskB;
                Reg::MaskReg specialMaskB;
                Reg::MaskReg zeroMask;
                Reg::RegTensor<int32_t> exp0A;
                Reg::RegTensor<int32_t> exp1A;
                Reg::RegTensor<int32_t> exp0B;
                Reg::RegTensor<int32_t> exp1B;

                Reg::Compare<int32_t, CMPMODE::EQ>(negInfMaskA, (Reg::RegTensor<int32_t>&)evenZeroFp32, negZero,
                                                   dataMaskB32);
                Reg::Compare<int32_t, CMPMODE::EQ>(negInfMaskB, (Reg::RegTensor<int32_t>&)evenOneFp32, negZero,
                                                   dataMaskB32);
                if constexpr (IsSameType<rowDataDtype, fp4x2_e1m2_t>::value) {
                    // E1M2: multiply by 4, truncate, divide by 4
                    Reg::Muls(evenZeroFp32, evenZeroFp32, FOUR, dataMaskB32);
                    Reg::Muls(evenOneFp32, evenOneFp32, FOUR, dataMaskB32);
                    Reg::Compares<float, CMPMODE::LT>(specialMaskA, evenZeroFp32, 0, dataMaskB32);
                    Reg::Compares<float, CMPMODE::LT>(specialMaskB, evenOneFp32, 0, dataMaskB32);
                    Reg::Truncate<float, roundMode>(evenZeroFp32, evenZeroFp32, dataMaskB32);
                    Reg::Truncate<float, roundMode>(evenOneFp32, evenOneFp32, dataMaskB32);
                    Reg::Muls(evenZeroFp32, evenZeroFp32, ONE_FOURTH, dataMaskB32);
                    Reg::Muls(evenOneFp32, evenOneFp32, ONE_FOURTH, dataMaskB32);
                } else {
                    // E2M1: align v to the 2^{e-1} grid.  The masked exponent field
                    // is kept in place (its float value is 2^e), so both powers of
                    // two are a single integer subtract:
                    //   raw = (e + 127) << 23
                    //   s   = 2^{e-1} = raw - (1 << 23)
                    //   1/s = 2^{1-e} = 0x7F800000 - raw
                    // The subnormal clamp e' = max(e, 0) becomes max(raw, 127 << 23).
                    Reg::And(exp0A, (Reg::RegTensor<int32_t>&)evenZeroFp32, expMaskFP32, dataMaskB32);
                    Reg::And(exp0B, (Reg::RegTensor<int32_t>&)evenOneFp32, expMaskFP32, dataMaskB32);
                    Reg::Max(exp0A, exp0A, expZeroFP32, dataMaskB32);
                    Reg::Max(exp0B, exp0B, expZeroFP32, dataMaskB32);
                    Reg::Sub(exp1A, expMaskFP32, exp0A, dataMaskB32); // 1/s
                    Reg::Sub(exp1B, expMaskFP32, exp0B, dataMaskB32);
                    Reg::Sub(exp0A, exp0A, expUnitFP32, dataMaskB32); // s
                    Reg::Sub(exp0B, exp0B, expUnitFP32, dataMaskB32);
                    Reg::Mul(evenZeroFp32, evenZeroFp32, (Reg::RegTensor<float>&)exp1A, dataMaskB32);
                    Reg::Mul(evenOneFp32, evenOneFp32, (Reg::RegTensor<float>&)exp1B, dataMaskB32);
                    Reg::Compares<float, CMPMODE::LT>(specialMaskA, evenZeroFp32, 0, dataMaskB32);
                    Reg::Compares<float, CMPMODE::LT>(specialMaskB, evenOneFp32, 0, dataMaskB32);
                    Reg::Truncate<float, roundMode>(evenZeroFp32, evenZeroFp32, dataMaskB32);
                    Reg::Truncate<float, roundMode>(evenOneFp32, evenOneFp32, dataMaskB32);
                    Reg::Mul(evenZeroFp32, evenZeroFp32, (Reg::RegTensor<float>&)exp0A, dataMaskB32);
                    Reg::Mul(evenOneFp32, evenOneFp32, (Reg::RegTensor<float>&)exp0B, dataMaskB32);
                }

                // Handle negative zero.  Reuse zeroMask between the two streams
                // to save one predicate register; the main chain is unchanged.
                Reg::Compares<float, CMPMODE::EQ>(zeroMask, evenZeroFp32, 0, dataMaskB32);
                Reg::And(zeroMask, specialMaskA, zeroMask, dataMaskB32);
                Reg::Or(zeroMask, negInfMaskA, zeroMask, dataMaskB32);
                Reg::Select<int32_t>((Reg::RegTensor<int32_t>&)evenZeroFp32, negZero,
                                     (Reg::RegTensor<int32_t>&)evenZeroFp32, zeroMask);
                Reg::Compares<float, CMPMODE::EQ>(zeroMask, evenOneFp32, 0, dataMaskB32);
                Reg::And(zeroMask, specialMaskB, zeroMask, dataMaskB32);
                Reg::Or(zeroMask, negInfMaskB, zeroMask, dataMaskB32);
                Reg::Select<int32_t>((Reg::RegTensor<int32_t>&)evenOneFp32, negZero,
                                     (Reg::RegTensor<int32_t>&)evenOneFp32, zeroMask);
            }

            // Pair 1: oddZeroFp32 + oddOneFp32, stage-major.
            {
                Reg::MaskReg negInfMaskA;
                Reg::MaskReg specialMaskA;
                Reg::MaskReg negInfMaskB;
                Reg::MaskReg specialMaskB;
                Reg::MaskReg zeroMask;
                Reg::RegTensor<int32_t> exp0A;
                Reg::RegTensor<int32_t> exp1A;
                Reg::RegTensor<int32_t> exp0B;
                Reg::RegTensor<int32_t> exp1B;

                Reg::Compare<int32_t, CMPMODE::EQ>(negInfMaskA, (Reg::RegTensor<int32_t>&)oddZeroFp32, negZero,
                                                   dataMaskB32);
                Reg::Compare<int32_t, CMPMODE::EQ>(negInfMaskB, (Reg::RegTensor<int32_t>&)oddOneFp32, negZero,
                                                   dataMaskB32);
                if constexpr (IsSameType<rowDataDtype, fp4x2_e1m2_t>::value) {
                    // E1M2: multiply by 4, truncate, divide by 4
                    Reg::Muls(oddZeroFp32, oddZeroFp32, FOUR, dataMaskB32);
                    Reg::Muls(oddOneFp32, oddOneFp32, FOUR, dataMaskB32);
                    Reg::Compares<float, CMPMODE::LT>(specialMaskA, oddZeroFp32, 0, dataMaskB32);
                    Reg::Compares<float, CMPMODE::LT>(specialMaskB, oddOneFp32, 0, dataMaskB32);
                    Reg::Truncate<float, roundMode>(oddZeroFp32, oddZeroFp32, dataMaskB32);
                    Reg::Truncate<float, roundMode>(oddOneFp32, oddOneFp32, dataMaskB32);
                    Reg::Muls(oddZeroFp32, oddZeroFp32, ONE_FOURTH, dataMaskB32);
                    Reg::Muls(oddOneFp32, oddOneFp32, ONE_FOURTH, dataMaskB32);
                } else {
                    // E2M1: same in-place exponent-field alignment as Pair 0.
                    Reg::And(exp0A, (Reg::RegTensor<int32_t>&)oddZeroFp32, expMaskFP32, dataMaskB32);
                    Reg::And(exp0B, (Reg::RegTensor<int32_t>&)oddOneFp32, expMaskFP32, dataMaskB32);
                    Reg::Max(exp0A, exp0A, expZeroFP32, dataMaskB32);
                    Reg::Max(exp0B, exp0B, expZeroFP32, dataMaskB32);
                    Reg::Sub(exp1A, expMaskFP32, exp0A, dataMaskB32); // 1/s
                    Reg::Sub(exp1B, expMaskFP32, exp0B, dataMaskB32);
                    Reg::Sub(exp0A, exp0A, expUnitFP32, dataMaskB32); // s
                    Reg::Sub(exp0B, exp0B, expUnitFP32, dataMaskB32);
                    Reg::Mul(oddZeroFp32, oddZeroFp32, (Reg::RegTensor<float>&)exp1A, dataMaskB32);
                    Reg::Mul(oddOneFp32, oddOneFp32, (Reg::RegTensor<float>&)exp1B, dataMaskB32);
                    Reg::Compares<float, CMPMODE::LT>(specialMaskA, oddZeroFp32, 0, dataMaskB32);
                    Reg::Compares<float, CMPMODE::LT>(specialMaskB, oddOneFp32, 0, dataMaskB32);
                    Reg::Truncate<float, roundMode>(oddZeroFp32, oddZeroFp32, dataMaskB32);
                    Reg::Truncate<float, roundMode>(oddOneFp32, oddOneFp32, dataMaskB32);
                    Reg::Mul(oddZeroFp32, oddZeroFp32, (Reg::RegTensor<float>&)exp0A, dataMaskB32);
                    Reg::Mul(oddOneFp32, oddOneFp32, (Reg::RegTensor<float>&)exp0B, dataMaskB32);
                }

                // Handle negative zero.  Reuse zeroMask between the two streams
                // to save one predicate register; the main chain is unchanged.
                Reg::Compares<float, CMPMODE::EQ>(zeroMask, oddZeroFp32, 0, dataMaskB32);
                Reg::And(zeroMask, specialMaskA, zeroMask, dataMaskB32);
                Reg::Or(zeroMask, negInfMaskA, zeroMask, dataMaskB32);
                Reg::Select<int32_t>((Reg::RegTensor<int32_t>&)oddZeroFp32, negZero,
                                     (Reg::RegTensor<int32_t>&)oddZeroFp32, zeroMask);
                Reg::Compares<float, CMPMODE::EQ>(zeroMask, oddOneFp32, 0, dataMaskB32);
                Reg::And(zeroMask, specialMaskB, zeroMask, dataMaskB32);
                Reg::Or(zeroMask, negInfMaskB, zeroMask, dataMaskB32);
                Reg::Select<int32_t>((Reg::RegTensor<int32_t>&)oddOneFp32, negZero,
                                     (Reg::RegTensor<int32_t>&)oddOneFp32, zeroMask);
            }

            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(evenZeroBf16, evenZeroFp32, dataMaskB32);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(evenOneBf16, evenOneFp32, dataMaskB32);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(oddZeroBf16, oddZeroFp32, dataMaskB32);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(oddOneBf16, oddOneFp32, dataMaskB32);

            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)evenZeroBf16,
                                                                    (Reg::RegTensor<uint32_t>&)evenZeroBf16);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)evenOneBf16,
                                                                    (Reg::RegTensor<uint32_t>&)evenOneBf16);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)oddZeroBf16,
                                                                    (Reg::RegTensor<uint32_t>&)oddZeroBf16);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)oddOneBf16,
                                                                    (Reg::RegTensor<uint32_t>&)oddOneBf16);

            Reg::Interleave(evenZeroBf16, evenOneBf16, evenZeroBf16, evenOneBf16);
            Reg::Interleave(oddZeroBf16, oddOneBf16, oddZeroBf16, oddOneBf16);
            Reg::Interleave(evenZeroBf16, oddZeroBf16, evenZeroBf16, oddZeroBf16);

            Reg::Cast<rowDataDtype, bfloat16_t, castTraitBF16toFp4>(evenFp4, evenZeroBf16, dataMaskB16);
            Reg::Cast<rowDataDtype, bfloat16_t, castTraitBF16toFp4>(oddFp4, oddZeroBf16, dataMaskB16);
        } else {
            // BF16 input path, unchanged.
            Reg::Mul(evenData, evenData, (Reg::RegTensor<xDtype>&)scaleForMulFP16, dataMaskB16);
            Reg::Mul(oddData, oddData, (Reg::RegTensor<xDtype>&)scaleForMulFP16, dataMaskB16);
            Reg::Interleave(evenData, oddData, evenData, oddData);
            Reg::Cast<rowDataDtype, xDtype, castTraitBF16toFp4>(evenFp4, evenData, dataMaskB16);
            Reg::Cast<rowDataDtype, xDtype, castTraitBF16toFp4>(oddFp4, oddData, dataMaskB16);
        }

        Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
            rowDataAddr, (Reg::RegTensor<uint8_t>&)evenFp4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
        Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
            rowDataAddr, (Reg::RegTensor<uint8_t>&)oddFp4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
    }
    return;
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__simd_vf__ inline void ComputeInterleaveVF(__ubuf__ uint8_t* dstAddr, __ubuf__ uint8_t* src0Addr,
                                            __ubuf__ uint8_t* src1Addr, bool hasSecondSlot)
{
    Reg::RegTensor<uint8_t> src0Reg;
    Reg::RegTensor<uint8_t> src1Reg;
    Reg::MaskReg maskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    Reg::LoadAlign(src0Reg, src0Addr);
    if (hasSecondSlot) {
        Reg::LoadAlign(src1Reg, src1Addr);
    } else {
        Reg::Duplicate(src1Reg, static_cast<uint8_t>(0));
    }
    Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_INTLV_B8>(dstAddr, src0Reg, src1Reg, maskB8);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__simd_vf__ inline void ComputeScaleOcpVF(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                          __ubuf__ uint8_t* rowScaleAddr, __ubuf__ uint16_t* rowScaleReciprocalAddr,
                                          __ubuf__ uint8_t* colScaleAddr, __ubuf__ uint16_t* colScaleReciprocalAddr,
                                          int64_t outputDtypeMaxExpIn)
{
    uint16_t outputDtypeMaxExp = outputDtypeMaxExpIn;

    Reg::RegTensor<xDtype> evenData;
    Reg::RegTensor<xDtype> oddData;
    Reg::RegTensor<bfloat16_t> evenBf16;
    Reg::RegTensor<bfloat16_t> oddBf16;
    Reg::RegTensor<uint16_t> evenExpBf16;
    Reg::RegTensor<uint16_t> oddExpBf16;
    Reg::RegTensor<uint16_t> expMaskBF16;
    Reg::RegTensor<uint16_t> expMaxDim1;
    Reg::RegTensor<uint16_t> colEvenMaxExp;
    Reg::RegTensor<uint16_t> colOddMaxExp;
    Reg::RegTensor<uint16_t> yMaxExp;
    Reg::RegTensor<uint16_t> nanE8M0;
    Reg::RegTensor<uint16_t> biasE8M0;
    Reg::RegTensor<uint16_t> zero;
    Reg::RegTensor<uint16_t> nanBF16;
    Reg::RegTensor<uint16_t> specialExp;
    Reg::RegTensor<uint16_t> mxScale1B16;
    Reg::RegTensor<uint8_t> mxScale1B8;
    Reg::RegTensor<uint16_t> reversedShareExp1;

    Reg::RegTensor<uint16_t> mxScale2ZeroB16;
    Reg::RegTensor<uint8_t> mxScale2ZeroB8;
    Reg::RegTensor<uint16_t> reversedShareExp2Zero;
    Reg::RegTensor<uint16_t> mxScale2OneB16;
    Reg::RegTensor<uint8_t> mxScale2OneB8;
    Reg::RegTensor<uint16_t> reversedShareExp2One;

    Reg::MaskReg infMask;
    Reg::MaskReg zeroMask;
    Reg::MaskReg invalidDataMask;
    Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();

    Reg::MaskReg maskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskReduceB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL8>();
    Reg::MaskReg maskReduceB16 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL16>();

    Reg::Duplicate(expMaskBF16, EXP_MASK_BF16);
    Reg::Duplicate(colEvenMaxExp, static_cast<uint16_t>(0));
    Reg::Duplicate(colOddMaxExp, static_cast<uint16_t>(0));
    Reg::Duplicate(yMaxExp, outputDtypeMaxExp);
    Reg::Duplicate(nanE8M0, NAN_FOR_FP8_E8M0);
    Reg::Duplicate(biasE8M0, BF16_EXP_BIAS);
    Reg::Duplicate(zero, static_cast<uint16_t>(0));
    Reg::Duplicate(nanBF16, NAN_CUSTOMIZATION);
    Reg::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);

    for (uint16_t i = 0; i < blockCount; i++) {
        // Interleaved load: splits blockW bf16/fp16 elements into even/odd halves
        Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(evenData, oddData,
                                                                                                   xAddr, ONCE_ROW_LEN);
        if constexpr (IsSameType<xDtype, half>::value) {
            // CAST_TRUNC preserves FP16 Inf/NaN as a BF16 all-ones exponent,
            // so the exponent can be extracted directly after conversion.
            Reg::Cast<bfloat16_t, xDtype, CAST_HALF_TO_BF16>(evenBf16, evenData, maskAll);
            Reg::Cast<bfloat16_t, xDtype, CAST_HALF_TO_BF16>(oddBf16, oddData, maskAll);
            Reg::And(evenExpBf16, (Reg::RegTensor<uint16_t>&)evenBf16, expMaskBF16, maskAll);
            Reg::And(oddExpBf16, (Reg::RegTensor<uint16_t>&)oddBf16, expMaskBF16, maskAll);
        } else {
            // BF16 path: extract exponent bits directly
            Reg::And(evenExpBf16, (Reg::RegTensor<uint16_t>&)evenData, expMaskBF16, maskAll);
            Reg::And(oddExpBf16, (Reg::RegTensor<uint16_t>&)oddData, expMaskBF16, maskAll);
        }

        // axis=-1: max of adjacent pair exponents
        Reg::Max(expMaxDim1, evenExpBf16, oddExpBf16, maskAll);
        Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(expMaxDim1, expMaxDim1, maskAll);

        // axis=-2: accumulate column-wise max exponents across rows
        Reg::Max(colEvenMaxExp, colEvenMaxExp, evenExpBf16, maskAll);
        Reg::Max(colOddMaxExp, colOddMaxExp, oddExpBf16, maskAll);

        // ---- axis=-1 scale computation ----
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxDim1, expMaskBF16, maskAll);
        // Test the original maximum, not the clamped shared exponent.  A
        // nonzero block below yMaxExp is clamped to E8M0 code 0 (scale 2^-127)
        // and still needs the corresponding 2^127 reciprocal for quantization.
        Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxDim1, zero, maskAll);
        Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMaxDim1, yMaxExp, maskAll);
        Reg::Select<uint16_t>(expMaxDim1, yMaxExp, expMaxDim1, invalidDataMask);

        Reg::Sub(expMaxDim1, expMaxDim1, yMaxExp, maskAll);
        Reg::ShiftRights(mxScale1B16, expMaxDim1, SHR_NUM_FOR_BF16, maskAll);
        Reg::Select<uint16_t>(mxScale1B16, mxScale1B16, nanE8M0, infMask);
        Reg::Select<uint16_t>(mxScale1B16, mxScale1B16, zero, zeroMask);

        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale1B8, mxScale1B16);
        Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(rowScaleAddr, mxScale1B8, UB_BLOCK_SIZE,
                                                                     maskReduceB8);
        // Compute 1/scale
        Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMaxDim1, biasE8M0, maskAll);

        Reg::Sub(reversedShareExp1, biasE8M0, expMaxDim1, maskAll);
        Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, nanBF16, infMask);
        Reg::Select<uint16_t>(reversedShareExp1, reversedShareExp1, zero, zeroMask);
        Reg::Select<uint16_t>(reversedShareExp1, specialExp, reversedShareExp1, invalidDataMask);
        Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(rowScaleReciprocalAddr, reversedShareExp1,
                                                                      ROW_SCALE_RECIPROCAL_ROW_ELEMS, maskReduceB16);
    }

    // ---- axis=-2 scale computation (interleaved part 1: even rows) ----
    Reg::Compare<uint16_t, CMPMODE::NE>(infMask, colEvenMaxExp, expMaskBF16, maskAll);
    Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, colEvenMaxExp, zero, maskAll);
    Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, colEvenMaxExp, yMaxExp, maskAll);
    Reg::Select<uint16_t>(colEvenMaxExp, yMaxExp, colEvenMaxExp, invalidDataMask);
    Reg::Sub(colEvenMaxExp, colEvenMaxExp, yMaxExp, maskAll);
    Reg::ShiftRights(mxScale2ZeroB16, colEvenMaxExp, SHR_NUM_FOR_BF16, maskAll);
    Reg::Select<uint16_t>(mxScale2ZeroB16, mxScale2ZeroB16, nanE8M0, infMask);
    Reg::Select<uint16_t>(mxScale2ZeroB16, mxScale2ZeroB16, zero, zeroMask);

    Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale2ZeroB8, mxScale2ZeroB16);

    Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, colEvenMaxExp, biasE8M0, maskAll);

    Reg::Sub(reversedShareExp2Zero, biasE8M0, colEvenMaxExp, maskAll);
    Reg::Select<uint16_t>(reversedShareExp2Zero, reversedShareExp2Zero, nanBF16, infMask);
    Reg::Select<uint16_t>(reversedShareExp2Zero, reversedShareExp2Zero, zero, zeroMask);
    Reg::Select<uint16_t>(reversedShareExp2Zero, specialExp, reversedShareExp2Zero, invalidDataMask);

    // ---- axis=-2 scale computation (interleaved part 2: odd rows) ----
    Reg::Compare<uint16_t, CMPMODE::NE>(infMask, colOddMaxExp, expMaskBF16, maskAll);
    Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, colOddMaxExp, zero, maskAll);
    Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, colOddMaxExp, yMaxExp, maskAll);
    Reg::Select<uint16_t>(colOddMaxExp, yMaxExp, colOddMaxExp, invalidDataMask);
    Reg::Sub(colOddMaxExp, colOddMaxExp, yMaxExp, maskAll);
    Reg::ShiftRights(mxScale2OneB16, colOddMaxExp, SHR_NUM_FOR_BF16, maskAll);
    Reg::Select<uint16_t>(mxScale2OneB16, mxScale2OneB16, nanE8M0, infMask);
    Reg::Select<uint16_t>(mxScale2OneB16, mxScale2OneB16, zero, zeroMask);

    Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale2OneB8, mxScale2OneB16);

    Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, colOddMaxExp, biasE8M0, maskAll);
    Reg::Sub(reversedShareExp2One, biasE8M0, colOddMaxExp, maskAll);
    Reg::Select<uint16_t>(reversedShareExp2One, reversedShareExp2One, nanBF16, infMask);
    Reg::Select<uint16_t>(reversedShareExp2One, reversedShareExp2One, zero, zeroMask);
    Reg::Select<uint16_t>(reversedShareExp2One, specialExp, reversedShareExp2One, invalidDataMask);

    // Interleaved store: merge even/odd scale and 1/scale for axis=-2
    Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_INTLV_B8>(colScaleAddr, mxScale2ZeroB8, mxScale2OneB8, maskB8);
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(colScaleReciprocalAddr, reversedShareExp2Zero,
                                                              reversedShareExp2One, maskAll);
}

template <typename xDtype>
__simd_callee__ inline void ComputeCuBLASSecondLastSlot(
    __ubuf__ uint16_t* slotAddr, Reg::RegTensor<uint8_t>& scale8Slot, Reg::RegTensor<uint32_t>& invMax,
    Reg::RegTensor<uint32_t>& manMaskReg, Reg::RegTensor<uint32_t>& expMaskReg, Reg::RegTensor<uint32_t>& zero32Reg,
    Reg::RegTensor<uint32_t>& scaleBiasReg, Reg::RegTensor<uint32_t>& nan32Reg, Reg::RegTensor<uint32_t>& fp8Nan32Reg,
    Reg::MaskReg& maskAll, Reg::MaskReg& maskAll32, Reg::MaskReg& maskB16)
{
    Reg::RegTensor<uint16_t> max16Reg;
    Reg::RegTensor<uint32_t> max32Reg;
    Reg::RegTensor<uint32_t> exp32Reg;
    Reg::RegTensor<uint32_t> man32Reg;
    Reg::RegTensor<uint32_t> expOne32Reg;
    Reg::RegTensor<uint32_t> extractExp;
    Reg::RegTensor<uint32_t> halfScale;
    Reg::RegTensor<uint16_t> scaleBf16;
    Reg::RegTensor<uint16_t> recip16;
    Reg::MaskReg cmpResult;
    Reg::MaskReg zeroMask;
    Reg::MaskReg p0;
    Reg::MaskReg p1;

    // The reciprocal is stored back over the very 32B slot the maximum was read from,
    // so keep a separate write base: slotAddr is post-incremented by the load.
    __ubuf__ uint16_t* slotWriteAddr = slotAddr;
    Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(max16Reg, slotAddr,
                                                                                                 VF_LEN_FP32);
    Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>((Reg::RegTensor<float>&)max32Reg, (Reg::RegTensor<xDtype>&)max16Reg,
                                                  maskAll);
    Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32Reg, expMaskReg, maskAll32);
    Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32Reg, zero32Reg, maskAll32);
    Reg::Mul((Reg::RegTensor<float>&)max32Reg, (Reg::RegTensor<float>&)max32Reg, (Reg::RegTensor<float>&)invMax,
             maskAll32);
    Reg::ShiftRights(exp32Reg, max32Reg, SHR_NUM_FOR_FP32, maskAll32);
    Reg::And(man32Reg, max32Reg, manMaskReg, maskAll32);
    Reg::Compares<uint32_t, CMPMODE::GT>(p0, exp32Reg, static_cast<uint32_t>(0), maskAll32);
    Reg::Compares<uint32_t, CMPMODE::LT>(p0, exp32Reg, EXP_254, p0);
    Reg::Compares<uint32_t, CMPMODE::GT>(p0, man32Reg, static_cast<uint32_t>(0), p0);
    Reg::Compares<uint32_t, CMPMODE::EQ>(p1, exp32Reg, static_cast<uint32_t>(0), maskAll32);
    Reg::Compares<uint32_t, CMPMODE::GT>(p1, man32Reg, HALF_FOR_MAN, p1);
    Reg::Or(p0, p0, p1, maskAll32);
    Reg::Adds(expOne32Reg, exp32Reg, 1, maskAll32);
    Reg::Select(extractExp, expOne32Reg, exp32Reg, p0);
    Reg::Select<uint32_t>(extractExp, extractExp, fp8Nan32Reg, cmpResult);
    Reg::Select<uint32_t>(extractExp, extractExp, zero32Reg, zeroMask);
    Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scaleBf16, extractExp);
    Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(scale8Slot, scaleBf16);

    Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskAll32);
    Reg::Sub(halfScale, scaleBiasReg, extractExp, maskAll32);
    Reg::Select<uint32_t>(halfScale, halfScale, nan32Reg, cmpResult);
    Reg::Select<uint32_t>(halfScale, halfScale, zero32Reg, zeroMask);
    Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(recip16, halfScale);
    Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(slotWriteAddr, recip16, VF_LEN_FP32, maskB16);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__simd_vf__ inline void ComputeScaleCuBLASSecondLastVF(uint16_t dataLen, uint32_t outputDtypeInvMax,
                                                       __ubuf__ uint16_t* colScaleReciprocalAddr,
                                                       __ubuf__ uint8_t* colScaleAddr)
{
    uint16_t times = dataLen / VF_LEN_FP32;

    Reg::RegTensor<uint32_t> invMax;
    Reg::RegTensor<uint32_t> manMaskReg;
    Reg::RegTensor<uint32_t> expMaskReg;
    Reg::RegTensor<uint32_t> zero32Reg;
    Reg::RegTensor<uint32_t> scaleBiasReg;
    Reg::RegTensor<uint32_t> nan32Reg;
    Reg::RegTensor<uint32_t> fp8Nan32Reg;
    Reg::RegTensor<uint8_t> scale8Slot0;
    Reg::RegTensor<uint8_t> scale8Slot1;
    Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::VL64>();
    Reg::MaskReg interleaveMask = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    Reg::Duplicate(scaleBiasReg, FP32_EXP_BIAS_CUBLAS);
    Reg::Duplicate(expMaskReg, MAX_EXP_FOR_FP32);
    Reg::Duplicate(zero32Reg, static_cast<uint32_t>(0));
    Reg::Duplicate(invMax, outputDtypeInvMax);
    Reg::Duplicate(manMaskReg, MAN_MASK_FLOAT);
    Reg::Duplicate(fp8Nan32Reg, MAX_EXP_FOR_FP8_IN_FP32);
    Reg::Duplicate(nan32Reg, static_cast<uint32_t>(NAN_CUSTOMIZATION));
    for (uint16_t i = 0; i < times; i++) {
        uint16_t colOffset = i * VF_LEN_FP32;
        __ubuf__ uint16_t* slot0Addr = colScaleReciprocalAddr + colOffset;
        __ubuf__ uint16_t* slot1Addr = colScaleReciprocalAddr + dataLen + colOffset;
        ComputeCuBLASSecondLastSlot<xDtype>(slot0Addr, scale8Slot0, invMax, manMaskReg, expMaskReg, zero32Reg,
                                            scaleBiasReg, nan32Reg, fp8Nan32Reg, maskAll, maskAll32, maskB16);
        ComputeCuBLASSecondLastSlot<xDtype>(slot1Addr, scale8Slot1, invMax, manMaskReg, expMaskReg, zero32Reg,
                                            scaleBiasReg, nan32Reg, fp8Nan32Reg, maskAll, maskAll32, maskB16);
        Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_INTLV_B8>(colScaleAddr + DIGIT_TWO * colOffset, scale8Slot0,
                                                                scale8Slot1, interleaveMask);
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__simd_vf__ inline void ComputeScaleCuBLASVF(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                             __ubuf__ uint8_t* rowDataAddr, __ubuf__ uint8_t* rowScaleAddr,
                                             __ubuf__ uint16_t* rowScaleReciprocalAddr, __ubuf__ uint8_t* colScaleAddr,
                                             __ubuf__ uint16_t* colScaleReciprocalAddr, uint32_t outputDtypeInvMaxIn)
{
    uint32_t outputDtypeInvMax = outputDtypeInvMaxIn;
    // Phase 1 appends one compacted block maximum per row through this cursor.
    // The col-scale reciprocal buffer is only temporarily reused as absmax scratch.
    __ubuf__ uint16_t* absMaxWriteAddr = colScaleReciprocalAddr;
    __ubuf__ xDtype* dataStartAddr = xAddr;
    __ubuf__ uint16_t* rowScaleReciprocalStartAddr = rowScaleReciprocalAddr;

    Reg::RegTensor<xDtype> evenData;
    Reg::RegTensor<xDtype> oddData;
    Reg::RegTensor<uint16_t> evenAbs;
    Reg::RegTensor<uint16_t> oddAbs;
    Reg::RegTensor<uint16_t> absMaxDim1;
    Reg::RegTensor<uint16_t> colScaleSlot0Part0;
    Reg::RegTensor<uint16_t> colScaleSlot0Part1;
    Reg::RegTensor<uint16_t> colScaleSlot1Part0;
    Reg::RegTensor<uint16_t> colScaleSlot1Part1;
    Reg::RegTensor<uint32_t> max32;
    Reg::RegTensor<uint32_t> exp32;
    Reg::RegTensor<uint32_t> man32;
    Reg::RegTensor<uint32_t> expAddOne32;
    Reg::RegTensor<uint32_t> extractExp;
    Reg::RegTensor<uint32_t> halfScale;
    Reg::RegTensor<uint16_t> scaleBf16;
    Reg::RegTensor<uint8_t> scaleU8Reg;
    Reg::RegTensor<uint8_t> scaleU8Row;
    Reg::RegTensor<uint16_t> reciprocalBf16Reg;
    Reg::RegTensor<uint16_t> reciprocalBf16Row;
    Reg::RegTensor<int8_t> extractIdx;
    Reg::RegTensor<uint16_t> absMask;
    Reg::RegTensor<uint32_t> invMax;
    Reg::RegTensor<uint32_t> manMaskReg;
    Reg::RegTensor<uint32_t> expMaskReg;
    Reg::RegTensor<uint32_t> zeroReg32;
    Reg::RegTensor<uint32_t> scaleBiasReg;
    Reg::RegTensor<uint32_t> nanReg32;
    Reg::RegTensor<uint32_t> fp8NanReg32;
    Reg::MaskReg cmpResult;
    Reg::MaskReg zeroMask;
    Reg::MaskReg p0;
    Reg::MaskReg p1;
    Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskReduceB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL8>();
    Reg::MaskReg maskReduceB16 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL16>();
    Reg::UnalignReg ureg;

    Reg::Duplicate(absMask, ABS_MASK_FOR_16BIT);
    Reg::Duplicate(colScaleSlot0Part0, static_cast<uint16_t>(0));
    Reg::Duplicate(colScaleSlot0Part1, static_cast<uint16_t>(0));
    Reg::Duplicate(colScaleSlot1Part0, static_cast<uint16_t>(0));
    Reg::Duplicate(colScaleSlot1Part1, static_cast<uint16_t>(0));

    uint16_t scaleCount = dataLen / BLOCK_SIZE;
    uint16_t totalScaleVf = static_cast<uint16_t>(blockCount * scaleCount);
    uint16_t tailScaleCount = totalScaleVf % static_cast<uint16_t>(VF_LEN_FP32);
    if (tailScaleCount != 0) {
        Reg::RegTensor<uint16_t> zeroTailBatch;
        Reg::MaskReg tailBatchMask = Reg::CreateMask<uint16_t, Reg::MaskPattern::VL64>();
        Reg::Duplicate(zeroTailBatch, static_cast<uint16_t>(0));
        __ubuf__ uint16_t* tailBatchAddr = colScaleReciprocalAddr + totalScaleVf - tailScaleCount;
        Reg::StoreAlign<uint16_t>(tailBatchAddr, zeroTailBatch, tailBatchMask);
    }

    // Phase 1: collect row-scale maxima compactly while preserving the two
    // independent 32-row col-scale slots.
    uint16_t slot0Rows = blockCount < BLOCK_SIZE ? blockCount : BLOCK_SIZE;
    for (uint16_t i = 0; i < slot0Rows; i++) {
        Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(evenData, oddData,
                                                                                                   xAddr, dataLen);
        Reg::And(evenAbs, (Reg::RegTensor<uint16_t>&)evenData, absMask, maskAll);
        Reg::And(oddAbs, (Reg::RegTensor<uint16_t>&)oddData, absMask, maskAll);
        Reg::Max(absMaxDim1, evenAbs, oddAbs, maskAll);
        Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(absMaxDim1, absMaxDim1, maskAll);
        Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(absMaxWriteAddr, absMaxDim1, ureg,
                                                                        dataLen / BLOCK_SIZE);
        Reg::Max(colScaleSlot0Part0, colScaleSlot0Part0, evenAbs, maskAll);
        Reg::Max(colScaleSlot0Part1, colScaleSlot0Part1, oddAbs, maskAll);
    }
    for (uint16_t i = slot0Rows; i < blockCount; i++) {
        Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(evenData, oddData,
                                                                                                   xAddr, dataLen);
        Reg::And(evenAbs, (Reg::RegTensor<uint16_t>&)evenData, absMask, maskAll);
        Reg::And(oddAbs, (Reg::RegTensor<uint16_t>&)oddData, absMask, maskAll);
        Reg::Max(absMaxDim1, evenAbs, oddAbs, maskAll);
        Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(absMaxDim1, absMaxDim1, maskAll);
        Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(absMaxWriteAddr, absMaxDim1, ureg,
                                                                        dataLen / BLOCK_SIZE);
        Reg::Max(colScaleSlot1Part0, colScaleSlot1Part0, evenAbs, maskAll);
        Reg::Max(colScaleSlot1Part1, colScaleSlot1Part1, oddAbs, maskAll);
    }
    Reg::StoreUnAlignPost(absMaxWriteAddr, ureg, 0);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    Reg::Duplicate(invMax, outputDtypeInvMax);
    Reg::Duplicate(manMaskReg, MAN_MASK_FLOAT);
    Reg::Duplicate(expMaskReg, MAX_EXP_FOR_FP32);
    Reg::Duplicate(zeroReg32, static_cast<uint32_t>(0));
    Reg::Duplicate(scaleBiasReg, FP32_EXP_BIAS_CUBLAS);
    Reg::Duplicate(nanReg32, static_cast<uint32_t>(NAN_CUSTOMIZATION));
    Reg::Duplicate(fp8NanReg32, MAX_EXP_FOR_FP8_IN_FP32);

    // Phase 2: convert 64 cached maxima per VF, then scatter eight rows
    // back to the original 32-byte-aligned scale/reciprocal layout.
    __ubuf__ uint16_t* absMaxReadAddr = colScaleReciprocalAddr;
    uint16_t batchCount = static_cast<uint16_t>((totalScaleVf + static_cast<uint16_t>(VF_LEN_FP32) - 1) /
                                                static_cast<uint16_t>(VF_LEN_FP32));
    for (uint16_t j = 0; j < batchCount; j++) {
        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
            absMaxDim1, absMaxReadAddr, VF_LEN_FP32);
        Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>((Reg::RegTensor<float>&)max32,
                                                      (Reg::RegTensor<xDtype>&)absMaxDim1, maskAll);
        Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMaskReg, maskAll32);
        Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroReg32, maskAll32);
        Reg::Mul((Reg::RegTensor<float>&)max32, (Reg::RegTensor<float>&)max32, (Reg::RegTensor<float>&)invMax,
                 maskAll32);
        Reg::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, maskAll32);
        Reg::And(man32, max32, manMaskReg, maskAll32);
        Reg::Compares<uint32_t, CMPMODE::GT>(p0, exp32, static_cast<uint32_t>(0), maskAll32);
        Reg::Compares<uint32_t, CMPMODE::LT>(p0, exp32, EXP_254, p0);
        Reg::Compares<uint32_t, CMPMODE::GT>(p0, man32, static_cast<uint32_t>(0), p0);
        Reg::Compares<uint32_t, CMPMODE::EQ>(p1, exp32, static_cast<uint32_t>(0), maskAll32);
        Reg::Compares<uint32_t, CMPMODE::GT>(p1, man32, HALF_FOR_MAN, p1);
        Reg::Or(p0, p0, p1, maskAll32);
        Reg::Adds(expAddOne32, exp32, 1, maskAll32);
        Reg::Select(extractExp, expAddOne32, exp32, p0);
        Reg::Select<uint32_t>(extractExp, extractExp, fp8NanReg32, cmpResult);
        Reg::Select<uint32_t>(extractExp, extractExp, zeroReg32, zeroMask);
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scaleBf16, extractExp);
        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(scaleU8Reg, scaleBf16);
        Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskAll32);
        Reg::Sub(halfScale, scaleBiasReg, extractExp, maskAll32);
        Reg::Select<uint32_t>(halfScale, halfScale, nanReg32, cmpResult);
        Reg::Select<uint32_t>(halfScale, halfScale, zeroReg32, zeroMask);
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(reciprocalBf16Reg, halfScale);

        for (uint16_t k = 0; k < 8; k++) {
            Reg::Arange(extractIdx, static_cast<int8_t>(k * 8));
            Reg::Gather(scaleU8Row, scaleU8Reg, (Reg::RegTensor<uint8_t>&)extractIdx);
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(rowScaleAddr, scaleU8Row, UB_BLOCK_SIZE,
                                                                         maskReduceB8);
            Reg::Arange(extractIdx, static_cast<int8_t>(k * 16));
            Reg::Gather((Reg::RegTensor<uint8_t>&)reciprocalBf16Row, (Reg::RegTensor<uint8_t>&)reciprocalBf16Reg,
                        (Reg::RegTensor<uint8_t>&)extractIdx);
            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                rowScaleReciprocalAddr, reciprocalBf16Row, ROW_SCALE_RECIPROCAL_ROW_ELEMS, maskReduceB16);
        }
    }
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    // Generate row_data immediately after row_scale while staying in the same vector scope.
    Reg::MaskReg maskAllB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint16_t> scaleForMulFP16;
    Reg::RegTensor<float> scaleForMulFP32;
    Reg::RegTensor<float> evenZeroFp32;
    Reg::RegTensor<float> evenOneFp32;
    Reg::RegTensor<float> oddZeroFp32;
    Reg::RegTensor<float> oddOneFp32;
    Reg::RegTensor<rowDataDtype> evenZeroFp8;
    Reg::RegTensor<rowDataDtype> evenOneFp8;
    Reg::RegTensor<rowDataDtype> oddZeroFp8;
    Reg::RegTensor<rowDataDtype> oddOneFp8;
    __ubuf__ xDtype* dataReadAddr = dataStartAddr;
    __ubuf__ uint16_t* rowScaleReciprocalReadAddr = rowScaleReciprocalStartAddr;

    for (uint16_t i = 0; i < blockCount; i++) {
        Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
            evenData, oddData, dataReadAddr, dataLen);
        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
            scaleForMulFP16, rowScaleReciprocalReadAddr, ROW_SCALE_RECIPROCAL_ROW_ELEMS);

        if constexpr (IsSameType<xDtype, half>::value) {
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(evenZeroFp32, evenData, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(evenOneFp32, evenData, maskAll);
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(scaleForMulFP32,
                                                              (Reg::RegTensor<bfloat16_t>&)scaleForMulFP16, maskAll);
            Reg::Mul(evenZeroFp32, evenZeroFp32, scaleForMulFP32, maskAll);
            Reg::Mul(evenOneFp32, evenOneFp32, scaleForMulFP32, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(oddZeroFp32, oddData, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(oddOneFp32, oddData, maskAll);
            Reg::Mul(oddZeroFp32, oddZeroFp32, scaleForMulFP32, maskAll);
            Reg::Mul(oddOneFp32, oddOneFp32, scaleForMulFP32, maskAll);
        } else {
            Reg::Mul(evenData, evenData, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
            Reg::Mul(oddData, oddData, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(evenZeroFp32, evenData, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(evenOneFp32, evenData, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(oddZeroFp32, oddData, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(oddOneFp32, oddData, maskAll);
        }
        Reg::Cast<rowDataDtype, float, CAST_32_TO_80>(evenZeroFp8, evenZeroFp32, maskAll);
        Reg::Cast<rowDataDtype, float, CAST_32_TO_82>(evenOneFp8, evenOneFp32, maskAll);
        Reg::Cast<rowDataDtype, float, CAST_32_TO_81>(oddZeroFp8, oddZeroFp32, maskAll);
        Reg::Cast<rowDataDtype, float, CAST_32_TO_83>(oddOneFp8, oddOneFp32, maskAll);
        Reg::Add((Reg::RegTensor<uint8_t>&)evenZeroFp8, (Reg::RegTensor<uint8_t>&)evenZeroFp8,
                 (Reg::RegTensor<uint8_t>&)evenOneFp8, maskAllB8);
        Reg::Add((Reg::RegTensor<uint8_t>&)evenZeroFp8, (Reg::RegTensor<uint8_t>&)evenZeroFp8,
                 (Reg::RegTensor<uint8_t>&)oddZeroFp8, maskAllB8);
        Reg::Add((Reg::RegTensor<uint8_t>&)evenZeroFp8, (Reg::RegTensor<uint8_t>&)evenZeroFp8,
                 (Reg::RegTensor<uint8_t>&)oddOneFp8, maskAllB8);
        Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
            rowDataAddr, (Reg::RegTensor<uint8_t>&)evenZeroFp8, dataLen, maskAllB8);
    }

    // Reuse the temporary area for the normal axis=-2 maxima output.
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(colScaleReciprocalAddr, colScaleSlot0Part0,
                                                              colScaleSlot0Part1, maskAll);
    Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(colScaleReciprocalAddr + dataLen, colScaleSlot1Part0,
                                                              colScaleSlot1Part1, maskAll);
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__simd_vf__ inline void ComputeColDataToFp8VF(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                              __ubuf__ uint16_t* colScaleReciprocalAddr, __ubuf__ uint8_t* colDataAddr,
                                              int64_t tileRowLength)
{
    int64_t rowLength = tileRowLength;
    constexpr uint32_t dualLoadLen = VF_LEN_B16 * DIGIT_TWO;

    Reg::RegTensor<xDtype> xEven;
    Reg::RegTensor<xDtype> xOdd;
    Reg::RegTensor<uint16_t> reciprocalEven;
    Reg::RegTensor<uint16_t> reciprocalOdd;
    Reg::RegTensor<float> reciprocalEvenFP32Layout0;
    Reg::RegTensor<float> reciprocalEvenFP32Layout1;
    Reg::RegTensor<float> reciprocalOddFP32Layout0;
    Reg::RegTensor<float> reciprocalOddFP32Layout1;
    Reg::RegTensor<float> xEvenFP32Layout0;
    Reg::RegTensor<float> xEvenFP32Layout1;
    Reg::RegTensor<float> xOddFP32Layout0;
    Reg::RegTensor<float> xOddFP32Layout1;
    Reg::RegTensor<rowDataDtype> yEvenFP8Layout0;
    Reg::RegTensor<rowDataDtype> yEvenFP8Layout2;
    Reg::RegTensor<rowDataDtype> yOddFP8Layout1;
    Reg::RegTensor<rowDataDtype> yOddFP8Layout3;

    Reg::MaskReg maskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskB16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    // Load all 256 reciprocal values once and reuse them for all 32 rows in this col-scale slot.
    Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
        reciprocalEven, reciprocalOdd, colScaleReciprocalAddr, dualLoadLen);
    if constexpr (IsSameType<xDtype, half>::value) {
        Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(reciprocalEvenFP32Layout0,
                                                          (Reg::RegTensor<bfloat16_t>&)reciprocalEven, maskB16);
        Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(reciprocalEvenFP32Layout1,
                                                         (Reg::RegTensor<bfloat16_t>&)reciprocalEven, maskB16);
        Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(reciprocalOddFP32Layout0,
                                                          (Reg::RegTensor<bfloat16_t>&)reciprocalOdd, maskB16);
        Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(reciprocalOddFP32Layout1,
                                                         (Reg::RegTensor<bfloat16_t>&)reciprocalOdd, maskB16);
    }

    for (uint16_t row = 0; row < blockCount; row++) {
        __ubuf__ xDtype* xCursor = xAddr + row * rowLength;
        Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(xEven, xOdd, xCursor,
                                                                                                   dualLoadLen);
        if constexpr (IsSameType<xDtype, half>::value) {
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xEvenFP32Layout0, xEven, maskB16);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xEvenFP32Layout1, xEven, maskB16);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xOddFP32Layout0, xOdd, maskB16);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xOddFP32Layout1, xOdd, maskB16);

            Reg::Mul(xEvenFP32Layout0, xEvenFP32Layout0, reciprocalEvenFP32Layout0, maskB32);
            Reg::Mul(xEvenFP32Layout1, xEvenFP32Layout1, reciprocalEvenFP32Layout1, maskB32);
            Reg::Mul(xOddFP32Layout0, xOddFP32Layout0, reciprocalOddFP32Layout0, maskB32);
            Reg::Mul(xOddFP32Layout1, xOddFP32Layout1, reciprocalOddFP32Layout1, maskB32);
        } else {
            Reg::Mul(xEven, xEven, (Reg::RegTensor<xDtype>&)reciprocalEven, maskB16);
            Reg::Mul(xOdd, xOdd, (Reg::RegTensor<xDtype>&)reciprocalOdd, maskB16);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xEvenFP32Layout0, xEven, maskB16);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xEvenFP32Layout1, xEven, maskB16);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(xOddFP32Layout0, xOdd, maskB16);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(xOddFP32Layout1, xOdd, maskB16);
        }

        Reg::Cast<rowDataDtype, float, CAST_32_TO_80>(yEvenFP8Layout0, xEvenFP32Layout0, maskB32);
        Reg::Cast<rowDataDtype, float, CAST_32_TO_82>(yEvenFP8Layout2, xEvenFP32Layout1, maskB32);
        Reg::Cast<rowDataDtype, float, CAST_32_TO_81>(yOddFP8Layout1, xOddFP32Layout0, maskB32);
        Reg::Cast<rowDataDtype, float, CAST_32_TO_83>(yOddFP8Layout3, xOddFP32Layout1, maskB32);

        Reg::Add((Reg::RegTensor<uint8_t>&)yEvenFP8Layout0, (Reg::RegTensor<uint8_t>&)yEvenFP8Layout0,
                 (Reg::RegTensor<uint8_t>&)yEvenFP8Layout2, maskB8);
        Reg::Add((Reg::RegTensor<uint8_t>&)yOddFP8Layout1, (Reg::RegTensor<uint8_t>&)yOddFP8Layout1,
                 (Reg::RegTensor<uint8_t>&)yOddFP8Layout3, maskB8);
        Reg::Add((Reg::RegTensor<uint8_t>&)yEvenFP8Layout0, (Reg::RegTensor<uint8_t>&)yEvenFP8Layout0,
                 (Reg::RegTensor<uint8_t>&)yOddFP8Layout1, maskB8);

        __ubuf__ uint8_t* yCursor = colDataAddr + row * rowLength;
        Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
            yCursor, (Reg::RegTensor<uint8_t>&)yEvenFP8Layout0, dualLoadLen, maskB8);
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__simd_vf__ inline void ComputeColDataToFp4VF(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                              __ubuf__ uint16_t* colScaleReciprocalAddr, __ubuf__ uint8_t* colDataAddr,
                                              int64_t tileRowLength)
{
    static constexpr Reg::CastTrait castTraitBF16toFp4 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                          Reg::MaskMergeMode::ZEROING, roundMode};
    static constexpr Reg::CastTrait castTraitFp32toBF16 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                           Reg::MaskMergeMode::ZEROING, roundMode};

    int64_t rowLength = tileRowLength;

    Reg::RegTensor<xDtype> x;
    Reg::RegTensor<bfloat16_t> evenBf16;
    Reg::RegTensor<bfloat16_t> oddBf16;
    Reg::RegTensor<bfloat16_t> xBF16;
    Reg::RegTensor<float> zeroHalfFp32;
    Reg::RegTensor<float> oneHalfFp32;
    Reg::RegTensor<uint16_t> reversedShareExp;
    Reg::RegTensor<float> reciprocalZeroFp32;
    Reg::RegTensor<float> reciprocalOneFp32;
    Reg::RegTensor<rowDataDtype> packedFp4;
    Reg::RegTensor<int32_t> negZero;

    Reg::MaskReg pregAll8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();

    Reg::Duplicate(negZero, NEG_ZERO);

    // Load per-column 1/scale
    Reg::LoadAlign<uint16_t, Reg::LoadDist::DIST_NORM>(reversedShareExp, colScaleReciprocalAddr);

    // FP4 packs two values per byte, so the packed column output of one
    // rowLength row occupies rowLength / 2 bytes.
    const uint32_t fp4RowBytes = static_cast<uint32_t>(rowLength / DIGIT_TWO);

    // E2M1 grid alignment constants, loop invariant; see ComputeRowDataToFp4VF.
    Reg::RegTensor<int32_t> expMaskFP32;
    Reg::RegTensor<int32_t> expUnitFP32;
    Reg::RegTensor<int32_t> expZeroFP32;
    if constexpr (IsSameType<xDtype, half>::value) {
        Reg::Duplicate(expMaskFP32, MAX_EXP_FOR_FP32);
        Reg::Duplicate(expUnitFP32, FP32_EXP_UNIT_BITS);
        Reg::Duplicate(expZeroFP32, FP32_EXP_ZERO_BITS);
    }

    for (uint16_t j = 0; j < blockCount; j++) {
        Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_NORM>(x, xAddr + j * rowLength);

        if constexpr (IsSameType<xDtype, half>::value) {
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(zeroHalfFp32, x, pregAll16);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(oneHalfFp32, x, pregAll16);

            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(reciprocalZeroFp32,
                                                              (Reg::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ONE>(reciprocalOneFp32,
                                                             (Reg::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);

            Reg::Mul(zeroHalfFp32, zeroHalfFp32, reciprocalZeroFp32, pregAll32);
            Reg::Mul(oneHalfFp32, oneHalfFp32, reciprocalOneFp32, pregAll32);

            {
                Reg::MaskReg negInfMaskZero;
                Reg::MaskReg specialMaskZero;
                Reg::MaskReg negInfMaskOne;
                Reg::MaskReg specialMaskOne;
                Reg::MaskReg zeroMask;
                Reg::RegTensor<int32_t> exp0Zero;
                Reg::RegTensor<int32_t> exp1Zero;
                Reg::RegTensor<int32_t> exp0One;
                Reg::RegTensor<int32_t> exp1One;

                Reg::Compare<int32_t, CMPMODE::EQ>(negInfMaskZero, (Reg::RegTensor<int32_t>&)zeroHalfFp32, negZero,
                                                   pregAll32);
                Reg::Compare<int32_t, CMPMODE::EQ>(negInfMaskOne, (Reg::RegTensor<int32_t>&)oneHalfFp32, negZero,
                                                   pregAll32);
                if constexpr (IsSameType<rowDataDtype, fp4x2_e1m2_t>::value) {
                    // E1M2: multiply by 4, truncate, divide by 4
                    Reg::Muls(zeroHalfFp32, zeroHalfFp32, FOUR, pregAll32);
                    Reg::Muls(oneHalfFp32, oneHalfFp32, FOUR, pregAll32);
                    Reg::Compares<float, CMPMODE::LT>(specialMaskZero, zeroHalfFp32, 0, pregAll32);
                    Reg::Compares<float, CMPMODE::LT>(specialMaskOne, oneHalfFp32, 0, pregAll32);
                    Reg::Truncate<float, roundMode>(zeroHalfFp32, zeroHalfFp32, pregAll32);
                    Reg::Truncate<float, roundMode>(oneHalfFp32, oneHalfFp32, pregAll32);
                    Reg::Muls(zeroHalfFp32, zeroHalfFp32, ONE_FOURTH, pregAll32);
                    Reg::Muls(oneHalfFp32, oneHalfFp32, ONE_FOURTH, pregAll32);
                } else {
                    // E2M1: align v to the 2^{e-1} grid.  The masked exponent field
                    // is kept in place (its float value is 2^e), so both powers of
                    // two are a single integer subtract:
                    //   raw = (e + 127) << 23
                    //   s   = 2^{e-1} = raw - (1 << 23)
                    //   1/s = 2^{1-e} = 0x7F800000 - raw
                    // The subnormal clamp e' = max(e, 0) becomes max(raw, 127 << 23).
                    Reg::And(exp0Zero, (Reg::RegTensor<int32_t>&)zeroHalfFp32, expMaskFP32, pregAll32);
                    Reg::And(exp0One, (Reg::RegTensor<int32_t>&)oneHalfFp32, expMaskFP32, pregAll32);
                    Reg::Max(exp0Zero, exp0Zero, expZeroFP32, pregAll32);
                    Reg::Max(exp0One, exp0One, expZeroFP32, pregAll32);
                    Reg::Sub(exp1Zero, expMaskFP32, exp0Zero, pregAll32); // 1/s
                    Reg::Sub(exp1One, expMaskFP32, exp0One, pregAll32);
                    Reg::Sub(exp0Zero, exp0Zero, expUnitFP32, pregAll32); // s
                    Reg::Sub(exp0One, exp0One, expUnitFP32, pregAll32);

                    Reg::Mul(zeroHalfFp32, zeroHalfFp32, (Reg::RegTensor<float>&)exp1Zero, pregAll32);
                    Reg::Mul(oneHalfFp32, oneHalfFp32, (Reg::RegTensor<float>&)exp1One, pregAll32);
                    Reg::Compares<float, CMPMODE::LT>(specialMaskZero, zeroHalfFp32, 0, pregAll32);
                    Reg::Compares<float, CMPMODE::LT>(specialMaskOne, oneHalfFp32, 0, pregAll32);
                    Reg::Truncate<float, roundMode>(zeroHalfFp32, zeroHalfFp32, pregAll32);
                    Reg::Truncate<float, roundMode>(oneHalfFp32, oneHalfFp32, pregAll32);
                    Reg::Mul(zeroHalfFp32, zeroHalfFp32, (Reg::RegTensor<float>&)exp0Zero, pregAll32);
                    Reg::Mul(oneHalfFp32, oneHalfFp32, (Reg::RegTensor<float>&)exp0One, pregAll32);
                }

                // Handle negative zero.  Reuse zeroMask between the two halves.
                Reg::Compares<float, CMPMODE::EQ>(zeroMask, zeroHalfFp32, 0, pregAll32);
                Reg::And(zeroMask, specialMaskZero, zeroMask, pregAll32);
                Reg::Or(zeroMask, negInfMaskZero, zeroMask, pregAll32);
                Reg::Select<int32_t>((Reg::RegTensor<int32_t>&)zeroHalfFp32, negZero,
                                     (Reg::RegTensor<int32_t>&)zeroHalfFp32, zeroMask);
                Reg::Compares<float, CMPMODE::EQ>(zeroMask, oneHalfFp32, 0, pregAll32);
                Reg::And(zeroMask, specialMaskOne, zeroMask, pregAll32);
                Reg::Or(zeroMask, negInfMaskOne, zeroMask, pregAll32);
                Reg::Select<int32_t>((Reg::RegTensor<int32_t>&)oneHalfFp32, negZero,
                                     (Reg::RegTensor<int32_t>&)oneHalfFp32, zeroMask);
            }

            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>((Reg::RegTensor<bfloat16_t>&)evenBf16, zeroHalfFp32,
                                                              pregAll32);
            Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>((Reg::RegTensor<bfloat16_t>&)oddBf16, oneHalfFp32,
                                                              pregAll32);

            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)evenBf16,
                                                                    (Reg::RegTensor<uint32_t>&)evenBf16);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>((Reg::RegTensor<uint16_t>&)oddBf16,
                                                                    (Reg::RegTensor<uint32_t>&)oddBf16);

            Reg::Interleave(evenBf16, oddBf16, evenBf16, oddBf16);
            Reg::Cast<rowDataDtype, bfloat16_t, castTraitBF16toFp4>(packedFp4, (Reg::RegTensor<bfloat16_t>&)evenBf16,
                                                                    pregAll16);
        } else {
            Reg::Mul(xBF16, x, (Reg::RegTensor<bfloat16_t>&)reversedShareExp, pregAll16);
            Reg::Cast<rowDataDtype, bfloat16_t, castTraitBF16toFp4>(packedFp4, xBF16, pregAll16);
        }

        Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_PACK4_B32>(colDataAddr + (j * fp4RowBytes),
                                                                 (Reg::RegTensor<uint8_t>&)packedFp4, pregAll8);
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__simd_vf__ inline void ComputeSigmoidSingleAxisVF(__ubuf__ xDtype* globalGateRow, __ubuf__ xDtype* localGateRow,
                                                   uint16_t gateCount, __ubuf__ uint8_t* sigmoidBroadcastAddr,
                                                   int64_t gateSideOffset)
{
    Reg::RegTensor<xDtype> globalGateReg, localGateReg;
    Reg::RegTensor<float> globalGateF, localGateF, globalTmpF, localTmpF, oneF;
    Reg::RegTensor<float> globalSigF, localSigF;
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();

    __ubuf__ uint8_t* sigmoidBytes = (__ubuf__ uint8_t*)sigmoidBroadcastAddr;
    __ubuf__ float* globalSigmoidUb = (__ubuf__ float*)sigmoidBytes;
    __ubuf__ float* localSigmoidUb = (__ubuf__ float*)(sigmoidBytes + gateSideOffset * sizeof(float));
    uint16_t gateVecLoops = static_cast<uint16_t>((static_cast<uint32_t>(gateCount) + VF_LEN_FP32 - 1) / VF_LEN_FP32);

    Reg::Duplicate(oneF, 1.0f, allMask);
    for (uint16_t i = 0; i < gateVecLoops; ++i) {
        uint32_t gateOffset = i * VF_LEN_FP32;
        Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(globalGateReg, globalGateRow + gateOffset);
        Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(localGateReg, localGateRow + gateOffset);
        Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(globalGateF, globalGateReg, allMask);
        Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(localGateF, localGateReg, allMask);
        Reg::Muls(globalTmpF, globalGateF, -1.0f, allMask);
        Reg::Muls(localTmpF, localGateF, -1.0f, allMask);
        Reg::Exp(globalTmpF, globalTmpF, allMask);
        Reg::Exp(localTmpF, localTmpF, allMask);
        Reg::Adds(globalTmpF, globalTmpF, 1.0f, allMask);
        Reg::Adds(localTmpF, localTmpF, 1.0f, allMask);
        Reg::Div(globalSigF, oneF, globalTmpF, allMask);
        Reg::Div(localSigF, oneF, localTmpF, allMask);
        Reg::StoreAlign<float>(globalSigmoidUb + gateOffset, globalSigF, allMask);
        Reg::StoreAlign<float>(localSigmoidUb + gateOffset, localSigF, allMask);
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__simd_vf__ inline void ComputeClaGateSingleAxisD256VF(__ubuf__ xDtype* globalAddr, __ubuf__ xDtype* localAddr,
                                                       __ubuf__ xDtype* outputAddr, uint16_t calcRows,
                                                       int64_t tileRowLength, __ubuf__ uint8_t* sigmoidBroadcastAddr,
                                                       int64_t gateSideOffset)
{
    Reg::RegTensor<xDtype> globalReg, localReg, outputReg;
    Reg::RegTensor<float> globalSigF, localSigF;
    Reg::RegTensor<float> globalRegF, localRegF;
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    __ubuf__ uint8_t* sigmoidBytes = (__ubuf__ uint8_t*)sigmoidBroadcastAddr;
    __ubuf__ float* globalSigmoidUb = (__ubuf__ float*)sigmoidBytes;
    __ubuf__ float* localSigmoidUb = (__ubuf__ float*)(sigmoidBytes + gateSideOffset * sizeof(float));

    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    for (uint16_t slot = 0; slot < calcRows; ++slot) {
        uint32_t tileOffset = slot * static_cast<uint32_t>(tileRowLength);
        __ubuf__ xDtype* tileGlobal = globalAddr + tileOffset;
        __ubuf__ xDtype* tileLocal = localAddr + tileOffset;
        __ubuf__ xDtype* tileOut = outputAddr + tileOffset;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(globalSigF, globalSigmoidUb + slot);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(localSigF, localSigmoidUb + slot);

        constexpr uint16_t d256VfsPerHead = static_cast<uint16_t>(256U / VF_LEN_FP32);
        for (uint16_t vf = 0; vf < d256VfsPerHead; ++vf) {
            uint32_t vfOffset = static_cast<uint32_t>(vf) * VF_LEN_FP32;
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(globalReg, tileGlobal + vfOffset);
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(localReg, tileLocal + vfOffset);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(globalRegF, globalReg, allMask);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(localRegF, localReg, allMask);
            Reg::Mul(globalRegF, globalRegF, globalSigF, allMask);
            Reg::MulAddDst(globalRegF, localRegF, localSigF, allMask);
            Reg::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(outputReg, globalRegF, allMask);
            Reg::StoreAlign<xDtype, Reg::StoreDist::DIST_PACK_B32>(tileOut + vfOffset, outputReg, allMask);
        }
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__simd_vf__ inline void ComputeClaGateSingleAxisD128VF(__ubuf__ xDtype* globalAddr, __ubuf__ xDtype* localAddr,
                                                       __ubuf__ xDtype* outputAddr, uint16_t calcRows,
                                                       __ubuf__ uint8_t* sigmoidBroadcastAddr, int64_t gateSideOffset)
{
    Reg::RegTensor<xDtype> globalReg0, localReg0, globalReg1, localReg1, outputReg0, outputReg1;
    Reg::RegTensor<float> globalSigF0, localSigF0, globalSigF1, localSigF1;
    Reg::RegTensor<float> globalRegF0, localRegF0, globalRegF1, localRegF1;
    Reg::MaskReg allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
    __ubuf__ uint8_t* sigmoidBytes = (__ubuf__ uint8_t*)sigmoidBroadcastAddr;
    __ubuf__ float* globalSigmoidUb = (__ubuf__ float*)sigmoidBytes;
    __ubuf__ float* localSigmoidUb = (__ubuf__ float*)(sigmoidBytes + gateSideOffset * sizeof(float));

    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    for (uint16_t slot = 0; slot < calcRows; ++slot) {
        uint32_t gateBase = static_cast<uint32_t>(slot) * DIGIT_TWO;
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(globalSigF0, globalSigmoidUb + gateBase);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(localSigF0, localSigmoidUb + gateBase);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(globalSigF1, globalSigmoidUb + gateBase + 1);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(localSigF1, localSigmoidUb + gateBase + 1);

        uint32_t tileOffset = static_cast<uint32_t>(slot) * ONCE_ROW_LEN;
        __ubuf__ xDtype* globalHead0 = globalAddr + tileOffset;
        __ubuf__ xDtype* globalHead1 = globalHead0 + D128_HEAD_ELEMS;
        __ubuf__ xDtype* localHead0 = localAddr + tileOffset;
        __ubuf__ xDtype* localHead1 = localHead0 + D128_HEAD_ELEMS;
        __ubuf__ xDtype* outputHead0 = outputAddr + tileOffset;
        __ubuf__ xDtype* outputHead1 = outputHead0 + D128_HEAD_ELEMS;

        constexpr uint16_t d128VfsPerHead = static_cast<uint16_t>(D128_HEAD_ELEMS / VF_LEN_FP32);
        for (uint16_t vf = 0; vf < d128VfsPerHead; ++vf) {
            uint32_t vfOffset = static_cast<uint32_t>(vf) * VF_LEN_FP32;
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(globalReg0, globalHead0 + vfOffset);
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(globalReg1, globalHead1 + vfOffset);
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(localReg0, localHead0 + vfOffset);
            Reg::LoadAlign<xDtype, Reg::LoadDist::DIST_UNPACK_B16>(localReg1, localHead1 + vfOffset);

            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(globalRegF0, globalReg0, allMask);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(globalRegF1, globalReg1, allMask);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(localRegF0, localReg0, allMask);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(localRegF1, localReg1, allMask);
            Reg::Mul(globalRegF0, globalRegF0, globalSigF0, allMask);
            Reg::Mul(globalRegF1, globalRegF1, globalSigF1, allMask);
            Reg::MulAddDst(globalRegF0, localRegF0, localSigF0, allMask);
            Reg::MulAddDst(globalRegF1, localRegF1, localSigF1, allMask);
            Reg::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(outputReg0, globalRegF0, allMask);
            Reg::Cast<xDtype, float, CAST_FP32_TO_FP16_BF16>(outputReg1, globalRegF1, allMask);
            Reg::StoreAlign<xDtype, Reg::StoreDist::DIST_PACK_B32>(outputHead0 + vfOffset, outputReg0, allMask);
            Reg::StoreAlign<xDtype, Reg::StoreDist::DIST_PACK_B32>(outputHead1 + vfOffset, outputReg1, allMask);
        }
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__simd_vf__ inline void ComputeRowScaleOcpBatchCompactVF(uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                                         __ubuf__ uint8_t* rowScaleAddr,
                                                         __ubuf__ uint16_t* rowScaleReciprocalAddr,
                                                         __ubuf__ uint16_t* maxExpAddr, int64_t outputDtypeMaxExpIn)
{
    // Phase 1 stores eight block maxima per 256 input values. Keep maxExpAddr at
    // the base for phase 2 and advance a separate POST_MODE_UPDATE write cursor.
    __ubuf__ uint16_t* maxExpWriteAddr = maxExpAddr;

    Reg::RegTensor<xDtype> evenData;
    Reg::RegTensor<xDtype> oddData;
    Reg::RegTensor<bfloat16_t> evenBf16;
    Reg::RegTensor<bfloat16_t> oddBf16;
    Reg::RegTensor<uint16_t> evenExpBf16;
    Reg::RegTensor<uint16_t> oddExpBf16;
    Reg::RegTensor<uint16_t> expMax;
    Reg::RegTensor<uint16_t> expMaskBF16;
    Reg::MaskReg maskAll = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::UnalignRegForStore ureg;

    Reg::Duplicate(expMaskBF16, EXP_MASK_BF16);
    for (uint16_t i = 0; i < blockCount; ++i) {
        Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(evenData, oddData,
                                                                                                   xAddr, ONCE_ROW_LEN);
        if constexpr (IsSameType<xDtype, half>::value) {
            // Same OCP CAST_TRUNC path as ComputeScaleOcpVF: special values
            // retain the BF16 all-ones exponent without a pre-cast repair mask.
            Reg::Cast<bfloat16_t, xDtype, CAST_HALF_TO_BF16>(evenBf16, evenData, maskAll);
            Reg::Cast<bfloat16_t, xDtype, CAST_HALF_TO_BF16>(oddBf16, oddData, maskAll);
            Reg::And(evenExpBf16, (Reg::RegTensor<uint16_t>&)evenBf16, expMaskBF16, maskAll);
            Reg::And(oddExpBf16, (Reg::RegTensor<uint16_t>&)oddBf16, expMaskBF16, maskAll);
        } else {
            Reg::And(evenExpBf16, (Reg::RegTensor<uint16_t>&)evenData, expMaskBF16, maskAll);
            Reg::And(oddExpBf16, (Reg::RegTensor<uint16_t>&)oddData, expMaskBF16, maskAll);
        }
        Reg::Max(expMax, evenExpBf16, oddExpBf16, maskAll);
        Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(expMax, expMax, maskAll);
        Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExpWriteAddr, expMax, ureg,
                                                                        ONCE_ROW_LEN / BLOCK_SIZE);
    }
    Reg::StoreUnAlignPost(maxExpWriteAddr, ureg, 0);

    // Phase 2 reads phase 1 output from UB; enforce the store-to-load dependency.
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    // Phase 2: transform up to 128 compact maxima per VF.  Scale is packed to
    // uint8 in UB and reciprocal remains a dense uint16 stream (eight/row).
    uint32_t totalScale = static_cast<uint32_t>(blockCount) * (ONCE_ROW_LEN / BLOCK_SIZE);
    uint16_t scaleLoops = static_cast<uint16_t>((totalScale + VF_LEN_B16 - 1) / VF_LEN_B16);
    uint16_t outputDtypeMaxExp = outputDtypeMaxExpIn;
    auto packedScaleAddr = (__ubuf__ uint16_t*)rowScaleAddr;

    Reg::RegTensor<uint16_t> maxExp;
    Reg::RegTensor<uint16_t> sharedExp;
    Reg::RegTensor<uint16_t> scale;
    Reg::RegTensor<uint16_t> reciprocal;
    Reg::RegTensor<uint16_t> expMask;
    Reg::RegTensor<uint16_t> dtypeMax;
    Reg::RegTensor<uint16_t> bias;
    Reg::RegTensor<uint16_t> nanScale;
    Reg::RegTensor<uint16_t> zero;
    Reg::RegTensor<uint16_t> nanReciprocal;
    Reg::RegTensor<uint16_t> specialExp;
    Reg::MaskReg validMask;
    Reg::MaskReg infMask;
    Reg::MaskReg nonZeroMask;
    Reg::MaskReg clampMask;
    Reg::MaskReg specialMask;

    Reg::Duplicate(expMask, EXP_MASK_BF16);
    Reg::Duplicate(dtypeMax, outputDtypeMaxExp);
    Reg::Duplicate(bias, BF16_EXP_BIAS);
    Reg::Duplicate(nanScale, NAN_FOR_FP8_E8M0);
    Reg::Duplicate(zero, static_cast<uint16_t>(0));
    Reg::Duplicate(nanReciprocal, NAN_CUSTOMIZATION);
    Reg::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);

    for (uint16_t i = 0; i < scaleLoops; ++i) {
        validMask = Reg::UpdateMask<uint16_t>(totalScale);
        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(maxExp, maxExpAddr, VF_LEN_B16);
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, maxExp, expMask, validMask);
        Reg::Compare<uint16_t, CMPMODE::NE>(nonZeroMask, maxExp, zero, validMask);
        Reg::Compare<uint16_t, CMPMODE::LE>(clampMask, maxExp, dtypeMax, validMask);
        Reg::Select<uint16_t>(maxExp, dtypeMax, maxExp, clampMask);
        Reg::Sub(sharedExp, maxExp, dtypeMax, validMask);
        Reg::ShiftRights(scale, sharedExp, SHR_NUM_FOR_BF16, validMask);
        Reg::Select<uint16_t>(scale, scale, nanScale, infMask);
        Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B16>(
            packedScaleAddr, scale, VF_LEN_FP32, validMask);

        Reg::Compare<uint16_t, CMPMODE::EQ>(specialMask, sharedExp, bias, validMask);
        Reg::Sub(reciprocal, bias, sharedExp, validMask);
        Reg::Select<uint16_t>(reciprocal, reciprocal, nanReciprocal, infMask);
        Reg::Select<uint16_t>(reciprocal, reciprocal, zero, nonZeroMask);
        Reg::Select<uint16_t>(reciprocal, specialExp, reciprocal, specialMask);
        Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(rowScaleReciprocalAddr, reciprocal, VF_LEN_B16,
                                                                      validMask);
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__simd_vf__ inline void ComputeRowDataToFp8CompactVF(uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                                     __ubuf__ uint16_t* rowScaleReciprocalAddr,
                                                     __ubuf__ uint8_t* rowDataAddr)
{
    Reg::MaskReg maskAll = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg maskAllB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint16_t> scaleForMulFP16;
    Reg::RegTensor<float> scaleForMulFP32;
    Reg::RegTensor<xDtype> evenData;
    Reg::RegTensor<xDtype> oddData;
    Reg::RegTensor<float> evenZeroFp32;
    Reg::RegTensor<float> evenOneFp32;
    Reg::RegTensor<float> oddZeroFp32;
    Reg::RegTensor<float> oddOneFp32;
    Reg::RegTensor<rowDataDtype> evenZeroFp8;
    Reg::RegTensor<rowDataDtype> evenOneFp8;
    Reg::RegTensor<rowDataDtype> oddZeroFp8;
    Reg::RegTensor<rowDataDtype> oddOneFp8;

    for (uint16_t i = 0; i < blockCount; ++i) {
        Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(evenData, oddData,
                                                                                                   xAddr, ONCE_ROW_LEN);
        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
            scaleForMulFP16, rowScaleReciprocalAddr, ONCE_ROW_LEN / BLOCK_SIZE);
        if constexpr (IsSameType<xDtype, half>::value) {
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(evenZeroFp32, evenData, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(evenOneFp32, evenData, maskAll);
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(scaleForMulFP32,
                                                              (Reg::RegTensor<bfloat16_t>&)scaleForMulFP16, maskAll);
            Reg::Mul(evenZeroFp32, evenZeroFp32, scaleForMulFP32, maskAll);
            Reg::Mul(evenOneFp32, evenOneFp32, scaleForMulFP32, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(oddZeroFp32, oddData, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(oddOneFp32, oddData, maskAll);
            Reg::Mul(oddZeroFp32, oddZeroFp32, scaleForMulFP32, maskAll);
            Reg::Mul(oddOneFp32, oddOneFp32, scaleForMulFP32, maskAll);
        } else {
            Reg::Mul(evenData, evenData, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
            Reg::Mul(oddData, oddData, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(evenZeroFp32, evenData, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(evenOneFp32, evenData, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(oddZeroFp32, oddData, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(oddOneFp32, oddData, maskAll);
        }
        Reg::Cast<rowDataDtype, float, CAST_32_TO_80>(evenZeroFp8, evenZeroFp32, maskAll);
        Reg::Cast<rowDataDtype, float, CAST_32_TO_82>(evenOneFp8, evenOneFp32, maskAll);
        Reg::Cast<rowDataDtype, float, CAST_32_TO_81>(oddZeroFp8, oddZeroFp32, maskAll);
        Reg::Cast<rowDataDtype, float, CAST_32_TO_83>(oddOneFp8, oddOneFp32, maskAll);
        Reg::Add((Reg::RegTensor<uint8_t>&)evenZeroFp8, (Reg::RegTensor<uint8_t>&)evenZeroFp8,
                 (Reg::RegTensor<uint8_t>&)evenOneFp8, maskAllB8);
        Reg::Add((Reg::RegTensor<uint8_t>&)evenZeroFp8, (Reg::RegTensor<uint8_t>&)evenZeroFp8,
                 (Reg::RegTensor<uint8_t>&)oddZeroFp8, maskAllB8);
        Reg::Add((Reg::RegTensor<uint8_t>&)evenZeroFp8, (Reg::RegTensor<uint8_t>&)evenZeroFp8,
                 (Reg::RegTensor<uint8_t>&)oddOneFp8, maskAllB8);
        Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
            rowDataAddr, (Reg::RegTensor<uint8_t>&)evenZeroFp8, ONCE_ROW_LEN, maskAllB8);
    }
}

template <typename xDtype, typename rowDataDtype, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__simd_vf__ inline void ComputeRowScaleCuBLASSingleAxisVF(uint16_t dataLen, uint16_t blockCount, __ubuf__ xDtype* xAddr,
                                                          __ubuf__ uint8_t* rowDataAddr, __ubuf__ uint8_t* rowScaleAddr,
                                                          __ubuf__ uint16_t* rowScaleReciprocalAddr,
                                                          __ubuf__ uint16_t* absMaxAddr, uint32_t outputDtypeInvMaxIn)
{
    uint32_t outputDtypeInvMax = outputDtypeInvMaxIn;
    __ubuf__ xDtype* dataStartAddr = xAddr;
    __ubuf__ uint16_t* rowScaleReciprocalStartAddr = rowScaleReciprocalAddr;
    // Phase 1 appends one compacted block maximum per row through this cursor;
    // absMaxAddr itself keeps pointing at the base for the phase-2 read cursor.
    __ubuf__ uint16_t* absMaxWriteAddr = absMaxAddr;

    // Keep the amax, scale/reciprocal and quantization phases in one vector scope,
    // consistent with the dual-axis cuBLAS path.

    // Phase 1: collect row maxima into the compact temporary layout.
    Reg::RegTensor<xDtype> evenData;
    Reg::RegTensor<xDtype> oddData;
    Reg::RegTensor<uint16_t> evenAbs;
    Reg::RegTensor<uint16_t> oddAbs;
    Reg::RegTensor<uint16_t> absMaxDim1;
    Reg::RegTensor<uint16_t> absMask;
    Reg::MaskReg maskAll = Reg::CreateMask<xDtype, Reg::MaskPattern::ALL>();
    Reg::UnalignRegForStore ureg;

    Reg::Duplicate(absMask, ABS_MASK_FOR_16BIT);
    for (uint16_t i = 0; i < blockCount; i++) {
        Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(evenData, oddData,
                                                                                                   xAddr, dataLen);
        Reg::And(evenAbs, (Reg::RegTensor<uint16_t>&)evenData, absMask, maskAll);
        Reg::And(oddAbs, (Reg::RegTensor<uint16_t>&)oddData, absMask, maskAll);
        Reg::Max(absMaxDim1, evenAbs, oddAbs, maskAll);
        Reg::ReduceDataBlock<AscendC::Reg::ReduceType::MAX>(absMaxDim1, absMaxDim1, maskAll);
        Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(absMaxWriteAddr, absMaxDim1, ureg,
                                                                        dataLen / BLOCK_SIZE);
    }
    Reg::StoreUnAlignPost(absMaxWriteAddr, ureg, 0);
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    // Phase 2: compute/scatter row scale and reciprocal from the compacted maxima.
    Reg::RegTensor<uint32_t> max32;
    Reg::RegTensor<uint32_t> exp32;
    Reg::RegTensor<uint32_t> man32;
    Reg::RegTensor<uint32_t> expAddOne32;
    Reg::RegTensor<uint32_t> extractExp;
    Reg::RegTensor<uint32_t> halfScale;
    Reg::RegTensor<uint16_t> scaleBf16;
    Reg::RegTensor<uint16_t> reciprocalBf16Reg;
    Reg::RegTensor<uint32_t> invMax;
    Reg::RegTensor<uint32_t> manMaskReg;
    Reg::RegTensor<uint32_t> expMaskReg;
    Reg::RegTensor<uint32_t> zeroReg32;
    Reg::RegTensor<uint32_t> scaleBiasReg;
    Reg::RegTensor<uint32_t> nanReg32;
    Reg::RegTensor<uint32_t> fp8NanReg32;
    Reg::MaskReg cmpResult;
    Reg::MaskReg zeroMask;
    Reg::MaskReg p0;
    Reg::MaskReg p1;
    Reg::MaskReg maskAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    uint32_t compactStoreElements = VF_LEN_FP32;
    Reg::MaskReg compactStoreMask = Reg::UpdateMask<uint16_t>(compactStoreElements);
    Reg::Duplicate(invMax, outputDtypeInvMax);
    Reg::Duplicate(manMaskReg, MAN_MASK_FLOAT);
    Reg::Duplicate(expMaskReg, MAX_EXP_FOR_FP32);
    Reg::Duplicate(zeroReg32, static_cast<uint32_t>(0));
    Reg::Duplicate(scaleBiasReg, FP32_EXP_BIAS_CUBLAS);
    Reg::Duplicate(nanReg32, static_cast<uint32_t>(NAN_CUSTOMIZATION));
    Reg::Duplicate(fp8NanReg32, MAX_EXP_FOR_FP8_IN_FP32);

    __ubuf__ uint16_t* absMaxReadAddr = absMaxAddr;
    uint16_t scaleCount = dataLen / BLOCK_SIZE;
    uint16_t totalScaleVf = static_cast<uint16_t>(blockCount * scaleCount);
    uint16_t batchCount = static_cast<uint16_t>((totalScaleVf + static_cast<uint16_t>(VF_LEN_FP32) - 1) /
                                                static_cast<uint16_t>(VF_LEN_FP32));
    for (uint16_t j = 0; j < batchCount; j++) {
        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
            absMaxDim1, absMaxReadAddr, VF_LEN_FP32);
        Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>((Reg::RegTensor<float>&)max32,
                                                      (Reg::RegTensor<xDtype>&)absMaxDim1, maskAll);
        Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMaskReg, maskAll32);
        Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroReg32, maskAll32);
        Reg::Mul((Reg::RegTensor<float>&)max32, (Reg::RegTensor<float>&)max32, (Reg::RegTensor<float>&)invMax,
                 maskAll32);
        Reg::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, maskAll32);
        Reg::And(man32, max32, manMaskReg, maskAll32);
        Reg::Compares<uint32_t, CMPMODE::GT>(p0, exp32, static_cast<uint32_t>(0), maskAll32);
        Reg::Compares<uint32_t, CMPMODE::LT>(p0, exp32, EXP_254, p0);
        Reg::Compares<uint32_t, CMPMODE::GT>(p0, man32, static_cast<uint32_t>(0), p0);
        Reg::Compares<uint32_t, CMPMODE::EQ>(p1, exp32, static_cast<uint32_t>(0), maskAll32);
        Reg::Compares<uint32_t, CMPMODE::GT>(p1, man32, HALF_FOR_MAN, p1);
        Reg::Or(p0, p0, p1, maskAll32);
        Reg::Adds(expAddOne32, exp32, 1, maskAll32);
        Reg::Select<uint32_t>(extractExp, expAddOne32, exp32, p0);
        Reg::Select<uint32_t>(extractExp, extractExp, fp8NanReg32, cmpResult);
        Reg::Select<uint32_t>(extractExp, extractExp, zeroReg32, zeroMask);
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(scaleBf16, extractExp);
        Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskAll32);
        Reg::Sub(halfScale, scaleBiasReg, extractExp, maskAll32);
        Reg::Select<uint32_t>(halfScale, halfScale, nanReg32, cmpResult);
        Reg::Select<uint32_t>(halfScale, halfScale, zeroReg32, zeroMask);
        Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(reciprocalBf16Reg, halfScale);

        auto packedScaleAddr = (__ubuf__ uint16_t*)rowScaleAddr;
        Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_PACK_B16>(packedScaleAddr + j * (VF_LEN_FP32 / DIGIT_TWO),
                                                                 scaleBf16, compactStoreMask);
        Reg::StoreAlign<uint16_t>(rowScaleReciprocalAddr + j * VF_LEN_FP32, reciprocalBf16Reg, compactStoreMask);
    }
    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();

    // Phase 3: quantize row data using the generated row-scale reciprocals.
    Reg::MaskReg maskAllB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    Reg::RegTensor<uint16_t> scaleForMulFP16;
    Reg::RegTensor<float> scaleForMulFP32;
    Reg::RegTensor<float> evenZeroFp32;
    Reg::RegTensor<float> evenOneFp32;
    Reg::RegTensor<float> oddZeroFp32;
    Reg::RegTensor<float> oddOneFp32;
    Reg::RegTensor<rowDataDtype> evenZeroFp8;
    Reg::RegTensor<rowDataDtype> evenOneFp8;
    Reg::RegTensor<rowDataDtype> oddZeroFp8;
    Reg::RegTensor<rowDataDtype> oddOneFp8;
    __ubuf__ xDtype* dataReadAddr = dataStartAddr;
    __ubuf__ uint16_t* rowScaleReciprocalReadAddr = rowScaleReciprocalStartAddr;
    constexpr uint16_t reciprocalStride = ONCE_ROW_LEN / BLOCK_SIZE;

    for (uint16_t i = 0; i < blockCount; i++) {
        Reg::LoadAlign<xDtype, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
            evenData, oddData, dataReadAddr, dataLen);
        Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
            scaleForMulFP16, rowScaleReciprocalReadAddr, reciprocalStride);
        if constexpr (IsSameType<xDtype, half>::value) {
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(evenZeroFp32, evenData, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(evenOneFp32, evenData, maskAll);
            Reg::Cast<float, bfloat16_t, CAST_X_TO_FP32_ZERO>(scaleForMulFP32,
                                                              (Reg::RegTensor<bfloat16_t>&)scaleForMulFP16, maskAll);
            Reg::Mul(evenZeroFp32, evenZeroFp32, scaleForMulFP32, maskAll);
            Reg::Mul(evenOneFp32, evenOneFp32, scaleForMulFP32, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(oddZeroFp32, oddData, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(oddOneFp32, oddData, maskAll);
            Reg::Mul(oddZeroFp32, oddZeroFp32, scaleForMulFP32, maskAll);
            Reg::Mul(oddOneFp32, oddOneFp32, scaleForMulFP32, maskAll);
        } else {
            Reg::Mul(evenData, evenData, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
            Reg::Mul(oddData, oddData, (Reg::RegTensor<xDtype>&)scaleForMulFP16, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(evenZeroFp32, evenData, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(evenOneFp32, evenData, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ZERO>(oddZeroFp32, oddData, maskAll);
            Reg::Cast<float, xDtype, CAST_X_TO_FP32_ONE>(oddOneFp32, oddData, maskAll);
        }
        Reg::Cast<rowDataDtype, float, CAST_32_TO_80>(evenZeroFp8, evenZeroFp32, maskAll);
        Reg::Cast<rowDataDtype, float, CAST_32_TO_82>(evenOneFp8, evenOneFp32, maskAll);
        Reg::Cast<rowDataDtype, float, CAST_32_TO_81>(oddZeroFp8, oddZeroFp32, maskAll);
        Reg::Cast<rowDataDtype, float, CAST_32_TO_83>(oddOneFp8, oddOneFp32, maskAll);
        Reg::Add((Reg::RegTensor<uint8_t>&)evenZeroFp8, (Reg::RegTensor<uint8_t>&)evenZeroFp8,
                 (Reg::RegTensor<uint8_t>&)evenOneFp8, maskAllB8);
        Reg::Add((Reg::RegTensor<uint8_t>&)evenZeroFp8, (Reg::RegTensor<uint8_t>&)evenZeroFp8,
                 (Reg::RegTensor<uint8_t>&)oddZeroFp8, maskAllB8);
        Reg::Add((Reg::RegTensor<uint8_t>&)evenZeroFp8, (Reg::RegTensor<uint8_t>&)evenZeroFp8,
                 (Reg::RegTensor<uint8_t>&)oddOneFp8, maskAllB8);
        Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
            rowDataAddr, (Reg::RegTensor<uint8_t>&)evenZeroFp8, dataLen, maskAllB8);
    }
}

} // namespace ClaGateQuant

#endif // OPS_NN_CLA_GATE_QUANT_VF_H
