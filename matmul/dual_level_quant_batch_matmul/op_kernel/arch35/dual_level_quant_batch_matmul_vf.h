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
 * \file dual_level_quant_batch_matmul_vf.h
 * \brief
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace Reg = AscendC::Reg;
using AscendC::IsSameType;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::Reg::AddrReg;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;

namespace DualLevelQuantBatchMatmul::Arch35 {
static constexpr uint64_t L0C_BASE_N = 128;
static constexpr Reg::CastTrait B32_TO_B16_TRAIT_PART_ONE = {
    Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
static constexpr uint64_t VEC_MAX_ELEM_B32 = VECTOR_REG_WIDTH >> 2;

__simd_vf__ inline void InitUbToBias(uint16_t count, __ubuf__ float* srcAddr, __ubuf__ float* dstAddr)
{
    // 一次初始化128个数据
    RegTensor<float> biasReg1;
    RegTensor<float> biasReg2;
    constexpr uint64_t storeDataOffset = 64;
    constexpr uint64_t biasStoreLen = 128;
    MaskReg maskAll = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(biasReg1, srcAddr);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(biasReg2, srcAddr + VEC_MAX_ELEM_B32);
    for (uint16_t idx = 0; idx < count; idx++) {
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(dstAddr + idx * biasStoreLen, biasReg1, maskAll);
        Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(dstAddr + storeDataOffset + idx * biasStoreLen, biasReg2,
                                                              maskAll);
    }
}

__simd_vf__ inline void InitUbToZero(uint16_t count, __ubuf__ int32_t* ubAddr)
{
    RegTensor<int32_t> dupVreg;
    MaskReg maskAll = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
    Reg::Duplicate(dupVreg, 0, maskAll);
    for (uint16_t idx = 0; idx < count; idx++) {
        Reg::StoreAlign<int32_t, Reg::StoreDist::DIST_NORM_B32>(ubAddr + idx * VECTOR_REG_WIDTH / sizeof(int32_t),
                                                                dupVreg, maskAll);
    }
}

template <typename yType>
__simd_vf__ inline void MulDoubleAdd(__ubuf__ float* cTmpFp32Addr, __ubuf__ float* cFp32Addr,
                                     __ubuf__ float* x1Level0ScaleFp32Addr, __ubuf__ float* x2Level0ScaleFp32Addr,
                                     uint16_t mL0Size, uint16_t x1ScaleGroupSizeNum)
{
    // 一个循环计算两个寄存器的值，64*2
    RegTensor<float> cTmpFp32Reg1;
    RegTensor<float> cTmpFp32Reg2;
    RegTensor<float> cFp32Reg1;
    RegTensor<float> cFp32Reg2;
    RegTensor<float> x1Level0Fp32Reg;
    RegTensor<float> x2Level0Fp32Reg1;
    RegTensor<float> x2Level0Fp32Reg2;

    RegTensor<float> level0ScaleMulFp32Reg1;
    RegTensor<float> level0ScaleMulFp32Reg2;
    MaskReg maskAll = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(x2Level0Fp32Reg1, x2Level0ScaleFp32Addr);
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(x2Level0Fp32Reg2, x2Level0ScaleFp32Addr + VEC_MAX_ELEM_B32);

    for (uint16_t mIdx = 0; mIdx < mL0Size; mIdx++) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(x1Level0Fp32Reg,
                                                           x1Level0ScaleFp32Addr + mIdx * x1ScaleGroupSizeNum);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(cTmpFp32Reg1, cTmpFp32Addr + mIdx * L0C_BASE_N);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(cTmpFp32Reg2,
                                                        cTmpFp32Addr + mIdx * L0C_BASE_N + VEC_MAX_ELEM_B32);

        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(cFp32Reg1, cFp32Addr + mIdx * L0C_BASE_N);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(cFp32Reg2, cFp32Addr + mIdx * L0C_BASE_N + VEC_MAX_ELEM_B32);

        Reg::Mul(level0ScaleMulFp32Reg1, x1Level0Fp32Reg, x2Level0Fp32Reg1, maskAll);
        Reg::Mul(level0ScaleMulFp32Reg2, x1Level0Fp32Reg, x2Level0Fp32Reg2, maskAll);

        Reg::MulAddDst(cFp32Reg1, cTmpFp32Reg1, level0ScaleMulFp32Reg1, maskAll);
        Reg::MulAddDst(cFp32Reg2, cTmpFp32Reg2, level0ScaleMulFp32Reg2, maskAll);

        if constexpr (!IsSameType<typename Reg::TypeGet<yType>::T, vector_f32>::value) {
            RegTensor<yType> cOutReg1;
            RegTensor<yType> cOutReg2;
            Reg::Cast<yType, float, B32_TO_B16_TRAIT_PART_ONE>(cOutReg1, cFp32Reg1, maskAll);
            Reg::Cast<yType, float, B32_TO_B16_TRAIT_PART_ONE>(cOutReg2, cFp32Reg2, maskAll);
            Reg::StoreAlign<yType, Reg::StoreDist::DIST_PACK_B32>((__ubuf__ yType*)cFp32Addr + mIdx * (L0C_BASE_N * 2),
                                                                  cOutReg1, maskAll);
            Reg::StoreAlign<yType, Reg::StoreDist::DIST_PACK_B32>(
                (__ubuf__ yType*)cFp32Addr + mIdx * (L0C_BASE_N * 2) + VEC_MAX_ELEM_B32, cOutReg2, maskAll);
        } else {
            Reg::StoreAlign<yType, Reg::StoreDist::DIST_NORM>(cFp32Addr + mIdx * L0C_BASE_N, cFp32Reg1, maskAll);
            Reg::StoreAlign<yType, Reg::StoreDist::DIST_NORM>(cFp32Addr + mIdx * L0C_BASE_N + VEC_MAX_ELEM_B32,
                                                              cFp32Reg2, maskAll);
        }
    }
}

template <typename yType>
__simd_vf__ inline void MulAdd(__ubuf__ float* cTmpFp32Addr, __ubuf__ float* cFp32Addr,
                               __ubuf__ float* x1Level0ScaleFp32Addr, __ubuf__ float* x2Level0ScaleFp32Addr,
                               uint16_t mL0Size, uint16_t x1ScaleGroupSizeNum)
{
    // 一个循环计算一个寄存器的值64
    RegTensor<float> cTmpFp32Reg1;
    RegTensor<float> cFp32Reg1;
    RegTensor<float> x1Level0Fp32Reg;
    RegTensor<float> x2Level0Fp32Reg1;

    RegTensor<float> level0ScaleMulFp32Reg1;
    RegTensor<float> level0ScaleMulFp32Reg2;
    MaskReg maskAll = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(x2Level0Fp32Reg1, x2Level0ScaleFp32Addr);

    for (uint16_t mIdx = 0; mIdx < mL0Size; mIdx++) {
        Reg::LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(x1Level0Fp32Reg,
                                                           x1Level0ScaleFp32Addr + mIdx * x1ScaleGroupSizeNum);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(cTmpFp32Reg1, cTmpFp32Addr + mIdx * L0C_BASE_N);
        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(cFp32Reg1, cFp32Addr + mIdx * L0C_BASE_N);
        Reg::Mul(level0ScaleMulFp32Reg1, x1Level0Fp32Reg, x2Level0Fp32Reg1, maskAll);
        Reg::MulAddDst(cFp32Reg1, cTmpFp32Reg1, level0ScaleMulFp32Reg1, maskAll);

        if constexpr (!IsSameType<typename Reg::TypeGet<yType>::T, vector_f32>::value) {
            RegTensor<yType> cOutReg1;
            Reg::Cast<yType, float, B32_TO_B16_TRAIT_PART_ONE>(cOutReg1, cFp32Reg1, maskAll);
            // 2:yType为bf16或fp16，在做地址偏移的时候需要乘2
            Reg::StoreAlign<yType, Reg::StoreDist::DIST_PACK_B32>((__ubuf__ yType*)cFp32Addr + mIdx * (L0C_BASE_N * 2),
                                                                  cOutReg1, maskAll);
        } else {
            Reg::StoreAlign<yType, Reg::StoreDist::DIST_NORM>(cFp32Addr + mIdx * L0C_BASE_N, cFp32Reg1, maskAll);
        }
    }
}

} // namespace DualLevelQuantBatchMatmul::Arch35
