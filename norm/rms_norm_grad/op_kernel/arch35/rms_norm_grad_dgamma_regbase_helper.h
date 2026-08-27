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
 * \file rms_norm_grad_dgamma_regbase_helper.h
 * \brief RmsNormGrad regbase dgamma helper file
 */
#ifndef RMS_NORM_GRAD_DGAMMA_REGBASE_HELPER_H
#define RMS_NORM_GRAD_DGAMMA_REGBASE_HELPER_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "rms_norm_grad_regbase_common.h"
namespace RmsNormGrad {
using namespace AscendC;
using AscendC::Reg::CreateMask;
using AscendC::Reg::LoadDist;
using AscendC::Reg::MaskPattern;
using AscendC::Reg::MaskReg;
using AscendC::Reg::MemType;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;

constexpr int32_t REDUCEBY8ELENUM = 16;
constexpr int32_t REDUCEBY4ELENUM = 8;
constexpr int32_t REDUCEBY2ELENUM = 4;
constexpr int32_t REDUCEBY1ELENUM = 2;
constexpr int32_t COMPRESSBY8ELENUM = 8;
constexpr int32_t RESERVESIZE = 32;

__aicore__ inline int64_t CEIL_DIV(int64_t x, int64_t y) { return (y > 0) ? (x + y - 1) / y : 0; }

__aicore__ inline uint32_t BLOCK_ALIGN(uint32_t x, uint32_t blockSize)
{
    return (blockSize > 0) ? (x + blockSize - 1) / blockSize * blockSize : 0;
}

template <typename DY_TYPE, typename X_TYPE, typename RSTD_TYPE, int TILING_KEY>
__aicore__ inline void CalcMulRes(__ubuf__ DY_TYPE* dyAddr, __ubuf__ X_TYPE* xAddr, __ubuf__ RSTD_TYPE* rstdAddr,
                                  __ubuf__ float* dgammaOutAddr, MaskReg& preg, uint32_t offset0, uint32_t k)
{
    RegTensor<float> xFp32, rstdFp32, dyFp32, temp_res, mul_res;

    if constexpr (IsSameType<DY_TYPE, float>::value) {
        LoadAlign<DY_TYPE, LoadDist::DIST_NORM>(dyFp32, (__ubuf__ float*)(dyAddr + offset0));
    } else {
        RegTensor<DY_TYPE> dstRegB16;
        LoadAlign<DY_TYPE, LoadDist::DIST_UNPACK_B16>(dstRegB16, (__ubuf__ DY_TYPE*)(dyAddr + offset0));
        Cast<float, DY_TYPE, castTraitB162B32>(dyFp32, dstRegB16, preg);
    }

    if constexpr (IsSameType<X_TYPE, float>::value) {
        LoadAlign<X_TYPE, LoadDist::DIST_NORM>(xFp32, (__ubuf__ float*)(xAddr + offset0));
    } else {
        RegTensor<X_TYPE> dstRegB16;
        LoadAlign<X_TYPE, LoadDist::DIST_UNPACK_B16>(dstRegB16, (__ubuf__ X_TYPE*)(xAddr + offset0));
        Cast<float, X_TYPE, castTraitB162B32>(xFp32, dstRegB16, preg);
    }

    LoadAlign<RSTD_TYPE, LoadDist::DIST_BRC_B32>(rstdFp32, ((__ubuf__ float*)rstdAddr + k));

    Mul(temp_res, xFp32, rstdFp32, preg);
    Mul(mul_res, dyFp32, temp_res, preg);

    StoreAlign<float, StoreDist::DIST_NORM_B32>((__ubuf__ float*)(dgammaOutAddr + offset0), mul_res, preg);
}

__aicore__ inline void reduceSumCompressedBy8(__ubuf__ float* dyAddr, MaskReg& preg, uint32_t offset,
                                              uint32_t ub_offset)
{
    RegTensor<float> temp_reg0_0, temp_reg0_1, temp_reg1_0, temp_reg1_1, temp_reg2_0, temp_reg2_1, temp_reg3_0,
        temp_reg3_1, temp_reg4_0, temp_reg4_1, temp_reg5_0, temp_reg5_1, temp_reg6_0, temp_reg6_1, temp_reg7_0,
        temp_reg7_1;
    __ubuf__ float* currentAddr = dyAddr + REDUCEBY8ELENUM * ub_offset;

    //
    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg0_0, (__ubuf__ float*)(currentAddr));
    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg0_1, (__ubuf__ float*)(currentAddr + offset));
    AscendC::Reg::Add(temp_reg0_0, temp_reg0_0, temp_reg0_1, preg);

    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg1_0, (__ubuf__ float*)(currentAddr + 2 * offset));
    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg1_1, (__ubuf__ float*)(currentAddr + 3 * offset));
    AscendC::Reg::Add(temp_reg1_0, temp_reg1_0, temp_reg1_1, preg);

    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg2_0, (__ubuf__ float*)(currentAddr + 4 * offset));
    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg2_1, (__ubuf__ float*)(currentAddr + 5 * offset));
    AscendC::Reg::Add(temp_reg2_0, temp_reg2_0, temp_reg2_1, preg);

    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg3_0, (__ubuf__ float*)(currentAddr + 6 * offset));
    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg3_1, (__ubuf__ float*)(currentAddr + 7 * offset));
    AscendC::Reg::Add(temp_reg3_0, temp_reg3_0, temp_reg3_1, preg);

    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg4_0, (__ubuf__ float*)(currentAddr + 8 * offset));
    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg4_1, (__ubuf__ float*)(currentAddr + 9 * offset));
    AscendC::Reg::Add(temp_reg4_0, temp_reg4_0, temp_reg4_1, preg);

    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg5_0, (__ubuf__ float*)(currentAddr + 10 * offset));
    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg5_1, (__ubuf__ float*)(currentAddr + 11 * offset));
    AscendC::Reg::Add(temp_reg5_0, temp_reg5_0, temp_reg5_1, preg);

    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg6_0, (__ubuf__ float*)(currentAddr + 12 * offset));
    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg6_1, (__ubuf__ float*)(currentAddr + 13 * offset));
    AscendC::Reg::Add(temp_reg6_0, temp_reg6_0, temp_reg6_1, preg);

    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg7_0, (__ubuf__ float*)(currentAddr + 14 * offset));
    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg7_1, (__ubuf__ float*)(currentAddr + 15 * offset));
    AscendC::Reg::Add(temp_reg7_0, temp_reg7_0, temp_reg7_1, preg);

    //
    AscendC::Reg::Add(temp_reg0_0, temp_reg0_0, temp_reg1_0, preg);
    AscendC::Reg::Add(temp_reg2_0, temp_reg2_0, temp_reg3_0, preg);
    AscendC::Reg::Add(temp_reg4_0, temp_reg4_0, temp_reg5_0, preg);
    AscendC::Reg::Add(temp_reg6_0, temp_reg6_0, temp_reg7_0, preg);

    AscendC::Reg::Add(temp_reg0_0, temp_reg0_0, temp_reg2_0, preg);
    AscendC::Reg::Add(temp_reg4_0, temp_reg4_0, temp_reg6_0, preg);

    AscendC::Reg::Add(temp_reg0_0, temp_reg0_0, temp_reg4_0, preg);

    StoreAlign<float, StoreDist::DIST_NORM_B32>((__ubuf__ float*)(dyAddr + ub_offset), temp_reg0_0, preg);
}

__aicore__ inline void reduceSumCompressedBy4(__ubuf__ float* dyAddr, MaskReg& preg, uint32_t offset,
                                              uint32_t ub_offset)
{
    RegTensor<float> temp_reg0_0, temp_reg0_1, temp_reg1_0, temp_reg1_1, temp_reg2_0, temp_reg2_1, temp_reg3_0,
        temp_reg3_1;
    __ubuf__ float* currentAddr = dyAddr + COMPRESSBY8ELENUM * ub_offset;
    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg0_0, (__ubuf__ float*)(currentAddr));
    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg0_1, (__ubuf__ float*)(currentAddr + offset));
    AscendC::Reg::Add(temp_reg0_0, temp_reg0_0, temp_reg0_1, preg);

    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg1_0, (__ubuf__ float*)(currentAddr + 2 * offset));
    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg1_1, (__ubuf__ float*)(currentAddr + 3 * offset));
    AscendC::Reg::Add(temp_reg1_0, temp_reg1_0, temp_reg1_1, preg);

    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg2_0, (__ubuf__ float*)(currentAddr + 4 * offset));
    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg2_1, (__ubuf__ float*)(currentAddr + 5 * offset));
    AscendC::Reg::Add(temp_reg2_0, temp_reg2_0, temp_reg2_1, preg);

    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg3_0, (__ubuf__ float*)(currentAddr + 6 * offset));
    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg3_1, (__ubuf__ float*)(currentAddr + 7 * offset));
    AscendC::Reg::Add(temp_reg3_0, temp_reg3_0, temp_reg3_1, preg);

    AscendC::Reg::Add(temp_reg0_0, temp_reg0_0, temp_reg1_0, preg);
    AscendC::Reg::Add(temp_reg2_0, temp_reg2_0, temp_reg3_0, preg);

    AscendC::Reg::Add(temp_reg0_0, temp_reg0_0, temp_reg2_0, preg);

    StoreAlign<float, StoreDist::DIST_NORM_B32>((__ubuf__ float*)(dyAddr + ub_offset), temp_reg0_0, preg);
}

__aicore__ inline void reduceSumCompressedBy2(__ubuf__ float* dyAddr, MaskReg& preg, uint32_t offset,
                                              uint32_t ub_offset)
{
    RegTensor<float> temp_reg0_0, temp_reg0_1, temp_reg1_0, temp_reg1_1;

    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg0_0, (__ubuf__ float*)(dyAddr));
    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg0_1, (__ubuf__ float*)(dyAddr + offset));
    AscendC::Reg::Add(temp_reg0_0, temp_reg0_0, temp_reg0_1, preg);

    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg1_0, (__ubuf__ float*)(dyAddr + 2 * offset));
    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg1_1, (__ubuf__ float*)(dyAddr + 3 * offset));
    AscendC::Reg::Add(temp_reg1_0, temp_reg1_0, temp_reg1_1, preg);

    AscendC::Reg::Add(temp_reg0_0, temp_reg0_0, temp_reg1_0, preg);
    StoreAlign<float, StoreDist::DIST_NORM_B32>((__ubuf__ float*)(dyAddr + ub_offset), temp_reg0_0, preg);
}

__aicore__ inline void reduceSumCompressedBy1(__ubuf__ float* dyAddr, MaskReg& preg, uint32_t offset)
{
    RegTensor<float> temp_reg0_0, temp_reg0_1;

    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg0_0, (__ubuf__ float*)(dyAddr));
    LoadAlign<float, LoadDist::DIST_NORM>(temp_reg0_1, (__ubuf__ float*)(dyAddr + offset));
    AscendC::Reg::Add(temp_reg0_0, temp_reg0_0, temp_reg0_1, preg);

    StoreAlign<float, StoreDist::DIST_NORM_B32>((__ubuf__ float*)(dyAddr), temp_reg0_0, preg);
}

__aicore__ inline void reduceSumCompressedBy8WithOutPad(__ubuf__ float* src1Addr, __ubuf__ float* src2Addr,
                                                        MaskReg& preg, uint32_t ub_offset, uint32_t vlFp32)
{
    for (uint16_t i = 0; i < 8; i++) {
        RegTensor<float> temp_reg0_0, temp_reg0_1;
        uint32_t tempOffset = i * vlFp32;
        LoadAlign<float, LoadDist::DIST_NORM>(temp_reg0_0, (__ubuf__ float*)(src1Addr + ub_offset + tempOffset));
        LoadAlign<float, LoadDist::DIST_NORM>(temp_reg0_1, (__ubuf__ float*)(src2Addr + ub_offset + tempOffset));
        AscendC::Reg::Add(temp_reg0_0, temp_reg0_0, temp_reg0_1, preg);
        StoreAlign<float, StoreDist::DIST_NORM_B32>((__ubuf__ float*)(src1Addr + ub_offset + tempOffset), temp_reg0_0,
                                                    preg);
    }
}

__aicore__ inline void reduceSumCompressedBy8WithPad(__ubuf__ float* src1Addr, __ubuf__ float* src2Addr, MaskReg& preg,
                                                     uint32_t ub_offset, uint32_t rowsBoundLine, uint32_t vlFp32,
                                                     uint32_t tailDataOffset)
{
    for (uint16_t i = 0; i < 8; i++) {
        RegTensor<float> temp_reg0_0, temp_reg0_1;
        uint32_t temp_off_set_0 = ub_offset + i * vlFp32;
        uint32_t temp_off_set_1 = tailDataOffset + temp_off_set_0 < rowsBoundLine ? tailDataOffset + temp_off_set_0 :
                                                                                    rowsBoundLine;
        LoadAlign<float, LoadDist::DIST_NORM>(temp_reg0_0, (__ubuf__ float*)(src1Addr + temp_off_set_0));
        LoadAlign<float, LoadDist::DIST_NORM>(temp_reg0_1, (__ubuf__ float*)(src2Addr + temp_off_set_1));

        AscendC::Reg::Add(temp_reg0_0, temp_reg0_0, temp_reg0_1, preg);
        StoreAlign<float, StoreDist::DIST_NORM_B32>((__ubuf__ float*)(src1Addr + temp_off_set_0), temp_reg0_0, preg);
    }
}

__aicore__ inline void UpdateCache(const AscendC::LocalTensor<float>& dstTensor, __ubuf__ float* srcAddr,
                                   const int64_t cacheID, const int64_t count)
{
    // UpdateCache
    uint16_t innerLoopTimes = cacheID;
    uint32_t innerLoopStride = count;
    __VEC_SCOPE__
    {
        __ubuf__ float* dst = (__ubuf__ float*)dstTensor.GetPhyAddr();
        __ubuf__ float* cah = (__ubuf__ float*)dstTensor.GetPhyAddr() + cacheID * count;
        uint32_t sreg = static_cast<uint32_t>(count);
        AscendC::Reg::RegTensor<float> aReg, bReg;
        AscendC::Reg::MaskReg pMask;
        pMask = AscendC::Reg::UpdateMask<float>(sreg);
        LoadAlign(aReg, (__ubuf__ float*)srcAddr);
        for (uint16_t j = 0; j < innerLoopTimes; ++j) {
            LoadAlign(bReg, (__ubuf__ float*)dst + static_cast<uint32_t>(j * innerLoopStride));
            Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
        }
        StoreAlign((__ubuf__ float*)cah, aReg, pMask);
    }
}

__aicore__ inline int64_t GetCacheID(const int64_t idx) { return ScalarGetCountOfValue<1>(idx ^ (idx + 1)) - 1; }
} // namespace RmsNormGrad
#endif // RMS_NORM_GRAD_REGBASE_DGAMMA_H
