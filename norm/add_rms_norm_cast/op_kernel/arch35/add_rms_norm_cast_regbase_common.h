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
 * \file add_rms_norm_cast_regbase_common.h
 * \brief AddRmsNormCast regbase common.
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_RMS_NORM_CAST_REGBASE_COMMON_H
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_RMS_NORM_CAST_REGBASE_COMMON_H

#include "kernel_operator.h"
#include "kernel_utils.h"
#include "../../norm_common/reduce_common_regbase.h"
#include "../../rms_norm/arch35/rms_norm_regbase_common.h"

namespace AddRmsNormCast {
using namespace AscendC;
using namespace AscendC::MicroAPI;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;
using namespace NormCommon;
using namespace NormCommon::NormCommonRegbase;

constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t B32_BLOCK_NUM = 8;
constexpr uint64_t B8_BLOCK_NUM = 32;
constexpr uint64_t ALIGN_32_FACTOR = 32;
constexpr uint64_t ALIGN_512_FACTOR = 512;
constexpr uint32_t SUM_COUNT = 2;
constexpr uint32_t CONST_FACTOR_2 = 2;
constexpr int32_t NDDMA_DIM = 5;
constexpr int32_t VL_SIZE = GetVRegSize();
constexpr int32_t V_LENGTH = (VL_SIZE / static_cast<int32_t>(sizeof(float)));
constexpr uint64_t MIGIC_AND_WONDERFUL_BYTES = 128;

template <HardEvent ent>
__aicore__ inline void SimpleNativePipeSync()
{
    event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(ent));
    SetFlag<ent>(event);
    WaitFlag<ent>(event);
}

/*!
 * @brief yLocal = xLocal * rstd * gammaLocal
 *
 * @param y1Local output y1Local
 * @param y2Local output y2Local
 * @param xLocal input xLocal
 * @param gammaLocal input gammaLocal
 * @param rstdLocal input rstdLocal
 * @param rstdOffset elem offset of rstdLocal
 * @param count vector commpute length
 * @return void
 */
template <typename U_X, typename U_GAMMA>
__aicore__ inline void ComputeY(
    LocalTensor<float>& y1Local, LocalTensor<U_X>& y2Local, LocalTensor<float>& xLocal,
    LocalTensor<U_GAMMA>& gammaLocal, LocalTensor<float>& rstdLocal, uint32_t rstdOffset, uint64_t count)
{
    uint32_t calCount = (uint32_t)(count / 2); // Unroll
    uint16_t repeatTimes = CeilDivision(calCount, V_LENGTH);

    __local_mem__ float* xAddr1 = (__ubuf__ float*)xLocal.GetPhyAddr();
    __local_mem__ float* xAddr2 = (__ubuf__ float*)xLocal.GetPhyAddr() + calCount;
    __local_mem__ U_GAMMA* gammaAddr1 = (__ubuf__ U_GAMMA*)gammaLocal.GetPhyAddr();
    __local_mem__ U_GAMMA* gammaAddr2 = (__ubuf__ U_GAMMA*)gammaLocal.GetPhyAddr() + calCount;
    __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
    __local_mem__ float* y1Addr1 = (__ubuf__ float*)y1Local.GetPhyAddr();
    __local_mem__ float* y1Addr2 = (__ubuf__ float*)y1Local.GetPhyAddr() + calCount;
    __local_mem__ U_X* y2Addr1 = (__ubuf__ U_X*)y2Local.GetPhyAddr();
    __local_mem__ U_X* y2Addr2 = (__ubuf__ U_X*)y2Local.GetPhyAddr() + calCount;

    __VEC_SCOPE__
    {
        uint32_t sreg = (uint32_t)calCount;
        RegTensor<U_X> yB16Reg1, yB16Reg2;
        RegTensor<float> rstdReg;
        RegTensor<float> xReg1, dst1Reg, gammaFp32Reg1, yReg1;
        RegTensor<float> xReg2, dst2Reg, gammaFp32Reg2, yReg2;
        MaskReg maskReg;
        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + rstdOffset);
        for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
            uint16_t offset = i * V_LENGTH;
            maskReg = UpdateMask<float>(sreg);
            DataCopy(xReg1, xAddr1 + offset);
            DataCopy(xReg2, xAddr2 + offset);
            NormCommon::LoadCastRegVF(gammaFp32Reg1, gammaAddr1, i, maskReg);
            NormCommon::LoadCastRegVF(gammaFp32Reg2, gammaAddr2, i, maskReg);
            Mul(dst1Reg, xReg1, rstdReg, maskReg);
            Mul(dst2Reg, xReg2, rstdReg, maskReg);
            Mul(yReg1, dst1Reg, gammaFp32Reg1, maskReg);
            Mul(yReg2, dst2Reg, gammaFp32Reg2, maskReg);
            DataCopy(y1Addr1 + offset, yReg1, maskReg);
            DataCopy(y1Addr2 + offset, yReg2, maskReg);
            Cast<U_X, float, castTraitB322B16>(yB16Reg1, yReg1, maskReg);
            Cast<U_X, float, castTraitB322B16>(yB16Reg2, yReg2, maskReg);
            DataCopy<U_X, StoreDist::DIST_PACK_B32>(y2Addr1 + offset, yB16Reg1, maskReg);
            DataCopy<U_X, StoreDist::DIST_PACK_B32>(y2Addr2 + offset, yB16Reg2, maskReg);
        }
    }
}

/*!
 * @brief Multi N yLocal = xLocal * rstd * gammaLocal
 *
 * @param y1Local output y1Local
 * @param y2Local output y2Local
 * @param xLocal input xLocal
 * @param gammaLocal input gammaLocal
 * @param rstdLocal input rstd
 * @param rstdOffsetStart elem offset of rstdLocal
 * @param count vector commpute length
 * @param curRows repeat
 * @return void
 */
template <typename U_X, typename U_GAMMA>
__aicore__ inline void ComputeYMulti(
    LocalTensor<float>& y1Local, LocalTensor<U_X>& y2Local, LocalTensor<float>& xLocal,
    LocalTensor<U_GAMMA>& gammaLocal, LocalTensor<float>& rstdLocal, uint64_t rstdOffsetStart, uint64_t count,
    uint64_t curRows)
{
    uint32_t rstdOffset = (uint32_t)rstdOffsetStart;
    uint32_t calCount = (uint32_t)(count / 2); // Unroll
    uint16_t repeatTimes = CeilDivision(calCount, V_LENGTH);

    __local_mem__ float* xAddr1 = (__ubuf__ float*)xLocal.GetPhyAddr();
    __local_mem__ float* xAddr2 = (__ubuf__ float*)xLocal.GetPhyAddr() + calCount;
    __local_mem__ U_GAMMA* gammaAddr1 = (__ubuf__ U_GAMMA*)gammaLocal.GetPhyAddr();
    __local_mem__ U_GAMMA* gammaAddr2 = (__ubuf__ U_GAMMA*)gammaLocal.GetPhyAddr() + calCount;
    __local_mem__ float* rstdAddr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
    __local_mem__ float* y1Addr1 = (__ubuf__ float*)y1Local.GetPhyAddr();
    __local_mem__ float* y1Addr2 = (__ubuf__ float*)y1Local.GetPhyAddr() + calCount;
    __local_mem__ U_X* y2Addr1 = (__ubuf__ U_X*)y2Local.GetPhyAddr();
    __local_mem__ U_X* y2Addr2 = (__ubuf__ U_X*)y2Local.GetPhyAddr() + calCount;

    __VEC_SCOPE__
    {
        for (uint16_t row = 0; row < (uint16_t)curRows; row++) {
            uint32_t sreg = calCount;
            RegTensor<U_X> yB16Reg1, yB16Reg2;
            RegTensor<float> rstdReg;
            RegTensor<float> xReg1, dst1Reg, gammaFp32Reg1, yReg1;
            RegTensor<float> xReg2, dst2Reg, gammaFp32Reg2, yReg2;
            MaskReg maskReg;
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + rstdOffset);
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {
                uint16_t offset = i * V_LENGTH;
                maskReg = UpdateMask<float>(sreg);
                DataCopy(xReg1, xAddr1 + offset);
                DataCopy(xReg2, xAddr2 + offset);
                NormCommon::LoadCastRegVF(gammaFp32Reg1, gammaAddr1, i, maskReg);
                NormCommon::LoadCastRegVF(gammaFp32Reg2, gammaAddr2, i, maskReg);
                Mul(dst1Reg, xReg1, rstdReg, maskReg);
                Mul(dst2Reg, xReg2, rstdReg, maskReg);
                Mul(yReg1, dst1Reg, gammaFp32Reg1, maskReg);
                Mul(yReg2, dst2Reg, gammaFp32Reg2, maskReg);
                DataCopy(y1Addr1 + offset, yReg1, maskReg);
                DataCopy(y1Addr2 + offset, yReg2, maskReg);
                Cast<U_X, float, castTraitB322B16>(yB16Reg1, yReg1, maskReg);
                Cast<U_X, float, castTraitB322B16>(yB16Reg2, yReg2, maskReg);
                DataCopy<U_X, StoreDist::DIST_PACK_B32>(y2Addr1 + offset, yB16Reg1, maskReg);
                DataCopy<U_X, StoreDist::DIST_PACK_B32>(y2Addr2 + offset, yB16Reg2, maskReg);
            }
            rstdOffset++;
            xAddr1 += count;
            xAddr2 += count;
            y1Addr1 += count;
            y1Addr2 += count;
            y2Addr1 += count;
            y2Addr2 += count;
        }
    }
}

} // namespace AddRmsNormCast
#endif // OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_RMS_NORM_CAST_REGBASE_COMMON_H
