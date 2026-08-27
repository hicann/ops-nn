/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v4_vf.h
 * \brief
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

namespace Reg = AscendC::Reg;
using AscendC::IsSameType;

namespace QuantBatchMatmulV4 {

template <typename xType, typename wType, typename scaleType>
struct ParamsGroupSize32 {
    uint32_t maskWeight;
    uint16_t outerExtend;
    uint16_t innerExtend;
    uint32_t outerStrideScale;
    uint32_t outerStrideWeight;
    uint32_t dataBlockStride;
    uint32_t repeatStride;
    int32_t outDimOffset;
    __local_mem__ xType* offsetBaseAddr;
    __local_mem__ scaleType* scaleBaseAddr;
    __local_mem__ int8_t* weightInBaseAddr0;
    __local_mem__ int8_t* weightInBaseAddr1;
    __local_mem__ xType* weightOutBaseAddr;
};

template <typename xType, typename wType, typename scaleType>
struct ParamsGroupKN {
    uint32_t groupNumUb;       // UB上一次计算多少个group
    uint32_t vLLoopNumInGroup; // 每个group中需要处理几次256个数据
    uint32_t n1LoopNum;        // UB上一次计算多少次n1
    uint32_t scaleN1Stride;    // scale addrReg n1轴stride
    uint32_t bubNLen;          // scale addrReg group轴stride
    uint32_t weightInGroupIdStride;
    uint32_t weighInN1Stride;
    uint32_t weightOutN1Stride;
    uint32_t weightOutGroupIdStride;
    uint32_t weightOutVlStride;
    __local_mem__ scaleType* scaleBaseAddr0;
    __local_mem__ scaleType* scaleBaseAddr1;
    __local_mem__ uint8_t* scaleMaskBaseAddr;
    __local_mem__ int8_t* weightInBaseAddr0;
    __local_mem__ int8_t* weightInBaseAddr1;
    __local_mem__ xType* weightOutBaseAddr;
};

constexpr Reg::CastTrait castF162F32Trait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                              AscendC::RoundMode::UNKNOWN};
constexpr Reg::CastTrait castF162F32Trait1 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                              AscendC::RoundMode::UNKNOWN};
constexpr Reg::CastTrait castF322F8Trait0 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                             AscendC::RoundMode::CAST_RINT};
constexpr Reg::CastTrait castF322F8Trait2 = {Reg::RegLayout::TWO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING,
                                             AscendC::RoundMode::CAST_RINT};
constexpr Reg::CastTrait castF42F16Trait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                             AscendC::RoundMode::UNKNOWN};
constexpr Reg::CastTrait castBF162FP16Trait0 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::NO_SAT,
                                                Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};

template <typename wType, typename scaleType>
__simd_callee__ inline void CastWeightF4ToF16(Reg::RegTensor<scaleType>& weightF16Reg0,
                                              Reg::RegTensor<scaleType>& weightF16Reg1,
                                              Reg::RegTensor<wType>& weightInReg0, Reg::RegTensor<wType>& weightInReg1,
                                              Reg::MaskReg& maskRegALL)
{
    if constexpr (AscendC::IsSameType<scaleType, half>::value) {
        Reg::RegTensor<bfloat16_t> weightBF16Reg0, weightBF16Reg1;
        // f4 -> bf16
        Reg::Cast<bfloat16_t, wType, castF42F16Trait0>(weightBF16Reg0, weightInReg0, maskRegALL);
        Reg::Cast<bfloat16_t, wType, castF42F16Trait0>(weightBF16Reg1, weightInReg1, maskRegALL);
        // bf16 -> fp16
        Reg::Cast<scaleType, bfloat16_t, castBF162FP16Trait0>(weightF16Reg0, weightBF16Reg0, maskRegALL);
        Reg::Cast<scaleType, bfloat16_t, castBF162FP16Trait0>(weightF16Reg1, weightBF16Reg1, maskRegALL);
    } else {
        Reg::Cast<scaleType, wType, castF42F16Trait0>(weightF16Reg0, weightInReg0, maskRegALL);
        Reg::Cast<scaleType, wType, castF42F16Trait0>(weightF16Reg1, weightInReg1, maskRegALL);
    }
}

template <typename xType, typename wType, typename scaleType, bool hasAntiquantOffset>
__aicore__ inline void AntiquantW4Pergroup32NK(ParamsGroupSize32<xType, wType, scaleType>& p)
{
    static constexpr Reg::CastTrait castTrait0 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                  Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait castTrait1 = {Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
                                                  Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait castTrait2 = {Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT,
                                                  Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait castTrait3 = {Reg::RegLayout::TWO, Reg::SatMode::NO_SAT,
                                                  Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    // +---------------------+------+
    // |      主块           | 尾块 |
    // |    (1)(512 对齐)    |  (2) |
    // |                     |      |
    // |                     |      |
    // |                     |      |
    // +---------------------+------+
    // |        尾块         | 尾块 |
    // |        (3)          |  (4) |
    // +---------------------+------+
    // 当前外轴按照1 unroll循环， >1的时候会有尾块(3)(4)

    Reg::RegTensor<scaleType> scaleLoad, scaleCompute0, scaleCompute1;
    Reg::RegTensor<wType> wLoad0, wLoad1;
    Reg::RegTensor<scaleType> wCvt0, wCvt1, wMul0, wMul1;
    Reg::RegTensor<xType> wCvtB8N0, wCvtB8N1, wCvtB8N2, wCvtB8N3, wSel0, wSel1, wSel2, wSel3;
    Reg::RegTensor<xType> wDIntlv0, wDIntlv1;
    Reg::RegTensor<float> wCvtF32N0, wCvtF32N1, wCvtF32N2, wCvtF32N3;

    Reg::MaskReg maskRegB4 = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
    Reg::MaskReg maskRegB16 = Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
    Reg::MaskReg maskRegVsel = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::M4>();
    Reg::MaskReg maskWeight;
    uint32_t maskWeightTmp;

    // (n, k) -> (k1, n1, n0, k0)
    for (uint16_t outerIdx = 0; outerIdx < p.outerExtend; ++outerIdx) {
        maskWeightTmp = p.maskWeight;
        // 按照一行一行处理
        for (uint16_t innerIdx = 0; innerIdx < p.innerExtend; ++innerIdx) {
            Reg::AddrReg addrRegScale = Reg::CreateAddrReg<scaleType>(outerIdx, p.outerStrideScale, innerIdx, OFFSET_8);
            Reg::AddrReg addrRegWeight = Reg::CreateAddrReg<uint8_t>(outerIdx, p.outerStrideWeight, innerIdx,
                                                                     OFFSET_FOR_4BITS);
            maskWeight = Reg::UpdateMask<xType>(maskWeightTmp);
            // DIST_E2B_B16 表示搬运模式为
            // SRC ： 0 1 2 3 4 5 6 7
            // DST ： 00000000000000001111111111111111222222222222222233333333333333333.............7777777777777777
            Reg::DataCopy<scaleType, Reg::LoadDist::DIST_E2B_B16>(scaleLoad, p.scaleBaseAddr, addrRegScale);
            // Interleave后变为
            // scale0:
            // 00000000000000000000000000000000111111111111111111111111111111111.............333333333333333333333333333333333
            // scale1:
            // 44444444444444444444444444444444555555555555555555555555555555555.............777777777777777777777777777777777
            Reg::Interleave(scaleCompute0, scaleCompute1, scaleLoad, scaleLoad);
            // DIST_UNPACK4_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
            // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
            // Vd 0 1 x x x x x x 2 3 x x x x x x
            // 对于256个数来说， 分2次处理， 每次处理128个数，即64B，应为地址按照int8存的，所以每次偏移64个int8的数
            Reg::DataCopy<uint8_t, Reg::LoadDist::DIST_UNPACK4_B8>(
                (Reg::RegTensor<uint8_t>&)wLoad0, (__local_mem__ uint8_t*)p.weightInBaseAddr0, addrRegWeight);
            Reg::DataCopy<uint8_t, Reg::LoadDist::DIST_UNPACK4_B8>(
                (Reg::RegTensor<uint8_t>&)wLoad1, (__local_mem__ uint8_t*)p.weightInBaseAddr1, addrRegWeight);

            CastWeightF4ToF16<wType, scaleType>(wCvt0, wCvt1, wLoad0, wLoad1, maskRegB4);

            Reg::Mul(wMul0, wCvt0, scaleCompute0, maskRegB16);
            Reg::Mul(wMul1, wCvt1, scaleCompute1, maskRegB16);
            // fp16 to fp32
            Reg::Cast<float, scaleType, castTrait0>(wCvtF32N0, wMul0, maskRegB16);
            Reg::Cast<float, scaleType, castTrait1>(wCvtF32N1, wMul0, maskRegB16);
            Reg::Cast<float, scaleType, castTrait0>(wCvtF32N2, wMul1, maskRegB16);
            Reg::Cast<float, scaleType, castTrait1>(wCvtF32N3, wMul1, maskRegB16);
            // fp32 to fp8
            Reg::Cast<xType, float, castTrait2>(wCvtB8N0, wCvtF32N0, maskRegB16);
            Reg::Cast<xType, float, castTrait3>(wCvtB8N1, wCvtF32N1, maskRegB16);
            Reg::Cast<xType, float, castTrait2>(wCvtB8N2, wCvtF32N2, maskRegB16);
            Reg::Cast<xType, float, castTrait3>(wCvtB8N3, wCvtF32N3, maskRegB16);
            // vsel M4
            Reg::Select(wSel0, wCvtB8N0, wCvtB8N1, maskRegVsel);
            Reg::Select(wSel1, wCvtB8N2, wCvtB8N3, maskRegVsel);
            // 去除无效数据
            Reg::DeInterleave(wDIntlv0, wDIntlv1, wSel0, wSel1);

            Reg::DataCopy<xType, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                p.weightOutBaseAddr, wDIntlv0, p.dataBlockStride, p.repeatStride, maskWeight);
        }
        p.weightOutBaseAddr += p.outDimOffset;
    }
}

template <typename xType, typename wType, typename scaleType, bool hasAntiquantOffset>
__aicore__ inline void AntiquantW4PergroupKN(ParamsGroupKN<xType, wType, scaleType>& param)
{
    Reg::RegTensor<scaleType> scaleRegCompute, scaleRegAssist, weightF16Reg0, weightF16Reg1;
    Reg::RegTensor<wType> weightInReg0, weightInReg1;
    Reg::RegTensor<float> weightF32Reg0, weightF32Reg1, weightF32Reg2, weightF32Reg3;
    Reg::RegTensor<xType> weightF8Reg0, weightF8Reg1, weightF8Reg2, weightF8Reg3, weightF8SelReg0, weightF8SelReg1;
    Reg::MaskReg scaleMaskReg = Reg::CreateMask<uint8_t>();
    Reg::MaskReg maskRegALL = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
    Reg::MaskReg maskRegVsel = Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::M4>();

    Reg::DataCopy(scaleMaskReg, param.scaleMaskBaseAddr);
    for (uint16_t n1Idx = 0; n1Idx < param.n1LoopNum; n1Idx++) {
        for (uint16_t groupIdx = 0; groupIdx < param.groupNumUb; groupIdx++) {
            Reg::AddrReg scaleAddrReg = Reg::CreateAddrReg<scaleType>(n1Idx, param.scaleN1Stride, groupIdx,
                                                                      param.bubNLen);
            Reg::DataCopy<scaleType, Reg::LoadDist::DIST_BLK>(scaleRegCompute, param.scaleBaseAddr0, scaleAddrReg);
            Reg::DataCopy<scaleType, Reg::LoadDist::DIST_BLK>(scaleRegAssist, param.scaleBaseAddr1, scaleAddrReg);
            Reg::Select(scaleRegCompute, scaleRegCompute, scaleRegAssist, scaleMaskReg);
            for (uint16_t vLIdx = 0; vLIdx < param.vLLoopNumInGroup; vLIdx++) {
                Reg::AddrReg weightInAddrReg = Reg::CreateAddrReg<uint8_t>(
                    n1Idx, param.weighInN1Stride, groupIdx, param.weightInGroupIdStride, vLIdx, OFFSET_FOR_4BITS);
                Reg::AddrReg weightOutAddrReg = Reg::CreateAddrReg<uint8_t>(n1Idx, param.weightOutN1Stride, groupIdx,
                                                                            param.weightOutGroupIdStride, vLIdx,
                                                                            param.weightOutVlStride);
                Reg::DataCopy<uint8_t, Reg::LoadDist::DIST_UNPACK4_B8>((Reg::RegTensor<uint8_t>&)weightInReg0,
                                                                       (__local_mem__ uint8_t*)param.weightInBaseAddr0,
                                                                       weightInAddrReg);
                Reg::DataCopy<uint8_t, Reg::LoadDist::DIST_UNPACK4_B8>((Reg::RegTensor<uint8_t>&)weightInReg1,
                                                                       (__local_mem__ uint8_t*)param.weightInBaseAddr1,
                                                                       weightInAddrReg);
                CastWeightF4ToF16<wType, scaleType>(weightF16Reg0, weightF16Reg1, weightInReg0, weightInReg1,
                                                    maskRegALL);
                Reg::Mul(weightF16Reg0, weightF16Reg0, scaleRegCompute, maskRegALL);
                Reg::Mul(weightF16Reg1, weightF16Reg1, scaleRegCompute, maskRegALL);

                Reg::Cast<float, scaleType, castF162F32Trait0>(weightF32Reg0, weightF16Reg0, maskRegALL);
                Reg::Cast<float, scaleType, castF162F32Trait1>(weightF32Reg1, weightF16Reg0, maskRegALL);
                Reg::Cast<float, scaleType, castF162F32Trait0>(weightF32Reg2, weightF16Reg1, maskRegALL);
                Reg::Cast<float, scaleType, castF162F32Trait1>(weightF32Reg3, weightF16Reg1, maskRegALL);
                Reg::Cast<xType, float, castF322F8Trait0>(weightF8Reg0, weightF32Reg0, maskRegALL);
                Reg::Cast<xType, float, castF322F8Trait2>(weightF8Reg1, weightF32Reg1, maskRegALL);
                Reg::Cast<xType, float, castF322F8Trait0>(weightF8Reg2, weightF32Reg2, maskRegALL);
                Reg::Cast<xType, float, castF322F8Trait2>(weightF8Reg3, weightF32Reg3, maskRegALL);
                Reg::Select(weightF8SelReg0, weightF8Reg0, weightF8Reg1, maskRegVsel);
                Reg::Select(weightF8SelReg1, weightF8Reg2, weightF8Reg3, maskRegVsel);

                Reg::DataCopy<xType, Reg::StoreDist::DIST_PACK_B16>(param.weightOutBaseAddr, weightF8SelReg0,
                                                                    weightOutAddrReg, maskRegALL);
                // 第二次数据搬运，UB偏置为128个fp8
                Reg::DataCopy<xType, Reg::StoreDist::DIST_PACK_B16>(param.weightOutBaseAddr + 128, weightF8SelReg1,
                                                                    weightOutAddrReg, maskRegALL);
            }
        }
    }
}

} // namespace QuantBatchMatmulV4
