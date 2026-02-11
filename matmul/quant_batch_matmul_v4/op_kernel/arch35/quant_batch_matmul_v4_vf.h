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

 #ifndef QUANT_BATCH_MATMUL_V4_VF_H
 #define QUANT_BATCH_MATMUL_V4_VF_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif

 namespace MicroAPI = AscendC::MicroAPI;

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
     __local_mem__ xType *offsetBaseAddr;
     __local_mem__ scaleType *scaleBaseAddr;
     __local_mem__ int8_t *weightInBaseAddr0;
     __local_mem__ int8_t *weightInBaseAddr1;
     __local_mem__ xType *weightOutBaseAddr;
 };

 template <typename xType, typename wType, typename scaleType>
 struct ParamsGroupKN {
     uint32_t groupNumUb; // UB上一次计算多少个group
     uint32_t vLLoopNumInGroup; // 每个group中需要处理几次256个数据
     uint32_t n1LoopNum; // UB上一次计算多少次n1
     uint32_t scaleN1Stride; // scale addrReg n1轴stride
     uint32_t bubNLen; // scale addrReg group轴stride
     uint32_t weightInGroupIdStride;
     uint32_t weighInN1Stride;
     uint32_t weightOutN1Stride;
     uint32_t weightOutGroupIdStride;
     uint32_t weightOutVlStride;
     __local_mem__ scaleType *scaleBaseAddr0;
     __local_mem__ scaleType *scaleBaseAddr1;
     __local_mem__ uint8_t *scaleMaskBaseAddr;
     __local_mem__ int8_t *weightInBaseAddr0;
     __local_mem__ int8_t *weightInBaseAddr1;
     __local_mem__ xType *weightOutBaseAddr;
 };

 constexpr MicroAPI::CastTrait castF162F32Trait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
     MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
 constexpr MicroAPI::CastTrait castF162F32Trait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
     MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
 constexpr MicroAPI::CastTrait castF322F8Trait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
     MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
 constexpr MicroAPI::CastTrait castF322F8Trait2 = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::NO_SAT,
     MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
 constexpr MicroAPI::CastTrait castF42F16Trait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
     MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};


 template <typename xType, typename wType, typename scaleType, bool hasAntiquantOffset>
 __aicore__ inline void AntiquantW4Pergroup32NK(ParamsGroupSize32<xType, wType, scaleType> &p)
 {
     static constexpr MicroAPI::CastTrait castTrait0 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
     static constexpr MicroAPI::CastTrait castTrait1 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
     static constexpr MicroAPI::CastTrait castTrait2 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
     static constexpr MicroAPI::CastTrait castTrait3 = {MicroAPI::RegLayout::TWO, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
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
     // 当前外轴按照1 unrool循环， >1的时候会有尾块(3)(4)

     MicroAPI::RegTensor<scaleType> scaleLoad, scaleCompute0, scaleCompute1;
     MicroAPI::RegTensor<wType> wLoad0, wLoad1;
     MicroAPI::RegTensor<scaleType> wCvt0, wCvt1, wMul0, wMul1;
     MicroAPI::RegTensor<xType> wCvtB8N0, wCvtB8N1, wCvtB8N2, wCvtB8N3, wSel0, wSel1, wSel2, wSel3;
     MicroAPI::RegTensor<xType> wDIntlv0, wDIntlv1;
     MicroAPI::RegTensor<float> wCvtF32N0, wCvtF32N1, wCvtF32N2, wCvtF32N3;

     MicroAPI::MaskReg maskRegB4 = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
     MicroAPI::MaskReg maskRegB16 = MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
     MicroAPI::MaskReg maskRegVsel = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::M4>();
     MicroAPI::MaskReg maskWeight;
     uint32_t maskWeightTmp;

     // (n, k) -> (k1, n1, n0, k0)
     for (uint16_t outerIdx = 0; outerIdx < p.outerExtend; ++outerIdx) {
         maskWeightTmp = p.maskWeight;
         // 按照一行一行处理
         for (uint16_t innerIdx = 0; innerIdx < p.innerExtend; ++innerIdx) {
             MicroAPI::AddrReg addrRegScale =
                 MicroAPI::CreateAddrReg<scaleType>(outerIdx, p.outerStrideScale, innerIdx, OFFSET_8);
             MicroAPI::AddrReg addrRegWeight =
                 MicroAPI::CreateAddrReg<uint8_t>(outerIdx, p.outerStrideWeight, innerIdx, OFFSET_FOR_4BITS);
             maskWeight = MicroAPI::UpdateMask<xType>(maskWeightTmp);
             // DIST_E2B_B16 表示搬运模式为
             // SRC ： 0 1 2 3 4 5 6 7
             // DST ： 00000000000000001111111111111111222222222222222233333333333333333.............7777777777777777
             MicroAPI::DataCopy<scaleType, MicroAPI::LoadDist::DIST_E2B_B16>(
                 scaleLoad, p.scaleBaseAddr, addrRegScale);
             // Interleave后变为
             // scale0:
             // 00000000000000000000000000000000111111111111111111111111111111111.............333333333333333333333333333333333
             // scale1:
             // 44444444444444444444444444444444555555555555555555555555555555555.............777777777777777777777777777777777
             MicroAPI::Interleave(scaleCompute0, scaleCompute1, scaleLoad, scaleLoad);
             // DIST_UNPACK4_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
             // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
             // Vd 0 1 x x x x x x 2 3 x x x x x x
             // 对于256个数来说， 分2次处理， 每次处理128个数，即64B，应为地址按照int8存的，所以每次偏移64个int8的数
             MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                 (MicroAPI::RegTensor<uint8_t>&)wLoad0,
                 (__local_mem__  uint8_t*)p.weightInBaseAddr0, addrRegWeight);
             MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                 (MicroAPI::RegTensor<uint8_t>&)wLoad1,
                 (__local_mem__  uint8_t*)p.weightInBaseAddr1, addrRegWeight);

             MicroAPI::Cast<scaleType, wType, castTrait0>(wCvt0, wLoad0, maskRegB4);
             MicroAPI::Cast<scaleType, wType, castTrait0>(wCvt1, wLoad1, maskRegB4);

             MicroAPI::Mul(wMul0, wCvt0, scaleCompute0, maskRegB16);
             MicroAPI::Mul(wMul1, wCvt1, scaleCompute1, maskRegB16);
             // fp16 to fp32
             MicroAPI::Cast<float, scaleType, castTrait0>(wCvtF32N0, wMul0, maskRegB16);
             MicroAPI::Cast<float, scaleType, castTrait1>(wCvtF32N1, wMul0, maskRegB16);
             MicroAPI::Cast<float, scaleType, castTrait0>(wCvtF32N2, wMul1, maskRegB16);
             MicroAPI::Cast<float, scaleType, castTrait1>(wCvtF32N3, wMul1, maskRegB16);
             // fp32 to fp8
             MicroAPI::Cast<xType, float, castTrait2>(wCvtB8N0, wCvtF32N0, maskRegB16);
             MicroAPI::Cast<xType, float, castTrait3>(wCvtB8N1, wCvtF32N1, maskRegB16);
             MicroAPI::Cast<xType, float, castTrait2>(wCvtB8N2, wCvtF32N2, maskRegB16);
             MicroAPI::Cast<xType, float, castTrait3>(wCvtB8N3, wCvtF32N3, maskRegB16);
             // vsel M4
             MicroAPI::Select(wSel0, wCvtB8N0, wCvtB8N1, maskRegVsel);
             MicroAPI::Select(wSel1, wCvtB8N2, wCvtB8N3, maskRegVsel);
             // 去除无效数据
             MicroAPI::DeInterleave(wDIntlv0, wDIntlv1, wSel0, wSel1);

             MicroAPI::DataCopy<xType, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                 p.weightOutBaseAddr, wDIntlv0, p.dataBlockStride, p.repeatStride, maskWeight);
         }
         p.weightOutBaseAddr += p.outDimOffset;
     }
 }

 template <typename xType, typename wType, typename scaleType, bool hasAntiquantOffset>
 __aicore__ inline void AntiquantW4PergroupKN(ParamsGroupKN<xType, wType, scaleType> &param)
 {
     MicroAPI::RegTensor<scaleType> scaleRegCompute, scaleRegAssist, weightF16Reg0, weightF16Reg1;
     MicroAPI::RegTensor<wType> weightInReg0, weightInReg1;
     MicroAPI::RegTensor<float> weightF32Reg0, weightF32Reg1, weightF32Reg2, weightF32Reg3;
     MicroAPI::RegTensor<xType> weightF8Reg0, weightF8Reg1, weightF8Reg2, weightF8Reg3, weightF8SelReg0, weightF8SelReg1;
     MicroAPI::MaskReg scaleMaskReg = MicroAPI::CreateMask<uint8_t>();
     MicroAPI::MaskReg maskRegALL = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
     MicroAPI::MaskReg maskRegVsel = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::M4>();

     MicroAPI::DataCopy(scaleMaskReg, param.scaleMaskBaseAddr);
     for (uint16_t n1Idx = 0; n1Idx < param.n1LoopNum; n1Idx++) {
         for (uint16_t groupIdx = 0; groupIdx < param.groupNumUb; groupIdx++) {
             MicroAPI::AddrReg scaleAddrReg =
                 MicroAPI::CreateAddrReg<scaleType>(n1Idx, param.scaleN1Stride, groupIdx, param.bubNLen);
             MicroAPI::DataCopy<scaleType, MicroAPI::LoadDist::DIST_BLK>(scaleRegCompute, param.scaleBaseAddr0, scaleAddrReg);
             MicroAPI::DataCopy<scaleType, MicroAPI::LoadDist::DIST_BLK>(scaleRegAssist, param.scaleBaseAddr1, scaleAddrReg);
             MicroAPI::Select(scaleRegCompute, scaleRegCompute, scaleRegAssist, scaleMaskReg);
             for (uint16_t vLIdx = 0; vLIdx < param.vLLoopNumInGroup; vLIdx++) {
                 MicroAPI::AddrReg weightInAddrReg =
                     MicroAPI::CreateAddrReg<uint8_t>(n1Idx, param.weighInN1Stride, groupIdx, param.weightInGroupIdStride,
                     vLIdx, OFFSET_FOR_4BITS);
                 MicroAPI::AddrReg weightOutAddrReg = MicroAPI::CreateAddrReg<uint8_t>(n1Idx, param.weightOutN1Stride,
                     groupIdx, param.weightOutGroupIdStride, vLIdx, param.weightOutVlStride);
                 MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                      (MicroAPI::RegTensor<uint8_t>&)weightInReg0, (__local_mem__  uint8_t*)param.weightInBaseAddr0,
                      weightInAddrReg);
                 MicroAPI::DataCopy<uint8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                      (MicroAPI::RegTensor<uint8_t>&)weightInReg1, (__local_mem__  uint8_t*)param.weightInBaseAddr1,
                      weightInAddrReg);
                 MicroAPI::Cast<scaleType, wType, castF42F16Trait0>(weightF16Reg0, weightInReg0, maskRegALL);
                 MicroAPI::Cast<scaleType, wType, castF42F16Trait0>(weightF16Reg1, weightInReg1, maskRegALL);
                 MicroAPI::Mul(weightF16Reg0, weightF16Reg0, scaleRegCompute, maskRegALL);
                 MicroAPI::Mul(weightF16Reg1, weightF16Reg1, scaleRegCompute, maskRegALL);

                 MicroAPI::Cast<float, scaleType, castF162F32Trait0>(weightF32Reg0, weightF16Reg0, maskRegALL);
                 MicroAPI::Cast<float, scaleType, castF162F32Trait1>(weightF32Reg1, weightF16Reg0, maskRegALL);
                 MicroAPI::Cast<float, scaleType, castF162F32Trait0>(weightF32Reg2, weightF16Reg1, maskRegALL);
                 MicroAPI::Cast<float, scaleType, castF162F32Trait1>(weightF32Reg3, weightF16Reg1, maskRegALL);
                 MicroAPI::Cast<xType, float, castF322F8Trait0>(weightF8Reg0, weightF32Reg0, maskRegALL);
                 MicroAPI::Cast<xType, float, castF322F8Trait2>(weightF8Reg1, weightF32Reg1, maskRegALL);
                 MicroAPI::Cast<xType, float, castF322F8Trait0>(weightF8Reg2, weightF32Reg2, maskRegALL);
                 MicroAPI::Cast<xType, float, castF322F8Trait2>(weightF8Reg3, weightF32Reg3, maskRegALL);
                 MicroAPI::Select(weightF8SelReg0, weightF8Reg0, weightF8Reg1, maskRegVsel);
                 MicroAPI::Select(weightF8SelReg1, weightF8Reg2, weightF8Reg3, maskRegVsel);

                 MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_PACK_B16>(
                    param.weightOutBaseAddr, weightF8SelReg0, weightOutAddrReg, maskRegALL);
                 // 第二次数据搬运，UB偏置为128个fp8
                 MicroAPI::DataCopy<xType, MicroAPI::StoreDist::DIST_PACK_B16>(
                    param.weightOutBaseAddr + 128, weightF8SelReg1, weightOutAddrReg, maskRegALL);
             }
         }
     }
 }

 }  // namespace QuantBatchMatmulV4

 #endif  // QUANT_BATCH_MATMUL_V4_VF_H