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
 * \file pool_index_without_pad_align.h
 * \brief MaxPoolWithArgmaxV3 / MaxPoolGrad NCHW 共用的线性索引到 (h, w) 索引换算接口（含 NC 与快速除法变体）。
 */

#ifndef POOL_UTILS_ARCH35_INDEX_POOL_INDEX_WITHOUT_PAD_ALIGN_H_
#define POOL_UTILS_ARCH35_INDEX_POOL_INDEX_WITHOUT_PAD_ALIGN_H_

#include <cstdint>
#include <type_traits>

#include "kernel_operator.h"
#include "pool_utils/arch35/compute/pool_fast_div.h"

namespace PoolUtils {
namespace Index {

/*
 * 功能：把线性索引换算为 hIndex * wInputActualNoPad + wIndex，IS_PAD 时把负索引钳到 0。
 */
template <typename T2, const uint32_t IS_PAD>
__aicore__ inline void ConvertIndexWithoutPadAlignCommon(AscendC::Reg::RegTensor<int32_t>& srcReg,
                                                         uint32_t wStrideOffset, T2 left, T2 wInputActualNoPad,
                                                         T2 hIndexBase, AscendC::Reg::RegTensor<T2>& dstReg,
                                                         int32_t ncInputOffset)
{
    AscendC::Reg::RegTensor<T2> hIndexReg;
    AscendC::Reg::RegTensor<int32_t> constReg;
    AscendC::Reg::RegTensor<int32_t> divResultReg;
    AscendC::Reg::RegTensor<T2> divResultRegUnpack;
    AscendC::Reg::RegTensor<T2> wIndexReg;
    AscendC::Reg::RegTensor<int32_t> wIndexRegUnpack;
    AscendC::Reg::RegTensor<T2> zeroReg;
    AscendC::Reg::MaskReg negInfMask;
    AscendC::Reg::MaskReg allMaskB32 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg allMaskT2 = AscendC::Reg::CreateMask<T2, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Duplicate(constReg, static_cast<int32_t>(wStrideOffset));
    AscendC::Reg::Duplicate(zeroReg, static_cast<T2>(0));
    AscendC::Reg::Adds(srcReg, srcReg, -ncInputOffset, allMaskB32);
    AscendC::Reg::Div(divResultReg, srcReg, constReg, allMaskB32);
    if constexpr (std::is_same<T2, int64_t>::value) {
        AscendC::Reg::UnPack(divResultRegUnpack, divResultReg);
        AscendC::Reg::Adds(hIndexReg, divResultRegUnpack, hIndexBase, allMaskT2);
    } else {
        AscendC::Reg::Adds(hIndexReg, divResultReg, hIndexBase, allMaskB32);
    }
    if constexpr (IS_PAD) {
        AscendC::Reg::Compare<T2, AscendC::CMPMODE::LT>(negInfMask, hIndexReg, zeroReg, allMaskT2);
        AscendC::Reg::Select(hIndexReg, zeroReg, hIndexReg, negInfMask);
    }
    AscendC::Reg::Muls(hIndexReg, hIndexReg, wInputActualNoPad, allMaskT2);
    AscendC::Reg::Mul(divResultReg, divResultReg, constReg, allMaskB32);
    AscendC::Reg::Sub(wIndexRegUnpack, srcReg, divResultReg, allMaskB32);
    if constexpr (std::is_same<T2, int64_t>::value) {
        AscendC::Reg::UnPack(wIndexReg, wIndexRegUnpack);
        AscendC::Reg::Adds(wIndexReg, wIndexReg, left, allMaskT2);
    } else {
        AscendC::Reg::Adds(wIndexReg, wIndexRegUnpack, left, allMaskB32);
    }
    if constexpr (IS_PAD) {
        AscendC::Reg::Compare<T2, AscendC::CMPMODE::LT>(negInfMask, wIndexReg, zeroReg, allMaskT2);
        AscendC::Reg::Select(wIndexReg, zeroReg, wIndexReg, negInfMask);
    }
    AscendC::Reg::Add(dstReg, hIndexReg, wIndexReg, allMaskT2);
    return;
}

/*
 * 功能：ConvertIndexWithoutPadAlignCommon 的多 NC 变体，先扣除 NC 维基址再做 (h, w) 换算。
 */
template <typename T2, const uint32_t IS_PAD>
__aicore__ inline void ConvertIndexWithoutPadAlignNcCommon(AscendC::Reg::RegTensor<int32_t>& srcReg,
                                                           uint32_t wStrideOffset, T2 left, T2 wInputActualNoPad,
                                                           T2 hIndexBase, AscendC::Reg::RegTensor<T2>& dstReg,
                                                           int32_t ncInputOffset, int32_t ncOutputCount,
                                                           int32_t inputNcSize)
{
    AscendC::Reg::RegTensor<int32_t> ncIndexReg;
    AscendC::Reg::RegTensor<int32_t> divResultReg;
    AscendC::Reg::RegTensor<int32_t> constReg;
    AscendC::Reg::MaskReg allMaskB32 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Arange(ncIndexReg, static_cast<int32_t>(0));
    AscendC::Reg::Duplicate(constReg, static_cast<int32_t>(ncOutputCount));
    AscendC::Reg::Div(divResultReg, ncIndexReg, constReg, allMaskB32);
    AscendC::Reg::Muls(divResultReg, divResultReg, inputNcSize, allMaskB32);
    AscendC::Reg::Sub(srcReg, srcReg, divResultReg, allMaskB32);

    ConvertIndexWithoutPadAlignCommon<T2, IS_PAD>(srcReg, wStrideOffset, left, wInputActualNoPad, hIndexBase, dstReg,
                                                  ncInputOffset);
}

/*
 * 功能：ConvertIndexWithoutPadAlignCommon 的快速除法变体，用 magic/shift 代替 Div 指令。
 */
template <const uint32_t IS_PAD>
__aicore__ inline void ConvertIndexWithoutPadAlignCommonFastDiv(AscendC::Reg::RegTensor<int32_t>& srcReg,
                                                                uint32_t wStrideOffset, int32_t left,
                                                                int32_t wInputActualNoPad, int32_t hIndexBase,
                                                                AscendC::Reg::RegTensor<int32_t>& dstReg,
                                                                int32_t ncInputOffset, uint32_t magic, uint32_t shift)
{
    AscendC::Reg::RegTensor<int32_t> hIndexReg;
    AscendC::Reg::RegTensor<int32_t> wIndexReg;
    AscendC::Reg::RegTensor<int32_t> zeroReg;
    AscendC::Reg::RegTensor<uint32_t> divResultU32;
    AscendC::Reg::RegTensor<uint32_t> magicReg;
    AscendC::Reg::MaskReg negInfMask;
    AscendC::Reg::MaskReg allMaskB32 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Duplicate(zeroReg, static_cast<int32_t>(0));
    AscendC::Reg::Duplicate(magicReg, magic);
    AscendC::Reg::Adds(srcReg, srcReg, -ncInputOffset, allMaskB32);

    PoolUtils::Compute::FastDivImpl(divResultU32, (AscendC::Reg::RegTensor<uint32_t>&)srcReg, magicReg,
                                    static_cast<int16_t>(shift), allMaskB32);

    AscendC::Reg::Adds(hIndexReg, (AscendC::Reg::RegTensor<int32_t>&)divResultU32, hIndexBase, allMaskB32);

    if constexpr (IS_PAD) {
        AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::LT>(negInfMask, hIndexReg, zeroReg, allMaskB32);
        AscendC::Reg::Select(hIndexReg, zeroReg, hIndexReg, negInfMask);
    }

    AscendC::Reg::Muls(hIndexReg, hIndexReg, wInputActualNoPad, allMaskB32);

    AscendC::Reg::Muls(divResultU32, divResultU32, wStrideOffset, allMaskB32);
    AscendC::Reg::Sub((AscendC::Reg::RegTensor<uint32_t>&)srcReg, (AscendC::Reg::RegTensor<uint32_t>&)srcReg,
                      divResultU32, allMaskB32);
    AscendC::Reg::Adds(wIndexReg, srcReg, left, allMaskB32);

    if constexpr (IS_PAD) {
        AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::LT>(negInfMask, wIndexReg, zeroReg, allMaskB32);
        AscendC::Reg::Select(wIndexReg, zeroReg, wIndexReg, negInfMask);
    }

    AscendC::Reg::Add(dstReg, hIndexReg, wIndexReg, allMaskB32);
}

/*
 * 功能：ConvertIndexWithoutPadAlignNcCommon 的快速除法变体。
 */
template <const uint32_t IS_PAD>
__aicore__ inline void ConvertIndexWithoutPadAlignNcCommonFastDiv(
    AscendC::Reg::RegTensor<int32_t>& srcReg, uint32_t wStrideOffset, int32_t left, int32_t wInputActualNoPad,
    int32_t hIndexBase, AscendC::Reg::RegTensor<int32_t>& dstReg, int32_t ncInputOffset, int32_t ncOutputCount,
    int32_t inputNcSize, uint32_t magicNc, uint32_t shiftNc, uint32_t magicWStride, uint32_t shiftWStride)
{
    AscendC::Reg::RegTensor<int32_t> ncIndexReg;
    AscendC::Reg::RegTensor<uint32_t> divResultU32;
    AscendC::Reg::RegTensor<uint32_t> magicReg;
    AscendC::Reg::MaskReg allMaskB32 = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();

    AscendC::Reg::Duplicate(magicReg, magicNc);
    AscendC::Reg::Arange(ncIndexReg, static_cast<int32_t>(0));
    PoolUtils::Compute::FastDivImpl(divResultU32, (AscendC::Reg::RegTensor<uint32_t>&)ncIndexReg, magicReg,
                                    static_cast<int16_t>(shiftNc), allMaskB32);
    AscendC::Reg::Muls(ncIndexReg, (AscendC::Reg::RegTensor<int32_t>&)divResultU32, inputNcSize, allMaskB32);
    AscendC::Reg::Sub(srcReg, srcReg, ncIndexReg, allMaskB32);

    ConvertIndexWithoutPadAlignCommonFastDiv<IS_PAD>(srcReg, wStrideOffset, left, wInputActualNoPad, hIndexBase, dstReg,
                                                     ncInputOffset, magicWStride, shiftWStride);
}

} // namespace Index
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_INDEX_POOL_INDEX_WITHOUT_PAD_ALIGN_H_
