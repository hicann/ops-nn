/**

 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PROLOGUE_TILE_CAST_MX_H
#define PROLOGUE_TILE_CAST_MX_H
#include "kernel_operator_intf.h"
#include "../../utils/math_utils.h"
#include "../../utils/underscore.h"

namespace MicroAPI = AscendC::MicroAPI;
using Act::Gemm::Get;
using AscendC::BLOCK_CUBE;
using AscendC::VECTOR_REG_WIDTH;
using AscendC::MicroAPI::AddrReg;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

namespace WeightQuantBatchMatmulV2::Arch35::Act::Prologue::Tile {
namespace detail {
using Act::CeilAlign;
using Act::CeilDiv;
// ND NK
template <class TensorTraitOut, class TensorTraitIn, class Shape>
struct TileCastImpl<
    Arch::Ascend910_95, AscendC::LocalTensor<TensorTraitOut>, AscendC::LocalTensor<TensorTraitIn>, Shape,
    typename AscendC::Std::enable_if_t<
        IsRowMajor2D<decltype(TensorTraitIn{}.GetLayout())>::value // 判断NK场景
        && AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, fp8_e8m0_t>>> {
    using DtypeOut = AscendC::PrimT<TensorTraitOut>;
    using DtypeIn = AscendC::PrimT<TensorTraitIn>;
    __aicore__ inline static void Run(
        const LocalTensor<TensorTraitOut>& tensorOut, const AscendC::LocalTensor<TensorTraitIn>& tensorIn,
        const Shape& shape)
    {
        uint16_t ubLoopN = CeilDiv(Get<0>(shape), static_cast<uint64_t>(4));
        __ubuf__ uint8_t* antiQuantScaleBasePhyAddr = (__ubuf__ uint8_t*)tensorIn.GetPhyAddr();
        __ubuf__ DtypeOut* antiQuantScaleF16PhyAddr0 = (__ubuf__ DtypeOut*)tensorOut.GetPhyAddr();
        __ubuf__ DtypeOut* antiQuantScaleF16PhyAddr1 = antiQuantScaleF16PhyAddr0 + (VECTOR_REG_SIZE<DtypeOut>);
        __VEC_SCOPE__
        {
            RegTensor<uint8_t> antiQuantScaleE8m0Vreg0;
            RegTensor<uint8_t> antiQuantScaleE8m0Vreg1;
            RegTensor<DtypeOut> antiQuantScaleF16Vreg0;
            RegTensor<DtypeOut> antiQuantScaleF16Vreg1;
            MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t ubLoopNIdx = 0; ubLoopNIdx < ubLoopN; ubLoopNIdx++) {
                // 搬运128个E8M0的antiquantscale, 通过两倍上采样变成256个E8M0， DIST_US_B8表示搬运模式如下：
                // Vn s1 s2 s3 s4 s5 s6 s7 s8 s9 ...... s125 s126 s127 s128
                // Vd s1 s1 s2 s2 s3 s3 s4 s4 s5 ...... s125 s125 s126 s126 s127 s127 s128 s128
                MicroAPI::LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_US_B8>(
                    antiQuantScaleE8m0Vreg0, antiQuantScaleBasePhyAddr + ubLoopNIdx * 128);
                RegTensor<uint8_t> zeroVreg;
                MicroAPI::Duplicate(zeroVreg, 0);
                // 通过数据重排指令， 交织 antiQuantScaleE8m0Vreg0 和 zeroVreg , Interleave后变为
                // antiQuantScaleE8m0Vreg0
                // Vn s1 0 s2 0 s3 0 s4 0 s5 0 s6 0 s7 0 s8 0....... s127 0 s128 0
                // antiQuantScaleE8m0Vreg1
                // Vd s128 0 s129 0 s130 0 s131 0 s132 0 s133 0....... s255 0 s256 0
                MicroAPI::Interleave(
                    antiQuantScaleE8m0Vreg0, antiQuantScaleE8m0Vreg1, zeroVreg, antiQuantScaleE8m0Vreg0);
                if constexpr (!IsSameType<typename MicroAPI::TypeGet<DtypeOut>::T, vector_f16>::value) {
                    // 逻辑右移一位,得到BF16{S1E8M7}的排列,比如s1 0 =[8bit, 8bit]= [10101011 00000000]
                    // 变为 S1 = [16bit] = [0101010110000000]
                    // antiQuantScaleF16Vreg0：
                    // S1 S2 S3 S4 S5 S6 S7 S8 ..... S127 S128
                    // antiQuantScaleF16Vreg1:
                    // S128 S129 S130 S131 S132 S133 S134 S135 ..... S255 S256
                    MicroAPI::ShiftRights(
                        (MicroAPI::RegTensor<uint16_t>&)antiQuantScaleF16Vreg0,
                        (MicroAPI::RegTensor<uint16_t>&)antiQuantScaleE8m0Vreg0, SHIFT_FOR_BF16, maskAll);
                    MicroAPI::ShiftRights(
                        (MicroAPI::RegTensor<uint16_t>&)antiQuantScaleF16Vreg1,
                        (MicroAPI::RegTensor<uint16_t>&)antiQuantScaleE8m0Vreg1, SHIFT_FOR_BF16, maskAll);
                } else {
                    // 需要转换为FP16格式时，先转换为BF16再转换为FP16
                    RegTensor<uint16_t> antiQuantScaleBF16Vreg0;
                    RegTensor<uint16_t> antiQuantScaleBF16Vreg1;
                    MicroAPI::ShiftRights(
                        (MicroAPI::RegTensor<uint16_t>&)antiQuantScaleBF16Vreg0,
                        (MicroAPI::RegTensor<uint16_t>&)antiQuantScaleE8m0Vreg0, SHIFT_FOR_BF16, maskAll);
                    MicroAPI::ShiftRights(
                        (MicroAPI::RegTensor<uint16_t>&)antiQuantScaleBF16Vreg1,
                        (MicroAPI::RegTensor<uint16_t>&)antiQuantScaleE8m0Vreg1, SHIFT_FOR_BF16, maskAll);
                    MicroAPI::Cast<DtypeOut, bfloat16_t, CAST_BF16_TO_FP16_TRAIT>(
                        antiQuantScaleF16Vreg0, (MicroAPI::RegTensor<bfloat16_t>&)antiQuantScaleBF16Vreg0, maskAll);
                    MicroAPI::Cast<DtypeOut, bfloat16_t, CAST_BF16_TO_FP16_TRAIT>(
                        antiQuantScaleF16Vreg1, (MicroAPI::RegTensor<bfloat16_t>&)antiQuantScaleBF16Vreg1, maskAll);
                }
                MicroAPI::StoreAlign<DtypeOut, MicroAPI::StoreDist::DIST_NORM_B16>(
                    antiQuantScaleF16PhyAddr0 + ubLoopNIdx * AscendC::VECTOR_REG_WIDTH, antiQuantScaleF16Vreg0,
                    maskAll);
                MicroAPI::StoreAlign<DtypeOut, MicroAPI::StoreDist::DIST_NORM_B16>(
                    antiQuantScaleF16PhyAddr1 + ubLoopNIdx * AscendC::VECTOR_REG_WIDTH, antiQuantScaleF16Vreg1,
                    maskAll);
            }
        }
    }
};

// ND/NZ KN
template <class TensorTraitOut, class TensorTraitIn, class Shape>
struct TileCastImpl<
    Arch::Ascend910_95, AscendC::LocalTensor<TensorTraitOut>, AscendC::LocalTensor<TensorTraitIn>, Shape,
    typename AscendC::Std::enable_if_t<
        IsColumnMajor2D<decltype(TensorTraitIn{}.GetLayout())>::value // 判断KN场景
        && AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, fp8_e8m0_t>>> {
    using DtypeOut = AscendC::PrimT<TensorTraitOut>;
    using DtypeIn = AscendC::PrimT<TensorTraitIn>;
    __aicore__ inline static void Run(
        const LocalTensor<TensorTraitOut>& tensorOut, const AscendC::LocalTensor<TensorTraitIn>& tensorIn,
        const Shape& shape)
    {
        uint16_t ubLoopK = static_cast<uint16_t>(Get<1>(shape));
        __ubuf__ uint8_t* antiQuantScaleBasePhyAddr = (__ubuf__ uint8_t*)tensorIn.GetPhyAddr();
        __ubuf__ DtypeOut* antiQuantScaleF16PhyAddr0 = (__ubuf__ DtypeOut*)tensorOut.GetPhyAddr();
        __ubuf__ DtypeOut* antiQuantScaleF16PhyAddr1 = antiQuantScaleF16PhyAddr0 + (AscendC::VECTOR_REG_WIDTH >> 1);
        __VEC_SCOPE__
        {
            // KN mte2搬运的antiquantscale的标准大小为(4，256), 按照一行的粒度处理
            RegTensor<uint8_t> antiQuantScaleE8m0Vreg0;
            RegTensor<uint8_t> antiQuantScaleE8m0Vreg1;
            RegTensor<DtypeOut> antiQuantScaleF16Vreg0;
            RegTensor<DtypeOut> antiQuantScaleF16Vreg1;
            MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t ubLoopKIdx = 0; ubLoopKIdx < ubLoopK; ubLoopKIdx++) {
                // 搬运256个E8M0的antiquantscale, DIST_NORM表示搬运模式如下：
                // Vn s1 s2 s3 s4 s5 s6 s7 s8 s9 ...... s254 s255 s256
                // Vd s1 s2 s3 s4 s5 s6 s7 s8 s9 ...... s254 s255 s256
                MicroAPI::LoadAlign<uint8_t, MicroAPI::LoadDist::DIST_NORM>(
                    antiQuantScaleE8m0Vreg0, antiQuantScaleBasePhyAddr + ubLoopKIdx * VECTOR_REG_WIDTH);
                RegTensor<uint8_t> zeroVreg;
                MicroAPI::Duplicate(zeroVreg, 0);
                MicroAPI::Interleave(
                    antiQuantScaleE8m0Vreg0, antiQuantScaleE8m0Vreg1, zeroVreg, antiQuantScaleE8m0Vreg0);
                if constexpr (!IsSameType<typename MicroAPI::TypeGet<DtypeOut>::T, vector_f16>::value) {
                    // 逻辑右移一位,得到BF16{S1E8M7}的排列,比如s1 0 =[8bit, 8bit]= [10101011 00000000]
                    // 变为 S1 = [16bit] = [0101010110000000]
                    // antiQuantScaleF16Vreg0：
                    // S1 S2 S3 S4 S5 S6 S7 S8 ..... S127 S128
                    // antiQuantScaleF16Vreg1:
                    // S128 S129 S130 S131 S132 S133 S134 S135 ..... S255 S256
                    MicroAPI::ShiftRights(
                        (MicroAPI::RegTensor<uint16_t>&)antiQuantScaleF16Vreg0,
                        (MicroAPI::RegTensor<uint16_t>&)antiQuantScaleE8m0Vreg0, SHIFT_FOR_BF16, maskAll);
                    MicroAPI::ShiftRights(
                        (MicroAPI::RegTensor<uint16_t>&)antiQuantScaleF16Vreg1,
                        (MicroAPI::RegTensor<uint16_t>&)antiQuantScaleE8m0Vreg1, SHIFT_FOR_BF16, maskAll);
                } else {
                    // 需要转换为FP16格式时，先转换为BF16再转换为FP16
                    RegTensor<uint16_t> antiQuantScaleBF16Vreg0;
                    RegTensor<uint16_t> antiQuantScaleBF16Vreg1;
                    MicroAPI::ShiftRights(
                        (MicroAPI::RegTensor<uint16_t>&)antiQuantScaleBF16Vreg0,
                        (MicroAPI::RegTensor<uint16_t>&)antiQuantScaleE8m0Vreg0, SHIFT_FOR_BF16, maskAll);
                    MicroAPI::ShiftRights(
                        (MicroAPI::RegTensor<uint16_t>&)antiQuantScaleBF16Vreg1,
                        (MicroAPI::RegTensor<uint16_t>&)antiQuantScaleE8m0Vreg1, SHIFT_FOR_BF16, maskAll);
                    MicroAPI::Cast<DtypeOut, bfloat16_t, CAST_BF16_TO_FP16_TRAIT>(
                        antiQuantScaleF16Vreg0, (MicroAPI::RegTensor<bfloat16_t>&)antiQuantScaleBF16Vreg0, maskAll);
                    MicroAPI::Cast<DtypeOut, bfloat16_t, CAST_BF16_TO_FP16_TRAIT>(
                        antiQuantScaleF16Vreg1, (MicroAPI::RegTensor<bfloat16_t>&)antiQuantScaleBF16Vreg1, maskAll);
                }
                MicroAPI::StoreAlign<DtypeOut, MicroAPI::StoreDist::DIST_NORM_B16>(
                    antiQuantScaleF16PhyAddr0 + ubLoopKIdx * VECTOR_REG_WIDTH, antiQuantScaleF16Vreg0, maskAll);
                MicroAPI::StoreAlign<DtypeOut, MicroAPI::StoreDist::DIST_NORM_B16>(
                    antiQuantScaleF16PhyAddr1 + ubLoopKIdx * VECTOR_REG_WIDTH, antiQuantScaleF16Vreg1, maskAll);
            }
        }
    }
};
} // namespace detail
} // namespace WeightQuantBatchMatmulV2::Arch35::Act::Prologue::Tile
#endif