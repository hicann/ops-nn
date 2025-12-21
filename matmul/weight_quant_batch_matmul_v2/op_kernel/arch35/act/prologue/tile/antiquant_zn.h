/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PROLOGUE_TILE_ANTIQUANT_ZN_H
#define PROLOGUE_TILE_ANTIQUANT_ZN_H
#include "../../utils/underscore.h"
#include "kernel_operator_intf.h"
namespace WeightQuantBatchMatmulV2::Arch35::Act::Prologue::Tile {
using AscendC::MicroAPI::RegTensor;
namespace MicroAPI = AscendC::MicroAPI;
namespace detail {
// kn
// int4
// per-channel
template <
    int32_t K, int32_t BufNum, class TensorOut, class TensorTraitIn, class TensorScale, class Shape,
    bool HasAntiQuantOffset>
struct AntiquantImpl<
    // fix n to 64
    Arch::Ascend910_95, AntiquantFixTilePrivate<64, K, BufNum, HasAntiQuantOffset>, TensorOut,
    AscendC::LocalTensor<TensorTraitIn>, TensorScale, Shape,
    typename AscendC::Std::enable_if_t<
        is_2d_zn<decltype(TensorTraitIn{}.GetLayout())>::value &&
        AscendC::Std::is_same_v<AscendC::PrimT<TensorTraitIn>, int4b_t>>> {
    using DtypeIn = int4x2_t;
    using DtypeOut = AscendC::PrimT<AscendC::Std::remove_cvref_t<decltype(TensorOut{}.GetTensorTrait())>>;
    // fix n to 64
    using Policy = AntiquantFixTilePrivate<64, K, BufNum, HasAntiQuantOffset>;
    static constexpr MicroAPI::LoadDist LD_DIST_SCALE = MicroAPI::LoadDist::DIST_BLK;
    static constexpr MicroAPI::LoadDist LD_DIST_W = MicroAPI::LoadDist::DIST_UNPACK4_B8;
    __aicore__ inline static void Run(
        const TensorOut& tensorOut, const AscendC::LocalTensor<TensorTraitIn>& tensorIn, const TensorScale& tensorScale,
        const TensorScale& tensorOffset, const Shape& shape)
    {
        uint64_t n = ::Act::Gemm::Get<0>(shape);
        uint64_t k = ::Act::Gemm::Get<1>(shape);
        __local_mem__ DtypeOut* antiQuantScaleBasePhyAddr = (__local_mem__ DtypeOut*)tensorScale.GetPhyAddr();
        __local_mem__ DtypeOut* antiQuantOffsetBasePhyAddr = (__local_mem__ DtypeOut*)tensorOffset.GetPhyAddr();
        __local_mem__ DtypeIn* weightLowBitPhyAddr = (__local_mem__ DtypeIn*)tensorIn.GetPhyAddr();
        __local_mem__ DtypeOut* weightHighBitPhyAddr = (__local_mem__ DtypeOut*)tensorOut.GetPhyAddr();
        uint64_t loopN1 = AscendC::CeilDiv(n, static_cast<uint64_t>(C0_SIZE<DtypeOut>));
        uint64_t innerDstStride = VECTOR_REG_SIZE<DtypeOut> * Policy::BUF_NUM;
        uint64_t loopInnerNum = AscendC::CeilDiv(
            CeilAlign(k, static_cast<uint64_t>(BLOCK_CUBE)) * C0_SIZE<DtypeOut>, VECTOR_REG_SIZE<DtypeOut>);
        uint64_t loopN1DstStride = loopInnerNum * innerDstStride;
        // (n1,k1,k0,n0)
        __VEC_SCOPE__
        {
            RegTensor<DtypeOut> antiQuantScaleVreg;
            RegTensor<DtypeOut> antiQuantOffsetVreg;
            RegTensor<DtypeIn> weightS4Vreg;
            RegTensor<DtypeOut> weightF16Vreg;
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();

            for (uint16_t loopN1Idx = 0; loopN1Idx < static_cast<uint16_t>(loopN1); loopN1Idx++) {
                // DIST_BLK 的含义为读取一个32B(即16个数)的数据，广播到256B(即128个数)
                MicroAPI::DataCopy<DtypeOut, LD_DIST_SCALE>(
                    antiQuantScaleVreg, antiQuantScaleBasePhyAddr + loopN1Idx * BLOCK_CUBE);
                if constexpr (HasAntiQuantOffset) {
                    MicroAPI::DataCopy<DtypeOut, LD_DIST_SCALE>(
                        antiQuantOffsetVreg, antiQuantOffsetBasePhyAddr + loopN1Idx * BLOCK_CUBE);
                }

                for (uint16_t LoopInnerNumIdx = 0; LoopInnerNumIdx < static_cast<uint16_t>(loopInnerNum);
                     LoopInnerNumIdx++) {
                    // DIST_UNPACK4_B8 表示搬运模式如下，Vn中一个数字4bit(0.5Byte)：
                    // Vn 0 1 2 3 4 5 6 7 8 9 a b c d e f
                    // Vd 0 1 x x x x x x 2 3 x x x x x x
                    MicroAPI::DataCopy<DtypeIn, LD_DIST_W>(
                        weightS4Vreg,
                        (__local_mem__ DtypeIn*)(weightLowBitPhyAddr + loopN1Idx * (BLOCK_CUBE >> 1) * Policy::K +
                                                 LoopInnerNumIdx * (VECTOR_REG_WIDTH >> 2)));
                    // PART_P0 表示按照如下形式处理做cast：
                    // Vn 0 1 x x x x x x 2 3 x x x x x x
                    // Vd 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3
                    MicroAPI::Cast<DtypeOut, DtypeIn, CAST_S4_TO_F16_TRAIT>(weightF16Vreg, weightS4Vreg, maskAll);
                    if constexpr (HasAntiQuantOffset) {
                        MicroAPI::Add(weightF16Vreg, weightF16Vreg, antiQuantOffsetVreg, maskAll);
                    }
                    MicroAPI::Mul(weightF16Vreg, weightF16Vreg, antiQuantScaleVreg, maskAll);

                    AddrReg weightF16PhyAddrReg =
                        MicroAPI::CreateAddrReg<DtypeOut>(loopN1Idx, loopN1DstStride, LoopInnerNumIdx, innerDstStride);
                    MicroAPI::DataCopy<DtypeOut, MicroAPI::StoreDist::DIST_NORM_B16>(
                        weightHighBitPhyAddr, weightF16Vreg, weightF16PhyAddrReg, maskAll);
                }
            }
        }
    }
};
} // namespace detail
} // namespace WeightQuantBatchMatmulV2::Arch35::Act::Prologue::Tile
#endif