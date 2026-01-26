/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ARCH35_CMCT_CONVERTOR_H
#define ARCH35_CMCT_CONVERTOR_H
#include "cmct/policy/dispatch_policy.h"
#include "cmct/kernel/kernel_matmul_a_prefetch_b_antiquant.h"
#include "cmct/block/block_scheduler_tail_resplit_expanded.h"
#include "cmct/block/block_mmad.h"
#include "cmct/prologue/block_prologue.h"
#include "cmct/prologue/dispatch_policy.h"
#include "cmct/utils/math_utils.h"
#include "cmct/utils/integral_constant.h"
#include "cmct/utils/gemm_type.h"
#include "cmct/utils/constant.h"

using AscendC::int4b_t;

namespace WeightQuantBatchMatmulV2 {
using Cmct::CeilAlign;
using Cmct::CeilDiv;
using Cmct::Gemm::GemmType;
using Cmct::Gemm::KB_ELEM;
using Cmct::Gemm::MmadAPrefetchBAntiquantScmc;
using Cmct::Gemm::Arch::Ascend910_95;
using Cmct::Gemm::Block::BlockMmad;
using Cmct::Gemm::Block::BlockSchedulerTailResplitExpanded;
using Cmct::Gemm::Kernel::KernelMatmulAPrefetchBAntiquant;
using Cmct::Prologue::BlockPrologue;

template <bool InnerK, bool IsNz>
struct XWeightShape {
    static_assert(AscendC::Std::always_false_v<decltype(InnerK)>, "can not find the specialization.");
};

template <bool InnerK>
struct XWeightShape<InnerK, false> {
    using Type = AscendC::Std::tuple<uint64_t, uint64_t>;
};

template <bool InnerK>
struct XWeightShape<InnerK, true> {
    // k1, n1, n0, k0
    using Type = AscendC::Std::tuple<
        AscendC::Std::tuple<Cmct::Gemm::_16, uint64_t>, AscendC::Std::tuple<Cmct::Gemm::_16, uint64_t>>;
};

template <bool InnerK, bool IsNz, typename DtypeA, typename DtypeB>
struct XWeightStride {};

template <typename DtypeA, typename DtypeB>
struct XWeightStride<false, false, DtypeA, DtypeB> {
    // k,m
    // k,n
    __aicore__ inline decltype(auto) operator()(uint64_t outer, uint64_t k)
    {
        return AscendC::Std::make_tuple(Cmct::Gemm::_1{}, outer);
    }
};

template <typename DtypeA, typename DtypeB>
struct XWeightStride<true, false, DtypeA, DtypeB> {
    // m,k
    // n,k
    __aicore__ inline decltype(auto) operator()(uint64_t outer, uint64_t k)
    {
        return AscendC::Std::make_tuple(k, Cmct::Gemm::_1{});
    }
};

template <bool InnerK, typename DtypeA, typename DtypeB>
struct XWeightStride<InnerK, true, DtypeA, DtypeB> {
    __aicore__ inline decltype(auto) operator()(uint64_t outer, uint64_t k)
    {
        if constexpr (InnerK) {
            // n, k -> k1, n1, n0(16), k0(16)
            // m, k -> k1, m1, m0(16), k0(16)
            // stride(以m,k为例) m1*m0*k0, m0*k0(256), k0(16), 1
            return AscendC::MakeStride(
                AscendC::MakeStride(Cmct::Gemm::_16{}, Cmct::Gemm::_256{}),
                AscendC::MakeStride(Cmct::Gemm::_1{}, CeilAlign(outer, 16UL) * 16UL));
        } else {
            // k, n -> n1, k1, k0(16), n0(16)
            // k, m -> m1, k1, k0(16), m0(16)
            // stride(以k,m为例) k1*k0*m0, k0*m0(256), m0(16), 1
            return AscendC::MakeStride(
                AscendC::MakeStride(Cmct::Gemm::_1{}, CeilAlign(k, 16UL) * 16UL),
                AscendC::MakeStride(Cmct::Gemm::_16{}, Cmct::Gemm::_256{}));
        }
    }
};

template <bool Trans, Cmct::Gemm::QuantType AntiquantType, typename Enable = void>
struct ScaleOffsetStride {};

template <bool Trans>
struct ScaleOffsetStride<Trans, Cmct::Gemm::QuantType::PER_TENSOR> {
    __aicore__ inline decltype(auto) operator()(uint64_t n, uint64_t k, int32_t groupSize)
    {
        return AscendC::Std::make_tuple(Cmct::Gemm::_1{});
    }
};

template <bool Trans>
struct ScaleOffsetStride<Trans, Cmct::Gemm::QuantType::PER_CHANNEL> {
    __aicore__ inline decltype(auto) operator()(uint64_t n, uint64_t k, int32_t groupSize)
    {
        return AscendC::Std::make_tuple(Cmct::Gemm::_1{}, n);
    }
};

template <Cmct::Gemm::QuantType AntiquantType>
struct ScaleOffsetStride<
    false, AntiquantType,
    typename AscendC::Std::enable_if_t<
        AntiquantType == Cmct::Gemm::QuantType::PER_GROUP || AntiquantType == Cmct::Gemm::QuantType::MX>> {
    __aicore__ inline decltype(auto) operator()(uint64_t n, uint64_t k, int32_t groupSize)
    {
        return AscendC::Std::make_tuple(Cmct::Gemm::_1{}, n);
    }
};

template <Cmct::Gemm::QuantType AntiquantType>
struct ScaleOffsetStride<
    true, AntiquantType,
    typename AscendC::Std::enable_if_t<
        AntiquantType == Cmct::Gemm::QuantType::PER_GROUP || AntiquantType == Cmct::Gemm::QuantType::MX>> {
    __aicore__ inline decltype(auto) operator()(uint64_t n, uint64_t k, int32_t groupSize)
    {
        return AscendC::Std::make_tuple(
            groupSize == 0 ? k : CeilDiv(k, static_cast<uint64_t>(groupSize)), Cmct::Gemm::_1{});
    }
};

// PS bias在编译期不清楚，通过tiling获取
using StrideBias = AscendC::Std::tuple<Cmct::Gemm::_1>;
using StrideC = AscendC::Std::tuple<uint64_t, Cmct::Gemm::_1>;

using ProblemShape = AscendC::Std::tuple<uint64_t, uint64_t, uint64_t>;          // m, n, k
using TileShapeL1 = AscendC::Std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>; // m, n, ka, kb
using TileShapeL0 = AscendC::Std::tuple<uint32_t, uint32_t, uint32_t>;           // m, n, k

template <bool WeightNz, bool InnerK>
struct StrideXWeight {
    static_assert(AscendC::Std::always_false_v<decltype(WeightNz)>, "Unsupported (WeightNz, InnerK) combination");
};

// n,k k1,n1,n0,k0
template <>
struct StrideXWeight<true, true> {
    using Type = AscendC::Std::tuple<
        AscendC::Std::tuple<Cmct::Gemm::_16, Cmct::Gemm::_256>, AscendC::Std::tuple<Cmct::Gemm::_1, uint64_t>>;
};

template <>
struct StrideXWeight<true, false> {
    using Type = AscendC::Std::tuple<
        AscendC::Std::tuple<Cmct::Gemm::_1, uint64_t>, AscendC::Std::tuple<Cmct::Gemm::_16, Cmct::Gemm::_256>>;
};

template <>
struct StrideXWeight<false, true> {
    using Type = AscendC::Std::tuple<uint64_t, Cmct::Gemm::_1>;
};

template <>
struct StrideXWeight<false, false> {
    using Type = AscendC::Std::tuple<Cmct::Gemm::_1, uint64_t>;
};

template <Cmct::Gemm::QuantType AntiquantType, bool InnerK, typename Enable = void>
struct StrideAntiquant {
    static_assert(
        AscendC::Std::always_false_v<decltype(AntiquantType)>, "Unsupported (AntiquantType, InnerK) combination");
};

template <bool InnerK>
struct StrideAntiquant<Cmct::Gemm::QuantType::PER_TENSOR, InnerK> {
    using Type = AscendC::Std::tuple<Cmct::Gemm::_1>;
};

template <bool InnerK>
struct StrideAntiquant<Cmct::Gemm::QuantType::PER_CHANNEL, InnerK> {
    using Type = AscendC::Std::tuple<Cmct::Gemm::_1, uint64_t>;
};

template <Cmct::Gemm::QuantType AntiquantType>
struct StrideAntiquant<
    AntiquantType, true,
    typename AscendC::Std::enable_if_t<
        AntiquantType == Cmct::Gemm::QuantType::PER_GROUP || AntiquantType == Cmct::Gemm::QuantType::MX>> {
    using Type = AscendC::Std::tuple<uint64_t, Cmct::Gemm::_1>;
};

template <Cmct::Gemm::QuantType AntiquantType>
struct StrideAntiquant<
    AntiquantType, false,
    typename AscendC::Std::enable_if_t<
        AntiquantType == Cmct::Gemm::QuantType::PER_GROUP || AntiquantType == Cmct::Gemm::QuantType::MX>> {
    using Type = AscendC::Std::tuple<Cmct::Gemm::_1, uint64_t>;
};

template <bool IsNz>
struct XWeightShapeObj {};

template <>
struct XWeightShapeObj<false> {
    // k,m
    // k,n
    __aicore__ inline decltype(auto) operator()(uint64_t outer, uint64_t k)
    {
        return AscendC::Std::make_tuple(outer, k);
    }
};

template <>
struct XWeightShapeObj<true> {
    // k, m  m1,k1,k0,n0
    __aicore__ inline decltype(auto) operator()(uint64_t outer, uint64_t k)
    {
        return AscendC::MakeShape(
            AscendC::MakeShape(Cmct::Gemm::_16{}, CeilDiv(outer, 16UL)),
            AscendC::MakeShape(Cmct::Gemm::_16{}, CeilDiv(k, 16UL)));
    }
};

template <Cmct::Gemm::QuantType AntiquantType, typename Enable = void>
struct ScaleOffsetShapeObj {};

template <>
struct ScaleOffsetShapeObj<Cmct::Gemm::QuantType::PER_TENSOR> {
    __aicore__ inline decltype(auto) operator()(uint64_t n, uint64_t k, int32_t groupSize)
    {
        return AscendC::Std::make_tuple(Cmct::Gemm::_1{});
    }
};

template <>
struct ScaleOffsetShapeObj<Cmct::Gemm::QuantType::PER_CHANNEL> {
    __aicore__ inline decltype(auto) operator()(uint64_t n, uint64_t k, int32_t groupSize)
    {
        return AscendC::Std::make_tuple(n, Cmct::Gemm::_1{});
    }
};

template <Cmct::Gemm::QuantType AntiquantType>
struct ScaleOffsetShapeObj<
    AntiquantType,
    typename AscendC::Std::enable_if_t<
        AntiquantType == Cmct::Gemm::QuantType::PER_GROUP || AntiquantType == Cmct::Gemm::QuantType::MX>> {
    __aicore__ inline decltype(auto) operator()(uint64_t n, uint64_t k, int32_t groupSize)
    {
        return AscendC::Std::make_tuple(n, CeilDiv(k, static_cast<uint64_t>(groupSize)));
    }
};

template <bool trans, Cmct::Gemm::QuantType AntiquantType, typename Enable = void>
struct ScaleOffsetShape {};

template <bool trans>
struct ScaleOffsetShape<trans, Cmct::Gemm::QuantType::PER_TENSOR> {
    using Type = AscendC::Std::tuple<Cmct::Gemm::_1>;
};

template <bool trans>
struct ScaleOffsetShape<trans, Cmct::Gemm::QuantType::PER_CHANNEL> {
    using Type = AscendC::Std::tuple<uint64_t, Cmct::Gemm::_1>;
};

template <Cmct::Gemm::QuantType AntiquantType, bool trans>
struct ScaleOffsetShape<
    trans, AntiquantType,
    typename AscendC::Std::enable_if_t<
        AntiquantType == Cmct::Gemm::QuantType::PER_GROUP || AntiquantType == Cmct::Gemm::QuantType::MX>> {
    using Type = AscendC::Std::tuple<uint64_t, uint64_t>;
};

template <
    typename HighBitType, typename ScaleType, bool IsWeightNz, bool bTrans, typename xType, typename wType,
    uint32_t ubMte2InnerSize, uint32_t ubMte2BufNum, bool hasAntiquantOffset, uint64_t AIV_NUM,
    Cmct::Gemm::QuantType QuantType>
struct ConfigBAntiquantScmc {};

// Dtype: A16
// Format: NZ
// Layout: weight column_major
// AntiquantType: PerChannel
template <
    typename HighBitType, typename ScaleType, bool bTrans, typename xType, typename wType, uint32_t ubMte2InnerSize,
    uint32_t ubMte2BufNum, bool hasAntiquantOffset, uint64_t AIV_NUM, Cmct::Gemm::QuantType QuantType>
struct ConfigBAntiquantScmc<
    HighBitType, ScaleType, true, bTrans, xType, wType, ubMte2InnerSize, ubMte2BufNum, hasAntiquantOffset, AIV_NUM,
    QuantType> {
    static constexpr uint64_t UB_OUT_BUF_NUM = Cmct::Gemm::QUADRUPLE_BUFFER_NUM;
    static constexpr uint64_t UB_IN_SIZE = 112;
    static constexpr uint64_t UB_OUT_SIZE = 128;
    static constexpr uint64_t SCALE_SIZE = 4;
    static constexpr uint64_t OFFSET_SIZE = 4;
    static constexpr uint64_t ANTIQUANT_SCALE_AFTER_CAST_SIZE = 0;
    static constexpr uint64_t VF_N = 64;
    static constexpr uint64_t VF_K = 256;
    static_assert(VF_K == ubMte2InnerSize);
    using Type = Cmct::Prologue::BAntiquantScmc<
        Ascend910_95, HighBitType, ScaleType, !bTrans, false, AIV_NUM, hasAntiquantOffset, ubMte2BufNum,
        ubMte2InnerSize, VF_N, VF_K, UB_OUT_BUF_NUM, UB_IN_SIZE, UB_OUT_SIZE, SCALE_SIZE, OFFSET_SIZE,
        ANTIQUANT_SCALE_AFTER_CAST_SIZE>;
};

// Dtype: A16
// Format: Nz
// Layout: weight column_major
// AntiquantType: PerGroup
template <
    typename HighBitType, typename ScaleType, bool bTrans, typename xType, uint32_t ubMte2InnerSize,
    uint32_t ubMte2BufNum, bool hasAntiquantOffset, uint64_t AIV_NUM>
struct ConfigBAntiquantScmc<
    HighBitType, ScaleType, true, bTrans, xType, int4b_t, ubMte2InnerSize, ubMte2BufNum, hasAntiquantOffset, AIV_NUM,
    Cmct::Gemm::QuantType::PER_GROUP> {
    static constexpr uint64_t UB_OUT_BUF_NUM = Cmct::Gemm::DOUBLE_BUFFER_NUM;
    static constexpr uint64_t UB_IN_SIZE = 128;
    static constexpr uint64_t UB_OUT_SIZE = 64;
    static constexpr uint64_t SCALE_SIZE = 16;
    static constexpr uint64_t OFFSET_SIZE = 16;
    static constexpr uint64_t ANTIQUANT_SCALE_AFTER_CAST_SIZE = 0;
    static constexpr uint64_t VF_N = 32;
    static constexpr uint64_t VF_K = 512;
    using Type = Cmct::Prologue::BAntiquantScmc<
        Ascend910_95, HighBitType, ScaleType, !bTrans, true, AIV_NUM, hasAntiquantOffset, ubMte2BufNum, ubMte2InnerSize,
        VF_N, VF_K, UB_OUT_BUF_NUM, UB_IN_SIZE, UB_OUT_SIZE, SCALE_SIZE, OFFSET_SIZE, ANTIQUANT_SCALE_AFTER_CAST_SIZE>;
};

// Dtype: A16
// Format: ND
// Layout: weight row_major/column_major
// AntiquantType: PerChannel/PerTensor
template <
    typename HighBitType, typename ScaleType, bool bTrans, typename xType, typename wType, uint32_t ubMte2InnerSize,
    uint32_t ubMte2BufNum, bool hasAntiquantOffset, uint64_t AIV_NUM, Cmct::Gemm::QuantType AntiQuantType>
struct ConfigBAntiquantScmc<
    HighBitType, ScaleType, false, bTrans, xType, wType, ubMte2InnerSize, ubMte2BufNum, hasAntiquantOffset, AIV_NUM,
    AntiQuantType> {
    static constexpr uint64_t UB_OUT_BUF_NUM = Cmct::Gemm::DOUBLE_BUFFER_NUM;
    static constexpr uint64_t UB_IN_SIZE = 174;
    static constexpr uint64_t UB_OUT_SIZE = 66;
    static constexpr uint64_t SCALE_SIZE = 4;
    static constexpr uint64_t OFFSET_SIZE = 4;
    static constexpr uint64_t ANTIQUANT_SCALE_AFTER_CAST_SIZE = 0;
    static constexpr uint64_t VF_N = bTrans ? 64 : 256;
    static constexpr uint64_t VF_K = bTrans ? 256 : 64;
    using Type = Cmct::Prologue::BAntiquantScmc<
        Ascend910_95, HighBitType, ScaleType, bTrans, false, AIV_NUM, hasAntiquantOffset, ubMte2BufNum, ubMte2InnerSize,
        VF_N, VF_K, UB_OUT_BUF_NUM, UB_IN_SIZE, UB_OUT_SIZE, SCALE_SIZE, OFFSET_SIZE, ANTIQUANT_SCALE_AFTER_CAST_SIZE>;
};

// Dtype: A16
// Format: ND
// Layout: weight row_major/column_major
// AntiquantType: MX
template <
    typename HighBitType, typename ScaleType, bool bTrans, typename xType, typename wType, uint32_t ubMte2InnerSize,
    uint32_t ubMte2BufNum, bool hasAntiquantOffset, uint64_t AIV_NUM>
struct ConfigBAntiquantScmc<
    HighBitType, ScaleType, false, bTrans, xType, wType, ubMte2InnerSize, ubMte2BufNum, hasAntiquantOffset, AIV_NUM,
    Cmct::Gemm::QuantType::MX> {
    static constexpr uint64_t UB_OUT_BUF_NUM = Cmct::Gemm::DOUBLE_BUFFER_NUM;
    static constexpr uint64_t UB_IN_SIZE = 128;
    static constexpr uint64_t UB_OUT_SIZE = 66;
    static constexpr uint64_t SCALE_SIZE = 8;
    static constexpr uint64_t OFFSET_SIZE = 0;
    static constexpr uint64_t ANTIQUANT_SCALE_AFTER_CAST_SIZE = 32;
    static constexpr uint64_t VF_N = bTrans ? 64 : 256;
    static constexpr uint64_t VF_K = bTrans ? 256 : 64;
    using Type = Cmct::Prologue::BAntiquantScmc<
        Ascend910_95, HighBitType, ScaleType, bTrans, false, AIV_NUM, hasAntiquantOffset, ubMte2BufNum, ubMte2InnerSize,
        VF_N, VF_K, UB_OUT_BUF_NUM, UB_IN_SIZE, UB_OUT_SIZE, SCALE_SIZE, OFFSET_SIZE, ANTIQUANT_SCALE_AFTER_CAST_SIZE>;
};

// Dtype: A16
// Format: NZ
// Layout: column_major
// AntiquantType: MX
template <
    typename HighBitType, typename ScaleType, bool bTrans, typename xType, typename wType, uint32_t ubMte2InnerSize,
    uint32_t ubMte2BufNum, bool hasAntiquantOffset, uint64_t AIV_NUM>
struct ConfigBAntiquantScmc<
    HighBitType, ScaleType, true, bTrans, xType, wType, ubMte2InnerSize, ubMte2BufNum, hasAntiquantOffset, AIV_NUM,
    Cmct::Gemm::QuantType::MX> {
    static constexpr uint64_t UB_OUT_BUF_NUM = Cmct::Gemm::QUADRUPLE_BUFFER_NUM;
    static constexpr uint64_t UB_IN_SIZE = 64;
    static constexpr uint64_t UB_OUT_SIZE = 128;
    static constexpr uint64_t SCALE_SIZE = 8;
    static constexpr uint64_t OFFSET_SIZE = 0;
    static constexpr uint64_t ANTIQUANT_SCALE_AFTER_CAST_SIZE = 16;
    static constexpr uint64_t VF_N = static_cast<uint64_t>(32 * KB_ELEM<HighBitType> / ubMte2InnerSize);
    static constexpr uint64_t VF_K = static_cast<uint64_t>(ubMte2InnerSize);
    using Type = Cmct::Prologue::BAntiquantScmc<
        Ascend910_95, HighBitType, ScaleType, !bTrans, true, AIV_NUM, hasAntiquantOffset, ubMte2BufNum, ubMte2InnerSize,
        VF_N, VF_K, UB_OUT_BUF_NUM, UB_IN_SIZE, UB_OUT_SIZE, SCALE_SIZE, OFFSET_SIZE, ANTIQUANT_SCALE_AFTER_CAST_SIZE>;
};

template <
    uint32_t UbMte2InnerSize, uint32_t UbMte2BufNum, bool TransA, bool TransB, Cmct::Gemm::QuantType AntiquantType,
    bool HasAntiquantOffset, bool IsBiasFp32, bool IsWeightNz>
struct ScmcKernel {
#if defined(__DAV_310R6__)
    static constexpr int SUB_BLOCK_NUM = 1;
#else
    static constexpr int SUB_BLOCK_NUM = 2;
#endif

    using ShapeA = typename XWeightShape<!TransA, false>::Type;
    using StrideA = typename StrideXWeight<false, !TransA>::Type;
    using AType = GemmType<DTYPE_X, AscendC::Layout<ShapeA, StrideA>>;

    using ShapeB = typename XWeightShape<TransB, IsWeightNz>::Type;
    using StrideB = typename StrideXWeight<IsWeightNz, TransB>::Type;
    using BType = GemmType<DTYPE_WEIGHT, AscendC::Layout<ShapeB, StrideB>>;

    using ShapeScale = typename ScaleOffsetShape<TransB, AntiquantType>::Type;
    using StrideAntiquantScale = typename StrideAntiquant<AntiquantType, TransB>::Type;
    using ScaleType = GemmType<DTYPE_ANTIQUANT_SCALE, AscendC::Layout<ShapeScale, StrideAntiquantScale>>;

    using ShapeC = AscendC::Std::tuple<uint64_t, uint64_t>;
    using StrideC = AscendC::Std::tuple<uint64_t, Cmct::Gemm::_1>;
    using CType = GemmType<DTYPE_Y, AscendC::Layout<ShapeC, StrideC>>;

    using DtypeBias = AscendC::Std::conditional_t<IsBiasFp32, float, DTYPE_X>;
    using BiasType =
        GemmType<DtypeBias, AscendC::Layout<AscendC::Std::tuple<uint64_t>, AscendC::Std::tuple<Cmct::Gemm::_1>>>;

    using DispatchPolicyMmad = MmadAPrefetchBAntiquantScmc<SUB_BLOCK_NUM>;
    using BlockMmad = BlockMmad<DispatchPolicyMmad, TileShapeL1, TileShapeL0, AType, BType, CType, BiasType>;

    using DispatchPolicyPrologue = typename ConfigBAntiquantScmc<
        DTYPE_X, DTYPE_ANTIQUANT_SCALE, IsWeightNz, TransB, DTYPE_X, DTYPE_WEIGHT, UbMte2InnerSize, UbMte2BufNum,
        HasAntiquantOffset, SUB_BLOCK_NUM, AntiquantType>::Type;
    using BlockPrologue = BlockPrologue<DispatchPolicyPrologue, BType, ScaleType, TileShapeL1>;

    using Kernel =
        KernelMatmulAPrefetchBAntiquant<ProblemShape, BlockMmad, BlockPrologue, BlockSchedulerTailResplitExpanded>;

    __aicore__ inline void operator()(
        GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale,
        GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
    {
        GET_TILING_DATA_WITH_STRUCT(wqbmmv2_tiling::WeightQuantBatchMatmulV2ASTilingDataParams, tilingDataIn, tiling);
        auto problemShape = AscendC::Std::make_tuple(tilingDataIn.mSize, tilingDataIn.nSize, tilingDataIn.kSize);

        typename BlockMmad::Arguments mmadArgs{
            .ptrA = x,
            .ptrC = y,
            .ptrBias = bias,
            .layoutA = AscendC::MakeLayout(
                XWeightShapeObj<false>{}(tilingDataIn.mSize, tilingDataIn.kSize),
                XWeightStride<!TransA, false, DTYPE_X, DTYPE_WEIGHT>{}(tilingDataIn.mSize, tilingDataIn.kSize)),
            .layoutC = AscendC::MakeLayout(
                AscendC::MakeShape(tilingDataIn.mSize, tilingDataIn.nSize),
                AscendC::MakeStride(tilingDataIn.nSize, Cmct::Gemm::_1{})),
            .layoutBias =
                AscendC::MakeLayout(AscendC::MakeShape(tilingDataIn.nSize), AscendC::MakeStride(Cmct::Gemm::_1{}))};
        typename BlockPrologue::Arguments prologueArgs{
            .ptrB = weight,
            .ptrScale = antiquantScale,
            .ptrOffset = antiquantOffset,
            .layoutB = AscendC::MakeLayout(
                XWeightShapeObj<IsWeightNz>{}(tilingDataIn.nSize, tilingDataIn.kSize),
                XWeightStride<TransB, IsWeightNz, DTYPE_X, DTYPE_WEIGHT>{}(tilingDataIn.nSize, tilingDataIn.kSize)),
            .layoutScale = AscendC::MakeLayout(
                ScaleOffsetShapeObj<AntiquantType>{}(tilingDataIn.nSize, tilingDataIn.kSize, tilingDataIn.groupSize),
                ScaleOffsetStride<TransB, AntiquantType>{}(
                    tilingDataIn.nSize, tilingDataIn.kSize, tilingDataIn.groupSize)),
            .antiQuantGroupSize = tilingDataIn.groupSize};

        typename BlockSchedulerTailResplitExpanded::Arguments schedulerArgs{};

        typename Kernel::Arguments args{
            .problemShape = problemShape, .mmad = mmadArgs, .prologue = prologueArgs, .scheduler = schedulerArgs};
        auto kernelParams = Kernel::ToUnderlyingArguments(args, &tilingDataIn);
        Kernel op;
        op(kernelParams);
    }
};

template <
    int TemplateCustom, bool TransA, bool TransB, int AntiquantType, bool HasAntiquantOffset, bool IsBiasFp32,
    bool IsWeightNz>
__aicore__ inline void InvokeKernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    static constexpr Cmct::Gemm::QuantType ANTIQUANT_TYPE = static_cast<Cmct::Gemm::QuantType>(AntiquantType);

    static constexpr bool B8 =
        IsSameType<DTYPE_WEIGHT, int8_t>::value || IsSameType<DTYPE_WEIGHT, float8_e5m2_t>::value ||
        IsSameType<DTYPE_WEIGHT, float8_e4m3_t>::value || IsSameType<DTYPE_WEIGHT, hifloat8_t>::value;
    static constexpr uint32_t UB_MTE2_INNER_SIZE = TemplateCustom == 0 || TemplateCustom == 1 || TemplateCustom == 4 ?
                                                       512 :
                                                   TemplateCustom == 2 || TemplateCustom == 6 ? 1024 :
                                                                                                256;
    static constexpr uint32_t UB_MTE2_BUF_NUM =
        TemplateCustom == 0 || TemplateCustom == 2 || (TemplateCustom == 4 && B8) || TemplateCustom == 5 ? 2 : 4;

    ScmcKernel<
        UB_MTE2_INNER_SIZE, UB_MTE2_BUF_NUM, TransA, TransB, ANTIQUANT_TYPE, HasAntiquantOffset, IsBiasFp32,
        IsWeightNz>{}(x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace, tiling);
}
} // namespace WeightQuantBatchMatmulV2

#define KERNEL_PARAMS x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace, tiling

#endif