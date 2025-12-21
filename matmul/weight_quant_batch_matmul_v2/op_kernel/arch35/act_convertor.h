/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ARCH35_ACT_CONVERTOR_H
#define ARCH35_ACT_CONVERTOR_H
#include "act/block/david_ub_antiquant_scmc_load_in_advance_aic_tail_resplit.h"
#include "act/block/david_ub_antiquant_scmc_load_in_advance_aiv_tail_resplit.h"
#include "act/dispatch_policy.h"
#include "act/kernel/david_wqbmm_load_in_advance.h"
#include "act/scheduler/tile_scheduler_tail_resplit.h"
#include "act/scheduler/tile_scheduler_tail_resplit_expanded.h"
#include "act/utils/constant.h"
#include "act/utils/math_utils.h"
#include "act/block/block_decl.h"
#include "act/prologue/block/block_prologue.h"
#include "act/prologue/dispatch_policy.h"
#include "act/kernel/kernel_matmul_a_prefetch_b_antiquant.h"
#include "weight_quant_batch_matmul_v2_arch35_tiling_data.h"
#include "act/utils/integral_constant.h"

using AscendC::int4b_t;

namespace WeightQuantBatchMatmulV2::Arch35::Act {

using Act::CeilDiv;
using Act::CeilAlign;
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
        AscendC::Std::tuple<::Act::Gemm::_16, uint64_t>, AscendC::Std::tuple<::Act::Gemm::_16, uint64_t>>;
};

template <bool InnerK, bool IsNz, typename DtypeA, typename DtypeB, bool IsPerGroup>
struct XWeightStride {};

template <typename DtypeA, typename DtypeB, bool IsPerGroup>
struct XWeightStride<false, false, DtypeA, DtypeB, IsPerGroup> {
    // k,m
    // k,n
    DEVICE decltype(auto) operator()(uint64_t outer, uint64_t k)
    {
        return AscendC::Std::make_tuple(::Act::Gemm::_1{}, outer);
    }
};

template <typename DtypeA, typename DtypeB>
struct XWeightStride<true, false, DtypeA, DtypeB, true> {
    // m,k
    // n,k
    DEVICE decltype(auto) operator()(uint64_t outer, uint64_t k)
    {
        return AscendC::Std::make_tuple(k, Act::_1{});
    }
};

template <typename DtypeA, typename DtypeB>
struct XWeightStride<true, false, DtypeA, DtypeB, false> {
    // m,k
    // n,k
    DEVICE decltype(auto) operator()(uint64_t outer, uint64_t k)
    {
        return AscendC::Std::make_tuple(k, ::Act::Gemm::_1{});
    }
};

template <bool InnerK, typename DtypeA, typename DtypeB>
struct XWeightStride<InnerK, true, DtypeA, DtypeB, true> {
    DEVICE decltype(auto) operator()(uint64_t outer, uint64_t k)
    {
        if constexpr (
            (AscendC::Std::is_same_v<DtypeA, half> || AscendC::Std::is_same_v<DtypeA, bfloat16_t>) &&
            AscendC::Std::is_same_v<DtypeB, AscendC::int4b_t>) {
            if constexpr (InnerK) {
                // n, k -> k1, n1, n0(16), k0(16)
                // m, k -> k1, m1, m0(16), k0(16)
                // stride(以m,k为例) m1*m0*k0, m0*k0(256), k0(16), 1
                return AscendC::Std::make_tuple( // 256数据分块
                    CeilDiv(outer, 16UL) * 256, Act::_256{}, Act::_16{}, Act::_1{});
            } else {
                // k, n -> n1, k1, k0(16), n0(16)
                // k, m -> m1, k1, k0(16), m0(16)
                // stride(以k,m为例) k1*k0*m0, k0*m0(256), m0(16), 1
                return AscendC::Std::make_tuple( // 256数据分块
                    CeilDiv(k, 16UL) * 256, Act::_256{}, Act::_16{}, Act::_1{});
            }
        }
    }
};

template <bool InnerK, typename DtypeA, typename DtypeB>
struct XWeightStride<InnerK, true, DtypeA, DtypeB, false> {
    DEVICE decltype(auto) operator()(uint64_t outer, uint64_t k)
    {
        if constexpr (InnerK) {
            // n, k -> k1, n1, n0(16), k0(16)
            // m, k -> k1, m1, m0(16), k0(16)
            // stride(以m,k为例) m1*m0*k0, m0*k0(256), k0(16), 1
            return AscendC::MakeStride(
                AscendC::MakeStride(::Act::Gemm::_16{}, ::Act::Gemm::_256{}),
                AscendC::MakeStride(::Act::Gemm::_1{}, CeilAlign(outer, 16UL) * 16UL));
        } else {
            // k, n -> n1, k1, k0(16), n0(16)
            // k, m -> m1, k1, k0(16), m0(16)
            // stride(以k,m为例) k1*k0*m0, k0*m0(256), m0(16), 1
            return AscendC::MakeStride(
                AscendC::MakeStride(::Act::Gemm::_1{}, CeilAlign(k, 16UL) * 16UL),
                AscendC::MakeStride(::Act::Gemm::_16{}, ::Act::Gemm::_256{}));
        }
    }
};

template <bool Trans, QuantType antiquantType>
struct ScaleOffsetStride {};

template <bool Trans>
struct ScaleOffsetStride<Trans, QuantType::PER_TENSOR> {
    DEVICE decltype(auto) operator()(uint64_t n, uint64_t k, int32_t groupSize)
    {
        return AscendC::Std::make_tuple(::Act::Gemm::_1{});
    }
};

template <bool Trans>
struct ScaleOffsetStride<Trans, QuantType::PER_CHANNEL> {
    DEVICE decltype(auto) operator()(uint64_t n, uint64_t k, int32_t groupSize)
    {
        return AscendC::Std::make_tuple(::Act::Gemm::_1{}, n);
    }
};

template <>
struct ScaleOffsetStride<false, QuantType::PER_GROUP> {
    DEVICE decltype(auto) operator()(uint64_t n, uint64_t k, int32_t groupSize)
    {
        return AscendC::Std::make_tuple(::Act::Gemm::_1{}, n);
    }
};

template <>
struct ScaleOffsetStride<true, QuantType::PER_GROUP> {
    DEVICE decltype(auto) operator()(uint64_t n, uint64_t k, int32_t groupSize)
    {
        return AscendC::Std::make_tuple(groupSize == 0 ? k : (k + groupSize - 1) / groupSize, ::Act::Gemm::_1{});
    }
};

// PS bias在编译期不清楚，通过tiling获取
using StrideBias = AscendC::Std::tuple<::Act::Gemm::_1>;
using StrideC = AscendC::Std::tuple<uint64_t, ::Act::Gemm::_1>;

using ProblemShape = AscendC::Std::tuple<uint64_t, uint64_t, uint64_t>;          // m, n, k
using TileShapeL1 = AscendC::Std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>; // m, n, ka, kb
using TileShapeL0 = AscendC::Std::tuple<uint32_t, uint32_t, uint32_t>;           // m, n, k
using TileShapeUb = AscendC::Std::tuple<uint32_t, uint32_t>;                     // n, k

template <bool WeightNz, bool InnerK, bool IsPerGroup>
struct StrideXWeight {
    static_assert(AscendC::Std::always_false_v<decltype(WeightNz)>, "Unsupported (WeightNz, InnerK) combination");
};

template <bool InnerK>
struct StrideXWeight<true, InnerK, true> {
    using Type = AscendC::Std::tuple<uint64_t, Act::_256, Act::_16, Act::_1>;
};

template <>
struct StrideXWeight<false, true, true> {
    using Type = AscendC::Std::tuple<uint64_t, Act::_1>;
};

// n,k k1,n1,n0,k0
template <>
struct StrideXWeight<true, true, false> {
    using Type = AscendC::Std::tuple<
        AscendC::Std::tuple<::Act::Gemm::_16, ::Act::Gemm::_256>, AscendC::Std::tuple<::Act::Gemm::_1, uint64_t>>;
};

template <>
struct StrideXWeight<true, false, false> {
    using Type = AscendC::Std::tuple<
        AscendC::Std::tuple<::Act::Gemm::_1, uint64_t>, AscendC::Std::tuple<::Act::Gemm::_16, ::Act::Gemm::_256>>;
};

template <>
struct StrideXWeight<false, true, false> {
    using Type = AscendC::Std::tuple<uint64_t, ::Act::Gemm::_1>;
};

template <>
struct StrideXWeight<false, false, false> {
    using Type = AscendC::Std::tuple<::Act::Gemm::_1, uint64_t>;
};

template <QuantType AntiquantType, bool InnerK>
struct StrideAntiquant {
    static_assert(
        AscendC::Std::always_false_v<decltype(AntiquantType)>, "Unsupported (AntiquantType, InnerK) combination");
};

template <bool InnerK>
struct StrideAntiquant<QuantType::PER_TENSOR, InnerK> {
    using Type = AscendC::Std::tuple<::Act::Gemm::_1>;
};

template <bool InnerK>
struct StrideAntiquant<QuantType::PER_CHANNEL, InnerK> {
    using Type = AscendC::Std::tuple<::Act::Gemm::_1, uint64_t>;
};

template <>
struct StrideAntiquant<QuantType::PER_GROUP, true> {
    using Type = AscendC::Std::tuple<uint64_t, ::Act::Gemm::_1>;
};

template <>
struct StrideAntiquant<QuantType::PER_GROUP, false> {
    using Type = AscendC::Std::tuple<::Act::Gemm::_1, uint64_t>;
};

template <int32_t UB_MTE2_INNER_SIZE>
struct TileShapeReg {
    static_assert(AscendC::Std::always_false_v<decltype(UB_MTE2_INNER_SIZE)>, "Unsupported UB_MTE2_INNER_SIZE");
};

template <>
// 寄存器k切分为256
struct TileShapeReg<256> {
    using Type = AscendC::Std::tuple<::Act::Gemm::_64, ::Act::Gemm::_256>;
};

template <>
// 寄存器k切分为512
struct TileShapeReg<512> {
    using Type = AscendC::Std::tuple<::Act::Gemm::_32, ::Act::Gemm::_512>;
};

template <>
// 寄存器k切分为1024
struct TileShapeReg<1024> {
    using Type = AscendC::Std::tuple<::Act::Gemm::_16, ::Act::Gemm::_1024>;
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
            AscendC::MakeShape(::Act::Gemm::_16{}, CeilDiv(outer, 16UL)),
            AscendC::MakeShape(::Act::Gemm::_16{}, CeilDiv(k, 16UL)));
    }
};

template <QuantType AntiquantType>
struct ScaleOffsetShapeObj {};

template <>
struct ScaleOffsetShapeObj<QuantType::PER_TENSOR> {
    __aicore__ inline decltype(auto) operator()(uint64_t n, uint64_t k, int32_t groupSize)
    {
        return AscendC::Std::make_tuple(::Act::Gemm::_1{});
    }
};

template <>
struct ScaleOffsetShapeObj<QuantType::PER_CHANNEL> {
    __aicore__ inline decltype(auto) operator()(uint64_t n, uint64_t k, int32_t groupSize)
    {
        return AscendC::Std::make_tuple(n, ::Act::Gemm::_1{});
    }
};

template <>
struct ScaleOffsetShapeObj<QuantType::PER_GROUP> {
    __aicore__ inline decltype(auto) operator()(uint64_t n, uint64_t k, int32_t groupSize)
    {
        return AscendC::Std::make_tuple(n, CeilDiv(k, static_cast<uint64_t>(groupSize)));
    }
};

template <bool trans, QuantType AntiquantType>
struct ScaleOffsetShape {};

template <bool trans>
struct ScaleOffsetShape<trans, QuantType::PER_TENSOR> {
    using Type = AscendC::Std::tuple<::Act::Gemm::_1>;
};

template <bool trans>
struct ScaleOffsetShape<trans, QuantType::PER_CHANNEL> {
    using Type = AscendC::Std::tuple<uint64_t, ::Act::Gemm::_1>;
};

template <bool trans>
struct ScaleOffsetShape<trans, QuantType::PER_GROUP> {
    using Type = AscendC::Std::tuple<uint64_t, uint64_t>;
};

template <
    uint32_t UB_MTE2_INNER_SIZE, uint32_t UB_MTE2_BUF_NUM, bool TRANS_A, bool TRANS_B, QuantType antiquantType,
    bool HAS_ANTIQUANT_OFFSET, bool IS_BIAS_FP32, bool IS_WEIGHT_NZ>
DEVICE void InvokeActA16W4PergroupKernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(
        wqbmmv2_tiling::WeightQuantBatchMatmulV2ASTilingDataParams, tilingDataIn, tiling);

#if defined(__DAV_310R6__)
    constexpr int SUB_BLOCK_NUM = 1;
#else
    constexpr int SUB_BLOCK_NUM = 2;
#endif
    using StrideA = typename StrideXWeight<false, !TRANS_A, true>::Type;
    using StrideB = typename StrideXWeight<IS_WEIGHT_NZ, TRANS_B, true>::Type;
    using StrideAntiquantScale = typename StrideAntiquant<antiquantType, TRANS_B>::Type;
    using StrideBOptionalTuple = AscendC::Std::conditional_t<
        HAS_ANTIQUANT_OFFSET, AscendC::Std::tuple<StrideB, StrideAntiquantScale, StrideAntiquantScale>,
        AscendC::Std::tuple<StrideB, StrideAntiquantScale>>;

    using DtypeBias = AscendC::Std::conditional_t<IS_BIAS_FP32, float, DTYPE_X>;
    using DispatchPolicy = MainloopDavidWqbmmUbAntiquantScmc<
        2, TileShapeUb, typename TileShapeReg<UB_MTE2_INNER_SIZE>::Type, g_coreType, SUB_BLOCK_NUM,
        UB_MTE2_BUF_NUM, 2, UB_MTE2_INNER_SIZE>;

    using BlockMmad = BlockMmad<
        DispatchPolicy, TileShapeL1, TileShapeL0, DTYPE_X, StrideA, DTYPE_WEIGHT, StrideBOptionalTuple, DtypeBias,
        StrideBias, DTYPE_Y, StrideC>;
    using Kernel = wqbmmv2<ProblemShape, BlockMmad, TileSchedulerTailResplit<SUB_BLOCK_NUM>>;

    typename BlockMmad::Arguments args{
        .aGmAddr = (__gm__ DTYPE_X*)x,
        .strideA =
            XWeightStride<!TRANS_A, false, DTYPE_X, DTYPE_WEIGHT, true>{}(tilingDataIn.mSize, tilingDataIn.kSize),
        .bGmAddr = (__gm__ DTYPE_WEIGHT*)weight,
        .strideB = XWeightStride<TRANS_B, IS_WEIGHT_NZ, DTYPE_X, DTYPE_WEIGHT, true>{}(
            tilingDataIn.nSize, tilingDataIn.kSize),
        .antiquantScaleGmAddr = (__gm__ DTYPE_X*)antiquantScale,
        .strideAntiquantScale = ScaleOffsetStride<TRANS_B, antiquantType>{}(
            tilingDataIn.nSize, tilingDataIn.kSize, tilingDataIn.groupSize),
        .antiquantOffsetGmAddr = (__gm__ DTYPE_X*)antiquantOffset,
        .biasGmAddr = (__gm__ DtypeBias*)bias,
        .strideBias = AscendC::Std::make_tuple(::Act::Gemm::_1{}),
        .cGmAddr = (__gm__ DTYPE_Y*)y,
        .strideC = AscendC::Std::make_tuple(tilingDataIn.nSize, ::Act::Gemm::_1{}),
        .groupSize = tilingDataIn.groupSize,
    };

    auto problemShape = AscendC::Std::make_tuple(tilingDataIn.mSize, tilingDataIn.nSize, tilingDataIn.kSize);
    typename BlockMmad::Params mainloopParams = BlockMmad::GetParams(problemShape, args, tilingDataIn);
    typename Kernel::Params kernelParams{problemShape, mainloopParams, mainloopParams.tileShapeL1};
    Kernel op;
    op(kernelParams);
}

template <
    typename HighBitType, typename ScaleType, bool IsWeightNz, bool bTrans, typename xType, uint32_t ubMte2InnerSize,
    uint32_t ubMte2BufNum, bool hasAntiquantOffset, uint64_t AIV_NUM>
struct ConfigBAntiquantScmc {};

template <
    typename HighBitType, typename ScaleType, bool bTrans, uint32_t ubMte2InnerSize, uint32_t ubMte2BufNum,
    bool hasAntiquantOffset, uint64_t AIV_NUM>
struct ConfigBAntiquantScmc<
    HighBitType, ScaleType, true, bTrans, int8_t, ubMte2InnerSize, ubMte2BufNum, hasAntiquantOffset, AIV_NUM> {
    static constexpr uint64_t UB_OUT_BUF_NUM = QUADRUPLE_BUFFER_NUM;
    static constexpr uint64_t UB_IN_SIZE = 96;
    static constexpr uint64_t UB_OUT_SIZE = 128;
    static constexpr uint64_t SCALE_SIZE = 12;
    static constexpr uint64_t OFFSET_SIZE = 0;
    static constexpr uint64_t VF_N = 64;
    static constexpr uint64_t VF_K = 512;
    using Type = Prologue::BAntiquantScmc<
        Arch::Ascend910_95, HighBitType, ScaleType, !bTrans, AIV_NUM, hasAntiquantOffset, ubMte2BufNum, ubMte2InnerSize,
        VF_N, VF_K, UB_OUT_BUF_NUM, UB_IN_SIZE, UB_OUT_SIZE, SCALE_SIZE, OFFSET_SIZE>;
};

template <
    typename HighBitType, typename ScaleType, bool bTrans, typename xType, uint32_t ubMte2InnerSize,
    uint32_t ubMte2BufNum, bool hasAntiquantOffset, uint64_t AIV_NUM>
struct ConfigBAntiquantScmc<
    HighBitType, ScaleType, true, bTrans, xType, ubMte2InnerSize, ubMte2BufNum, hasAntiquantOffset, AIV_NUM> {
    static constexpr uint64_t UB_OUT_BUF_NUM = QUADRUPLE_BUFFER_NUM;
    static constexpr uint64_t UB_IN_SIZE = 112;
    static constexpr uint64_t UB_OUT_SIZE = 128;
    static constexpr uint64_t SCALE_SIZE = 4;
    static constexpr uint64_t OFFSET_SIZE = 4;
    static constexpr uint64_t VF_N = 64;
    static constexpr uint64_t VF_K = 256;
    using Type = Prologue::BAntiquantScmc<
        Arch::Ascend910_95, HighBitType, ScaleType, !bTrans, AIV_NUM, hasAntiquantOffset, ubMte2BufNum, ubMte2InnerSize,
        VF_N, VF_K, UB_OUT_BUF_NUM, UB_IN_SIZE, UB_OUT_SIZE, SCALE_SIZE, OFFSET_SIZE>;
};

template <
    typename HighBitType, typename ScaleType, bool bTrans, typename xType, uint32_t ubMte2InnerSize,
    uint32_t ubMte2BufNum, bool hasAntiquantOffset, uint64_t AIV_NUM>
struct ConfigBAntiquantScmc<
    HighBitType, ScaleType, false, bTrans, xType, ubMte2InnerSize, ubMte2BufNum, hasAntiquantOffset, AIV_NUM> {
    static constexpr uint64_t UB_OUT_BUF_NUM = DOUBLE_BUFFER_NUM;
    static constexpr uint64_t UB_IN_SIZE = 174;
    static constexpr uint64_t UB_OUT_SIZE = 66;
    static constexpr uint64_t SCALE_SIZE = 4;
    static constexpr uint64_t OFFSET_SIZE = 4;
    static constexpr uint64_t VF_N = bTrans ? 64 : 256;
    static constexpr uint64_t VF_K = bTrans ? 256 : 64;
    using Type = Prologue::BAntiquantScmc<
        Arch::Ascend910_95, HighBitType, ScaleType, bTrans, AIV_NUM, hasAntiquantOffset, ubMte2BufNum, ubMte2InnerSize,
        VF_N, VF_K, UB_OUT_BUF_NUM, UB_IN_SIZE, UB_OUT_SIZE, SCALE_SIZE, OFFSET_SIZE>;
};

template <
    uint32_t UB_MTE2_INNER_SIZE, uint32_t UB_MTE2_BUF_NUM, bool TRANS_A, bool TRANS_B, QuantType antiquantType,
    bool HAS_ANTIQUANT_OFFSET, bool IS_BIAS_FP32, bool IS_WEIGHT_NZ>
struct ScmcKernel {
#if defined(__DAV_310R6__)
    static constexpr int SUB_BLOCK_NUM = 1;
#else
    static constexpr int SUB_BLOCK_NUM = 2;
#endif

    using ShapeA = typename XWeightShape<!TRANS_A, false>::Type;
    using StrideA = typename StrideXWeight<false, !TRANS_A, antiquantType == QuantType::PER_GROUP>::Type;
    using AType = GemmType<DTYPE_X, AscendC::Layout<ShapeA, StrideA>>;

    using ShapeB = typename XWeightShape<TRANS_B, IS_WEIGHT_NZ>::Type;
    using StrideB =
        typename StrideXWeight<IS_WEIGHT_NZ, TRANS_B, antiquantType == QuantType::PER_GROUP>::Type;
    using BType = GemmType<DTYPE_WEIGHT, AscendC::Layout<ShapeB, StrideB>>;

    using ShapeScale = typename ScaleOffsetShape<TRANS_B, antiquantType>::Type;
    using StrideAntiquantScale = typename StrideAntiquant<antiquantType, TRANS_B>::Type;
    using ScaleType = GemmType<DTYPE_ANTIQUANT_SCALE, AscendC::Layout<ShapeScale, StrideAntiquantScale>>;

    using ShapeC = AscendC::Std::tuple<uint64_t, uint64_t>;
    using StrideC = AscendC::Std::tuple<uint64_t, ::Act::Gemm::_1>;
    using CType = GemmType<DTYPE_Y, AscendC::Layout<ShapeC, StrideC>>;

    using DtypeBias = AscendC::Std::conditional_t<IS_BIAS_FP32, float, DTYPE_X>;
    using BiasType =
        GemmType<DtypeBias, AscendC::Layout<AscendC::Std::tuple<uint64_t>, AscendC::Std::tuple<::Act::Gemm::_1>>>;

    using DispatchPolicyMmad = MmadAPrefetchBAntiquantScmc<SUB_BLOCK_NUM>;
    using BlockMmad = Block::BlockMmadAct<DispatchPolicyMmad, TileShapeL1, TileShapeL0, AType, BType, CType, BiasType>;

    using DispatchPolicyPrologue = typename ConfigBAntiquantScmc<
        DTYPE_X, DTYPE_ANTIQUANT_SCALE, IS_WEIGHT_NZ, TRANS_B, DTYPE_X, UB_MTE2_INNER_SIZE,
        UB_MTE2_BUF_NUM, HAS_ANTIQUANT_OFFSET, SUB_BLOCK_NUM>::Type;
    using BlockPrologue = Prologue::Block::BlockPrologue<DispatchPolicyPrologue, BType, ScaleType, TileShapeL1>;

    using Kernel = Kernel::KernelMatmulAPrefetchBAntiquant<
        ProblemShape, BlockMmad, BlockPrologue, Scheduler::TileSchedulerTailResplitExpanded>;

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
                XWeightStride<
                    !TRANS_A, false, DTYPE_X, DTYPE_WEIGHT, antiquantType == QuantType::PER_GROUP>{}(
                    tilingDataIn.mSize, tilingDataIn.kSize)),
            .layoutC = AscendC::MakeLayout(
                AscendC::MakeShape(tilingDataIn.mSize, tilingDataIn.nSize),
                AscendC::MakeStride(tilingDataIn.nSize, ::Act::Gemm::_1{})),
            .layoutBias =
                AscendC::MakeLayout(AscendC::MakeShape(tilingDataIn.nSize), AscendC::MakeStride(::Act::Gemm::_1{}))};
        typename BlockPrologue::Arguments prologueArgs{
            .ptrB = weight,
            .ptrScale = antiquantScale,
            .ptrOffset = antiquantOffset,
            .layoutB = AscendC::MakeLayout(
                XWeightShapeObj<IS_WEIGHT_NZ>{}(tilingDataIn.nSize, tilingDataIn.kSize),
                XWeightStride<
                    TRANS_B, IS_WEIGHT_NZ, DTYPE_X, DTYPE_WEIGHT,
                    antiquantType == QuantType::PER_GROUP>{}(tilingDataIn.nSize, tilingDataIn.kSize)),
            .layoutScale = AscendC::MakeLayout(
                ScaleOffsetShapeObj<antiquantType>{}(
                    tilingDataIn.nSize, tilingDataIn.kSize, tilingDataIn.groupSize),
                ScaleOffsetStride<TRANS_B, antiquantType>{}(
                    tilingDataIn.nSize, tilingDataIn.kSize, tilingDataIn.groupSize)),
            .antiQuantGroupSize = tilingDataIn.groupSize};

        typename Scheduler::TileSchedulerTailResplitExpanded::Arguments schedulerArgs{};

        typename Kernel::Arguments args{
            .problemShape = problemShape, .mmad = mmadArgs, .prologue = prologueArgs, .scheduler = schedulerArgs};
        auto kernelParams = Kernel::ToUnderlyingArguments(args, &tilingDataIn);
        Kernel op;
        op(kernelParams);
    }
};

template <
    int TEMPLATE_CUSTOM, bool TRANS_A, bool TRANS_B, int ANTIQUANT_TYPE, bool HAS_ANTIQUANT_OFFSET,
    bool IS_BIAS_FP32, bool IS_WEIGHT_NZ>
DEVICE void InvokeActKernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    static constexpr QuantType antiquantType = static_cast<QuantType>(ANTIQUANT_TYPE);

    static constexpr uint32_t UB_MTE2_INNER_SIZE =
        TEMPLATE_CUSTOM == 0 || TEMPLATE_CUSTOM == 1 || TEMPLATE_CUSTOM == 4 ? 512 :
        TEMPLATE_CUSTOM == 2 || TEMPLATE_CUSTOM == 6                         ? 1024 :
                                                                               256;
    static constexpr bool B8 =
        IsSameType<DTYPE_WEIGHT, int8_t>::value || IsSameType<DTYPE_WEIGHT, float8_e5m2_t>::value ||
        IsSameType<DTYPE_WEIGHT, float8_e4m3_t>::value || IsSameType<DTYPE_WEIGHT, hifloat8_t>::value;
    static constexpr uint32_t UB_MTE2_BUF_NUM =
        TEMPLATE_CUSTOM == 0 || TEMPLATE_CUSTOM == 2 || (TEMPLATE_CUSTOM == 4 && B8) || TEMPLATE_CUSTOM == 5 ? 2 : 4;
    if constexpr (
        antiquantType == QuantType::PER_GROUP && (ORIG_DTYPE_X == DT_BF16 || ORIG_DTYPE_X == DT_FLOAT16) &&
        (ORIG_DTYPE_WEIGHT == DT_INT4)) {
        InvokeActA16W4PergroupKernel<
            UB_MTE2_INNER_SIZE, UB_MTE2_BUF_NUM, TRANS_A, TRANS_B, antiquantType, HAS_ANTIQUANT_OFFSET, IS_BIAS_FP32,
            IS_WEIGHT_NZ>(
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace, tiling);
    } else {
        ScmcKernel<
            UB_MTE2_INNER_SIZE, UB_MTE2_BUF_NUM, TRANS_A, TRANS_B, antiquantType, HAS_ANTIQUANT_OFFSET, IS_BIAS_FP32,
            IS_WEIGHT_NZ>{}(
            x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace, tiling);
    }
}
} // namespace WeightQuantBatchMatmulV2::Arch35::Act

#define KERNEL_PARAMS x, weight, antiquantScale, antiquantOffset, quantScale, quantOffset, bias, y, workspace, tiling

#endif