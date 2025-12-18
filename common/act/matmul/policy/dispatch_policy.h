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
 * \file dispatch_policy.h
 * \brief
 */
#ifndef MATMUL_POLICY_DISPATCH_POLICY_H
#define MATMUL_POLICY_DISPATCH_POLICY_H

#include "../../utils/integral_constant.h"
#include "../../utils/arch.h"

namespace Act {
namespace Gemm {
/* block schedule policies */
struct KernelMultiBlock {};            // Multi-block pipelined data transfer
struct KernelMmadPerBaseK {};          // Perform matrix multiplication with baseK granularity
struct KernelMmadWithScale {};         // Multi-block with scale
struct KernelMixWithWeightPrologue {}; // Mix with weight prologue

/**
 * @struct GMMPerTile
 * @brief Define a template struct GMMPerTile for configuring block matrix multiplication policies
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct GMMPerTile {
    using ScheduleType = KernelMmadPerBaseK;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

/**
 * @struct QuantMatmulMultiBlock
 * @brief Define a template struct QuantMatmulMultiBlock for multi-block matrix multiplication policies
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct QuantMatmulMultiBlock {
    using ScheduleType = KernelMmadWithScale;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

/**
 * @struct QuantMatmulWithTileMultiBlock
 * @brief Define a template struct QuantMatmulWithTileMultiBlock for multi-block matrix multiplication policies
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct QuantMatmulWithTileMultiBlock {
    using ScheduleType = KernelMmadWithScale;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

/**
 * @struct MatmulNaivePipelineWithLayout
 * @brief Structure for a naive matrix multiplication pipeline with layout
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulNaivePipelineWithLayout {
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;

    constexpr static int l0CStages = 1;
    constexpr static bool enalbeL0CUnitFlag = false;
};

/**
 * @struct MatmulWithScale
 * @brief Matrix multiplication with scaleA and scaleB
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>, uint64_t FULL_LOAD_MODE_ = 0>
struct MatmulWithScale {
    using ScheduleType = KernelMmadWithScale;
    using SingleShape = SingleCoreShape;
    constexpr static uint64_t fullLoadMode = FULL_LOAD_MODE_;
};

/**
 * @struct MatmulMultiBlockWithLayout
 * @brief Matrix multiplication multi-block layout structure, no bias, no quant
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulMultiBlockWithLayout {
    using ScheduleType = KernelMultiBlock;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

/**
 * @struct MatmulMultiBlockBiasWithLayout
 * @brief Matrix multiplication multi-block layout structure, support bias, no quant
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulMultiBlockBiasWithLayout {
    using ScheduleType = KernelMultiBlock;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

/**
 * @struct MatmulMultiBlockBiasWithLayout
 * @brief Matrix multiplication multi-block structure, no bias, no quant, implemented based on highlevel api
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulMultiBlock {
    using ScheduleType = KernelMultiBlock;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

/**
 * @struct MatmulMultiBlockWithOutQue
 * @brief Matrix multiplication multi-block structure, no quant, implemented based on Layout
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulMultiBlockWithOutQue {
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

/**
 * @struct MatmulIterBatch
 * @brief Matrix multiplication iteration batch processing structure, no quant, no bias, implemented base on layout
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulIterBatch {
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

/**
 * @struct MatmulMultiBlockBias
 * @brief Matrix multiplication multi-block structure, support bias, no quant, implemented based on highlevel api
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulMultiBlockBias {
    using ScheduleType = KernelMultiBlock;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

/**
 * @struct MatmulMultiBlockOnKAxisWithLayout
 * @brief Matrix multiplication multi-block structure, with K-axis caching, no quant, implemented based on Layout
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulMultiBlockOnKAxisWithLayout {
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

/**
 * @struct SparseMatmulMultiBlockOnKAxisWithLayout
 * @brief Sparse matrix multiplication multi-block structure, with K-axis caching, no quant, no bias, implemented based on Layout
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct SparseMatmulMultiBlockOnKAxisWithLayout {
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

/**
 * @struct MatmulL1Input
 * @brief Matrix multiplication L1 input structure, no bias, implemented based on highlevel api
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulL1Input {
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

/**
 * @struct MatmulL1InputBias
 * @brief Matrix multiplication L1 input bias structure, implemented based on highlevel api
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulL1InputBias {
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

/**
 * @struct MatmulL1InputWithLayout
 * @brief Matrix multiplication L1 input structure, no bias, implemented based on layout
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulL1InputWithLayout {
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

/**
 * @struct MatmulL1InputBiasWithLayout
 * @brief Matrix multiplication L1 input bias structure, implemented based on layout
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulL1InputBiasWithLayout {
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

/**
 * @struct MatmulL0COutputWithLayout
 * @brief Matrix multiplication L0C output structure, implemented based on layout
 * @param [in] SingleCoreShape: the shape of a single core, default is AscendC::Shape<_0, _0, _0, _0>
 */
template <class SingleCoreShape = AscendC::Shape<_0, _0, _0, _0>>
struct MatmulL0COutputWithLayout {
    using ScheduleType = KernelMultiBlock;
    using SingleShape = SingleCoreShape;
    constexpr static bool enableInputDataLenCheck = false;
};

// Antiquant in ub with single communication single computation
struct UbAntiquantWithScSc {
    using ScheduleType = KernelMixWithWeightPrologue;
    using ArchTag = Act::Gemm::Arch::Ascend910_95;
};
} // namespace Gemm
} // namespace Act
#endif
